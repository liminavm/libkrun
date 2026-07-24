// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The `UsbDeviceModel` trait: the mechanism/policy seam between the emulated
//! xHCI controller (in libkrun) and the software-defined gadgets limina drives
//! (FIDO HID, fingerprint reader). See `docs/design/usb-xhci.md` §3.3.
//!
//! The controller answers the *standard* enumeration requests itself from the
//! descriptors the model hands it (device / config / string). Class- and
//! vendor-specific control requests — and, in Stage B2, interrupt/bulk data — are
//! forwarded to the model. The one semantic that matters most is **deferred
//! completion**: a [`ControlTransfer`] (or [`Transfer`]) carries a completion
//! handle that may be invoked from *any* thread at *any* later time, so a gadget
//! that blocks (the FIDO gadget waits on a Touch ID prompt for seconds) never
//! stalls the controller's worker or the guest's vcpu threads.

use std::sync::Mutex;

/// USB device speed, encoded into PORTSC and the slot context at enumeration.
/// B1 only exercises full-speed (the FIDO gadget is a full-speed HID key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsbSpeed {
    Low,
    Full,
    High,
}

/// A parsed USB SETUP packet (the 8 bytes of a control transfer's Setup Stage).
#[derive(Clone, Copy, Debug)]
pub struct SetupPacket {
    pub request_type: u8,
    pub request: u8,
    pub value: u16,
    pub index: u16,
    pub length: u16,
}

impl SetupPacket {
    /// Parse the 8 wire bytes (all little-endian multi-byte fields).
    pub fn from_bytes(b: [u8; 8]) -> Self {
        SetupPacket {
            request_type: b[0],
            request: b[1],
            value: u16::from_le_bytes([b[2], b[3]]),
            index: u16::from_le_bytes([b[4], b[5]]),
            length: u16::from_le_bytes([b[6], b[7]]),
        }
    }

    /// Data-stage direction: true = device-to-host (IN).
    pub fn is_in(&self) -> bool {
        self.request_type & 0x80 != 0
    }

    /// Request kind: 0 = standard, 1 = class, 2 = vendor, 3 = reserved.
    pub fn kind(&self) -> u8 {
        (self.request_type >> 5) & 0x3
    }

    /// Recipient: 0 = device, 1 = interface, 2 = endpoint, 3 = other.
    pub fn recipient(&self) -> u8 {
        self.request_type & 0x1f
    }
}

/// The outcome the controller turns into a Transfer Event once a control
/// transfer completes (possibly on another thread).
#[derive(Clone, Debug)]
pub enum XferOutcome {
    /// Device-to-host data (IN). Written to the guest's data-stage buffers,
    /// truncated to the buffer length; the residue is reported in the event.
    In(Vec<u8>),
    /// A host-to-device / no-data transfer completed successfully (ACK).
    Ack,
    /// The endpoint stalled (the request is unsupported): a Stall Error event.
    Stall,
}

/// A one-shot, thread-safe completion sink. The controller builds it with a
/// closure that posts the Transfer Event(s) + interrupt; the gadget invokes it
/// exactly once (now or later). Dropping it without completing stalls the
/// transfer, so a gadget can never wedge the guest by forgetting a request.
/// A boxed one-shot completion callback (posts the Transfer Event + interrupt).
type CompletionFn = Box<dyn FnOnce(XferOutcome) + Send>;

pub struct Completion {
    inner: Mutex<Option<CompletionFn>>,
}

impl Completion {
    pub(crate) fn new(f: impl FnOnce(XferOutcome) + Send + 'static) -> Self {
        Completion {
            inner: Mutex::new(Some(Box::new(f))),
        }
    }

    fn fire(&self, outcome: XferOutcome) {
        // take() makes this idempotent: a second call is a no-op, and Drop won't
        // re-fire after an explicit completion.
        let f = self.inner.lock().unwrap().take();
        if let Some(f) = f {
            f(outcome);
        }
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        // A gadget that drops the transfer without completing it stalls the
        // endpoint rather than leaving the guest waiting forever.
        let f = self.inner.lock().unwrap().take();
        if let Some(f) = f {
            f(XferOutcome::Stall);
        }
    }
}

/// A control (EP0) transfer handed to a gadget for a class/vendor request the
/// controller doesn't answer itself. Completion may be deferred to any thread.
pub struct ControlTransfer {
    setup: SetupPacket,
    /// For an OUT data stage: the bytes the host sent (already read from guest
    /// memory). Empty for IN or no-data transfers.
    data_out: Vec<u8>,
    completion: Completion,
}

impl ControlTransfer {
    pub(crate) fn new(setup: SetupPacket, data_out: Vec<u8>, completion: Completion) -> Self {
        ControlTransfer {
            setup,
            data_out,
            completion,
        }
    }

    pub fn setup(&self) -> &SetupPacket {
        &self.setup
    }

    /// The host→device payload for an OUT transfer's data stage.
    pub fn data_out(&self) -> &[u8] {
        &self.data_out
    }

    /// Complete an IN transfer, supplying the device→host bytes.
    pub fn complete_in(self, data: Vec<u8>) {
        self.completion.fire(XferOutcome::In(data));
    }

    /// Complete a no-data / OUT transfer successfully.
    pub fn ack(self) {
        self.completion.fire(XferOutcome::Ack);
    }

    /// Stall the endpoint (unsupported request).
    pub fn stall(self) {
        self.completion.fire(XferOutcome::Stall);
    }
}

/// An endpoint address (`bEndpointAddress`): endpoint number + direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpAddr {
    pub num: u8,
    pub dir_in: bool,
}

/// A non-control (interrupt/bulk) transfer handed to a gadget. An **IN** transfer
/// with no data ready is simply *held* (the gadget keeps this object; the TRBs stay
/// outstanding on the ring) until data arrives — xHCI's natural NAK analogue and the
/// FIDO/HID gadget's core need. An **OUT** transfer carries the guest's bytes in
/// `data_out`. Completion may be deferred to any thread (see [`Completion`]).
pub struct Transfer {
    /// For an OUT transfer: the host→device bytes. Empty for IN.
    data_out: Vec<u8>,
    /// For an IN transfer: how many bytes the guest is willing to receive.
    in_len: usize,
    completion: Completion,
}

impl Transfer {
    pub(crate) fn new(data_out: Vec<u8>, in_len: usize, completion: Completion) -> Self {
        Transfer {
            data_out,
            in_len,
            completion,
        }
    }

    pub fn data_out(&self) -> &[u8] {
        &self.data_out
    }

    pub fn in_len(&self) -> usize {
        self.in_len
    }

    pub fn complete_in(self, data: Vec<u8>) {
        self.completion.fire(XferOutcome::In(data));
    }

    pub fn ack(self) {
        self.completion.fire(XferOutcome::Ack);
    }

    pub fn stall(self) {
        self.completion.fire(XferOutcome::Stall);
    }
}

/// The descriptors a gadget exposes. The controller serves GET_DESCRIPTOR from
/// these, so a gadget never reimplements standard enumeration. `config` blocks
/// are the full concatenated descriptor set (config + interfaces + endpoints).
#[derive(Clone, Debug, Default)]
pub struct DeviceDescriptors {
    /// The 18-byte device descriptor.
    pub device: Vec<u8>,
    /// One or more configuration blocks (config + interface + endpoint descriptors).
    pub configs: Vec<Vec<u8>>,
    /// String descriptors indexed by descriptor index; index 0 is the LANGID list.
    pub strings: Vec<Vec<u8>>,
}

/// A software-defined USB device driven by the emulated controller. All methods
/// take `&self`: the model is shared (`Arc`) across the vcpu and worker threads
/// and completes transfers through the thread-safe [`Completion`] handles, so it
/// must be `Send + Sync` and internally synchronize any mutable state.
pub trait UsbDeviceModel: Send + Sync {
    /// The device / configuration / string descriptors (used by the controller
    /// to answer standard enumeration requests).
    fn descriptors(&self) -> DeviceDescriptors;

    /// The device's attachment speed (encoded into PORTSC + the slot context).
    fn speed(&self) -> UsbSpeed {
        UsbSpeed::Full
    }

    /// Handle a class- or vendor-specific EP0 control request. The default
    /// stalls (unsupported); gadgets override to answer their own requests.
    fn handle_control(&self, xfer: ControlTransfer) {
        xfer.stall();
    }

    /// Handle a non-EP0 (interrupt/bulk) transfer. B1 never calls this for the
    /// mock (zero non-EP0 endpoints); B2 gadgets hold the transfer until data is
    /// ready. The default stalls.
    fn handle_transfer(&self, _ep: EpAddr, xfer: Transfer) {
        xfer.stall();
    }

    /// Reset device state (Reset Device command / port reset). Default: no-op.
    fn reset(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn setup_packet_parses_little_endian_fields() {
        // GET_DESCRIPTOR(device), wLength = 0x0040.
        let s = SetupPacket::from_bytes([0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x40, 0x00]);
        assert!(s.is_in());
        assert_eq!(s.kind(), 0, "standard request");
        assert_eq!(s.recipient(), 0, "device recipient");
        assert_eq!(s.value, 0x0100);
        assert_eq!(s.length, 0x0040);
    }

    #[test]
    fn completion_fires_exactly_once() {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        c.fire(XferOutcome::Ack);
        c.fire(XferOutcome::Stall); // ignored — already fired
        assert!(matches!(rx.recv().unwrap(), XferOutcome::Ack));
        assert!(rx.try_recv().is_err(), "only one outcome delivered");
    }

    #[test]
    fn dropping_a_transfer_stalls_it() {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        let xfer = ControlTransfer::new(
            SetupPacket::from_bytes([0x40, 0x01, 0, 0, 0, 0, 0, 0]),
            Vec::new(),
            c,
        );
        drop(xfer);
        assert!(matches!(rx.recv().unwrap(), XferOutcome::Stall));
    }
}
