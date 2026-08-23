// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A generic **HID report pipe** gadget: a full-speed HID device whose fixed-size IN/OUT
//! reports are shuttled verbatim over a caller-supplied channel, with zero knowledge of
//! what the frames mean. It is the reusable, upstream-shaped mechanism the emulated xHCI
//! controller drives; the *policy* (what a frame is, who produces it) lives in the caller.
//! limina wires it to the CTAPHID authenticator to present the Touch-ID FIDO key to a stock
//! guest (`crates/limina-vmm/src/fido_usb.rs`); a future fingerprint reader could reuse it.
//!
//! [`HidReportPipe::new_in_only`] builds the **input-device** shape: one interrupt-IN
//! endpoint and no interrupt-OUT (a keyboard's only host→device report is the LED byte,
//! which arrives as a SET_REPORT control — an OUT endpoint with no matching Output item in
//! the report descriptor is a descriptor mismatch). Input reports are also *perishable*: the
//! FIFO is capped and is emptied when a driver (re)reads the HID report descriptor, so
//! keystrokes produced while nothing was listening cannot replay into a live console the
//! instant the guest's driver binds. A data pipe (`new`) keeps the unbounded, never-dropped
//! FIFO its message framing needs.
//!
//! It is [`HidMockDevice`](super::HidMockDevice) generalised: instead of echoing an OUT
//! report back on the next IN, it forwards each guest→host frame (interrupt-OUT or a
//! SET_REPORT control) to the [`ReportSink`], and delivers host→guest frames pushed via
//! [`HidReportPipe::push_in`] by completing a held interrupt-IN transfer (xHCI's NAK
//! analogue). Frame ordering is preserved: a single held-IN slot drains a FIFO of queued
//! host→guest frames.
//!
//! **Held-transfer discipline (see `docs/design/usb-xhci.md` §3.3 and the Stage-C
//! INVARIANTS).** FIDO clients open/close hidraw repeatedly (fido2-token probes every
//! node), so Stop/Reset Endpoint churn — and thus stale held INs — is routine. The gadget
//! keeps **at most one** held IN (the HID-first "one outstanding interrupt-IN TRB per
//! device" scope): a new IN supersedes any previously held one, and `reset()` drops it.
//! The superseded transfer's completion is a no-op once the controller's generation guard
//! has invalidated it (a Stop/Reset Endpoint bumps the generation), so dropping it is
//! harmless — and it bounds the held set across open/close churn instead of leaking one
//! dead transfer per hidraw open.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use super::model::{
    ControlTransfer, DeviceDescriptors, EpAddr, Transfer, UsbDeviceModel, UsbSpeed,
};

// Descriptor types.
const DT_STRING: u8 = 0x03;
const DT_HID: u8 = 0x21;
const DT_HID_REPORT: u8 = 0x22;

// HID class / endpoints.
const CLASS_HID: u8 = 0x03;
const EP_IN_ADDR: u8 = 0x81; // interrupt IN, EP1 (DCI 3)
const EP_OUT_ADDR: u8 = 0x01; // interrupt OUT, EP1 (DCI 2)

/// FIFO cap for an input-device pipe (see [`HidReportPipe::new_in_only`]). Deep enough to
/// absorb a typing burst against the 5 ms interrupt interval, shallow enough that a backlog
/// built up while no driver was polling is bounded rather than replayed wholesale.
const INPUT_QUEUE_CAP: usize = 32;

/// A sink for guest→host report frames (delivered on the interrupt-OUT endpoint or via a
/// SET_REPORT control transfer). Invoked with the gadget's state lock released, from the
/// controller's worker thread. Frames are exactly `report_len` bytes.
pub type ReportSink = Arc<dyn Fn(Vec<u8>) + Send + Sync>;

/// Mutable pipe state: the FIFO of host→guest frames waiting for an IN, and the single held
/// interrupt-IN transfer (bounded — see the module docs).
#[derive(Default)]
struct PipeState {
    /// Host→guest frames received (via [`HidReportPipe::push_in`]) with no IN transfer ready
    /// to carry them; drained oldest-first onto the next IN. Preserves frame order.
    queued: VecDeque<Vec<u8>>,
    /// The one interrupt-IN transfer the gadget is holding (no frame ready when it arrived).
    held_in: Option<Transfer>,
}

/// A generic HID gadget that pipes fixed-size reports to/from a channel. Construct with
/// [`HidReportPipe::new`]; feed host→guest frames with [`HidReportPipe::push_in`]; receive
/// guest→host frames through the [`ReportSink`] passed at construction.
pub struct HidReportPipe {
    descriptors: DeviceDescriptors,
    report_descriptor: Vec<u8>,
    report_len: usize,
    state: Mutex<PipeState>,
    out_sink: ReportSink,
    /// Input-device semantics (see [`HidReportPipe::new_in_only`]): no interrupt-OUT
    /// endpoint, a capped FIFO, and a flush when a driver reads the report descriptor.
    input_device: bool,
}

impl HidReportPipe {
    /// Build a pipe gadget with the given USB identity and HID report descriptor.
    ///
    /// - `vid`/`pid`: the device's `idVendor`/`idProduct`.
    /// - `report_descriptor`: the HID report descriptor bytes (the caller's policy — e.g. the
    ///   FIDO usage-page-0xF1D0 descriptor). `report_len` is the fixed IN/OUT report size.
    /// - `strings`: `[manufacturer, product, serial, interface]` string descriptors.
    /// - `out_sink`: receives every guest→host frame.
    pub fn new(
        vid: u16,
        pid: u16,
        report_descriptor: Vec<u8>,
        report_len: usize,
        strings: [&str; 4],
        out_sink: ReportSink,
    ) -> Arc<HidReportPipe> {
        Self::build(
            vid,
            pid,
            report_descriptor,
            report_len,
            strings,
            out_sink,
            false,
        )
    }

    /// Build an **input-device** pipe: a single interrupt-IN endpoint (no interrupt-OUT),
    /// a capped host→guest FIFO, and a flush of that FIFO whenever the guest reads the HID
    /// report descriptor — the moment its driver binds. `out_sink` still receives host→device
    /// reports, which for an input device arrive only as SET_REPORT controls (a keyboard's
    /// LED byte). Everything else matches [`HidReportPipe::new`].
    pub fn new_in_only(
        vid: u16,
        pid: u16,
        report_descriptor: Vec<u8>,
        report_len: usize,
        strings: [&str; 4],
        out_sink: ReportSink,
    ) -> Arc<HidReportPipe> {
        Self::build(
            vid,
            pid,
            report_descriptor,
            report_len,
            strings,
            out_sink,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        vid: u16,
        pid: u16,
        report_descriptor: Vec<u8>,
        report_len: usize,
        strings: [&str; 4],
        out_sink: ReportSink,
        input_device: bool,
    ) -> Arc<HidReportPipe> {
        let descriptors = build_descriptors(vid, pid, &report_descriptor, strings, input_device);
        Arc::new(HidReportPipe {
            descriptors,
            report_descriptor,
            report_len,
            state: Mutex::new(PipeState::default()),
            out_sink,
            input_device,
        })
    }

    /// Deliver a host→guest frame: complete the held interrupt-IN transfer now if one is
    /// waiting, otherwise queue the frame (FIFO) for the next IN. Short frames are
    /// zero-padded and long frames truncated to `report_len`.
    pub fn push_in(&self, mut frame: Vec<u8>) {
        frame.truncate(self.report_len);
        frame.resize(self.report_len, 0);
        let mut st = self.state.lock().unwrap();
        if let Some(held) = st.held_in.take() {
            // Complete outside the state lock (the completion re-locks the controller).
            drop(st);
            held.complete_in(frame);
        } else {
            // An input device's reports are perishable: keep the newest, drop the oldest,
            // rather than growing a backlog nothing is draining.
            if self.input_device && st.queued.len() >= INPUT_QUEUE_CAP {
                st.queued.pop_front();
            }
            st.queued.push_back(frame);
        }
    }
}

impl UsbDeviceModel for HidReportPipe {
    fn descriptors(&self) -> DeviceDescriptors {
        self.descriptors.clone()
    }

    fn speed(&self) -> UsbSpeed {
        UsbSpeed::Full
    }

    fn handle_control(&self, xfer: ControlTransfer) {
        let s = *xfer.setup();
        // GET_DESCRIPTOR(HID report) — a standard request the controller forwards to us.
        if s.kind() == 0 && s.request == 0x06 && (s.value >> 8) as u8 == DT_HID_REPORT {
            // A driver is binding. Drop anything queued for the previous one: an input
            // report produced while nobody was listening is stale by the time it would be
            // delivered, and delivering it types into whatever now holds the console.
            // (The controller only calls `reset()` on a Reset Device command, which Linux
            // does not issue during a normal enumeration — this control is the binding
            // signal a gadget actually sees.)
            if self.input_device {
                self.state.lock().unwrap().queued.clear();
            }
            xfer.complete_in(self.report_descriptor.clone());
            return;
        }
        // HID class requests.
        if s.kind() == 1 {
            match s.request {
                0x09 => {
                    // SET_REPORT: an alternative guest→host path (some FIDO stacks use it
                    // instead of the interrupt-OUT endpoint). Forward the payload.
                    let frame = xfer.data_out().to_vec();
                    xfer.ack();
                    (self.out_sink)(frame);
                    return;
                }
                0x01 => {
                    // GET_REPORT: hand back a queued host→guest frame if any, else zeros.
                    let frame = { self.state.lock().unwrap().queued.pop_front() }
                        .unwrap_or_else(|| vec![0u8; self.report_len]);
                    xfer.complete_in(frame);
                    return;
                }
                // SET_IDLE / SET_PROTOCOL: accept benignly.
                0x0a | 0x0b => {
                    xfer.ack();
                    return;
                }
                0x02 => {
                    xfer.complete_in(vec![0]); // GET_IDLE
                    return;
                }
                0x03 => {
                    xfer.complete_in(vec![0]); // GET_PROTOCOL
                    return;
                }
                _ => {}
            }
        }
        xfer.stall();
    }

    fn handle_transfer(&self, ep: EpAddr, xfer: Transfer) {
        if ep.dir_in {
            // Interrupt IN: deliver a queued host→guest frame, else hold the transfer (at
            // most one — a new IN supersedes any stale prior hold; see the module docs).
            let mut st = self.state.lock().unwrap();
            if let Some(frame) = st.queued.pop_front() {
                drop(st);
                xfer.complete_in(frame);
            } else {
                let stale = st.held_in.replace(xfer);
                // Drop the superseded hold OUTSIDE the state lock: its Drop fires the
                // completion, which re-locks the controller (never the gadget). Once the
                // controller's generation guard has invalidated it, that completion is a
                // no-op; if somehow still live, it stalls cleanly (the guest re-submits).
                drop(st);
                drop(stale);
            }
        } else {
            // Interrupt OUT: the guest's report bytes → forward host-ward and ack the OUT.
            let frame = xfer.data_out().to_vec();
            xfer.ack();
            (self.out_sink)(frame);
        }
    }

    fn reset(&self) {
        let mut st = self.state.lock().unwrap();
        st.queued.clear();
        let stale = st.held_in.take();
        drop(st);
        drop(stale); // stall the held IN outside the lock (see handle_transfer)
    }
}

/// Assemble the device/config/string descriptors for a HID gadget — interrupt IN + OUT, or
/// interrupt IN alone when `in_only`. Generic plumbing: the caller supplies identity and the
/// report descriptor.
fn build_descriptors(
    vid: u16,
    pid: u16,
    report_descriptor: &[u8],
    strings: [&str; 4],
    in_only: bool,
) -> DeviceDescriptors {
    DeviceDescriptors {
        device: device_descriptor(vid, pid),
        configs: vec![config_descriptor(report_descriptor.len() as u16, in_only)],
        strings: vec![
            vec![0x04, DT_STRING, 0x09, 0x04], // index 0: LANGID 0x0409
            string_descriptor(strings[0]),     // iManufacturer
            string_descriptor(strings[1]),     // iProduct
            string_descriptor(strings[2]),     // iSerialNumber
            string_descriptor(strings[3]),     // iInterface
        ],
    }
}

/// The 18-byte device descriptor (class defined at the interface, EP0 max packet 64).
fn device_descriptor(vid: u16, pid: u16) -> Vec<u8> {
    vec![
        0x12, // bLength = 18
        0x01, // bDescriptorType = DEVICE
        0x00,
        0x02, // bcdUSB 2.00
        0x00, // bDeviceClass (per-interface)
        0x00, // bDeviceSubClass
        0x00, // bDeviceProtocol
        0x40, // bMaxPacketSize0 = 64
        (vid & 0xff) as u8,
        (vid >> 8) as u8,
        (pid & 0xff) as u8,
        (pid >> 8) as u8,
        0x00,
        0x01, // bcdDevice 1.00
        0x01, // iManufacturer
        0x02, // iProduct
        0x03, // iSerialNumber
        0x01, // bNumConfigurations
    ]
}

/// config + HID interface + HID descriptor + interrupt-IN (+ interrupt-OUT unless `in_only`).
/// wTotalLength = 9 + 9 + 9 + 7 (+ 7) = 34 or 41.
fn config_descriptor(report_len: u16, in_only: bool) -> Vec<u8> {
    let total_len: u16 = if in_only { 34 } else { 41 };
    let mut c = vec![
        0x09,
        0x02, // CONFIGURATION
        (total_len & 0xff) as u8,
        (total_len >> 8) as u8, // wTotalLength
        0x01,                   // bNumInterfaces
        0x01,                   // bConfigurationValue
        0x00,                   // iConfiguration
        0x80,                   // bmAttributes (bus-powered)
        0x32,                   // bMaxPower = 100 mA
    ];
    // Interface 0: HID, one or two endpoints. bInterfaceSubClass/bInterfaceProtocol stay
    // 0/0 (report-only, NOT the boot keyboard/mouse protocol): Linux's hid-generic binds on
    // the class alone, while EDK2's UsbKbDxe binds only boot-protocol keyboards — so a boot
    // gadget would be aggregated into EFI ConIn alongside VirtioKeyboardDxe and type every
    // pre-boot keystroke twice.
    c.extend_from_slice(&[
        0x09,
        0x04,                              // INTERFACE
        0x00,                              // bInterfaceNumber
        0x00,                              // bAlternateSetting
        if in_only { 0x01 } else { 0x02 }, // bNumEndpoints
        CLASS_HID,                         // bInterfaceClass = HID
        0x00,                              // bInterfaceSubClass (no boot)
        0x00,                              // bInterfaceProtocol
        0x04,                              // iInterface
    ]);
    // HID descriptor.
    c.extend_from_slice(&[
        0x09,
        DT_HID, // HID
        0x11,
        0x01,          // bcdHID 1.11
        0x00,          // bCountryCode
        0x01,          // bNumDescriptors
        DT_HID_REPORT, // bDescriptorType (report)
        (report_len & 0xff) as u8,
        (report_len >> 8) as u8, // wDescriptorLength
    ]);
    // Interrupt-IN endpoint (0x81), wMaxPacketSize 64, bInterval 5.
    c.extend_from_slice(&[0x07, 0x05, EP_IN_ADDR, 0x03, 0x40, 0x00, 0x05]);
    // Interrupt-OUT endpoint (0x01), wMaxPacketSize 64, bInterval 5 — absent on an input
    // device, whose only host→device report is a SET_REPORT control.
    if !in_only {
        c.extend_from_slice(&[0x07, 0x05, EP_OUT_ADDR, 0x03, 0x40, 0x00, 0x05]);
    }
    debug_assert_eq!(c.len(), total_len as usize);
    c
}

fn string_descriptor(s: &str) -> Vec<u8> {
    let utf16: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let mut d = vec![(2 + utf16.len()) as u8, DT_STRING];
    d.extend_from_slice(&utf16);
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usb::model::{Completion, SetupPacket, XferOutcome};
    use std::sync::mpsc;

    const REPORT_LEN: usize = 64;

    fn pipe() -> (Arc<HidReportPipe>, mpsc::Receiver<Vec<u8>>) {
        let (tx, rx) = mpsc::channel();
        let sink: ReportSink = Arc::new(move |f: Vec<u8>| tx.send(f).unwrap());
        let rd = vec![0x06, 0xd0, 0xf1, 0x09, 0x01, 0xa1, 0x01, 0xc0];
        (
            HidReportPipe::new(
                0x1d6b,
                0x0f1d,
                rd,
                REPORT_LEN,
                ["limina", "pipe", "SN", "iface"],
                sink,
            ),
            rx,
        )
    }

    const IN_REPORT_LEN: usize = 8;

    /// An input-device pipe (one interrupt-IN endpoint), the keyboard shape.
    fn in_pipe() -> (Arc<HidReportPipe>, mpsc::Receiver<Vec<u8>>) {
        let (tx, rx) = mpsc::channel();
        let sink: ReportSink = Arc::new(move |f: Vec<u8>| tx.send(f).unwrap());
        let rd = vec![0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0xc0];
        (
            HidReportPipe::new_in_only(
                0x1d6b,
                0x0f1e,
                rd,
                IN_REPORT_LEN,
                ["limina", "kbd", "SN", "iface"],
                sink,
            ),
            rx,
        )
    }

    /// GET_DESCRIPTOR(HID report) — what a guest driver issues as it binds.
    fn read_report_descriptor(p: &HidReportPipe) {
        let (tx, _rx) = mpsc::channel();
        let c = Completion::new(move |o| {
            let _ = tx.send(o);
        });
        p.handle_control(ControlTransfer::new(
            SetupPacket::from_bytes([0x81, 0x06, 0x00, 0x22, 0x00, 0x00, 0xff, 0x00]),
            Vec::new(),
            c,
        ));
    }

    fn in_transfer(len: usize) -> (Transfer, mpsc::Receiver<XferOutcome>) {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        (Transfer::new(Vec::new(), len, c), rx)
    }

    fn out_transfer(data: Vec<u8>) -> (Transfer, mpsc::Receiver<XferOutcome>) {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        (Transfer::new(data, 0, c), rx)
    }

    #[test]
    fn descriptors_carry_identity_and_two_endpoints() {
        let (p, _rx) = pipe();
        let d = p.descriptors();
        assert_eq!(d.device.len(), 18);
        assert_eq!(u16::from_le_bytes([d.device[8], d.device[9]]), 0x1d6b);
        assert_eq!(u16::from_le_bytes([d.device[10], d.device[11]]), 0x0f1d);
        let c = &d.configs[0];
        assert_eq!(u16::from_le_bytes([c[2], c[3]]) as usize, c.len());
        assert_eq!(c[9 + 4], 0x02, "two endpoints");
        assert_eq!(c[9 + 5], CLASS_HID, "HID class");
    }

    #[test]
    fn out_report_reaches_the_sink() {
        let (p, sink_rx) = pipe();
        let mut report = vec![0u8; REPORT_LEN];
        report[0] = 0xAB;
        let (out, out_rx) = out_transfer(report.clone());
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: false,
            },
            out,
        );
        assert!(matches!(out_rx.recv().unwrap(), XferOutcome::Ack));
        assert_eq!(sink_rx.recv().unwrap(), report);
    }

    #[test]
    fn set_report_control_reaches_the_sink() {
        let (p, sink_rx) = pipe();
        let mut report = vec![0u8; REPORT_LEN];
        report[1] = 0x99;
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        let xfer = ControlTransfer::new(
            SetupPacket::from_bytes([0x21, 0x09, 0x00, 0x02, 0x00, 0x00, 0x40, 0x00]),
            report.clone(),
            c,
        );
        p.handle_control(xfer);
        assert!(
            matches!(rx.recv().unwrap(), XferOutcome::Ack),
            "SET_REPORT acked"
        );
        assert_eq!(sink_rx.recv().unwrap(), report);
    }

    #[test]
    fn held_in_completes_when_a_frame_is_pushed() {
        let (p, _rx) = pipe();
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        assert!(in_rx.try_recv().is_err(), "IN held, not completed");
        let mut frame = vec![0u8; REPORT_LEN];
        frame[0] = 0xCC;
        p.push_in(frame.clone());
        match in_rx.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, frame),
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn queued_frame_drains_to_a_later_in() {
        let (p, _rx) = pipe();
        let mut frame = vec![0u8; REPORT_LEN];
        frame[7] = 0x5A;
        p.push_in(frame.clone());
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        match in_rx.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, frame),
            other => panic!("expected In, got {other:?}"),
        }
    }

    /// Repeated INs with nothing to deliver never accumulate more than one held transfer:
    /// each new IN supersedes (and stalls) the prior — the open/close-churn guard.
    #[test]
    fn a_new_in_supersedes_a_stale_held_in() {
        let (p, _rx) = pipe();
        let (in1, rx1) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in1,
        );
        // A second IN arrives before any frame — the first is superseded and stalled.
        let (in2, rx2) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in2,
        );
        assert!(
            matches!(rx1.recv().unwrap(), XferOutcome::Stall),
            "stale IN stalled"
        );
        // The frame goes to the live (second) IN, in order.
        let frame = vec![0x11u8; REPORT_LEN];
        p.push_in(frame.clone());
        match rx2.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, frame),
            other => panic!("expected In, got {other:?}"),
        }
    }

    /// The input-device shape: one interrupt-IN endpoint, no interrupt-OUT (its host→device
    /// report is a SET_REPORT control), and a config block whose wTotalLength matches.
    #[test]
    fn an_input_pipe_exposes_a_single_in_endpoint() {
        let (p, _rx) = in_pipe();
        let c = &p.descriptors().configs[0];
        assert_eq!(u16::from_le_bytes([c[2], c[3]]) as usize, c.len());
        assert_eq!(c.len(), 34, "config + interface + HID + one endpoint");
        assert_eq!(c[9 + 4], 0x01, "one endpoint");
        assert_eq!(c[9 + 5], CLASS_HID, "HID class");
        assert_eq!(
            c[9 + 6],
            0x00,
            "no boot subclass (EFI ConIn must not bind it)"
        );
        assert_eq!(c[9 + 7], 0x00, "no boot protocol");
        // The only endpoint descriptor present is the IN one.
        let eps: Vec<u8> = c[9 + 9 + 9..]
            .chunks(7)
            .map(|e| e[2]) // bEndpointAddress
            .collect();
        assert_eq!(eps, vec![EP_IN_ADDR]);
    }

    /// A keyboard's reports are perishable: with nothing draining the FIFO it keeps the most
    /// recent [`INPUT_QUEUE_CAP`] and drops the oldest, instead of growing without bound.
    /// A data pipe (`new`) keeps everything — a CTAPHID response is many framed packets and
    /// dropping one corrupts the message.
    #[test]
    fn an_input_pipe_caps_its_backlog_and_a_data_pipe_does_not() {
        let (p, _rx) = in_pipe();
        for i in 0..(INPUT_QUEUE_CAP + 5) {
            p.push_in(vec![i as u8; IN_REPORT_LEN]);
        }
        let st = p.state.lock().unwrap();
        assert_eq!(st.queued.len(), INPUT_QUEUE_CAP);
        assert_eq!(st.queued[0][0], 5, "oldest dropped, newest kept");
        drop(st);

        let (d, _rx) = pipe();
        for i in 0..(INPUT_QUEUE_CAP + 5) {
            d.push_in(vec![i as u8; REPORT_LEN]);
        }
        assert_eq!(d.state.lock().unwrap().queued.len(), INPUT_QUEUE_CAP + 5);
    }

    /// The replay guard. Keys pressed before any driver was listening must not be delivered
    /// to the driver that finally binds — it would type them into whatever now owns the
    /// console (a LUKS passphrase prompt). The report-descriptor read IS that bind moment:
    /// the controller only calls `reset()` for a Reset Device command, which Linux does not
    /// issue during a normal enumeration.
    #[test]
    fn reading_the_report_descriptor_flushes_a_stale_input_backlog() {
        let (p, _rx) = in_pipe();
        p.push_in(vec![0x04; IN_REPORT_LEN]); // typed while nothing was bound
        assert_eq!(p.state.lock().unwrap().queued.len(), 1);
        read_report_descriptor(&p);
        assert!(
            p.state.lock().unwrap().queued.is_empty(),
            "stale input reports dropped when the driver bound"
        );

        // A data pipe's frames are message-framed, not perishable: the same read keeps them.
        let (d, _rx) = pipe();
        d.push_in(vec![0x04; REPORT_LEN]);
        read_report_descriptor(&d);
        assert_eq!(d.state.lock().unwrap().queued.len(), 1);
    }

    #[test]
    fn reset_drops_held_in_and_queue() {
        let (p, _rx) = pipe();
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        p.push_in(vec![1u8; REPORT_LEN]); // completes the held IN
        assert!(matches!(in_rx.recv().unwrap(), XferOutcome::In(_)));
        // Now hold another and reset: it stalls, and any queue is cleared.
        let (in2, rx2) = in_transfer(REPORT_LEN);
        p.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in2,
        );
        p.reset();
        assert!(
            matches!(rx2.recv().unwrap(), XferOutcome::Stall),
            "reset stalls held IN"
        );
    }
}
