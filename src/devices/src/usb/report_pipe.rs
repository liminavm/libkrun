// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A generic **HID report pipe** gadget: a full-speed HID device whose fixed-size IN/OUT
//! reports are shuttled verbatim over a caller-supplied channel, with zero knowledge of
//! what the frames mean. It is the reusable, upstream-shaped mechanism the emulated xHCI
//! controller drives; the *policy* (what a frame is, who produces it) lives in the caller.
//! limina wires it to the CTAPHID authenticator to present the Touch-ID FIDO key to a stock
//! guest (`crates/limina-vmm/src/fido_usb.rs`); a future fingerprint reader could reuse it.
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
        let descriptors = build_descriptors(vid, pid, &report_descriptor, strings);
        Arc::new(HidReportPipe {
            descriptors,
            report_descriptor,
            report_len,
            state: Mutex::new(PipeState::default()),
            out_sink,
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

/// Assemble the device/config/string descriptors for a two-endpoint (interrupt IN + OUT)
/// HID gadget. Generic plumbing: the caller supplies identity and the report descriptor.
fn build_descriptors(
    vid: u16,
    pid: u16,
    report_descriptor: &[u8],
    strings: [&str; 4],
) -> DeviceDescriptors {
    DeviceDescriptors {
        device: device_descriptor(vid, pid),
        configs: vec![config_descriptor(report_descriptor.len() as u16)],
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

/// config + HID interface + HID descriptor + interrupt-IN + interrupt-OUT endpoints.
/// wTotalLength = 9 + 9 + 9 + 7 + 7 = 41.
fn config_descriptor(report_len: u16) -> Vec<u8> {
    let total_len: u16 = 41;
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
    // Interface 0: HID, 2 endpoints.
    c.extend_from_slice(&[
        0x09, 0x04,      // INTERFACE
        0x00,      // bInterfaceNumber
        0x00,      // bAlternateSetting
        0x02,      // bNumEndpoints
        CLASS_HID, // bInterfaceClass = HID
        0x00,      // bInterfaceSubClass (no boot)
        0x00,      // bInterfaceProtocol
        0x04,      // iInterface
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
    // Interrupt-OUT endpoint (0x01), wMaxPacketSize 64, bInterval 5.
    c.extend_from_slice(&[0x07, 0x05, EP_OUT_ADDR, 0x03, 0x40, 0x00, 0x05]);
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
