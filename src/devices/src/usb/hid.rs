// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A mock full-speed **HID** gadget that exercises the emulated xHCI controller's
//! non-EP0 data path end-to-end (Stage B2). It deliberately uses the same shape the
//! shipped FIDO gadget will (usage page 0xF1D0, 64-byte IN/OUT reports), so the
//! machinery this proves — held IN transfers, deferred completion from *another* code
//! path, and OUT delivery — is exactly what FIDO needs.
//!
//! Behaviour is a simple **echo**: each 64-byte OUT report (interrupt-OUT endpoint or a
//! SET_REPORT control) is returned on the next interrupt-IN transfer. With nothing
//! queued, an interrupt-IN transfer is *held* (its [`Transfer`] kept here, its TRBs left
//! outstanding on the ring) until an OUT report arrives to complete it. That single
//! round-trip covers a held IN, its completion fired from the OUT path, and both
//! endpoint directions — the guest sees a working `/dev/hidrawN`.
//!
//! Distinct identity (`0x1d6b:0x0f11`) from the enumeration-only [`MockUsbDevice`], which
//! stays as-is for the unit tests.

use std::collections::VecDeque;
use std::sync::Mutex;

use super::model::{
    ControlTransfer, DeviceDescriptors, EpAddr, Transfer, UsbDeviceModel, UsbSpeed,
};

const VID: u16 = 0x1d6b;
const PID: u16 = 0x0f11;

/// The IN/OUT report size (bytes). Matches the FIDO CTAPHID frame size.
pub const REPORT_LEN: usize = 64;

// Descriptor types.
const DT_STRING: u8 = 0x03;
const DT_HID: u8 = 0x21;
const DT_HID_REPORT: u8 = 0x22;

// HID class / endpoints.
const CLASS_HID: u8 = 0x03;
const EP_IN_ADDR: u8 = 0x81; // interrupt IN, EP1 (DCI 3)
const EP_OUT_ADDR: u8 = 0x01; // interrupt OUT, EP1 (DCI 2)

/// The 18-byte device descriptor (class defined at the interface).
fn device_descriptor() -> Vec<u8> {
    vec![
        0x12, // bLength = 18
        0x01, // bDescriptorType = DEVICE
        0x00, 0x02, // bcdUSB 2.00
        0x00, // bDeviceClass (per-interface)
        0x00, // bDeviceSubClass
        0x00, // bDeviceProtocol
        0x40, // bMaxPacketSize0 = 64
        (VID & 0xff) as u8,
        (VID >> 8) as u8,
        (PID & 0xff) as u8,
        (PID >> 8) as u8,
        0x00, 0x01, // bcdDevice 1.00
        0x01, // iManufacturer
        0x02, // iProduct
        0x03, // iSerialNumber
        0x01, // bNumConfigurations
    ]
}

/// The report descriptor: a vendor-defined usage page (0xF1D0, the FIDO page) with a
/// 64-byte input report and a 64-byte output report, no report IDs.
fn report_descriptor() -> Vec<u8> {
    vec![
        0x06, 0xd0, 0xf1, // Usage Page (0xF1D0, vendor-defined)
        0x09, 0x01, //       Usage (0x01)
        0xa1, 0x01, //       Collection (Application)
        0x09, 0x20, //         Usage (0x20, data in)
        0x15, 0x00, //         Logical Minimum (0)
        0x26, 0xff, 0x00, //   Logical Maximum (255)
        0x75, 0x08, //         Report Size (8)
        0x95, REPORT_LEN as u8, // Report Count (64)
        0x81, 0x02, //         Input (Data, Var, Abs)
        0x09, 0x21, //         Usage (0x21, data out)
        0x15, 0x00, //         Logical Minimum (0)
        0x26, 0xff, 0x00, //   Logical Maximum (255)
        0x75, 0x08, //         Report Size (8)
        0x95, REPORT_LEN as u8, // Report Count (64)
        0x91, 0x02, //         Output (Data, Var, Abs)
        0xc0, //             End Collection
    ]
}

/// The configuration block: config + HID interface + HID descriptor + interrupt-IN and
/// interrupt-OUT endpoints. wTotalLength = 9 + 9 + 9 + 7 + 7 = 41.
fn config_descriptor() -> Vec<u8> {
    let report_len = report_descriptor().len() as u16;
    let total_len: u16 = 41;
    let mut c = vec![
        0x09, 0x02, // CONFIGURATION
        (total_len & 0xff) as u8,
        (total_len >> 8) as u8, // wTotalLength
        0x01, // bNumInterfaces
        0x01, // bConfigurationValue
        0x00, // iConfiguration
        0x80, // bmAttributes (bus-powered)
        0x32, // bMaxPower = 100 mA
    ];
    // Interface 0: HID, 2 endpoints.
    c.extend_from_slice(&[
        0x09, 0x04, // INTERFACE
        0x00, // bInterfaceNumber
        0x00, // bAlternateSetting
        0x02, // bNumEndpoints
        CLASS_HID, // bInterfaceClass = HID
        0x00, // bInterfaceSubClass (no boot)
        0x00, // bInterfaceProtocol
        0x04, // iInterface
    ]);
    // HID descriptor.
    c.extend_from_slice(&[
        0x09, DT_HID, // HID
        0x11, 0x01, // bcdHID 1.11
        0x00, // bCountryCode
        0x01, // bNumDescriptors
        DT_HID_REPORT, // bDescriptorType (report)
        (report_len & 0xff) as u8,
        (report_len >> 8) as u8, // wDescriptorLength
    ]);
    // Interrupt-IN endpoint (0x81), wMaxPacketSize 64, bInterval 5.
    c.extend_from_slice(&[
        0x07, 0x05, // ENDPOINT
        EP_IN_ADDR, // bEndpointAddress
        0x03, // bmAttributes = interrupt
        REPORT_LEN as u8, 0x00, // wMaxPacketSize = 64
        0x05, // bInterval
    ]);
    // Interrupt-OUT endpoint (0x01), wMaxPacketSize 64, bInterval 5.
    c.extend_from_slice(&[
        0x07, 0x05, // ENDPOINT
        EP_OUT_ADDR, // bEndpointAddress
        0x03, // bmAttributes = interrupt
        REPORT_LEN as u8, 0x00, // wMaxPacketSize = 64
        0x05, // bInterval
    ]);
    debug_assert_eq!(c.len(), total_len as usize);
    c
}

fn string_descriptor(s: &str) -> Vec<u8> {
    let utf16: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let mut d = vec![(2 + utf16.len()) as u8, DT_STRING];
    d.extend_from_slice(&utf16);
    d
}

/// Mutable gadget state: at most a queue of pending outbound reports and a queue of
/// held IN transfers, protected by one mutex.
#[derive(Default)]
struct HidState {
    /// Reports received on OUT / SET_REPORT, awaiting an IN transfer to carry them back.
    queued: VecDeque<Vec<u8>>,
    /// Interrupt-IN transfers the gadget is holding (no report ready when they arrived).
    held_in: VecDeque<Transfer>,
}

/// The mock full-speed HID echo device.
pub struct HidMockDevice {
    state: Mutex<HidState>,
}

impl HidMockDevice {
    pub fn new() -> Self {
        HidMockDevice {
            state: Mutex::new(HidState::default()),
        }
    }

    /// Accept a report from an OUT path (interrupt-OUT or SET_REPORT): if an IN transfer
    /// is held, complete it now; otherwise queue the bytes for the next IN. Returns the
    /// IN transfer to complete (outside the state lock), if any.
    fn accept_report(&self, mut report: Vec<u8>) {
        report.truncate(REPORT_LEN);
        report.resize(REPORT_LEN, 0);
        let pending_in = {
            let mut st = self.state.lock().unwrap();
            match st.held_in.pop_front() {
                Some(xfer) => Some(xfer),
                None => {
                    st.queued.push_back(report.clone());
                    None
                }
            }
        };
        // Complete the held IN outside the state lock (its completion re-locks the
        // controller, never this gadget — no lock-order cycle, but keep it narrow).
        if let Some(xfer) = pending_in {
            xfer.complete_in(report);
        }
    }
}

impl Default for HidMockDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl UsbDeviceModel for HidMockDevice {
    fn descriptors(&self) -> DeviceDescriptors {
        DeviceDescriptors {
            device: device_descriptor(),
            configs: vec![config_descriptor()],
            strings: vec![
                vec![0x04, DT_STRING, 0x09, 0x04], // LANGID 0x0409
                string_descriptor("limina"),
                string_descriptor("limina Mock HID Device"),
                string_descriptor("LIMINA-HID-0001"),
                string_descriptor("limina hid echo interface"),
            ],
        }
    }

    fn speed(&self) -> UsbSpeed {
        UsbSpeed::Full
    }

    fn handle_control(&self, xfer: ControlTransfer) {
        let s = *xfer.setup();
        // GET_DESCRIPTOR(HID report) — standard request the controller forwards to us.
        if s.kind() == 0 && s.request == 0x06 && (s.value >> 8) as u8 == DT_HID_REPORT {
            xfer.complete_in(report_descriptor());
            return;
        }
        // HID class requests.
        if s.kind() == 1 {
            match s.request {
                0x09 => {
                    // SET_REPORT: the report bytes arrive in the data-out stage. Echo them.
                    let report = xfer.data_out().to_vec();
                    xfer.ack();
                    self.accept_report(report);
                    return;
                }
                0x01 => {
                    // GET_REPORT: return a queued report if any, else zeros.
                    let report = {
                        let mut st = self.state.lock().unwrap();
                        st.queued.pop_front()
                    }
                    .unwrap_or_else(|| vec![0u8; REPORT_LEN]);
                    xfer.complete_in(report);
                    return;
                }
                // SET_IDLE / SET_PROTOCOL / GET_IDLE / GET_PROTOCOL: accept benignly.
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
            // Interrupt IN: deliver a queued report, else hold the transfer.
            let report = {
                let mut st = self.state.lock().unwrap();
                st.queued.pop_front()
            };
            match report {
                Some(report) => xfer.complete_in(report),
                None => self.state.lock().unwrap().held_in.push_back(xfer),
            }
        } else {
            // Interrupt OUT: the guest's report bytes; echo them and ack the OUT.
            let report = xfer.data_out().to_vec();
            xfer.ack();
            self.accept_report(report);
        }
    }

    fn reset(&self) {
        // Drop any held IN transfers (their Drop stalls them, harmlessly — the
        // controller's generation guard drops the completion for a torn-down endpoint)
        // and clear the queue.
        let mut st = self.state.lock().unwrap();
        st.queued.clear();
        st.held_in.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usb::model::{Completion, XferOutcome};
    use std::sync::mpsc;

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
    fn descriptors_are_well_formed() {
        let d = HidMockDevice::new().descriptors();
        assert_eq!(d.device.len(), 18);
        assert_eq!(u16::from_le_bytes([d.device[8], d.device[9]]), VID);
        assert_eq!(u16::from_le_bytes([d.device[10], d.device[11]]), PID);
        let c = &d.configs[0];
        // wTotalLength matches the concatenated block.
        assert_eq!(u16::from_le_bytes([c[2], c[3]]) as usize, c.len());
        // Interface advertises 2 endpoints and HID class.
        assert_eq!(c[9 + 4], 0x02, "two endpoints");
        assert_eq!(c[9 + 5], CLASS_HID, "HID class");
    }

    #[test]
    fn held_in_completes_when_an_out_report_arrives() {
        let dev = HidMockDevice::new();
        // An IN transfer arrives with nothing queued -> held (no completion yet).
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        dev.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        assert!(in_rx.try_recv().is_err(), "IN is held, not completed");

        // An OUT report arrives -> completes the held IN with those bytes, and acks OUT.
        let mut report = vec![0u8; REPORT_LEN];
        report[0] = 0xAB;
        report[63] = 0xCD;
        let (out_xfer, out_rx) = out_transfer(report.clone());
        dev.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: false,
            },
            out_xfer,
        );
        assert!(matches!(out_rx.recv().unwrap(), XferOutcome::Ack), "OUT acked");
        match in_rx.recv().unwrap() {
            XferOutcome::In(bytes) => {
                assert_eq!(bytes, report, "held IN echoes the OUT report");
            }
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn queued_out_report_is_delivered_to_a_later_in() {
        let dev = HidMockDevice::new();
        // OUT arrives first (no IN held) -> queued.
        let mut report = vec![0u8; REPORT_LEN];
        report[7] = 0x5A;
        let (out_xfer, out_rx) = out_transfer(report.clone());
        dev.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: false,
            },
            out_xfer,
        );
        assert!(matches!(out_rx.recv().unwrap(), XferOutcome::Ack));
        // A later IN drains the queued report immediately.
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        dev.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        match in_rx.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, report),
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn get_descriptor_report_is_served() {
        let dev = HidMockDevice::new();
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        // GET_DESCRIPTOR (report), recipient interface.
        let xfer = ControlTransfer::new(
            crate::usb::model::SetupPacket::from_bytes([0x81, 0x06, 0x00, 0x22, 0x00, 0x00, 0x00, 0x01]),
            Vec::new(),
            c,
        );
        dev.handle_control(xfer);
        match rx.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, report_descriptor()),
            other => panic!("expected report descriptor, got {other:?}"),
        }
    }

    #[test]
    fn set_report_control_echoes_on_the_next_in() {
        let dev = HidMockDevice::new();
        // Hold an IN first.
        let (in_xfer, in_rx) = in_transfer(REPORT_LEN);
        dev.handle_transfer(
            EpAddr {
                num: 1,
                dir_in: true,
            },
            in_xfer,
        );
        // SET_REPORT over EP0 with a 64-byte payload.
        let mut report = vec![0u8; REPORT_LEN];
        report[1] = 0x99;
        let (tx, ctl_rx) = mpsc::channel();
        let c = Completion::new(move |o| tx.send(o).unwrap());
        let xfer = ControlTransfer::new(
            crate::usb::model::SetupPacket::from_bytes([0x21, 0x09, 0x00, 0x02, 0x00, 0x00, 0x40, 0x00]),
            report.clone(),
            c,
        );
        dev.handle_control(xfer);
        assert!(matches!(ctl_rx.recv().unwrap(), XferOutcome::Ack), "SET_REPORT acked");
        match in_rx.recv().unwrap() {
            XferOutcome::In(bytes) => assert_eq!(bytes, report),
            other => panic!("expected In, got {other:?}"),
        }
    }
}
