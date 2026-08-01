// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A generic **bulk pipe** gadget: a USB device whose vendor-specific bulk endpoints are
//! shuttled verbatim over a caller-supplied channel, with zero knowledge of what the bytes
//! mean. It is the reusable, upstream-shaped mechanism the emulated xHCI controller drives;
//! the *policy* (the device identity + what a packet means) lives in the caller. limina wires
//! it to the elanmoc fingerprint-reader protocol to present a Touch-ID-backed match-on-chip
//! reader to a stock guest (`crates/limina-vmm/src/moc_usb.rs`); see
//! `docs/design/usb-moc-fingerprint.md`.
//!
//! It is [`HidReportPipe`](super::HidReportPipe) generalised along the three axes a bulk
//! vendor device needs and a fixed-size HID interrupt pair does not:
//!
//! 1. **Identity-free.** The caller supplies the full [`DeviceDescriptors`] (device / config /
//!    interface / endpoint / string blocks), so this type carries no VID/PID or report
//!    descriptor of its own.
//! 2. **Variable-length frames.** Bulk transfers are not padded to a fixed report size; each
//!    guest→host frame is delivered exactly as pushed and each host→guest frame forwarded
//!    exactly as the guest wrote it.
//! 3. **Multiple IN endpoints, each independently held.** A device may expose several IN
//!    endpoints with different semantics (the elanmoc reader uses `0x83` for immediate command
//!    replies and `0x84` for finger-wait replies). Each IN endpoint keeps its own held-IN slot
//!    and FIFO, keyed by endpoint address.
//!
//! **Held-transfer discipline (see `docs/design/usb-xhci.md` §3.3 and the Stage-C INVARIANTS),**
//! carried over from [`HidReportPipe`]: an IN read with no frame ready is *held* (xHCI's NAK
//! analogue); a new IN supersedes any stale prior hold on the same endpoint (bounding the held
//! set to one per endpoint across open/close churn); `reset()` drops every hold. The superseded
//! or reset transfer's completion is a no-op once the controller's generation guard has
//! invalidated it, so dropping it is harmless.
//!
//! **Stall signal.** Unlike the data-only HID pipe, a bulk policy must be able to *fail* a held
//! IN (an elanmoc enroll-decline or any protocol error must error the guest's finger-wait read,
//! not deliver bytes — a data reply would be misread as a retry and loop forever). [`stall_in`]
//! stalls the endpoint's held read, or — if none is posted yet — enqueues the stall. Frames and
//! stalls share one per-endpoint FIFO, so they are delivered in the exact order the policy signals
//! them (a "frame then error" sequence never reorders to error-then-stale-frame) and repeated
//! stalls are never coalesced (each is consumed by its own read; a dropped stall would wedge a
//! later read forever).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use super::model::{DeviceDescriptors, EpAddr, Transfer, UsbDeviceModel, UsbSpeed};

/// A sink for guest→host bulk frames (delivered on an OUT endpoint). Invoked with the gadget's
/// state lock released, from the controller's worker thread. The first argument is the OUT
/// endpoint address (`bEndpointAddress`, e.g. `0x01`); the second is the bytes the guest wrote.
pub type BulkSink = Arc<dyn Fn(u8, Vec<u8>) + Send + Sync>;

/// A sink for **endpoint-cancel** notifications: the guest quiesced an endpoint (xHCI Stop
/// Endpoint — a `usb_kill_urb` / libusb transfer cancel), abandoning whatever read it had posted
/// there. Invoked with the gadget's state lock released, with the endpoint address (e.g. `0x84`).
/// A policy that started long-running host work for a held read (the fingerprint reader's Touch ID
/// prompt) uses this to abort it; policies that don't need it pass none.
pub type BulkCancelSink = Arc<dyn Fn(u8) + Send + Sync>;

/// A host→guest event waiting for a read to carry it: either data bytes or a stall. Ordering
/// matters — a policy that signals "frame then error" (or two errors) must have the guest observe
/// them in that exact order, so both live in one FIFO rather than a queue plus a separate latch.
enum InEvent {
    Frame(Vec<u8>),
    Stall,
}

/// Per-IN-endpoint state: the FIFO of pending host→guest events, and the single held read (no
/// event ready when the read arrived). Invariant: `held.is_some()` ⇒ `events.is_empty()` (a read is
/// only held when there was nothing to give it, and a new event completes the held read directly).
#[derive(Default)]
struct InEndpoint {
    /// Events received (via [`BulkPipe::push_in`] / [`BulkPipe::stall_in`]) with no read ready to
    /// carry them; drained oldest-first onto subsequent reads. Order- and count-preserving.
    events: VecDeque<InEvent>,
    /// The one read transfer the gadget is holding (no event ready when it arrived).
    held: Option<Transfer>,
}

/// A generic bulk gadget that pipes variable-length frames to/from a channel. Construct with
/// [`BulkPipe::new`]; feed host→guest frames with [`BulkPipe::push_in`] (or fail one with
/// [`BulkPipe::stall_in`]); receive guest→host frames through the [`BulkSink`] passed at
/// construction. Keyed by endpoint address, so a device with several IN/OUT endpoints multiplexes
/// over one instance.
pub struct BulkPipe {
    descriptors: DeviceDescriptors,
    speed: UsbSpeed,
    /// IN-endpoint state keyed by `bEndpointAddress` (e.g. `0x83`, `0x84`).
    ins: Mutex<HashMap<u8, InEndpoint>>,
    out_sink: BulkSink,
    /// Notified when the guest cancels an endpoint (see [`BulkCancelSink`]).
    cancel_sink: Option<BulkCancelSink>,
}

impl BulkPipe {
    /// Build a bulk-pipe gadget with the given descriptors and attachment speed. The descriptors
    /// are the caller's policy (identity + endpoint layout); `out_sink` receives every guest→host
    /// frame tagged with its OUT endpoint address.
    pub fn new(
        descriptors: DeviceDescriptors,
        speed: UsbSpeed,
        out_sink: BulkSink,
    ) -> Arc<BulkPipe> {
        BulkPipe::with_cancel_sink(descriptors, speed, out_sink, None)
    }

    /// As [`BulkPipe::new`], plus a [`BulkCancelSink`] notified when the guest cancels an
    /// endpoint's outstanding read (Stop Endpoint). Policies that start host work on behalf of a
    /// held IN need this to learn the guest walked away — nothing is sent on the wire when libusb
    /// cancels a transfer, so the controller's Stop Endpoint is the only evidence.
    pub fn with_cancel_sink(
        descriptors: DeviceDescriptors,
        speed: UsbSpeed,
        out_sink: BulkSink,
        cancel_sink: Option<BulkCancelSink>,
    ) -> Arc<BulkPipe> {
        Arc::new(BulkPipe {
            descriptors,
            speed,
            ins: Mutex::new(HashMap::new()),
            out_sink,
            cancel_sink,
        })
    }

    /// Deliver a host→guest frame on IN endpoint `addr`: complete a held read now if one is
    /// waiting, else enqueue the frame (in order, after any earlier pending events) for the next
    /// read. Frames are delivered verbatim (the controller truncates to the guest's read buffer
    /// and reports residue).
    pub fn push_in(&self, addr: u8, frame: Vec<u8>) {
        let mut st = self.ins.lock().unwrap();
        let ep = st.entry(addr).or_default();
        if let Some(held) = ep.held.take() {
            // Complete outside the state lock (the completion re-locks the controller).
            debug_assert!(ep.events.is_empty(), "held read ⇒ no queued events");
            drop(st);
            held.complete_in(frame);
        } else {
            ep.events.push_back(InEvent::Frame(frame));
        }
    }

    /// Fail IN endpoint `addr`'s outstanding read with a stall (the policy signals an error /
    /// declined operation). If a read is held it is stalled now; otherwise the stall is enqueued
    /// **in order** behind any earlier pending frames, so a "frame then error" sequence reaches the
    /// guest as frame-then-error (never reordered), and repeated stalls are never coalesced (each is
    /// consumed by its own read — a lost stall would wedge a later read forever).
    pub fn stall_in(&self, addr: u8) {
        let mut st = self.ins.lock().unwrap();
        let ep = st.entry(addr).or_default();
        if let Some(held) = ep.held.take() {
            // Dropping the Transfer fires XferOutcome::Stall (see model::Completion::drop).
            debug_assert!(ep.events.is_empty(), "held read ⇒ no queued events");
            drop(st);
            drop(held);
        } else {
            ep.events.push_back(InEvent::Stall);
        }
    }
}

impl UsbDeviceModel for BulkPipe {
    fn descriptors(&self) -> DeviceDescriptors {
        self.descriptors.clone()
    }

    fn speed(&self) -> UsbSpeed {
        self.speed
    }

    // handle_control is left as the trait default (stall): the controller answers standard
    // enumeration from our descriptors, and the elanmoc protocol issues no class/vendor EP0
    // requests (the interface is vendor-class 0xFF; usbhid never binds, so the HID report
    // descriptor is never requested).

    fn handle_transfer(&self, ep: EpAddr, xfer: Transfer) {
        if ep.dir_in {
            let addr = 0x80 | ep.num;
            let mut st = self.ins.lock().unwrap();
            let epst = st.entry(addr).or_default();
            // Deliver the next pending event in order (frame or stall), else hold the read.
            // NOTE: at most one read is held per endpoint — the elanmoc driver is strictly
            // lockstep (one transfer outstanding at a time), which this relies on. A guest that
            // keeps two reads posted on one bulk IN would see the earlier one superseded (stalled);
            // if BulkPipe is ever driven by such a policy, hold a `VecDeque<Transfer>` here instead.
            match epst.events.pop_front() {
                Some(InEvent::Frame(frame)) => {
                    drop(st);
                    xfer.complete_in(frame);
                }
                Some(InEvent::Stall) => {
                    drop(st);
                    xfer.stall();
                }
                None => {
                    // Hold it; a new read supersedes any stale prior hold on this endpoint.
                    let stale = epst.held.replace(xfer);
                    // Drop the superseded hold OUTSIDE the state lock: its Drop fires the
                    // completion, which re-locks the controller (never the gadget). Once the
                    // controller's generation guard has invalidated it, that completion is a no-op;
                    // if somehow still live, it stalls cleanly (the guest re-submits).
                    drop(st);
                    drop(stale);
                }
            }
        } else {
            // Bulk OUT: the guest's bytes → forward host-ward (tagged with the endpoint) and ack.
            let frame = xfer.data_out().to_vec();
            xfer.ack();
            (self.out_sink)(ep.num, frame);
        }
    }

    /// The guest cancelled this endpoint's outstanding read (Stop Endpoint). Drop the held
    /// transfer — its completion is already stale (the controller bumped the generation) — and
    /// discard anything still queued for it: those events answer a transaction the guest has
    /// abandoned, and handing them to the *next* read would desync the policy (a queued stall
    /// would fail the guest's next, unrelated request). Then notify the policy so it can abort
    /// whatever it started for that read. OUT endpoints carry no held state; the notification
    /// still goes through, since the policy may care.
    fn endpoint_stopped(&self, ep: EpAddr) {
        let addr = if ep.dir_in { 0x80 | ep.num } else { ep.num };
        let mut st = self.ins.lock().unwrap();
        let stale = st.remove(&addr);
        drop(st);
        drop(stale); // the held read's Drop stalls it; the generation guard drops that outcome
        if let Some(sink) = &self.cancel_sink {
            sink(addr);
        }
    }

    fn reset(&self) {
        let mut st = self.ins.lock().unwrap();
        // Take every held read and clear all pending events; stall the holds outside the lock.
        let held: Vec<Transfer> = st.values_mut().filter_map(|e| e.held.take()).collect();
        st.clear();
        drop(st);
        drop(held); // each Drop stalls its held IN (see handle_transfer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usb::model::{Completion, XferOutcome};
    use std::sync::mpsc;

    // elanmoc-shaped: one OUT (0x01), two IN (0x83 immediate, 0x84 finger-wait).
    const EP_OUT: u8 = 1; // address 0x01
    const EP_IN_CMD: u8 = 3; // address 0x83
    const EP_IN_MOC: u8 = 4; // address 0x84

    // The completion closures ignore a closed channel (`let _ = send`): a transfer still held
    // when the pipe is dropped at end of scope stalls into an already-dropped receiver, which is
    // benign — real controller completions don't panic on it either.
    fn pipe() -> (Arc<BulkPipe>, mpsc::Receiver<(u8, Vec<u8>)>) {
        let (tx, rx) = mpsc::channel();
        let sink: BulkSink = Arc::new(move |addr: u8, f: Vec<u8>| {
            let _ = tx.send((addr, f));
        });
        let d = DeviceDescriptors {
            device: vec![0x12, 0x01],
            configs: vec![vec![0x09, 0x02]],
            strings: vec![vec![0x04, 0x03, 0x09, 0x04]],
        };
        (BulkPipe::new(d, UsbSpeed::Full, sink), rx)
    }

    /// A pipe that also reports endpoint cancels; the second receiver yields the cancelled
    /// endpoint address.
    fn pipe_with_cancel() -> (Arc<BulkPipe>, mpsc::Receiver<u8>) {
        let sink: BulkSink = Arc::new(|_addr: u8, _f: Vec<u8>| {});
        let (ctx, crx) = mpsc::channel();
        let cancel: BulkCancelSink = Arc::new(move |addr: u8| {
            let _ = ctx.send(addr);
        });
        let d = DeviceDescriptors {
            device: vec![0x12, 0x01],
            configs: vec![vec![0x09, 0x02]],
            strings: vec![vec![0x04, 0x03, 0x09, 0x04]],
        };
        (
            BulkPipe::with_cancel_sink(d, UsbSpeed::Full, sink, Some(cancel)),
            crx,
        )
    }

    fn in_transfer(len: usize) -> (Transfer, mpsc::Receiver<XferOutcome>) {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| {
            let _ = tx.send(o);
        });
        (Transfer::new(Vec::new(), len, c), rx)
    }

    fn out_transfer(data: Vec<u8>) -> (Transfer, mpsc::Receiver<XferOutcome>) {
        let (tx, rx) = mpsc::channel();
        let c = Completion::new(move |o| {
            let _ = tx.send(o);
        });
        (Transfer::new(data, 0, c), rx)
    }

    #[test]
    fn speed_and_descriptors_are_the_caller_policy() {
        let (p, _rx) = pipe();
        assert_eq!(p.speed(), UsbSpeed::Full);
        assert_eq!(p.descriptors().device, vec![0x12, 0x01]);
    }

    #[test]
    fn out_frame_reaches_the_sink_tagged_with_its_endpoint() {
        let (p, sink_rx) = pipe();
        let payload = vec![0x40, 0x19]; // an elanmoc get-version command
        let (out, out_rx) = out_transfer(payload.clone());
        p.handle_transfer(
            EpAddr {
                num: EP_OUT,
                dir_in: false,
            },
            out,
        );
        assert!(matches!(out_rx.recv().unwrap(), XferOutcome::Ack));
        assert_eq!(sink_rx.recv().unwrap(), (0x01, payload));
    }

    #[test]
    fn variable_length_frames_are_delivered_verbatim() {
        let (p, _rx) = pipe();
        // A 97-byte get-userid reply and a 2-byte ACK — no fixed report size, no padding.
        for frame in [vec![0xABu8; 97], vec![0x40, 0x00]] {
            let (in_xfer, in_rx) = in_transfer(frame.len());
            p.handle_transfer(
                EpAddr {
                    num: EP_IN_CMD,
                    dir_in: true,
                },
                in_xfer,
            );
            p.push_in(0x83, frame.clone());
            match in_rx.recv().unwrap() {
                XferOutcome::In(bytes) => assert_eq!(bytes, frame),
                other => panic!("expected In, got {other:?}"),
            }
        }
    }

    #[test]
    fn held_in_completes_when_a_frame_is_pushed() {
        let (p, _rx) = pipe();
        let (in_xfer, in_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            in_xfer,
        );
        assert!(in_rx.try_recv().is_err(), "IN held, not completed");
        p.push_in(0x83, vec![0x40, 0x03]);
        match in_rx.recv().unwrap() {
            XferOutcome::In(b) => assert_eq!(b, vec![0x40, 0x03]),
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn queued_frame_drains_to_a_later_in() {
        let (p, _rx) = pipe();
        p.push_in(0x83, vec![0x40, 0x00]);
        let (in_xfer, in_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            in_xfer,
        );
        match in_rx.recv().unwrap() {
            XferOutcome::In(b) => assert_eq!(b, vec![0x40, 0x00]),
            other => panic!("expected In, got {other:?}"),
        }
    }

    /// The two IN endpoints are independent: a frame pushed to 0x83 must not complete a read held
    /// on 0x84, and vice versa. This is the multi-endpoint property the reader relies on.
    #[test]
    fn in_endpoints_are_independent() {
        let (p, _rx) = pipe();
        let (moc_in, moc_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            moc_in,
        ); // held on 0x84
           // A frame for 0x83 must not touch the 0x84 hold.
        p.push_in(0x83, vec![0x40, 0x03]);
        assert!(
            moc_rx.try_recv().is_err(),
            "0x84 hold untouched by a 0x83 frame"
        );
        // The 0x84 frame completes the 0x84 hold.
        p.push_in(0x84, vec![0x40, 0x05]);
        match moc_rx.recv().unwrap() {
            XferOutcome::In(b) => assert_eq!(b, vec![0x40, 0x05]),
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn stall_in_fails_a_held_read() {
        let (p, _rx) = pipe();
        let (in_xfer, in_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            in_xfer,
        );
        p.stall_in(0x84); // enroll-decline: fail the finger-wait read
        assert!(matches!(in_rx.recv().unwrap(), XferOutcome::Stall));
    }

    /// A stall signalled before the guest posts its read must not be lost: it is enqueued and the
    /// next read consumes it. (Guards the stall/read ordering race.)
    #[test]
    fn stall_before_read_is_delivered_to_the_next_read() {
        let (p, _rx) = pipe();
        p.stall_in(0x84); // no read held yet
        let (in_xfer, in_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            in_xfer,
        );
        assert!(
            matches!(in_rx.recv().unwrap(), XferOutcome::Stall),
            "queued stall consumed"
        );
    }

    /// A frame then a stall, both signalled before any read, must reach the guest IN THAT ORDER —
    /// never stall-first-then-stale-frame. This is the ordering the event queue exists to preserve
    /// (a reordered stale frame would be misread as the next command's reply).
    #[test]
    fn frame_then_stall_is_delivered_in_order() {
        let (p, _rx) = pipe();
        p.push_in(0x84, vec![0x40, 0x00]); // frame first
        p.stall_in(0x84); // then a stall
        let (r1x, r1) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            r1x,
        );
        match r1.recv().unwrap() {
            XferOutcome::In(b) => assert_eq!(b, vec![0x40, 0x00], "frame delivered first"),
            other => panic!("expected the frame first, got {other:?}"),
        }
        let (r2x, r2) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            r2x,
        );
        assert!(
            matches!(r2.recv().unwrap(), XferOutcome::Stall),
            "stall delivered second"
        );
    }

    /// Two stalls with no intervening read must both be delivered — the second read must NOT hang
    /// (a coalesced/lost stall would wedge it until timeout/endpoint reset).
    #[test]
    fn two_stalls_are_not_coalesced() {
        let (p, _rx) = pipe();
        p.stall_in(0x84);
        p.stall_in(0x84);
        for _ in 0..2 {
            let (rx_xfer, rx) = in_transfer(2);
            p.handle_transfer(
                EpAddr {
                    num: EP_IN_MOC,
                    dir_in: true,
                },
                rx_xfer,
            );
            assert!(
                matches!(rx.recv().unwrap(), XferOutcome::Stall),
                "each stall consumed by its own read"
            );
        }
    }

    /// reset() clears queued events (not just held reads): a stall queued before reset is gone, so a
    /// post-reset read holds rather than stalling on stale state.
    #[test]
    fn reset_clears_a_queued_stall() {
        let (p, _rx) = pipe();
        p.stall_in(0x84); // queued, no read
        p.reset();
        let (rx_xfer, rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            rx_xfer,
        );
        assert!(
            rx.try_recv().is_err(),
            "post-reset read holds, stale stall cleared"
        );
    }

    /// Repeated INs with nothing to deliver never accumulate more than one held transfer per
    /// endpoint: each new IN supersedes (and stalls) the prior — the open/close-churn guard.
    #[test]
    fn a_new_in_supersedes_a_stale_held_in() {
        let (p, _rx) = pipe();
        let (in1, rx1) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            in1,
        );
        let (in2, rx2) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            in2,
        );
        assert!(
            matches!(rx1.recv().unwrap(), XferOutcome::Stall),
            "stale IN stalled"
        );
        p.push_in(0x83, vec![0x40, 0x00]);
        match rx2.recv().unwrap() {
            XferOutcome::In(b) => assert_eq!(b, vec![0x40, 0x00]),
            other => panic!("expected In, got {other:?}"),
        }
    }

    #[test]
    fn reset_stalls_all_held_ins_and_clears_queues() {
        let (p, _rx) = pipe();
        let (cmd_in, cmd_rx) = in_transfer(2);
        let (moc_in, moc_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            cmd_in,
        ); // held on 0x83
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            moc_in,
        ); // held on 0x84
        p.push_in(0x83, vec![0x40, 0x00]); // completes the held 0x83 read
        assert!(
            matches!(cmd_rx.recv().unwrap(), XferOutcome::In(_)),
            "0x83 read completed"
        );
        // Queue a frame on an endpoint with no read posted, and re-hold 0x83, then reset.
        p.push_in(0x83, vec![0x99]); // no read held on 0x83 now → queued
        let (cmd_in2, cmd_rx2) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            cmd_in2,
        ); // drains the queue!
        assert!(
            matches!(cmd_rx2.recv().unwrap(), XferOutcome::In(_)),
            "queued frame drained"
        );
        // Now hold both again and reset: both holds stall, and no stale queue remains.
        let (cmd_in3, cmd_rx3) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            cmd_in3,
        );
        p.push_in(0x82, vec![0x11]); // queued on an endpoint with no reader
        p.reset();
        assert!(
            matches!(cmd_rx3.recv().unwrap(), XferOutcome::Stall),
            "0x83 hold stalled"
        );
        assert!(
            matches!(moc_rx.recv().unwrap(), XferOutcome::Stall),
            "0x84 hold stalled"
        );
        // The 0x82 queue was cleared: a fresh 0x82 read holds rather than draining [0x11].
        let (fresh, fresh_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: 2,
                dir_in: true,
            },
            fresh,
        );
        assert!(
            fresh_rx.try_recv().is_err(),
            "post-reset read is held, not draining stale queue"
        );
    }

    /// A guest transfer cancel (Stop Endpoint) drops that endpoint's held read and tells the
    /// policy, which is the only way it can learn the guest walked away from work it started.
    /// Other endpoints are untouched.
    #[test]
    fn endpoint_stop_drops_the_held_read_and_notifies_the_policy() {
        let (p, cancel_rx) = pipe_with_cancel();
        let (moc_in, moc_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            moc_in,
        );
        let (cmd_in, cmd_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_CMD,
                dir_in: true,
            },
            cmd_in,
        );

        p.endpoint_stopped(EpAddr {
            num: EP_IN_MOC,
            dir_in: true,
        });

        assert_eq!(
            cancel_rx.recv().unwrap(),
            0x84,
            "policy told which endpoint"
        );
        assert!(
            matches!(moc_rx.recv().unwrap(), XferOutcome::Stall),
            "the cancelled endpoint's hold is released"
        );
        assert!(
            cmd_rx.try_recv().is_err(),
            "the untouched endpoint keeps its hold"
        );
    }

    /// A reply the policy emits *after* the guest cancelled must not be delivered to the next,
    /// unrelated read on that endpoint — the stop clears the endpoint's queue.
    #[test]
    fn a_stop_discards_events_queued_for_the_abandoned_transaction() {
        let (p, _cancel_rx) = pipe_with_cancel();
        p.push_in(0x84, vec![0x40, 0x00]); // queued: no read posted yet
        p.endpoint_stopped(EpAddr {
            num: EP_IN_MOC,
            dir_in: true,
        });
        let (fresh, fresh_rx) = in_transfer(2);
        p.handle_transfer(
            EpAddr {
                num: EP_IN_MOC,
                dir_in: true,
            },
            fresh,
        );
        assert!(
            fresh_rx.try_recv().is_err(),
            "post-cancel read holds rather than draining the abandoned reply"
        );
    }
}
