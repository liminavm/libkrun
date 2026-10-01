// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use utils::byte_order;
use utils::eventfd::EventFd;
use vm_memory::GuestMemoryMmap;

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, Queue as VirtQueue, QueueConfig,
    VirtioDevice,
};
use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;

use super::TsiFlags;
use super::muxer::{VsockMuxer, pop_rx};
use super::packet::VsockPacket;
use super::{defs, defs::uapi};
use crate::virtio::InterruptTransport;

pub(crate) const RXQ_INDEX: usize = 0;
pub(crate) const TXQ_INDEX: usize = 1;
pub(crate) const EVQ_INDEX: usize = 2;

/// The virtio features supported by our vsock device:
/// - VIRTIO_F_VERSION_1: the device conforms to at least version 1.0 of the VirtIO spec.
/// - VIRTIO_F_IN_ORDER: the device returns used buffers in the same order that the driver makes
///   them available.
/// - VIRTIO_RING_F_EVENT_IDX: each side tells the other how far to run before the next doorbell
///   or interrupt, so a busy queue raises neither per packet.
pub(crate) const AVAIL_FEATURES: u64 = (1 << uapi::VIRTIO_F_VERSION_1 as u64)
    | (1 << uapi::VIRTIO_F_IN_ORDER as u64)
    | (1 << uapi::VIRTIO_VSOCK_F_DGRAM)
    | (1 << VIRTIO_RING_F_EVENT_IDX);

pub struct Vsock {
    cid: u64,
    pub(crate) muxer: VsockMuxer,
    pub(crate) queue_rx: Option<Arc<Mutex<VirtQueue>>>,
    pub(crate) queue_tx: Option<Arc<Mutex<VirtQueue>>>,
    // Queue events are stored separately for event handling.
    pub(crate) queue_events: Vec<Arc<EventFd>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
}

impl Vsock {
    /// Create a new virtio-vsock device with the given VM CID.
    pub fn new(
        cid: u64,
        host_port_map: Option<HashMap<u16, u16>>,
        unix_ipc_port_map: Option<HashMap<u32, (PathBuf, bool)>>,
        tsi_flags: TsiFlags,
        timesync: bool,
    ) -> super::Result<Vsock> {
        Ok(Vsock {
            cid,
            muxer: VsockMuxer::new(cid, host_port_map, unix_ipc_port_map, tsi_flags, timesync),
            queue_rx: None,
            queue_tx: None,
            queue_events: Vec::new(),
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(utils::eventfd::EFD_NONBLOCK)
                .map_err(super::VsockError::EventFd)?,
            device_state: DeviceState::Inactive,
        })
    }

    pub fn id(&self) -> &str {
        defs::VSOCK_DEV_ID
    }

    pub fn cid(&self) -> u64 {
        self.cid
    }

    /// Walk the driver-provided RX queue buffers and attempt to fill them up with any data that we
    /// have pending. Return `true` if descriptors have been added to the used ring, and `false`
    /// otherwise.
    pub fn process_stream_rx(&mut self) -> bool {
        debug!("process_stream_rx()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let mut have_used = false;

        debug!("process_rx before while");
        let queue_rx = self
            .queue_rx
            .as_ref()
            .expect("queue_rx should exist when activated");
        let mut queue_rx = queue_rx.lock().unwrap();
        while let Some(head) = pop_rx(&mut queue_rx, mem) {
            debug!("process_rx inside while");
            let used_len = match VsockPacket::from_rx_virtq_head(&head) {
                Ok(mut pkt) => {
                    if self.muxer.recv_pkt(&mut pkt).is_ok() {
                        pkt.hdr().len() as u32 + pkt.len()
                    } else {
                        // We are using a consuming iterator over the virtio buffers, so, if we can't
                        // fill in this buffer, we'll need to undo the last iterator step.
                        queue_rx.undo_pop();
                        break;
                    }
                }
                Err(e) => {
                    warn!("RX queue error: {e:?}");
                    0
                }
            };

            debug!("process_rx: something to queue");
            have_used = true;
            if let Err(e) = queue_rx.add_used(mem, head.index, used_len) {
                error!("failed to add used elements to the queue: {e:?}");
            }
        }

        have_used
    }

    /// Walk the driver-provided TX queue buffers, package them up as vsock packets, and process
    /// them. Return `true` if descriptors have been added to the used ring, and `false` otherwise.
    pub fn process_stream_tx(&mut self) -> bool {
        debug!("process_stream_tx()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let mut have_used = false;

        let queue_tx = self
            .queue_tx
            .as_ref()
            .expect("queue_tx should exist when activated");
        let mut queue_tx = queue_tx.lock().unwrap();
        loop {
            let mut stalled = false;
            while let Some(head) = queue_tx.pop(mem) {
                let pkt = match VsockPacket::from_tx_virtq_head(&head) {
                    Ok(pkt) => pkt,
                    Err(e) => {
                        error!("error reading TX packet: {e:?}");
                        have_used = true;
                        if let Err(e) = queue_tx.add_used(mem, head.index, 0) {
                            error!("failed to add used elements to the queue: {e:?}");
                        }
                        continue;
                    }
                };

                let sent = if pkt.type_() == uapi::VSOCK_TYPE_DGRAM {
                    debug!("process_stream_tx() is DGRAM");
                    self.muxer.send_dgram_pkt(&pkt)
                } else {
                    debug!("process_stream_tx() is STREAM");
                    self.muxer.send_stream_pkt(&pkt)
                };
                if sent.is_err() {
                    queue_tx.undo_pop();
                    stalled = true;
                    break;
                }

                have_used = true;
                if let Err(e) = queue_tx.add_used(mem, head.index, 0) {
                    error!("failed to add used elements to the queue: {e:?}");
                }
            }

            // With EVENT_IDX the guest rings again only once its avail index passes the position
            // published here; packets that raced the publish are drained in another pass.
            let raced = queue_tx.enable_notification(mem).unwrap_or(false);
            if stalled || !raced {
                break;
            }
        }

        have_used
    }

    /// Whether the guest wants an interrupt for the buffers used on either queue since the last
    /// check. Both queues are asked, since asking resets each one's count; without EVENT_IDX the
    /// answer is always yes.
    pub(crate) fn needs_interrupt(&self) -> bool {
        let DeviceState::Activated(ref mem, _) = self.device_state else {
            return false;
        };
        let asks = |queue: &Option<Arc<Mutex<VirtQueue>>>| {
            queue
                .as_ref()
                .is_some_and(|q| q.lock().unwrap().needs_notification(mem).unwrap_or(true))
        };
        asks(&self.queue_rx) | asks(&self.queue_tx)
    }
}

impl VirtioDevice for Vsock {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features
    }

    fn device_type(&self) -> u32 {
        uapi::VIRTIO_ID_VSOCK
    }

    fn device_name(&self) -> &str {
        "vsock"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        match offset {
            0 if data.len() == 8 => byte_order::write_le_u64(data, self.cid()),
            0 if data.len() == 4 => {
                byte_order::write_le_u32(data, (self.cid() & 0xffff_ffff) as u32)
            }
            4 if data.len() == 4 => {
                byte_order::write_le_u32(data, ((self.cid() >> 32) & 0xffff_ffff) as u32)
            }
            _ => warn!(
                "virtio-vsock received invalid read request of {} bytes at offset {}",
                data.len(),
                offset
            ),
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "guest driver attempted to write device config (offset={:x}, len={:x})",
            offset,
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if queues.len() != defs::NUM_QUEUES {
            error!(
                "Cannot perform activate. Expected {} queue(s), got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }

        if self.activate_evt.write(1).is_err() {
            error!("Cannot write to activate_evt",);
            return Err(ActivateError::BadActivate);
        }

        // Store queue events for event handling.
        self.queue_events = queues.iter().map(|dq| dq.event.clone()).collect();

        // Extract queues from DeviceQueues and wrap in Arc<Mutex<>>.
        let mut queues_vec: Vec<VirtQueue> = queues.into_iter().map(|dq| dq.queue).collect();
        // Note: EVQ (index 2) is currently unused, we just take it to maintain the vec.
        let _evq = queues_vec.pop().unwrap();
        let tx_queue = queues_vec.pop().unwrap();
        let rx_queue = queues_vec.pop().unwrap();

        self.queue_tx = Some(Arc::new(Mutex::new(tx_queue)));
        self.queue_rx = Some(Arc::new(Mutex::new(rx_queue)));
        self.muxer.activate(
            mem.clone(),
            self.queue_rx.clone().unwrap(),
            interrupt.clone(),
        );

        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    /// Deactivate on the virtio reset the guest issues when re-initialising the device — notably on
    /// resume from suspend-to-idle. Returning `false` here (the trait default) leaves the transport
    /// marking the device FAILED (`device_status 0x8f`), so the guest's re-init writes get dropped
    /// and vsock (the limina control plane) never comes back. Like the balloon, vsock runs under the
    /// shared EventManager with queue eventfds that are stable across a transport reset, so they stay
    /// registered and route to a fresh `activate`.
    ///
    /// Crucially we must ALSO tear down the muxer's per-activation worker threads
    /// (timesync/muxer/reaper): the guest recreates the RX queue on re-activate, and a stale thread
    /// from the previous activation would keep writing used-ring entries into the now-freed ring
    /// (guest-memory corruption) and leak a thread trio every suspend cycle. `muxer.deactivate()`
    /// signals them to stop (fire-and-forget — this runs on the vCPU thread and must not block) and
    /// drops stale host-side proxy state.
    fn reset(&mut self) -> bool {
        self.muxer.deactivate();
        self.device_state = DeviceState::Inactive;
        true
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;
    use std::os::unix::io::AsRawFd;
    use std::sync::atomic::Ordering;

    use polly::event_manager::{EventManager, Subscriber};
    use utils::epoll::{EpollEvent, EventSet};
    use utils::eventfd::EFD_NONBLOCK;
    use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
    use vm_memory::GuestAddress;

    use super::super::muxer::{MuxerRx, push_packet};
    use super::super::muxer_rxq::MuxerRxQ;
    use super::super::packet::VSOCK_PKT_HDR_SIZE;
    use super::*;
    use crate::legacy::DummyIrqChip;
    use crate::virtio::VIRTIO_MMIO_INT_VRING;
    use crate::virtio::queue::VIRTQ_DESC_F_WRITE;
    use crate::virtio::queue::tests::VirtQueue as GuestQueue;

    const HOST_PORT: u32 = 1025;
    const GUEST_PORT: u32 = 50000;

    fn memory() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap()
    }

    /// Activate the device the way the MMIO transport does: queues carry EVENT_IDX exactly when
    /// the driver acked it.
    fn activated(
        mem: &GuestMemoryMmap,
        queues: [&GuestQueue; 3],
        acked: u64,
    ) -> (Vsock, InterruptTransport) {
        let mut vsock = Vsock::new(3, None, None, TsiFlags::empty(), false).unwrap();
        vsock.set_acked_features(acked & vsock.avail_features());
        let event_idx = vsock.acked_features() & (1 << VIRTIO_RING_F_EVENT_IDX) != 0;
        let interrupt =
            InterruptTransport::new(DummyIrqChip::new().into(), "vsock-test".to_string()).unwrap();
        let queues = queues
            .iter()
            .map(|q| {
                let mut queue = q.create_queue();
                queue.set_event_idx(event_idx);
                DeviceQueue::new(queue, Arc::new(EventFd::new(EFD_NONBLOCK).unwrap()))
            })
            .collect();
        vsock
            .activate(mem.clone(), interrupt.clone(), queues)
            .unwrap();
        (vsock, interrupt)
    }

    /// Post a header-only stream packet in TX slot `slot` and ring the doorbell.
    fn guest_sends(vsock: &mut Vsock, mem: &GuestMemoryMmap, tx: &GuestQueue, slot: u16) {
        let addr = 0x8000 + slot as u64 * 0x100;
        tx.dtable[slot as usize].set(addr, VSOCK_PKT_HDR_SIZE as u32, 0, 0);
        tx.avail.ring[slot as usize].set(slot);
        tx.avail.idx.set(slot + 1);
        let mut scratch = tx.create_queue();
        scratch.next_avail = Wrapping(slot);
        let head = scratch.pop(mem).unwrap();
        VsockPacket::from_tx_virtq_head(&head)
            .unwrap()
            .set_op(uapi::VSOCK_OP_CREDIT_UPDATE)
            .set_type(uapi::VSOCK_TYPE_STREAM)
            .set_src_port(GUEST_PORT)
            .set_dst_port(HOST_PORT)
            .set_dst_cid(uapi::VSOCK_HOST_CID);

        let kick = vsock.queue_events[TXQ_INDEX].clone();
        kick.write(1).unwrap();
        let mut event_manager = EventManager::new().unwrap();
        vsock.process(
            &EpollEvent::new(EventSet::IN, kick.as_raw_fd() as u64),
            &mut event_manager,
        );
    }

    fn take_interrupt(interrupt: &InterruptTransport) -> bool {
        interrupt.status().swap(0, Ordering::SeqCst) & VIRTIO_MMIO_INT_VRING as usize != 0
    }

    #[test]
    fn a_drained_tx_queue_tells_the_guest_where_to_ring_next() {
        let mem = memory();
        let (rx, tx, ev) = (
            GuestQueue::new(GuestAddress(0x1000), &mem, 16),
            GuestQueue::new(GuestAddress(0), &mem, 16),
            GuestQueue::new(GuestAddress(0x2000), &mem, 16),
        );
        let (mut vsock, _interrupt) = activated(&mem, [&rx, &tx, &ev], EVERYTHING);

        guest_sends(&mut vsock, &mem, &tx, 0);

        // With EVENT_IDX the guest rings only when its avail index passes this value; left
        // behind, the guest would stop announcing packets.
        assert_eq!(tx.used.event.get(), 1);
        vsock.reset();
    }

    #[test]
    fn the_guest_is_interrupted_only_once_past_the_index_it_asked_for() {
        let mem = memory();
        let (rx, tx, ev) = (
            GuestQueue::new(GuestAddress(0x1000), &mem, 16),
            GuestQueue::new(GuestAddress(0), &mem, 16),
            GuestQueue::new(GuestAddress(0x2000), &mem, 16),
        );
        let (mut vsock, interrupt) = activated(&mem, [&rx, &tx, &ev], EVERYTHING);

        tx.avail.event.set(0);
        guest_sends(&mut vsock, &mem, &tx, 0);
        assert!(
            take_interrupt(&interrupt),
            "used index 1 passes used_event 0"
        );

        tx.avail.event.set(8);
        guest_sends(&mut vsock, &mem, &tx, 1);
        assert!(
            !take_interrupt(&interrupt),
            "used index 2 has not passed used_event 8"
        );
        vsock.reset();
    }

    #[test]
    fn a_guest_without_event_idx_is_interrupted_for_every_completion() {
        let mem = memory();
        let (rx, tx, ev) = (
            GuestQueue::new(GuestAddress(0x1000), &mem, 16),
            GuestQueue::new(GuestAddress(0), &mem, 16),
            GuestQueue::new(GuestAddress(0x2000), &mem, 16),
        );
        let acked = EVERYTHING & !(1 << VIRTIO_RING_F_EVENT_IDX);
        let (mut vsock, interrupt) = activated(&mem, [&rx, &tx, &ev], acked);

        tx.avail.event.set(8);
        guest_sends(&mut vsock, &mem, &tx, 0);
        assert!(take_interrupt(&interrupt));
        guest_sends(&mut vsock, &mem, &tx, 1);
        assert!(take_interrupt(&interrupt));
        vsock.reset();
    }

    #[test]
    fn an_empty_rx_ring_tells_the_guest_to_ring_on_refill() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        rx.dtable[0].set(
            0x8000,
            VSOCK_PKT_HDR_SIZE as u32 + 64,
            VIRTQ_DESC_F_WRITE,
            0,
        );
        rx.avail.ring[0].set(0);
        rx.avail.idx.set(1);
        let mut queue = rx.create_queue();
        queue.set_event_idx(true);
        let queue = Arc::new(Mutex::new(queue));
        let pending = Arc::new(Mutex::new(MuxerRxQ::new()));
        let reset = || MuxerRx::Reset {
            local_port: HOST_PORT,
            peer_port: GUEST_PORT,
        };

        push_packet(3, reset(), &pending, &queue, &mem);
        push_packet(3, reset(), &pending, &queue, &mem);

        // The second packet found no buffer and waits in the pending queue; only a refill kick
        // brings it out, and the guest kicks only if the avail index it writes passes this value.
        assert_eq!(pending.lock().unwrap().len(), 1);
        assert_eq!(rx.used.event.get(), 1);
    }

    /// A driver that acks whatever the device offers.
    const EVERYTHING: u64 = u64::MAX;
}
