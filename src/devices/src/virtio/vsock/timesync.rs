use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time;

use super::super::Queue as VirtQueue;
use super::defs::uapi;
use super::packet::VsockPacket;

use crate::virtio::InterruptTransport;
use vm_memory::GuestMemoryMmap;

const UPDATE_INTERVAL: u64 = 60 * 1000 * 1000 * 1000;
const SLEEP_NSECS: u64 = 2 * 1000 * 1000 * 1000;
const TSYNC_PORT: u32 = 123;

/*
 * We send a time sync packet if we slept for 3 times more nanoseconds than expected
 * (which is an indication the system forced us to take a long nap), or if
 * UPDATE_INTERVAL has been reached.
 *
 * All three inputs are CLOCK_REALTIME reads, which can step BACKWARD between samples
 * (NTP correction, manual clock set, host-sleep wall-clock adjustment) — saturate to
 * zero elapsed instead of panicking (debug) or wrapping to a huge value (release).
 * A backward step itself doesn't need a packet from here: the guest learns the new
 * wall clock through the next periodic sync / the control-plane TimeSync path.
 */
fn sync_due(now: u64, last_awake: u64, last_update: u64) -> bool {
    now.saturating_sub(last_awake) >= (SLEEP_NSECS * 3)
        || now.saturating_sub(last_update) >= UPDATE_INTERVAL
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A backward wall-clock step between the `last_awake` sample and the `now` sample
    /// (NTP, manual set) must not panic (debug subtract-with-overflow — seen in the wild
    /// as `attempt to subtract with overflow` at timesync.rs:83) or wrap to a huge
    /// elapsed value in release.
    #[test]
    fn backward_clock_step_does_not_underflow() {
        let now = 1_000_000_000u64;
        let last_awake = now + 5_000_000_000; // clock stepped back ~5s
        let last_update = now;
        assert!(!sync_due(now, last_awake, last_update));
    }

    #[test]
    fn long_nap_triggers_sync() {
        let last_awake = 1_000_000_000_000u64;
        let now = last_awake + SLEEP_NSECS * 3;
        assert!(sync_due(now, last_awake, now));
    }

    #[test]
    fn update_interval_triggers_sync() {
        let last_update = 1_000_000_000_000u64;
        let now = last_update + UPDATE_INTERVAL;
        assert!(sync_due(now, now, last_update));
    }
}

pub struct TimesyncThread {
    cid: u64,
    mem: GuestMemoryMmap,
    queue_mutex: Arc<Mutex<VirtQueue>>,
    interrupt: InterruptTransport,
    // Set true by `VsockMuxer::deactivate` on a device reset (suspend/resume). Checked before every
    // wake/queue-write so a stale timesync thread from a previous activation never writes into the
    // now-recreated (freed/reallocated) guest RX ring. Fire-and-forget: the thread self-exits.
    stop: Arc<AtomicBool>,
}

impl TimesyncThread {
    pub fn new(
        cid: u64,
        mem: GuestMemoryMmap,
        queue_mutex: Arc<Mutex<VirtQueue>>,
        interrupt: InterruptTransport,
        stop: Arc<AtomicBool>,
    ) -> Self {
        Self {
            cid,
            mem,
            queue_mutex,
            interrupt,
            stop,
        }
    }

    fn send_time(&self, time: u64) {
        let mut queue = self.queue_mutex.lock().unwrap();
        if let Some(head) = queue.pop(&self.mem)
            && let Ok(mut pkt) = VsockPacket::from_rx_virtq_head(&head)
        {
            pkt.set_op(uapi::VSOCK_OP_RW)
                .set_src_cid(uapi::VSOCK_HOST_CID)
                .set_dst_cid(self.cid)
                .set_src_port(TSYNC_PORT)
                .set_dst_port(TSYNC_PORT)
                .set_type(uapi::VSOCK_TYPE_DGRAM);

            pkt.write_time_sync(time);
            pkt.set_len(pkt.buf().unwrap().len() as u32);
            if let Err(e) =
                queue.add_used(&self.mem, head.index, pkt.hdr().len() as u32 + pkt.len())
            {
                error!("failed to add used elements to the queue: {e:?}");
            }
            self.interrupt.signal_used_queue();
        }
    }

    fn work(&mut self) {
        let mut last_update = 0u64;
        let mut last_awake = utils::time::get_time(utils::time::ClockType::Real);
        loop {
            // Bail before touching the queue if this activation has been torn down (device reset).
            if self.stop.load(Ordering::Acquire) {
                break;
            }
            let now = utils::time::get_time(utils::time::ClockType::Real);
            if sync_due(now, last_awake, last_update) {
                self.send_time(now);
                last_update = now;
            }

            last_awake = utils::time::get_time(utils::time::ClockType::Real);
            thread::sleep(time::Duration::from_nanos(SLEEP_NSECS));
        }
    }

    pub fn run(mut self) {
        thread::Builder::new()
            .name("vsock timesync".into())
            .spawn(move || self.work())
            .unwrap();
    }
}
