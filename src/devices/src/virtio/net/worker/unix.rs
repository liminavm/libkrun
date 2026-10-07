use std::cmp;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::result;
use std::thread;
use std::time::{Duration, Instant};

use utils::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use utils::eventfd::EventFd;
use virtio_bindings::virtio_net::VIRTIO_NET_HDR_F_DATA_VALID;
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

use crate::virtio::net::backend::ConnectError;
use crate::virtio::net::backend::{NetBackend, ReadError, Reconnect, WriteError};
use crate::virtio::net::device::{FrontendError, RxError, TxError, VirtioNetBackend};
#[cfg(target_os = "linux")]
use crate::virtio::net::tap::Tap;
use crate::virtio::net::unixgram::Unixgram;
use crate::virtio::net::unixstream::Unixstream;
use crate::virtio::net::{MAX_BUFFER_SIZE, QUEUE_SIZE, VNET_HDR_LEN};
use crate::virtio::{DeviceQueue, InterruptTransport};

/// The first wait before reconnecting to a proxy that went away, doubled after every failed
/// attempt up to [`RECONNECT_MAX`]. The supervisor respawns gvproxy within about a second, so
/// most outages end on one of the first few attempts.
const RECONNECT_FIRST: Duration = Duration::from_millis(100);
const RECONNECT_MAX: Duration = Duration::from_secs(5);

/// The backend's proxy went away; the worker drops the guest's frames and retries the
/// connection.
struct Lost {
    next_try: Instant,
    delay: Duration,
    /// The backend cannot reconnect, so nothing is retried.
    for_good: bool,
}

pub struct NetWorker {
    rx_q: DeviceQueue,
    tx_q: DeviceQueue,
    interrupt: InterruptTransport,

    mem: GuestMemoryMmap,
    backend: Box<dyn NetBackend + Send>,
    // Signalled by `Net::reset` (suspend/resume): the worker drains it, drops out of its epoll
    // loop, and exits so the device can be re-activated with fresh queues.
    stop_fd: EventFd,

    rx_frame_buf: [u8; MAX_BUFFER_SIZE],
    rx_frame_buf_len: usize,
    rx_has_deferred_frame: bool,
    // The guest negotiated GUEST_CSUM, so a frame from a proxy can reach it marked checksum-valid.
    rx_data_valid: bool,

    tx_iovec: Vec<(GuestAddress, usize)>,
    tx_frame_buf: [u8; MAX_BUFFER_SIZE],
    tx_frame_len: usize,
    tx_has_deferred_frame: bool,

    lost: Option<Lost>,
}

/// Open the network backend from its config. Kept separate from [`NetWorker`] so the backend
/// connection (e.g. the gvproxy unixgram socket) is created once by the [`Net`] device and
/// **preserved across suspend/resume** — the worker borrows it and hands it back on reset, instead
/// of tearing the gateway connection down and reconnecting (which would drop gvproxy).
pub(crate) fn connect_backend(
    cfg_backend: VirtioNetBackend,
    vnet_features: u64,
) -> Result<Box<dyn NetBackend + Send>, ConnectError> {
    let _ = vnet_features; // only the Tap backend (Linux) uses it
    Ok(match cfg_backend {
        VirtioNetBackend::UnixstreamFd(fd) => {
            // SAFETY: we need to trust that the library user has configured
            // the backend with a healthy file descriptor.
            let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
            Box::new(Unixstream::new(owned_fd)) as Box<dyn NetBackend + Send>
        }
        VirtioNetBackend::UnixstreamPath(path) => {
            Box::new(Unixstream::open(path)?) as Box<dyn NetBackend + Send>
        }
        VirtioNetBackend::UnixgramFd(fd) => {
            // SAFETY: we need to trust that the library user has configured
            // the backend with a healthy file descriptor.
            let owned_fd = unsafe { OwnedFd::from_raw_fd(fd) };
            Box::new(Unixgram::new(owned_fd)) as Box<dyn NetBackend + Send>
        }
        VirtioNetBackend::UnixgramPath(path, vfkit_magic) => {
            Box::new(Unixgram::open(path, vfkit_magic)?) as Box<dyn NetBackend + Send>
        }
        #[cfg(target_os = "linux")]
        VirtioNetBackend::Tap(tap_name) => {
            Box::new(Tap::new(tap_name, vnet_features)?) as Box<dyn NetBackend + Send>
        }
    })
}

impl NetWorker {
    pub fn new(
        rx_q: DeviceQueue,
        tx_q: DeviceQueue,
        interrupt: InterruptTransport,
        mem: GuestMemoryMmap,
        backend: Box<dyn NetBackend + Send>,
        stop_fd: EventFd,
        rx_data_valid: bool,
    ) -> Self {
        Self {
            rx_q,
            tx_q,

            mem,
            backend,
            interrupt,
            stop_fd,

            rx_frame_buf: [0u8; MAX_BUFFER_SIZE],
            rx_frame_buf_len: 0,
            rx_has_deferred_frame: false,
            rx_data_valid,

            tx_frame_buf: [0u8; MAX_BUFFER_SIZE],
            tx_frame_len: 0,
            tx_iovec: Vec::with_capacity(QUEUE_SIZE as usize),
            tx_has_deferred_frame: false,

            lost: None,
        }
    }

    /// Run the worker on its own thread. The join handle yields the backend back when the worker
    /// stops (on reset), so the device can reuse the same gateway connection on re-activate.
    pub fn run(self) -> thread::JoinHandle<Box<dyn NetBackend + Send>> {
        thread::Builder::new()
            .name("virtio-net worker".into())
            .spawn(|| self.work())
            .unwrap()
    }

    fn work(mut self) -> Box<dyn NetBackend + Send> {
        #[cfg(target_os = "macos")]
        const TX_TIMER_FD: RawFd = -2;

        let virtq_rx_ev_fd = self.rx_q.event.as_raw_fd();
        let virtq_tx_ev_fd = self.tx_q.event.as_raw_fd();
        let mut backend_socket = self.backend.raw_socket_fd();
        let stop_ev_fd = self.stop_fd.as_raw_fd();

        let mut epoll = Epoll::new().unwrap();

        let _ = epoll.ctl(
            ControlOperation::Add,
            stop_ev_fd,
            &EpollEvent::new(EventSet::IN, stop_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            virtq_rx_ev_fd,
            &EpollEvent::new(EventSet::IN, virtq_rx_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            virtq_tx_ev_fd,
            &EpollEvent::new(EventSet::IN, virtq_tx_ev_fd as u64),
        );
        register_backend(&epoll, backend_socket);

        let mut epoll_events = vec![EpollEvent::new(EventSet::empty(), 0); 32];
        'poll: loop {
            let timeout = self.reconnect_timeout(Instant::now());
            match epoll.wait(epoll_events.len(), timeout, epoll_events.as_mut_slice()) {
                Ok(ev_cnt) => {
                    for event in &epoll_events[0..ev_cnt] {
                        let source = event.fd();
                        let event_set = event.event_set();
                        match event_set {
                            EventSet::IN if source == stop_ev_fd => {
                                // Reset/suspend: drain the signal and exit so the device can be
                                // cleanly re-activated (new queues) on resume.
                                let _ = self.stop_fd.read();
                                break 'poll;
                            }
                            EventSet::IN if source == virtq_rx_ev_fd => {
                                self.process_rx_queue_event();
                            }
                            EventSet::IN if source == virtq_tx_ev_fd => {
                                self.process_tx_queue_event();
                            }
                            _ if source == backend_socket => {
                                if event_set.contains(EventSet::HANG_UP)
                                    || event_set.contains(EventSet::READ_HANG_UP)
                                {
                                    self.backend_lost(&format!("{event_set:?} on its socket"));
                                } else {
                                    if event_set.contains(EventSet::IN) {
                                        self.process_backend_socket_readable()
                                    }

                                    if event_set.contains(EventSet::OUT) {
                                        self.process_backend_socket_writeable()
                                    }
                                }
                            }
                            #[cfg(target_os = "macos")]
                            _ if event_set.is_empty() && source == TX_TIMER_FD => {
                                self.process_tx_loop();
                            }
                            _ => {
                                log::warn!(
                                    "Received unknown event: {event_set:?} from fd: {source:?}"
                                );
                            }
                        }
                    }

                    if self.try_reconnect(Instant::now()) {
                        let fd = self.backend.raw_socket_fd();
                        if fd != backend_socket {
                            let _ = epoll.ctl(
                                ControlOperation::Delete,
                                backend_socket,
                                &EpollEvent::new(EventSet::empty(), backend_socket as u64),
                            );
                            backend_socket = fd;
                            register_backend(&epoll, backend_socket);
                        }
                        self.process_tx_loop();
                        self.process_backend_socket_readable();
                    }

                    // Arm the retry timer after processing all events, so it
                    // reflects the final state of tx_has_deferred_frame.
                    #[cfg(target_os = "macos")]
                    if self.tx_has_deferred_frame {
                        let delay = self.backend.write_retry_delay_us();
                        if delay > 0 {
                            epoll.add_oneshot_timer(delay, TX_TIMER_FD as u64);
                        }
                    }
                }
                Err(e) => {
                    debug!("vsock: failed to consume muxer epoll event: {e}");
                }
            }
        }

        // The poll loop broke on `stop_fd` (reset): hand the backend connection back so the device
        // can reuse it when the guest re-activates the NIC on resume, instead of reconnecting.
        self.backend
    }

    /// The proxy went away: drop the guest's frames from here on and start reconnecting.
    fn backend_lost(&mut self, why: &str) {
        if self.lost.is_some() {
            return;
        }
        log::warn!(
            "virtio-net: the network backend went away ({why}); the guest's frames are dropped \
             until it can be reconnected"
        );
        self.lost = Some(Lost {
            next_try: Instant::now(),
            delay: RECONNECT_FIRST,
            for_good: false,
        });
    }

    /// The epoll timeout that wakes the worker for the next reconnect attempt, or -1.
    fn reconnect_timeout(&self, now: Instant) -> i32 {
        match &self.lost {
            Some(lost) if !lost.for_good => {
                lost.next_try.saturating_duration_since(now).as_millis() as i32
            }
            _ => -1,
        }
    }

    /// Reconnect a lost backend if an attempt is due. True when the backend is back.
    fn try_reconnect(&mut self, now: Instant) -> bool {
        let Some(lost) = &mut self.lost else {
            return false;
        };
        if lost.for_good || now < lost.next_try {
            return false;
        }
        match self.backend.reconnect() {
            Reconnect::Done => {
                log::info!("virtio-net: reconnected to the network backend");
                self.lost = None;
                true
            }
            Reconnect::NotYet(e) => {
                log::debug!(
                    "virtio-net: the network backend is not back yet ({e}); retrying in {:?}",
                    lost.delay
                );
                lost.next_try = now + lost.delay;
                lost.delay = cmp::min(lost.delay * 2, RECONNECT_MAX);
                false
            }
            Reconnect::Unsupported => {
                log::error!(
                    "virtio-net: the network backend was handed over as a descriptor and cannot \
                     be reconnected; networking is now disabled"
                );
                lost.for_good = true;
                false
            }
        }
    }

    fn process_rx_queue_event(&mut self) {
        if let Err(e) = self.rx_q.event.read() {
            log::error!("Failed to get rx event from queue: {e:?}");
        }
        if let Err(e) = self.rx_q.queue.disable_notification(&self.mem) {
            error!("error disabling queue notifications: {e:?}");
        }
        if let Err(e) = self.process_rx() {
            log::error!("Failed to process rx: {e:?} (triggered by queue event)")
        };
        if let Err(e) = self.rx_q.queue.enable_notification(&self.mem) {
            error!("error disabling queue notifications: {e:?}");
        }
    }

    fn process_tx_queue_event(&mut self) {
        match self.tx_q.event.read() {
            Ok(_) => self.process_tx_loop(),
            Err(e) => {
                log::error!("Failed to get tx queue event from queue: {e:?}");
            }
        }
    }

    fn process_backend_socket_readable(&mut self) {
        if let Err(e) = self.rx_q.queue.enable_notification(&self.mem) {
            error!("error disabling queue notifications: {e:?}");
        }
        if let Err(e) = self.process_rx() {
            log::error!("Failed to process rx: {e:?} (triggered by backend socket readable)");
        };
        if let Err(e) = self.rx_q.queue.disable_notification(&self.mem) {
            error!("error disabling queue notifications: {e:?}");
        }
    }

    fn process_backend_socket_writeable(&mut self) {
        match self
            .backend
            .try_finish_write(VNET_HDR_LEN, &self.tx_frame_buf[..self.tx_frame_len])
        {
            Ok(()) => self.process_tx_loop(),
            Err(WriteError::PartialWrite | WriteError::NothingWritten) => {}
            Err(e @ WriteError::Internal(_)) => {
                log::error!("Failed to finish write: {e:?}");
            }
            Err(e @ WriteError::ProcessNotRunning) => {
                log::debug!("Failed to finish write: {e:?}");
            }
        }
    }

    fn process_rx(&mut self) -> result::Result<(), RxError> {
        // if we have a deferred frame we try to process it first,
        // if that is not possible, we don't continue processing other frames
        if self.rx_has_deferred_frame {
            if self.write_frame_to_guest() {
                self.rx_has_deferred_frame = false;
            } else {
                return Ok(());
            }
        }

        let mut signal_queue = false;

        // Read as many frames as possible.
        let result = loop {
            match self.read_into_rx_frame_buf_from_backend() {
                Ok(()) => {
                    if self.write_frame_to_guest() {
                        signal_queue = true;
                    } else {
                        self.rx_has_deferred_frame = true;
                        break Ok(());
                    }
                }
                Err(ReadError::NothingRead) => break Ok(()),
                Err(e @ ReadError::Internal(_)) => break Err(RxError::Backend(e)),
            }
        };

        // At this point we processed as many Rx frames as possible. Wake the guest only if it is
        // waiting: with EVENT_IDX, NAPI parks `used_event` while it polls and trusts the device
        // not to interrupt again until it moves, so signalling every drain costs the guest an
        // interrupt per frame.
        if signal_queue
            && self
                .rx_q
                .queue
                .needs_notification(&self.mem)
                .unwrap_or(true)
        {
            self.interrupt
                .try_signal_used_queue()
                .map_err(RxError::DeviceError)?;
        }

        result
    }

    fn process_tx_loop(&mut self) {
        loop {
            self.tx_q.queue.disable_notification(&self.mem).unwrap();

            self.tx_has_deferred_frame = match self.process_tx() {
                Err(TxError::Backend(WriteError::NothingWritten)) => true,
                Err(e) => {
                    log::error!("Failed to process tx: {e:?}");
                    false
                }
                _ => false,
            };

            let has_new_entries = self.tx_q.queue.enable_notification(&self.mem).unwrap();
            if self.tx_has_deferred_frame || !has_new_entries {
                break;
            }
        }
    }

    fn process_tx(&mut self) -> result::Result<(), TxError> {
        if self.lost.is_some() && self.try_reconnect(Instant::now()) {
            self.process_backend_socket_readable();
        }
        let tx_queue = &mut self.tx_q.queue;

        if self.backend.has_unfinished_write()
            && self
                .backend
                .try_finish_write(VNET_HDR_LEN, &self.tx_frame_buf[..self.tx_frame_len])
                .is_err()
        {
            log::trace!("Cannot process tx because of unfinished partial write!");
            return Ok(());
        }

        let mut raise_irq = false;
        let mut result = Ok(());
        let mut went_away = false;

        while let Some(head) = tx_queue.pop(&self.mem) {
            let head_index = head.index;
            let mut next_desc = Some(head);

            self.tx_iovec.clear();
            while let Some(desc) = next_desc {
                if desc.is_write_only() {
                    self.tx_iovec.clear();
                    break;
                }
                self.tx_iovec.push((desc.addr, desc.len as usize));
                next_desc = desc.next_descriptor();
            }

            // Copy buffer from across multiple descriptors.
            let mut read_count = 0;
            for (desc_addr, desc_len) in self.tx_iovec.drain(..) {
                let limit = cmp::min(read_count + desc_len, self.tx_frame_buf.len());

                let read_result = self
                    .mem
                    .read_slice(&mut self.tx_frame_buf[read_count..limit], desc_addr);
                match read_result {
                    Ok(()) => {
                        read_count += limit - read_count;
                    }
                    Err(e) => {
                        log::error!("Failed to read slice: {e:?}");
                        read_count = 0;
                        break;
                    }
                }
            }

            self.tx_frame_len = read_count;
            match self
                .backend
                .write_frame(VNET_HDR_LEN, &mut self.tx_frame_buf[..read_count])
            {
                Ok(()) => {
                    self.tx_frame_len = 0;
                    tx_queue
                        .add_used(&self.mem, head_index, 0)
                        .map_err(TxError::QueueError)?;
                    raise_irq = true;
                }
                Err(WriteError::NothingWritten) => {
                    tx_queue.undo_pop();
                    result = Err(TxError::Backend(WriteError::NothingWritten));
                    break;
                }
                Err(WriteError::PartialWrite) => {
                    log::trace!("process_tx: partial write");
                    /*
                    This situation should be pretty rare, assuming reasonably sized socket buffers.
                    We have written only a part of a frame to the backend socket (the socket is full).

                    The frame we have read from the guest remains in tx_frame_buf, and will be sent
                    later.

                    Note that we cannot wait for the backend to process our sending frames, because
                    the backend could be blocked on sending a remainder of a frame to us - us waiting
                    for backend would cause a deadlock.
                     */
                    tx_queue
                        .add_used(&self.mem, head_index, 0)
                        .map_err(TxError::QueueError)?;
                    raise_irq = true;
                    break;
                }
                // The frame is lost, as on a cable with nobody at the other end, but its
                // descriptor goes back to the guest: one kept would shrink the TX ring for good.
                Err(WriteError::ProcessNotRunning) => {
                    self.tx_frame_len = 0;
                    tx_queue
                        .add_used(&self.mem, head_index, 0)
                        .map_err(TxError::QueueError)?;
                    raise_irq = true;
                    went_away = true;
                }
                Err(e @ WriteError::Internal(_)) => {
                    return Err(TxError::Backend(e));
                }
            }
        }

        if raise_irq && tx_queue.needs_notification(&self.mem).unwrap() {
            self.interrupt
                .try_signal_used_queue()
                .map_err(TxError::DeviceError)?;
        }
        if went_away {
            self.backend_lost("a write found the proxy gone");
        }

        result
    }

    // Copies a single frame from `self.rx_frame_buf` into the guest.
    fn write_frame_to_guest_impl(&mut self) -> result::Result<(), FrontendError> {
        let mut result: std::result::Result<(), FrontendError> = Ok(());

        let queue = &mut self.rx_q.queue;
        let head_descriptor = queue.pop(&self.mem).ok_or(FrontendError::EmptyQueue)?;
        let head_index = head_descriptor.index;

        let mut frame_slice = &self.rx_frame_buf[..self.rx_frame_buf_len];

        let frame_len = frame_slice.len();
        let mut maybe_next_descriptor = Some(head_descriptor);
        while let Some(descriptor) = &maybe_next_descriptor {
            if frame_slice.is_empty() {
                break;
            }

            if !descriptor.is_write_only() {
                result = Err(FrontendError::ReadOnlyDescriptor);
                break;
            }

            let len = std::cmp::min(frame_slice.len(), descriptor.len as usize);
            match self.mem.write_slice(&frame_slice[..len], descriptor.addr) {
                Ok(()) => {
                    frame_slice = &frame_slice[len..];
                }
                Err(e) => {
                    log::error!("Failed to write slice: {e:?}");
                    result = Err(FrontendError::GuestMemory(e));
                    break;
                }
            };

            maybe_next_descriptor = descriptor.next_descriptor();
        }
        if result.is_ok() && !frame_slice.is_empty() {
            log::warn!("Receiving buffer is too small to hold frame of current size");
            result = Err(FrontendError::DescriptorChainTooSmall);
        }

        // Mark the descriptor chain as used. If an error occurred, skip the descriptor chain.
        let used_len = if result.is_err() { 0 } else { frame_len as u32 };
        queue
            .add_used(&self.mem, head_index, used_len)
            .map_err(FrontendError::QueueError)?;
        result
    }

    // Copies a single frame from `self.rx_frame_buf` into the guest. In case of an error retries
    // the operation if possible. Returns true if the operation was successfull.
    fn write_frame_to_guest(&mut self) -> bool {
        let max_iterations = self.rx_q.queue.actual_size();
        for _ in 0..max_iterations {
            match self.write_frame_to_guest_impl() {
                Ok(()) => return true,
                Err(FrontendError::EmptyQueue) => {
                    // retry
                    continue;
                }
                Err(_) => {
                    // retry
                    continue;
                }
            }
        }

        false
    }

    /// Fills self.rx_frame_buf with an ethernet frame from backend and prepends virtio_net_hdr to it
    fn read_into_rx_frame_buf_from_backend(&mut self) -> result::Result<(), ReadError> {
        self.rx_frame_buf_len = self.backend.read_frame(&mut self.rx_frame_buf)?;
        // A proxy's frames come out of its own network stack over a local socket, checksums
        // already right; without the flag the guest re-sums every byte it receives.
        if self.rx_data_valid && self.backend.synthesizes_vnet_hdr() {
            self.rx_frame_buf[0] = VIRTIO_NET_HDR_F_DATA_VALID as u8;
        }
        Ok(())
    }
}

fn register_backend(epoll: &Epoll, fd: RawFd) {
    let _ = epoll.ctl(
        ControlOperation::Add,
        fd,
        &EpollEvent::new(
            EventSet::IN | EventSet::OUT | EventSet::EDGE_TRIGGERED | EventSet::READ_HANG_UP,
            fd as u64,
        ),
    );
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use utils::eventfd::EFD_NONBLOCK;
    use vm_memory::GuestAddress;

    use super::*;
    use crate::legacy::DummyIrqChip;
    use crate::virtio::VIRTIO_MMIO_INT_VRING;
    use crate::virtio::net::write_virtio_net_hdr;
    use crate::virtio::queue::VIRTQ_DESC_F_WRITE;
    use crate::virtio::queue::tests::VirtQueue as GuestQueue;

    /// A backend holding the frames the proxy has queued for the guest; `synthesizes` false
    /// stands in for a tap, which relays the host kernel's header (blank here).
    struct Proxy {
        frames: VecDeque<Vec<u8>>,
        synthesizes: bool,
        link: Arc<Link>,
    }

    /// Whether the proxy is there, and what reached it.
    #[derive(Default)]
    struct Link {
        down: AtomicBool,
        written: AtomicUsize,
    }

    impl NetBackend for Proxy {
        fn synthesizes_vnet_hdr(&self) -> bool {
            self.synthesizes
        }
        fn read_frame(&mut self, buf: &mut [u8]) -> result::Result<usize, ReadError> {
            let frame = self.frames.pop_front().ok_or(ReadError::NothingRead)?;
            let hdr_len = write_virtio_net_hdr(buf);
            buf[hdr_len..hdr_len + frame.len()].copy_from_slice(&frame);
            Ok(hdr_len + frame.len())
        }
        fn write_frame(&mut self, _: usize, _: &mut [u8]) -> result::Result<(), WriteError> {
            if self.link.down.load(Ordering::SeqCst) {
                return Err(WriteError::ProcessNotRunning);
            }
            self.link.written.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn reconnect(&mut self) -> Reconnect {
            if self.link.down.load(Ordering::SeqCst) {
                Reconnect::NotYet(nix::Error::ENOENT)
            } else {
                Reconnect::Done
            }
        }
        fn has_unfinished_write(&self) -> bool {
            false
        }
        fn try_finish_write(&mut self, _: usize, _: &[u8]) -> result::Result<(), WriteError> {
            Ok(())
        }
        fn raw_socket_fd(&self) -> RawFd {
            -1
        }
    }

    fn memory() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x20000)]).unwrap()
    }

    /// A worker whose RX ring holds `buffers` posted 4 KiB buffers, with EVENT_IDX negotiated,
    /// and whose proxy has `frames` small frames waiting.
    fn worker(
        mem: &GuestMemoryMmap,
        rx: &GuestQueue,
        tx: &GuestQueue,
        buffers: u16,
        frames: usize,
    ) -> (NetWorker, InterruptTransport) {
        worker_with(mem, rx, tx, buffers, frames, false, true)
    }

    fn worker_with(
        mem: &GuestMemoryMmap,
        rx: &GuestQueue,
        tx: &GuestQueue,
        buffers: u16,
        frames: usize,
        guest_csum: bool,
        synthesizes: bool,
    ) -> (NetWorker, InterruptTransport) {
        for i in 0..buffers {
            rx.dtable[i as usize].set(0x8000 + i as u64 * 0x1000, 0x1000, VIRTQ_DESC_F_WRITE, 0);
            rx.avail.ring[i as usize].set(i);
        }
        rx.avail.idx.set(buffers);
        let queue = |q: &GuestQueue| {
            let mut queue = q.create_queue();
            queue.set_event_idx(true);
            DeviceQueue::new(queue, Arc::new(EventFd::new(EFD_NONBLOCK).unwrap()))
        };
        let interrupt =
            InterruptTransport::new(DummyIrqChip::new().into(), "net-test".to_string()).unwrap();
        let proxy = Proxy {
            frames: (0..frames).map(|_| vec![0xab; 60]).collect(),
            synthesizes,
            link: Arc::default(),
        };
        let worker = NetWorker::new(
            queue(rx),
            queue(tx),
            interrupt.clone(),
            mem.clone(),
            Box::new(proxy),
            EventFd::new(EFD_NONBLOCK).unwrap(),
            guest_csum,
        );
        (worker, interrupt)
    }

    fn take_interrupt(interrupt: &InterruptTransport) -> bool {
        interrupt.status().swap(0, Ordering::SeqCst) & VIRTIO_MMIO_INT_VRING as usize != 0
    }

    #[test]
    fn a_guest_still_polling_its_rx_ring_is_not_interrupted() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, interrupt) = worker(&mem, &rx, &tx, 8, 2);

        // NAPI parks used_event while it polls; the frames land, the doorbell stays quiet.
        rx.avail.event.set(8);
        worker.process_backend_socket_readable();

        assert_eq!(rx.used.idx.get(), 2, "both frames were delivered");
        assert!(
            !take_interrupt(&interrupt),
            "used index 2 has not passed used_event 8"
        );
    }

    /// The virtio-net header flags the guest found on the first RX buffer.
    fn delivered_hdr_flags(mem: &GuestMemoryMmap) -> u8 {
        mem.read_obj(GuestAddress(0x8000)).unwrap()
    }

    #[test]
    fn a_frame_from_the_proxy_reaches_a_csum_guest_marked_valid() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, _interrupt) = worker_with(&mem, &rx, &tx, 8, 1, true, true);

        worker.process_backend_socket_readable();

        // The proxy built this frame in its own stack and handed it over a local socket; with
        // the flag the guest skips re-summing every byte.
        assert_eq!(delivered_hdr_flags(&mem), VIRTIO_NET_HDR_F_DATA_VALID as u8);
    }

    #[test]
    fn a_guest_without_guest_csum_gets_a_zero_header() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, _interrupt) = worker_with(&mem, &rx, &tx, 8, 1, false, true);

        worker.process_backend_socket_readable();

        assert_eq!(delivered_hdr_flags(&mem), 0);
    }

    #[test]
    fn a_backend_that_brings_its_own_header_keeps_it() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, _interrupt) = worker_with(&mem, &rx, &tx, 8, 1, true, false);

        worker.process_backend_socket_readable();

        // A tap hands over the kernel's own header; its flags are the kernel's to set.
        assert_eq!(delivered_hdr_flags(&mem), 0);
    }

    #[test]
    fn a_guest_waiting_on_its_rx_ring_is_interrupted() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, interrupt) = worker(&mem, &rx, &tx, 8, 2);

        rx.avail.event.set(0);
        worker.process_backend_socket_readable();

        assert!(
            take_interrupt(&interrupt),
            "used index 2 passes used_event 0"
        );
    }

    /// Queue `count` guest frames on the TX ring, after the `posted` already there.
    fn post_tx(tx: &GuestQueue, posted: u16, count: u16) {
        for i in posted..posted + count {
            tx.dtable[i as usize].set(0x10000 + i as u64 * 0x100, 0x60, 0, 0);
            tx.avail.ring[i as usize].set(i);
        }
        tx.avail.idx.set(posted + count);
    }

    /// A proxy that goes away loses the guest's frames but not its descriptors, and once it is
    /// back the worker reconnects on the backoff and frames reach it again.
    #[test]
    fn frames_sent_while_the_proxy_is_gone_are_dropped_and_returned_to_the_guest() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let link = Arc::new(Link::default());
        let (mut worker, _interrupt) = worker(&mem, &rx, &tx, 0, 0);
        worker.backend = Box::new(Proxy {
            frames: VecDeque::new(),
            synthesizes: true,
            link: link.clone(),
        });

        link.down.store(true, Ordering::SeqCst);
        post_tx(&tx, 0, 4);
        worker.process_tx_loop();
        assert_eq!(
            tx.used.idx.get(),
            4,
            "every frame's descriptor went back to the guest"
        );
        assert_eq!(link.written.load(Ordering::SeqCst), 0);
        assert!(
            worker.lost.is_some(),
            "and the worker knows the proxy is gone"
        );

        let now = Instant::now();
        assert!(!worker.try_reconnect(now), "the proxy is not back yet");
        link.down.store(false, Ordering::SeqCst);
        assert!(
            !worker.try_reconnect(now),
            "the next attempt waits for the backoff"
        );
        assert!(worker.try_reconnect(now + RECONNECT_FIRST));
        assert!(worker.lost.is_none());

        post_tx(&tx, 4, 2);
        worker.process_tx_loop();
        assert_eq!(tx.used.idx.get(), 6);
        assert_eq!(
            link.written.load(Ordering::SeqCst),
            2,
            "frames reach the proxy again"
        );
    }

    /// A backend that cannot reconnect is given up on once, not retried.
    #[test]
    fn a_backend_that_cannot_reconnect_is_given_up_on() {
        let mem = memory();
        let rx = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        let tx = GuestQueue::new(GuestAddress(0x3000), &mem, 16);
        let (mut worker, _interrupt) = worker(&mem, &rx, &tx, 0, 0);
        worker.backend_lost("a test");
        // The default `reconnect` of a backend that does not override it.
        struct Fd;
        impl NetBackend for Fd {
            fn read_frame(&mut self, _: &mut [u8]) -> result::Result<usize, ReadError> {
                Err(ReadError::NothingRead)
            }
            fn write_frame(&mut self, _: usize, _: &mut [u8]) -> result::Result<(), WriteError> {
                Err(WriteError::ProcessNotRunning)
            }
            fn has_unfinished_write(&self) -> bool {
                false
            }
            fn try_finish_write(&mut self, _: usize, _: &[u8]) -> result::Result<(), WriteError> {
                Ok(())
            }
            fn raw_socket_fd(&self) -> RawFd {
                -1
            }
        }
        worker.backend = Box::new(Fd);
        let now = Instant::now();
        assert!(!worker.try_reconnect(now));
        assert_eq!(
            worker.reconnect_timeout(now),
            -1,
            "nothing left to wake up for"
        );
        assert!(!worker.try_reconnect(now + RECONNECT_MAX));
    }
}
