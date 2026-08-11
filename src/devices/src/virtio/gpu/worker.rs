use std::collections::VecDeque;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use utils::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use utils::eventfd::EventFd;

#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
use rutabaga_gfx::{ResourceCreate3D, ResourceCreateBlob, RutabagaFence, Transfer3D};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;
use vm_memory::{GuestAddress, GuestMemoryMmap};

use super::super::descriptor_utils::{Reader, Writer};
use super::super::{DeviceQueue, GpuError, Queue as VirtQueue};
use super::protocol::{
    GpuCommand, GpuResponse, VirtioGpuResult, virtio_gpu_ctrl_hdr, virtio_gpu_mem_entry,
};
use super::virtio_gpu::{GpuActivation, VirtioGpu};
use crate::display::DisplayInfo;
use crate::virtio::fs::ExportTable;
use crate::virtio::gpu::device::DisplayUpdate;
use crate::virtio::gpu::protocol::{
    VIRTIO_GPU_EVENT_DISPLAY, VIRTIO_GPU_FLAG_FENCE, VIRTIO_GPU_FLAG_INFO_RING_IDX,
};
use crate::virtio::gpu::virtio_gpu::VirtioGpuRing;
use crate::virtio::{InterruptTransport, VirtioShmRegion};
use krun_display::DisplayBackend;
use krun_display::Rect;

/// Commands the device sends to the long-lived worker thread.
///
/// limina (renderer-singleton fix): `virgl_renderer_init` is a process-global, init-once,
/// thread-bound singleton, so the worker (which owns the `Rutabaga`/renderer) must live for the
/// whole VMM process and stay on one thread. `activate()`/`reset()` no longer spawn/join it —
/// they message it to bind/unbind the per-activation transport (the control/cursor queues, guest
/// memory, and interrupt, which are recreated on every activation).
pub enum WorkerCmd {
    /// (Re)bind the transport and start servicing the queues.
    Activate {
        mem: GuestMemoryMmap,
        control_q: DeviceQueue,
        cursor_q: DeviceQueue,
        interrupt: InterruptTransport,
        /// limina (host-sleep s2idle): the transport re-armed this activation's queues
        /// from the previous activation's register file — a no-PM-ops driver's
        /// bus-fallback thaw, not a real re-init. Classifies a parked session (below).
        thaw: bool,
    },
    /// Stop servicing and release the transport, but keep the renderer alive (device reset).
    Deactivate,
    /// Tear the worker down (device drop / VMM teardown).
    Shutdown,
    /// limina M9.3: serialize the GPU re-creation state (rutabaga journal + venus wire
    /// journals + mapped-blob contents) for the VMM snapshot. Sent with the vCPUs quiesced,
    /// so no queue traffic races the capture.
    Snapshot {
        reply: std::sync::mpsc::Sender<Option<Vec<u8>>>,
    },
}

/// How the inner service loop ended.
enum InnerExit {
    Deactivate,
    Shutdown,
}

pub struct Worker {
    /// Wakes the inner epoll loop when a [`WorkerCmd`] is waiting on `cmd_rx` (the queue fds
    /// epoll can't observe the channel). Persistent across activations.
    stop_fd: EventFd,
    /// Activation/teardown commands from the device.
    cmd_rx: std::sync::mpsc::Receiver<WorkerCmd>,
    /// The current activation's transport, shared with the renderer's fence handler (which is
    /// registered once and outlives any activation). `None` while inactive.
    active: Arc<Mutex<Option<GpuActivation>>>,
    shm_region: VirtioShmRegion,
    virgl_flags: u32,
    /// limina: serve only the software-2D scanout path; don't init virglrenderer/rutabaga.
    software_2d: bool,
    #[cfg(target_os = "macos")]
    map_sender: Sender<WorkerMessage>,
    export_table: Option<ExportTable>,
    displays: Box<[DisplayInfo]>,
    display_backend: DisplayBackend<'static>,
    /// limina runtime resize: config-space `events_read`, shared with the device. The worker
    /// sets VIRTIO_GPU_EVENT_DISPLAY here before raising a config-change; the guest clears it.
    events_read: Arc<AtomicU32>,
    /// Wakes the service loop's epoll when a resize is pending (shared with the device + handle).
    resize_evt: Arc<EventFd>,
    /// Queued display updates; one is taken per `resize_evt` wake so each reaches the guest
    /// as its own config-change event.
    resize_pending: Arc<Mutex<VecDeque<DisplayUpdate>>>,
    /// limina M9.3: a staged GPU snapshot payload ('LGPU'), taken and replayed on the next
    /// activation. Staged by the restore path; the guest's thaw resets the device (bus
    /// fallback: reset → features → DRIVER_OK), so replay must run AFTER that reset's
    /// `reset_session` and before any queue command — i.e. exactly at activation.
    pending_restore: Arc<Mutex<Option<Vec<u8>>>>,
    /// limina M9.3: flipped true (+ notify) once the staged replay finished; the device's
    /// `activate` blocks the guest's DRIVER_OK on it (see `Gpu::restore_done`).
    restore_done: Arc<(Mutex<bool>, Condvar)>,
}

impl Worker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        shm_region: VirtioShmRegion,
        virgl_flags: u32,
        software_2d: bool,
        stop_fd: EventFd,
        cmd_rx: std::sync::mpsc::Receiver<WorkerCmd>,
        active: Arc<Mutex<Option<GpuActivation>>>,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
        export_table: Option<ExportTable>,
        displays: Box<[DisplayInfo]>,
        display_backend: DisplayBackend<'static>,
        events_read: Arc<AtomicU32>,
        resize_evt: Arc<EventFd>,
        resize_pending: Arc<Mutex<VecDeque<DisplayUpdate>>>,
        pending_restore: Arc<Mutex<Option<Vec<u8>>>>,
        restore_done: Arc<(Mutex<bool>, Condvar)>,
    ) -> Self {
        Self {
            stop_fd,
            cmd_rx,
            active,
            shm_region,
            virgl_flags,
            software_2d,
            #[cfg(target_os = "macos")]
            map_sender,
            export_table,
            displays,
            display_backend,
            events_read,
            resize_evt,
            resize_pending,
            pending_restore,
            restore_done,
        }
    }

    pub fn run(self) -> thread::JoinHandle<()> {
        thread::Builder::new()
            .name("gpu worker".into())
            .spawn(|| self.work())
            .unwrap()
    }

    /// The worker thread body: build the renderer ONCE (it's a process-global, thread-bound
    /// singleton), then loop activating/deactivating as the device binds/unbinds transport.
    fn work(mut self) {
        // The renderer + fence handler are created once and live for the whole process. The
        // fence handler reaches the current activation's queue/mem/interrupt through `active`.
        let mut virtio_gpu = VirtioGpu::new(
            self.active.clone(),
            self.virgl_flags,
            self.software_2d,
            #[cfg(target_os = "macos")]
            self.map_sender.clone(),
            self.export_table.take(),
            self.displays.clone(),
            self.display_backend,
        );

        // limina (host-sleep s2idle, defer-and-classify): a device reset no longer wipes the
        // session immediately — if it was quiescent, the bookkeeping is PARKED and this flag
        // set, and the NEXT activation classifies it: a re-armed activation (`thaw`, the
        // bus-fallback s2idle signature) adopts the parked world unchanged (the thawing guest
        // is the same live kernel — its ids still match); a queue-programming activation is a
        // real re-init (reboot, rebind, firmware hand-off) and runs the deferred wipe first.
        let mut parked_session = false;

        loop {
            // Inactive: block on the next command from the device.
            let (mem, control_q, cursor_q, interrupt, thaw) = match self.cmd_rx.recv() {
                Ok(WorkerCmd::Activate {
                    mem,
                    control_q,
                    cursor_q,
                    interrupt,
                    thaw,
                }) => (mem, control_q, cursor_q, interrupt, thaw),
                // A Deactivate while already inactive is a no-op; Shutdown / closed channel ends.
                Ok(WorkerCmd::Deactivate) => continue,
                Ok(WorkerCmd::Shutdown) | Err(_) => {
                    if parked_session {
                        virtio_gpu.reset_session();
                    }
                    return;
                }
                // limina M9.3: a snapshot request can arrive while inactive (unlikely — the
                // production suspend path snapshots with the device DRIVER_OK — but harmless).
                Ok(WorkerCmd::Snapshot { reply }) => {
                    Self::drain_fences(&virtio_gpu);
                    let _ = reply.send(virtio_gpu.snapshot_gpu_payload());
                    continue;
                }
            };

            // Classify a parked session against this activation (see `parked_session`).
            if parked_session {
                parked_session = false;
                if thaw {
                    info!(
                        "gpu worker: adopting the parked session across a bus-fallback thaw \
                         (in-place s2idle; no replay needed)"
                    );
                } else {
                    info!(
                        "gpu worker: dropping the parked session — this activation programmed \
                         its queues (real driver re-init)"
                    );
                    virtio_gpu.reset_session();
                }
            }

            // Own each activation's queue eventfds + queues; publish the transport so the
            // long-lived fence handler retires guest fences into THIS activation's queue.
            // limina wake-probe: watch the TRANSPORT's fd, not the try_clone below — try_clone
            // dups, so the worker's copy has a different raw fd and would never match the
            // doorbell the vCPU stamps. (This is exactly what the first run got wrong: every
            // wake landed in `no_kick`.)
            crate::virtio::wake_probe::watch(control_q.event.as_raw_fd());
            let control_evt = control_q.event.try_clone().unwrap();
            let cursor_evt = cursor_q.event.try_clone().unwrap();
            let control_queue = Arc::new(Mutex::new(control_q.queue));
            let cursor_queue = Arc::new(Mutex::new(cursor_q.queue));
            *self.active.lock().unwrap() = Some(GpuActivation {
                mem: mem.clone(),
                control_queue: control_queue.clone(),
                interrupt: interrupt.clone(),
            });

            // limina M9.3: a staged GPU snapshot payload replays NOW — after the thaw
            // reset's `reset_session` (which would have wiped it) and before the first
            // queue command (whose venus state it re-creates). The guest's DRIVER_OK write
            // is BLOCKED on `restore_done` for the duration (see `Gpu::activate`), keeping
            // the thaw — and with it guest userspace — frozen until the blob mappings are
            // re-established; without that hold, mesa's first post-thaw writes raced the
            // replay and landed in unmapped shmem (run-14 ring FATAL).
            let staged = self.pending_restore.lock().unwrap().take();
            if let Some(data) = staged {
                if self.gpu_restore(&mut virtio_gpu, &data, &mem) {
                    info!("gpu worker: staged snapshot replay complete");
                } else {
                    warn!("gpu worker: staged snapshot replay failed; guest session will restart");
                }
                let (done_lock, done_cv) = &*self.restore_done;
                *done_lock.lock().unwrap() = true;
                done_cv.notify_all();
            }

            let exit = self.service(
                &mut virtio_gpu,
                &mem,
                &control_evt,
                &control_queue,
                &cursor_evt,
                &cursor_queue,
                &interrupt,
            );

            // Unbind the transport. What happens to the session bookkeeping depends on how
            // this activation ended (defer-and-classify, docs/design/host-sleep-s2idle.md §3):
            //
            // - Deactivate (guest device reset): DON'T wipe yet. If the session is quiescent —
            //   fence ledger drained AND nothing in the present plumbing references the freed
            //   queue — PARK it and let the next activation classify (thaw → adopt, re-init →
            //   deferred wipe). A thaw's reset arrives after the whole suspend, so the drain
            //   returns immediately there; a dirty reset with work in flight fails the check
            //   and takes the fail-closed wipe (exactly the old behavior). This is what keeps
            //   a seated venus session alive across in-place s2idle: the world was never lost,
            //   only this wipe destroyed it.
            // - Shutdown (device drop / VMM teardown): wipe as before — teardown ordering
            //   frees per-session state while the renderer thread is still alive.
            match exit {
                InnerExit::Deactivate => {
                    // Drain BEFORE unbinding the transport (§22 hardening, 2026-07-30):
                    // fence retirement (the async fence callbacks), the guest-hold latch
                    // (500 ms ceiling) and the present pump all need the live activation —
                    // once `active` is cleared the fence handler drops completions on the
                    // floor, so the old post-unbind drain could only ever succeed
                    // vacuously. Draining here lets a session that was merely MID-FRAME
                    // at the reset become quiescent and PARK instead of losing its whole
                    // venus world: the couve aborted-sleep wipe was an s2idle entry
                    // landing inside the compositor's fade-out animation.
                    Self::drain_for_classify(&mut virtio_gpu);
                    *self.active.lock().unwrap() = None;
                    let (outstanding, summary) = virtio_gpu.outstanding_fences();
                    if outstanding == 0 && virtio_gpu.present_quiescent() {
                        let (cookies, shown) = virtio_gpu.discard_stale_present_bookkeeping();
                        if shown > 100 {
                            // A backlog this deep means shown acks stopped draining long
                            // ago (every ack either missing or naming an unknown surface)
                            // — the leak we want field logs to surface at default level.
                            warn!(
                                "gpu worker: dropped {shown} awaiting-shown entries at \
                                 reset — the shown-ack loop was not draining this session \
                                 ({cookies} flush-parked cookie(s) also discarded)"
                            );
                        } else if cookies != 0 || shown != 0 {
                            info!(
                                "gpu worker: dropped stale present bookkeeping stranded by \
                                 the reset ({cookies} flush-parked cookie(s), {shown} \
                                 awaiting-shown entr(y/ies)) — forward-looking state with \
                                 no queue references"
                            );
                        }
                        parked_session = true;
                        info!(
                            "gpu worker: device reset with a quiescent session — PARKED for \
                             classification at the next activation"
                        );
                    } else {
                        warn!(
                            "gpu worker: device reset with the session NOT quiescent \
                             (fences: {outstanding} outstanding [{summary}]; {}) — wiping \
                             (fail-closed)",
                            virtio_gpu.present_pending_summary()
                        );
                        virtio_gpu.reset_session();
                    }
                    continue;
                }
                InnerExit::Shutdown => {
                    *self.active.lock().unwrap() = None;
                    virtio_gpu.reset_session();
                    return;
                }
            }
        }
    }

    /// limina (§22 hardening): bounded present+fence drain before classifying a device
    /// reset, run with the activation still bound. Pumps retired presents (this thread
    /// owns the display backend, same as the service loop's present_ev handling) while
    /// the async fence callbacks and the guest-hold latch retire the rest. The deadline
    /// comfortably covers the guest-hold ceiling (500 ms) plus a shown-ack round trip;
    /// a session that cannot drain (a guest that died mid-frame) pays it once and takes
    /// the fail-closed wipe exactly as before.
    fn drain_for_classify(virtio_gpu: &mut VirtioGpu) {
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_millis(1500);
        loop {
            virtio_gpu.process_retired_presents();
            let (outstanding, _) = virtio_gpu.outstanding_fences();
            if outstanding == 0 && virtio_gpu.present_quiescent() {
                info!("gpu worker: reset drain quiesced in {:?}", start.elapsed());
                return;
            }
            if std::time::Instant::now() >= deadline {
                warn!(
                    "gpu worker: reset drain timed out after {:?}",
                    start.elapsed()
                );
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    /// limina M9.3: drain in-flight guest fences before a snapshot capture. The guest
    /// froze mid-frame and its waiters hold sync_files backed by these fences; a fence
    /// whose completion misses the snapshot never signals in the restored epoch (eyeball
    /// 2026-07-19: gnome-shell resumed wedged in `vn_GetSemaphoreFdKHR → poll()`). The
    /// vCPUs are parked, so no new fences arrive; retirement runs on virglrenderer's
    /// async fence-callback threads (ASYNC_FENCE_CB), so it proceeds while we poll. The
    /// used-ring writes land in guest RAM (dumped after this) and the raised IRQs latch
    /// in the GIC (saved after the GPU capture) — restored waiters see signaled fences.
    fn drain_fences(virtio_gpu: &VirtioGpu) {
        let start = std::time::Instant::now();
        let deadline = start + std::time::Duration::from_secs(5);
        loop {
            let (n, summary) = virtio_gpu.outstanding_fences();
            if n == 0 {
                info!(
                    "gpu snapshot: fence ledger drained in {:?}",
                    start.elapsed()
                );
                return;
            }
            if std::time::Instant::now() >= deadline {
                warn!("gpu snapshot: fence drain TIMED OUT with {n} outstanding: {summary}");
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// limina M9.3: replay a GPU snapshot section into the (fresh) renderer. macOS-only
    /// mechanism today — the venus/KK stack this exists for; elsewhere report failure so
    /// the caller falls back to fresh-renderer behavior.
    fn gpu_restore(&self, virtio_gpu: &mut VirtioGpu, data: &[u8], mem: &GuestMemoryMmap) -> bool {
        #[cfg(target_os = "macos")]
        {
            virtio_gpu.restore_gpu_payload(data, &self.shm_region, mem)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (virtio_gpu, data, mem);
            false
        }
    }

    /// Service the bound transport's queues until the device deactivates or shuts down.
    #[allow(clippy::too_many_arguments)]
    fn service(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        mem: &GuestMemoryMmap,
        control_evt: &EventFd,
        control_queue: &Arc<Mutex<VirtQueue>>,
        cursor_evt: &EventFd,
        cursor_queue: &Arc<Mutex<VirtQueue>>,
        interrupt: &InterruptTransport,
    ) -> InnerExit {
        let control_ev_fd = control_evt.as_raw_fd();
        let cursor_ev_fd = cursor_evt.as_raw_fd();
        let stop_ev_fd = self.stop_fd.as_raw_fd();
        // limina runtime resize: the host (limina-vmm) kicks this fd to push a new display size.
        let resize_ev_fd = self.resize_evt.as_raw_fd();
        // limina (#8): retired present fences wake this fd; -1 (never matched) when
        // the renderer is absent.
        let present_ev_fd = virtio_gpu.present_event_fd().unwrap_or(-1);
        // limina (vrend fence honesty): virglrenderer's poll eventfd. vrend's sync thread
        // parks in `wait_sync` whenever GL queries are pending until this thread (the GL
        // thread) pumps `virgl_renderer_poll()`; unpumped, every fence behind the parked
        // one wedges — which matters now that a vrend session's Global-ring fences route
        // through virglrenderer (see `vrend_ctx_seen`). -1 when there's no renderer.
        let renderer_poll_fd = virtio_gpu.renderer_poll_fd().unwrap_or(-1);

        // A stop signal left over from a Deactivate that arrived while we were inactive (the
        // outer loop consumed the command but not the kick) just produces one spurious wake
        // here, handled by the `TryRecvError::Empty` branch below — no entry drain needed (and
        // draining here could eat a Deactivate kick that races this activation).
        let mut epoll = Epoll::new().unwrap();
        let _ = epoll.ctl(
            ControlOperation::Add,
            control_ev_fd,
            &EpollEvent::new(EventSet::IN, control_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            cursor_ev_fd,
            &EpollEvent::new(EventSet::IN, cursor_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            stop_ev_fd,
            &EpollEvent::new(EventSet::IN, stop_ev_fd as u64),
        );
        let _ = epoll.ctl(
            ControlOperation::Add,
            resize_ev_fd,
            &EpollEvent::new(EventSet::IN, resize_ev_fd as u64),
        );
        if present_ev_fd >= 0 {
            let _ = epoll.ctl(
                ControlOperation::Add,
                present_ev_fd,
                &EpollEvent::new(EventSet::IN, present_ev_fd as u64),
            );
        }
        if renderer_poll_fd >= 0 {
            let _ = epoll.ctl(
                ControlOperation::Add,
                renderer_poll_fd,
                &EpollEvent::new(EventSet::IN, renderer_poll_fd as u64),
            );
        }

        let mut epoll_events = vec![EpollEvent::new(EventSet::empty(), 0); 32];
        // limina wake-trace (LIMINA_WAKE_TRACE=1): per-source wake accounting for this
        // loop, ~5s cadence — see docs/perf/overhead-inventory.md.
        let mut wake_trace = std::env::var("LIMINA_WAKE_TRACE")
            .ok()
            .map(|_| (std::time::Instant::now(), [0u64; 6]));
        // limina wake-probe (LIMINA_WAKE_PROBE=1): the VMM half of the venus wake chain —
        // guest doorbell -> this thread scheduled -> control queue drained (which is where
        // vkr_ring_notify happens) -> used-queue interrupt raised. Joins virglrenderer's
        // LIMINA_RING_WAKE_PROFILE, which covers everything after cnd_signal.
        let mut wake_probe = crate::virtio::wake_probe::Profile::new();
        loop {
            let ev_cnt = match epoll.wait(epoll_events.len(), -1, epoll_events.as_mut_slice()) {
                Ok(n) => n,
                Err(e) => {
                    debug!("gpu worker epoll wait failed: {e}");
                    continue;
                }
            };
            // Stamped before any per-event work, so `kick -> wake` is the scheduling latency
            // alone and nothing this loop does afterwards leaks into it.
            let wake_ns = if wake_probe.is_some() {
                crate::virtio::wake_probe::now_ns()
            } else {
                0
            };
            if let Some((last, counts)) = wake_trace.as_mut() {
                counts[0] += 1;
                for event in &epoll_events[0..ev_cnt] {
                    let source = event.fd();
                    if source == control_ev_fd {
                        counts[1] += 1;
                    } else if source == cursor_ev_fd {
                        counts[2] += 1;
                    } else if present_ev_fd >= 0 && source == present_ev_fd {
                        counts[3] += 1;
                    } else if source == resize_ev_fd {
                        counts[4] += 1;
                    } else {
                        counts[5] += 1;
                    }
                }
                let secs = last.elapsed().as_secs_f64();
                if secs >= 5.0 {
                    eprintln!(
                        "[WAKETRACE gpu_worker] wakes={:.0}/s control={:.0}/s cursor={:.0}/s present={:.0}/s resize={:.0}/s other={:.0}/s",
                        counts[0] as f64 / secs,
                        counts[1] as f64 / secs,
                        counts[2] as f64 / secs,
                        counts[3] as f64 / secs,
                        counts[4] as f64 / secs,
                        counts[5] as f64 / secs,
                    );
                    *counts = [0; 6];
                    *last = std::time::Instant::now();
                }
            }
            for event in &epoll_events[0..ev_cnt] {
                let source = event.fd();
                if source == stop_ev_fd {
                    // A command awaits (the queue fds can't observe the channel). Drain the
                    // wake and act on it.
                    let _ = self.stop_fd.read();
                    match self.cmd_rx.try_recv() {
                        Ok(WorkerCmd::Deactivate) => return InnerExit::Deactivate,
                        Ok(WorkerCmd::Shutdown) => return InnerExit::Shutdown,
                        // limina M9.3: snapshot with the transport bound (vCPUs quiesced by
                        // the caller, so the queues are idle). Drain in-flight fences first —
                        // see drain_fences.
                        Ok(WorkerCmd::Snapshot { reply }) => {
                            Self::drain_fences(virtio_gpu);
                            let _ = reply.send(virtio_gpu.snapshot_gpu_payload());
                        }
                        // An Activate while already active shouldn't happen (the lifecycle is
                        // activate→reset→activate); ignore it. Empty = spurious wake.
                        Ok(WorkerCmd::Activate { .. })
                        | Err(std::sync::mpsc::TryRecvError::Empty) => {}
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            return InnerExit::Shutdown
                        }
                    }
                }
                if source == control_ev_fd {
                    if let Err(e) = control_evt.read() {
                        error!("Failed to read control_evt: {e:?}");
                        continue;
                    }
                    // EVENT_IDX: drain inside a disable/enable-notification bracket so the
                    // guest stops ringing the doorbell (a vmexit + this wake) while we're
                    // already here; `enable_notification` re-arms and reports entries that
                    // raced the re-arm. Signal only when the guest asked (used_event) —
                    // without EVENT_IDX `needs_notification` is always true (stock behavior).
                    // limina wake-probe: the doorbell behind this wake, if any. Taken before
                    // the drain so a kick that lands mid-drain stays for the next wake rather
                    // than being credited to this one.
                    let kick_ns = wake_probe
                        .as_mut()
                        .and_then(|_| crate::virtio::wake_probe::consume_kick());
                    // First pass only: that is the pass carrying the command the doorbell
                    // announced (vkNotifyRingMESA included). Later passes are the
                    // enable_notification re-arm race, a different question.
                    let mut probe_pass = kick_ns;
                    if let (Some(p), None) = (wake_probe.as_mut(), kick_ns) {
                        p.record_no_kick();
                    }
                    loop {
                        let _ = control_queue.lock().unwrap().disable_notification(mem);
                        let used_any = self.process_queue(virtio_gpu, control_queue, mem);
                        let drained_ns = probe_pass.map(|_| crate::virtio::wake_probe::now_ns());
                        if let (Some(p), Some(k)) = (wake_probe.as_mut(), probe_pass) {
                            p.arm(k);
                        }
                        if used_any
                            && control_queue
                                .lock()
                                .unwrap()
                                .needs_notification(mem)
                                .unwrap_or(true)
                        {
                            // Arm the return-path measurement here, not with `arm` above: this
                            // is the only branch that actually interrupts the guest, and the
                            // guest can ack before this thread runs again.
                            if let Some(p) = wake_probe.as_ref() {
                                p.arm_irq();
                            }
                            if let Err(e) = interrupt.try_signal_used_queue() {
                                error!("Error signaling queue: {e:?}");
                            }
                        }
                        if let (Some(p), Some(k), Some(d)) =
                            (wake_probe.as_mut(), probe_pass, drained_ns)
                        {
                            p.record(k, wake_ns, d, crate::virtio::wake_probe::now_ns());
                            probe_pass = None;
                        }
                        match control_queue.lock().unwrap().enable_notification(mem) {
                            Ok(true) => continue,
                            Ok(false) => break,
                            Err(e) => {
                                error!("Error re-enabling control queue notification: {e:?}");
                                break;
                            }
                        }
                    }
                }
                if source == cursor_ev_fd {
                    if let Err(e) = cursor_evt.read() {
                        error!("Failed to read cursor_evt: {e:?}");
                        continue;
                    }
                    loop {
                        let _ = cursor_queue.lock().unwrap().disable_notification(mem);
                        let used_any = self.process_cursor_queue(virtio_gpu, cursor_queue, mem);
                        if used_any
                            && cursor_queue
                                .lock()
                                .unwrap()
                                .needs_notification(mem)
                                .unwrap_or(true)
                        {
                            if let Err(e) = interrupt.try_signal_used_queue() {
                                error!("Error signaling cursor queue: {e:?}");
                            }
                        }
                        match cursor_queue.lock().unwrap().enable_notification(mem) {
                            Ok(true) => continue,
                            Ok(false) => break,
                            Err(e) => {
                                error!("Error re-enabling cursor queue notification: {e:?}");
                                break;
                            }
                        }
                    }
                }
                // limina runtime resize: the host pushed a new display size. Apply it to the
                // scanout's preferred mode, set the config-space display-event bit, and raise a
                // virtio-gpu config-change interrupt so the guest re-reads GET_DISPLAY_INFO and
                // re-modesets (the backend reallocates on the resulting SET_SCANOUT geometry).
                if source == resize_ev_fd {
                    let _ = self.resize_evt.read();
                    // Exactly one update per wake: a connect/disconnect pair must reach the
                    // guest as two distinct config-change events, or it would only ever observe
                    // the final state and never notice the display went away. Anything still
                    // queued re-kicks the eventfd so the next loop iteration picks it up.
                    let (pending, more) = {
                        let mut queue = self.resize_pending.lock().unwrap();
                        (queue.pop_front(), !queue.is_empty())
                    };
                    if let Some(update) = pending {
                        let id = update.display_id;
                        if let Some((w, h)) = update.size {
                            virtio_gpu.set_display_size(id, w, h);
                        }
                        if let Some(params) = update.edid {
                            virtio_gpu.set_display_edid(id, params);
                        }
                        if let Some(connected) = update.connected {
                            virtio_gpu.set_display_connected(id, connected);
                        }
                        self.events_read
                            .fetch_or(VIRTIO_GPU_EVENT_DISPLAY, Ordering::SeqCst);
                        if let Err(e) = interrupt.try_signal_config_change() {
                            error!("display update: failed to signal config change: {e:?}");
                        }
                    }
                    if more {
                        let _ = self.resize_evt.write(1);
                    }
                }
                // limina (#8): present fences retired — show their parked frames.
                if present_ev_fd >= 0 && source == present_ev_fd {
                    virtio_gpu.process_retired_presents();
                }
                // limina (vrend fence honesty): vrend's sync thread asked for a GL-thread
                // query check — pump virgl_renderer_poll() (flushes the eventfd, checks
                // pending queries, signals the parked sync thread).
                if renderer_poll_fd >= 0 && source == renderer_poll_fd {
                    virtio_gpu.renderer_event_poll();
                }
            }
            if let Some(p) = wake_probe.as_mut() {
                p.maybe_report();
            }
        }
    }

    fn process_gpu_command(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        mem: &GuestMemoryMmap,
        hdr: virtio_gpu_ctrl_hdr,
        cmd: GpuCommand,
        reader: &mut Reader,
    ) -> VirtioGpuResult {
        virtio_gpu.force_ctx_0();

        match cmd {
            GpuCommand::GetDisplayInfo => virtio_gpu.display_info(),
            GpuCommand::GetEdid(info) => virtio_gpu.get_edid(info.scanout),
            GpuCommand::ResourceCreate2d(info) => {
                // limina: handle 2D resources in software (host CPU memory) instead of
                // mapping them onto a virgl GL render target — the stock path fails on a
                // GL-less host (e.g. macOS). See VirtioGpu::resource_create_2d.
                let r = virtio_gpu.resource_create_2d(
                    info.resource_id,
                    info.format,
                    info.width,
                    info.height,
                );
                if r.is_ok() {
                    virtio_gpu.journal_mut().resource_create_2d(
                        info.resource_id,
                        info.format,
                        info.width,
                        info.height,
                    );
                }
                r
            }
            GpuCommand::ResourceUnref(info) => {
                let r = virtio_gpu.unref_resource(info.resource_id);
                if r.is_ok() {
                    let pinned = virtio_gpu.journal_mut().resource_unref(info.resource_id);
                    // The GLOBAL unref releases the vkr journal pin the blob create
                    // took on its backing memory (per-ctx detach must not).
                    if let Some((ctx_id, blob_id)) = pinned {
                        virtio_gpu.journal_unpin(ctx_id, blob_id);
                    }
                }
                r
            }
            GpuCommand::SetScanout(info) => {
                let r = virtio_gpu.set_scanout(
                    info.scanout_id,
                    info.resource_id,
                    info.r.width,
                    info.r.height,
                );
                if r.is_ok() {
                    virtio_gpu.journal_mut().set_scanout(
                        info.scanout_id,
                        info.resource_id,
                        info.r.width,
                        info.r.height,
                    );
                }
                r
            }
            GpuCommand::ResourceFlush(info) => {
                let rect = Rect {
                    x: info.r.x,
                    y: info.r.y,
                    width: info.r.width,
                    height: info.r.height,
                };
                virtio_gpu.flush_resource(info.resource_id, rect)
            }
            GpuCommand::TransferToHost2d(info) => {
                let resource_id = info.resource_id;
                let transfer = Transfer3D::new_2d(info.r.x, info.r.y, info.r.width, info.r.height);
                virtio_gpu.transfer_write(0, resource_id, transfer)
            }
            GpuCommand::ResourceAttachBacking(info) => {
                let available_bytes = reader.available_bytes();
                if available_bytes != 0 {
                    let entry_count = info.nr_entries as usize;
                    let mut vecs = Vec::with_capacity(entry_count);
                    for _ in 0..entry_count {
                        match reader.read_obj::<virtio_gpu_mem_entry>() {
                            Ok(entry) => {
                                let addr = GuestAddress(entry.addr);
                                let len = entry.length as usize;
                                vecs.push((addr, len))
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }
                    let backing: Vec<(u64, usize)> =
                        vecs.iter().map(|(a, l)| (a.0, *l)).collect();
                    let r = virtio_gpu.attach_backing(info.resource_id, mem, vecs);
                    if r.is_ok() {
                        virtio_gpu
                            .journal_mut()
                            .attach_backing(info.resource_id, backing);
                    }
                    r
                } else {
                    error!("missing data for command {cmd:?}");
                    Err(GpuResponse::ErrUnspec)
                }
            }
            GpuCommand::ResourceDetachBacking(info) => {
                let r = virtio_gpu.detach_backing(info.resource_id);
                if r.is_ok() {
                    virtio_gpu.journal_mut().detach_backing(info.resource_id);
                }
                r
            }
            GpuCommand::UpdateCursor(info) => virtio_gpu.update_cursor(
                info.resource_id,
                info.hot_x,
                info.hot_y,
                info.pos.x,
                info.pos.y,
            ),
            GpuCommand::MoveCursor(info) => virtio_gpu.move_cursor(info.pos.x, info.pos.y),
            GpuCommand::ResourceAssignUuid(info) => {
                let resource_id = info.resource_id;
                virtio_gpu.resource_assign_uuid(resource_id)
            }
            GpuCommand::GetCapsetInfo(info) => virtio_gpu.get_capset_info(info.capset_index),
            GpuCommand::GetCapset(info) => {
                virtio_gpu.get_capset(info.capset_id, info.capset_version)
            }

            GpuCommand::CtxCreate(info) => {
                let context_name: Option<String> = String::from_utf8(info.debug_name.to_vec()).ok();
                let r = virtio_gpu.create_context(
                    hdr.ctx_id,
                    info.context_init,
                    context_name.as_deref(),
                );
                if r.is_ok() {
                    virtio_gpu.journal_mut().ctx_create(
                        hdr.ctx_id,
                        info.context_init,
                        context_name,
                    );
                }
                r
            }
            GpuCommand::CtxDestroy(_info) => {
                let r = virtio_gpu.destroy_context(hdr.ctx_id);
                if r.is_ok() {
                    virtio_gpu.journal_mut().ctx_destroy(hdr.ctx_id);
                }
                r
            }
            GpuCommand::CtxAttachResource(info) => {
                let r = virtio_gpu.context_attach_resource(hdr.ctx_id, info.resource_id);
                if r.is_ok() {
                    virtio_gpu
                        .journal_mut()
                        .ctx_attach_resource(hdr.ctx_id, info.resource_id);
                }
                r
            }
            GpuCommand::CtxDetachResource(info) => {
                let r = virtio_gpu.context_detach_resource(hdr.ctx_id, info.resource_id);
                if r.is_ok() {
                    virtio_gpu
                        .journal_mut()
                        .ctx_detach_resource(hdr.ctx_id, info.resource_id);
                }
                r
            }
            GpuCommand::ResourceCreate3d(info) => {
                let resource_id = info.resource_id;
                let resource_create_3d = ResourceCreate3D {
                    target: info.target,
                    format: info.format,
                    bind: info.bind,
                    width: info.width,
                    height: info.height,
                    depth: info.depth,
                    array_size: info.array_size,
                    last_level: info.last_level,
                    nr_samples: info.nr_samples,
                    flags: info.flags,
                };

                let r = virtio_gpu.resource_create_3d(resource_id, resource_create_3d);
                if r.is_ok() {
                    virtio_gpu.journal_mut().resource_create_3d(
                        resource_id,
                        info.target,
                        info.format,
                        info.bind,
                        info.width,
                        info.height,
                        info.depth,
                        info.array_size,
                        info.last_level,
                        info.nr_samples,
                        info.flags,
                    );
                }
                r
            }
            GpuCommand::TransferToHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_write(ctx_id, resource_id, transfer)
            }
            GpuCommand::TransferFromHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_read(ctx_id, resource_id, transfer, None)
            }
            GpuCommand::CmdSubmit3d(info) => {
                // limina M9.3 trace: total submissions, the denominator for unknown_ctx.
                virtio_gpu.trace().submits.fetch_add(1, Ordering::Relaxed);
                if reader.available_bytes() != 0 {
                    let num_in_fences = info.num_in_fences as usize;
                    let cmd_size = info.size as usize;
                    let mut cmd_buf = vec![0; cmd_size];
                    let mut fence_ids: Vec<u64> = Vec::with_capacity(num_in_fences);

                    for _ in 0..num_in_fences {
                        match reader.read_obj::<u64>() {
                            Ok(fence_id) => {
                                fence_ids.push(fence_id);
                            }
                            Err(_) => return Err(GpuResponse::ErrUnspec),
                        }
                    }

                    if reader.read_exact(&mut cmd_buf[..]).is_ok() {
                        virtio_gpu.submit_command(hdr.ctx_id, &mut cmd_buf[..], &fence_ids)
                    } else {
                        Err(GpuResponse::ErrInvalidParameter)
                    }
                } else {
                    // Silently accept empty command buffers to allow for
                    // benchmarking.
                    Ok(GpuResponse::OkNoData)
                }
            }
            GpuCommand::ResourceCreateBlob(info) => {
                let resource_id = info.resource_id;
                let ctx_id = hdr.ctx_id;

                let resource_create_blob = ResourceCreateBlob {
                    blob_mem: info.blob_mem,
                    blob_flags: info.blob_flags,
                    blob_id: info.blob_id,
                    size: info.size,
                };

                let entry_count = info.nr_entries;
                if reader.available_bytes() == 0 && entry_count > 0 {
                    return Err(GpuResponse::ErrUnspec);
                }

                let mut vecs = Vec::with_capacity(entry_count as usize);
                for _ in 0..entry_count {
                    match reader.read_obj::<virtio_gpu_mem_entry>() {
                        Ok(entry) => {
                            let addr = GuestAddress(entry.addr);
                            let len = entry.length as usize;
                            vecs.push((addr, len))
                        }
                        Err(_) => return Err(GpuResponse::ErrUnspec),
                    }
                }

                let backing: Vec<(u64, usize)> = vecs.iter().map(|(a, l)| (a.0, *l)).collect();
                let r = virtio_gpu.resource_create_blob(
                    ctx_id,
                    resource_id,
                    resource_create_blob,
                    vecs,
                    mem,
                );
                if r.is_ok() {
                    // The vkr wire watermark is the cross-layer replay fence: everything the
                    // blob depends on (its vkAllocateMemory) is at or below this seq.
                    let vkr_seq = virtio_gpu.journal_vkr_seq(ctx_id);
                    debug!(
                        "gpu journal: CREATE_BLOB res {} ctx {} blob_id {} recorded with vkr fence {}",
                        resource_id, ctx_id, info.blob_id, vkr_seq
                    );
                    virtio_gpu.journal_mut().create_blob(
                        ctx_id,
                        resource_id,
                        info.blob_mem,
                        info.blob_flags,
                        info.blob_id,
                        info.size,
                        backing,
                        vkr_seq,
                    );
                }
                r
            }
            GpuCommand::SetScanoutBlob(info) => {
                // limina tier-2 (macOS): present an IOSurface-backed blob scanout zero-copy.
                #[cfg(target_os = "macos")]
                {
                    let r = virtio_gpu.set_scanout_blob(
                        info.scanout_id,
                        info.resource_id,
                        info.width,
                        info.height,
                        info.format,
                    );
                    if r.is_ok() {
                        virtio_gpu.journal_mut().set_scanout_blob(
                            info.scanout_id,
                            info.resource_id,
                            info.width,
                            info.height,
                            info.format,
                        );
                    }
                    r
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = info;
                    Err(GpuResponse::ErrUnspec)
                }
            }
            GpuCommand::ResourceMapBlob(info) => {
                let resource_id = info.resource_id;
                let offset = info.offset;
                let r = virtio_gpu.resource_map_blob(resource_id, &self.shm_region, offset);
                if r.is_ok() {
                    virtio_gpu.journal_mut().map_blob(resource_id, offset);
                }
                r
            }
            GpuCommand::ResourceUnmapBlob(info) => {
                let resource_id = info.resource_id;
                let r = virtio_gpu.resource_unmap_blob(resource_id, &self.shm_region);
                if r.is_ok() {
                    virtio_gpu.journal_mut().unmap_blob(resource_id);
                }
                r
            }
        }
    }

    fn process_queue(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        control_queue: &Arc<Mutex<VirtQueue>>,
        mem: &GuestMemoryMmap,
    ) -> bool {
        let mut used_any = false;
        let mem = mem.clone();

        // limina M9.3 trace: a renderer state dump was requested (first stale-ctx
        // event, or a periodic LIMINA_GPU_TRACE_VKR tick). Service it here — this is
        // the only thread allowed to call into the renderer singleton. Piggybacking
        // on control-queue activity is fine: the interesting moments (guest submits
        // into a fresh renderer; a healthy session under measurement) both kick.
        if virtio_gpu
            .trace()
            .dump_requested
            .swap(false, Ordering::Relaxed)
        {
            virtio_gpu.journal().dump();
            virtio_gpu.dump_renderer_state();
        }

        loop {
            let head = control_queue.lock().unwrap().pop(&mem);

            if let Some(head) = head {
                let mut reader = Reader::new(&mem, head.clone())
                    .map_err(GpuError::QueueReader)
                    .unwrap();
                let mut writer = Writer::new(&mem, head.clone())
                    .map_err(GpuError::QueueWriter)
                    .unwrap();

                let mut resp = Err(GpuResponse::ErrUnspec);
                let mut gpu_cmd = None;
                let mut ctrl_hdr = None;
                let mut len = 0;

                match GpuCommand::decode(&mut reader) {
                    Ok((hdr, cmd)) => {
                        // limina wake-probe: time each command so a drain that overruns can
                        // name the command that caused it. Two clock reads per command, only
                        // while the probe is on.
                        let t0 = crate::virtio::wake_probe::cmd_start();
                        resp = self.process_gpu_command(virtio_gpu, &mem, hdr, cmd, &mut reader);
                        crate::virtio::wake_probe::cmd_end(t0, hdr.type_);
                        ctrl_hdr = Some(hdr);
                        gpu_cmd = Some(cmd);
                    }
                    Err(e) => debug!("descriptor decode error: {e:?}"),
                }

                let mut gpu_response = match resp {
                    Ok(gpu_response) => gpu_response,
                    Err(gpu_response) => {
                        // limina M9.3 trace: count the error by class (stale ctx /
                        // unknown resource / other) for the tick reporter.
                        virtio_gpu.trace().classify_error(&gpu_response);
                        // limina: error responses are load-bearing diagnostics (the guest sees
                        // RESP_ERR and degrades silently — e.g. Firefox WebGL EGL_CREATE
                        // failures); keep them visible at default log level.
                        warn!("{gpu_cmd:?} -> {gpu_response:?}");
                        gpu_response
                    }
                };

                let mut add_to_queue = true;

                if writer.available_bytes() != 0 {
                    let mut fence_id = 0;
                    let mut ctx_id = 0;
                    let mut flags = 0;
                    let mut ring_idx = 0;
                    if let Some(cmd) = gpu_cmd {
                        let ctrl_hdr = ctrl_hdr.unwrap();
                        if ctrl_hdr.flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                            flags = ctrl_hdr.flags;
                            fence_id = ctrl_hdr.fence_id;
                            ctx_id = ctrl_hdr.ctx_id;
                            ring_idx = ctrl_hdr.ring_idx;

                            let fence = RutabagaFence {
                                flags,
                                fence_id,
                                ctx_id,
                                ring_idx,
                            };
                            // limina (#8 half 2): only a flush's fence may be held for
                            // fence-accurate presents.
                            if std::env::var_os("LIMINA_GPU_TEST_TRACE_FENCES").is_some() {
                                warn!(
                                    "FENCECMD id={fence_id} ctx={ctx_id} ring={ring_idx} \
                                     cmd={cmd:?}"
                                );
                            }
                            let is_flush = matches!(cmd, GpuCommand::ResourceFlush(_));
                            gpu_response = match virtio_gpu.create_fence(fence, is_flush) {
                                Ok(_) => gpu_response,
                                Err(fence_resp) => {
                                    warn!("create_fence {fence_id} -> {fence_resp:?}");
                                    fence_resp
                                }
                            };
                        }
                    }

                    // Prepare the response now, even if it is going to wait until
                    // fence is complete.
                    match gpu_response.encode(flags, fence_id, ctx_id, ring_idx, &mut writer) {
                        Ok(l) => len = l,
                        Err(e) => debug!("ctrl queue response encode error: {e:?}"),
                    }

                    if flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                        let ring = match flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                            0 => VirtioGpuRing::Global,
                            _ => VirtioGpuRing::ContextSpecific { ctx_id, ring_idx },
                        };

                        add_to_queue = virtio_gpu.process_fence(ring, fence_id, head.index, len);
                    }
                }

                if add_to_queue {
                    if let Err(e) = control_queue
                        .lock()
                        .unwrap()
                        .add_used(&mem, head.index, len)
                    {
                        error!("failed to add used elements to the queue: {e:?}");
                    }
                    used_any = true;
                }
            } else {
                break;
            }
        }

        debug!("gpu: process_queue exit");
        used_any
    }

    /// limina: drain the cursor queue. Each entry is an `UPDATE_CURSOR`/`MOVE_CURSOR` command
    /// (read-only from the guest, no response payload), dispatched to the display backend as a
    /// host overlay update. Draining it is what lets the guest use its hardware cursor plane;
    /// an unserviced cursor queue makes the guest fall back to compositing the cursor into the
    /// scanout (or stall once the ring fills).
    fn process_cursor_queue(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        cursor_queue: &Arc<Mutex<VirtQueue>>,
        mem: &GuestMemoryMmap,
    ) -> bool {
        let mut used_any = false;
        let mem = mem.clone();
        let cursor_queue = cursor_queue.clone();

        loop {
            let head = cursor_queue.lock().unwrap().pop(&mem);
            let Some(head) = head else { break };

            let mut reader = Reader::new(&mem, head.clone())
                .map_err(GpuError::QueueReader)
                .unwrap();
            match GpuCommand::decode(&mut reader) {
                Ok((hdr, cmd)) => {
                    if let Err(resp) =
                        self.process_gpu_command(virtio_gpu, &mem, hdr, cmd, &mut reader)
                    {
                        debug!("cursor cmd {cmd:?} -> {resp:?}");
                    }
                }
                Err(e) => debug!("cursor descriptor decode error: {e:?}"),
            }

            // Cursor commands carry no response data; return the descriptor with len 0 so the
            // guest's cursor queue keeps making progress.
            if let Err(e) = cursor_queue.lock().unwrap().add_used(&mem, head.index, 0) {
                error!("failed to add used elements to the cursor queue: {e:?}");
            }
            used_any = true;
        }
        used_any
    }
}
