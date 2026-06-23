use std::io::Read;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
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
    },
    /// Stop servicing and release the transport, but keep the renderer alive (device reset).
    Deactivate,
    /// Tear the worker down (device drop / VMM teardown).
    Shutdown,
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
    /// The pending resize `(display_id, width, height)`, taken on a `resize_evt` wake.
    resize_pending: Arc<Mutex<Option<(u32, u32, u32)>>>,
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
        resize_pending: Arc<Mutex<Option<(u32, u32, u32)>>>,
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

        loop {
            // Inactive: block on the next command from the device.
            let (mem, control_q, cursor_q, interrupt) = match self.cmd_rx.recv() {
                Ok(WorkerCmd::Activate {
                    mem,
                    control_q,
                    cursor_q,
                    interrupt,
                }) => (mem, control_q, cursor_q, interrupt),
                // A Deactivate while already inactive is a no-op; Shutdown / closed channel ends.
                Ok(WorkerCmd::Deactivate) => continue,
                Ok(WorkerCmd::Shutdown) | Err(_) => return,
            };

            // Own each activation's queue eventfds + queues; publish the transport so the
            // long-lived fence handler retires guest fences into THIS activation's queue.
            let control_evt = control_q.event.try_clone().unwrap();
            let cursor_evt = cursor_q.event.try_clone().unwrap();
            let control_queue = Arc::new(Mutex::new(control_q.queue));
            let cursor_queue = Arc::new(Mutex::new(cursor_q.queue));
            *self.active.lock().unwrap() = Some(GpuActivation {
                mem: mem.clone(),
                control_queue: control_queue.clone(),
                interrupt: interrupt.clone(),
            });

            let exit = self.service(
                &mut virtio_gpu,
                &mem,
                &control_evt,
                &control_queue,
                &cursor_evt,
                &cursor_queue,
                &interrupt,
            );

            // Unbind the transport and drop this session's resource/scanout/fence bookkeeping
            // (its descriptors index the now-freed queue) — but keep the renderer alive.
            *self.active.lock().unwrap() = None;
            virtio_gpu.reset_session();

            match exit {
                InnerExit::Deactivate => continue,
                InnerExit::Shutdown => return,
            }
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

        let mut epoll_events = vec![EpollEvent::new(EventSet::empty(), 0); 32];
        loop {
            let ev_cnt = match epoll.wait(epoll_events.len(), -1, epoll_events.as_mut_slice()) {
                Ok(n) => n,
                Err(e) => {
                    debug!("gpu worker epoll wait failed: {e}");
                    continue;
                }
            };
            for event in &epoll_events[0..ev_cnt] {
                let source = event.fd();
                if source == stop_ev_fd {
                    // A command awaits (the queue fds can't observe the channel). Drain the
                    // wake and act on it.
                    let _ = self.stop_fd.read();
                    match self.cmd_rx.try_recv() {
                        Ok(WorkerCmd::Deactivate) => return InnerExit::Deactivate,
                        Ok(WorkerCmd::Shutdown) => return InnerExit::Shutdown,
                        // An Activate while already active shouldn't happen (the lifecycle is
                        // activate→reset→activate); ignore it. Empty = spurious wake.
                        Ok(WorkerCmd::Activate { .. }) | Err(std::sync::mpsc::TryRecvError::Empty) => {}
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
                    if self.process_queue(virtio_gpu, control_queue, mem) {
                        if let Err(e) = interrupt.try_signal_used_queue() {
                            error!("Error signaling queue: {e:?}");
                        }
                    }
                }
                if source == cursor_ev_fd {
                    if let Err(e) = cursor_evt.read() {
                        error!("Failed to read cursor_evt: {e:?}");
                        continue;
                    }
                    if self.process_cursor_queue(virtio_gpu, cursor_queue, mem) {
                        if let Err(e) = interrupt.try_signal_used_queue() {
                            error!("Error signaling cursor queue: {e:?}");
                        }
                    }
                }
                // limina runtime resize: the host pushed a new display size. Apply it to the
                // scanout's preferred mode, set the config-space display-event bit, and raise a
                // virtio-gpu config-change interrupt so the guest re-reads GET_DISPLAY_INFO and
                // re-modesets (the backend reallocates on the resulting SET_SCANOUT geometry).
                if source == resize_ev_fd {
                    let _ = self.resize_evt.read();
                    let pending = self.resize_pending.lock().unwrap().take();
                    if let Some((id, w, h)) = pending {
                        virtio_gpu.set_display_size(id, w, h);
                        self.events_read
                            .fetch_or(VIRTIO_GPU_EVENT_DISPLAY, Ordering::SeqCst);
                        if let Err(e) = interrupt.try_signal_config_change() {
                            error!("display resize: failed to signal config change: {e:?}");
                        }
                    }
                }
                // limina (#8): present fences retired — show their parked frames.
                if present_ev_fd >= 0 && source == present_ev_fd {
                    virtio_gpu.process_retired_presents();
                }
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
                virtio_gpu.resource_create_2d(
                    info.resource_id,
                    info.format,
                    info.width,
                    info.height,
                )
            }
            GpuCommand::ResourceUnref(info) => virtio_gpu.unref_resource(info.resource_id),
            GpuCommand::SetScanout(info) => virtio_gpu.set_scanout(
                info.scanout_id,
                info.resource_id,
                info.r.width,
                info.r.height,
            ),
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
                    virtio_gpu.attach_backing(info.resource_id, mem, vecs)
                } else {
                    error!("missing data for command {cmd:?}");
                    Err(GpuResponse::ErrUnspec)
                }
            }
            GpuCommand::ResourceDetachBacking(info) => virtio_gpu.detach_backing(info.resource_id),
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
                virtio_gpu.create_context(hdr.ctx_id, info.context_init, context_name.as_deref())
            }
            GpuCommand::CtxDestroy(_info) => virtio_gpu.destroy_context(hdr.ctx_id),
            GpuCommand::CtxAttachResource(info) => {
                virtio_gpu.context_attach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::CtxDetachResource(info) => {
                virtio_gpu.context_detach_resource(hdr.ctx_id, info.resource_id)
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

                virtio_gpu.resource_create_3d(resource_id, resource_create_3d)
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

                virtio_gpu.resource_create_blob(
                    ctx_id,
                    resource_id,
                    resource_create_blob,
                    vecs,
                    mem,
                )
            }
            GpuCommand::SetScanoutBlob(info) => {
                // limina tier-2 (macOS): present an IOSurface-backed blob scanout zero-copy.
                #[cfg(target_os = "macos")]
                {
                    virtio_gpu.set_scanout_blob(
                        info.scanout_id,
                        info.resource_id,
                        info.width,
                        info.height,
                        info.format,
                    )
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
                virtio_gpu.resource_map_blob(resource_id, &self.shm_region, offset)
            }
            GpuCommand::ResourceUnmapBlob(info) => {
                let resource_id = info.resource_id;
                virtio_gpu.resource_unmap_blob(resource_id, &self.shm_region)
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
                        resp = self.process_gpu_command(virtio_gpu, &mem, hdr, cmd, &mut reader);
                        ctrl_hdr = Some(hdr);
                        gpu_cmd = Some(cmd);
                    }
                    Err(e) => debug!("descriptor decode error: {e:?}"),
                }

                let mut gpu_response = match resp {
                    Ok(gpu_response) => gpu_response,
                    Err(gpu_response) => {
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
