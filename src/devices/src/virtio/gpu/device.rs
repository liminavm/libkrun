use std::io::Write;
use std::sync::mpsc::Sender as CmdSender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

#[cfg(target_os = "macos")]
use crossbeam_channel::Sender;
use utils::eventfd::{EventFd, EFD_NONBLOCK};
use vm_memory::{ByteValued, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, QueueConfig, VirtioDevice,
    VirtioShmRegion, fs::ExportTable,
};
use super::defs;
use super::defs::uapi;
use super::defs::uapi::virtio_gpu_config;
use super::virtio_gpu::GpuActivation;
use super::worker::{Worker, WorkerCmd};
use crate::display::DisplayInfo;
use crate::virtio::InterruptTransport;
use krun_display::DisplayBackend;
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;

// Supported features.
pub(crate) const AVAIL_FEATURES: u64 = (1u64 << uapi::VIRTIO_F_VERSION_1)
    | (1u64 << uapi::VIRTIO_GPU_F_VIRGL)
    | (1u64 << uapi::VIRTIO_GPU_F_EDID)
    | (1u64 << uapi::VIRTIO_GPU_F_RESOURCE_UUID)
    | (1u64 << uapi::VIRTIO_GPU_F_RESOURCE_BLOB)
    | (1u64 << uapi::VIRTIO_GPU_F_CONTEXT_INIT);

// limina software-2D-only mode: advertise a plain 2D virtio-gpu — no VIRGL/BLOB/CONTEXT_INIT
// (all renderer-backed) and no capsets — so the guest never probes for or issues 3D
// commands and falls back to the 2D scanout path (efifb/simpledrm/fbcon). See `Gpu::new`.
pub(crate) const AVAIL_FEATURES_SOFTWARE_2D: u64 = (1u64 << uapi::VIRTIO_F_VERSION_1)
    | (1u64 << uapi::VIRTIO_GPU_F_EDID)
    | (1u64 << uapi::VIRTIO_GPU_F_RESOURCE_UUID);

const QUEUE_SIZE: u16 = 256;
static QUEUE_CONFIG: [QueueConfig; defs::NUM_QUEUES] =
    [QueueConfig::new(QUEUE_SIZE); defs::NUM_QUEUES];

pub struct Gpu {
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) device_state: DeviceState,
    shm_region: Option<VirtioShmRegion>,
    virgl_flags: u32,
    /// limina: serve only software-2D scanout; don't init virglrenderer/rutabaga (see module).
    software_2d: bool,
    #[cfg(target_os = "macos")]
    map_sender: Sender<WorkerMessage>,
    export_table: Option<ExportTable>,
    displays: Box<[DisplayInfo]>,
    display_backend: DisplayBackend<'static>,
    /// The long-lived gpu worker thread. Spawned once (on first `activate()`) and kept across
    /// resets — it owns the process-global, thread-bound renderer for the whole VMM lifetime.
    /// Joined on `Drop`. See [`WorkerCmd`].
    worker_thread: Option<JoinHandle<()>>,
    /// Sends activate/deactivate/shutdown commands to the worker; `None` until first activate.
    worker_tx: Option<CmdSender<WorkerCmd>>,
    /// Wakes the worker's inner epoll loop when a command is queued (reset / Drop).
    worker_stopfd: EventFd,
}

impl Gpu {
    pub fn new(
        virgl_flags: u32,
        software_2d: bool,
        displays: Box<[DisplayInfo]>,
        display_backend: DisplayBackend<'static>,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
    ) -> super::Result<Gpu> {
        Ok(Gpu {
            avail_features: if software_2d {
                AVAIL_FEATURES_SOFTWARE_2D
            } else {
                AVAIL_FEATURES
            },
            acked_features: 0,
            device_state: DeviceState::Inactive,
            shm_region: None,
            virgl_flags,
            software_2d,
            #[cfg(target_os = "macos")]
            map_sender,
            export_table: None,
            displays,
            display_backend,
            worker_thread: None,
            worker_tx: None,
            worker_stopfd: EventFd::new(EFD_NONBLOCK).map_err(super::GpuError::EventFd)?,
        })
    }

    pub fn id(&self) -> &str {
        defs::GPU_DEV_ID
    }

    pub fn set_shm_region(&mut self, shm_region: VirtioShmRegion) {
        debug!("virtio_gpu: set_shm_region");
        self.shm_region = Some(shm_region);
    }

    pub fn set_export_table(&mut self, export_table: ExportTable) {
        self.export_table = Some(export_table);
    }

    /*
    pub fn process_ctl(&mut self) -> bool {
        debug!("gpu: process_ctl()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let mut have_used = false;

        //while let Some(head) = self.queues[CTL_INDEX].pop(mem) {
        if let Some(head) = self.queues[CTL_INDEX].pop(mem) {
            let index = head.index;
            let mut written = 0;
            for desc in head.into_iter() {
                error!("gpu: process_ctl() unimplemented");
                self.queues[CTL_INDEX].go_to_previous_position();
                break;
            }

            have_used = true;
            self.queues[CTL_INDEX].add_used(mem, index, written);
        }

        have_used
    }

    pub fn process_cur(&mut self) -> bool {
        debug!("gpu: process_cur()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let mut have_used = false;

        while let Some(head) = self.queues[CTL_INDEX].pop(mem) {
            let index = head.index;
            let mut written = 0;
            for desc in head.into_iter() {
                error!("gpu: process_cur() unimplemented");
                self.queues[CTL_INDEX].go_to_previous_position();
                break;
            }

            have_used = true;
            self.queues[CTL_INDEX].add_used(mem, index, written);
        }

        have_used
    }
    */
}

impl VirtioDevice for Gpu {
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
        uapi::VIRTIO_ID_GPU
    }

    fn device_name(&self) -> &str {
        "gpu"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config = virtio_gpu_config {
            events_read: 0,
            events_clear: 0,
            num_scanouts: self.displays.len() as u32,
            // No renderer in software-2D-only mode → no capsets to advertise.
            num_capsets: if self.software_2d { 0 } else { 5 },
        };

        let config_slice = config.as_slice();
        let config_len = config_slice.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(&config_slice[offset as usize..std::cmp::min(end, config_len) as usize])
                .unwrap();
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "gpu: guest driver attempted to write device config (offset={:x}, len={:x})",
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
        let [control_q, cursor_q]: [_; defs::NUM_QUEUES] = queues.try_into().map_err(|_| {
            error!(
                "Cannot perform activate. Expected {} queue(s)",
                defs::NUM_QUEUES
            );
            ActivateError::BadActivate
        })?;

        // Spawn the worker ONCE, on the first activation. It owns the process-global,
        // thread-bound renderer for the whole VMM lifetime; later activations (driver rebind,
        // EFI→kernel hand-off, reboot) reuse the same thread+renderer and just rebind transport.
        // This is the renderer-singleton fix: `virgl_renderer_init` can't be re-run, so we must
        // never drop+recreate the renderer on reset.
        let tx = match &self.worker_tx {
            Some(tx) => tx,
            None => {
                let shm_region = match self.shm_region.as_ref() {
                    Some(s) => s.clone(),
                    None => panic!("virtio_gpu: missing SHM region"),
                };
                let (tx, rx) = std::sync::mpsc::channel();
                let active: Arc<Mutex<Option<GpuActivation>>> = Arc::new(Mutex::new(None));
                let worker = Worker::new(
                    shm_region,
                    self.virgl_flags,
                    self.software_2d,
                    self.worker_stopfd.try_clone().unwrap(),
                    rx,
                    active,
                    #[cfg(target_os = "macos")]
                    self.map_sender.clone(),
                    self.export_table.take(),
                    self.displays.clone(),
                    self.display_backend,
                );
                self.worker_thread = Some(worker.run());
                self.worker_tx = Some(tx);
                self.worker_tx.as_ref().unwrap()
            }
        };

        // Bind this activation's transport. The worker is in its outer recv() (first activate)
        // or just deactivated, so a plain send wakes it — no stop_fd kick needed here.
        if tx
            .send(WorkerCmd::Activate {
                mem: mem.clone(),
                control_q,
                cursor_q,
                interrupt: interrupt.clone(),
            })
            .is_err()
        {
            error!("virtio_gpu: worker thread is gone; cannot activate");
            return Err(ActivateError::BadActivate);
        }

        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        // Tell the worker to stop servicing the (about-to-be-freed) queues and drop this
        // session's bookkeeping, WITHOUT tearing down the renderer — so the guest's re-init
        // (firmware→kernel hand-off, driver rebind, reboot) gets a fresh device backed by the
        // same long-lived renderer instead of a failed second `virgl_renderer_init`. The
        // stop_fd kick wakes the worker's inner epoll loop to pick up the queued command.
        if let Some(tx) = &self.worker_tx {
            let _ = tx.send(WorkerCmd::Deactivate);
            let _ = self.worker_stopfd.write(1);
        }
        self.device_state = DeviceState::Inactive;
        true
    }

    fn shm_region(&self) -> Option<&VirtioShmRegion> {
        debug!("virtio_gpu: GET_shm_region");
        self.shm_region.as_ref()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // Reap the long-lived worker thread (it isn't joined on reset anymore). Closing the
        // command channel would also end it, but the explicit Shutdown + stop_fd kick wakes it
        // promptly whether it's parked in recv() or its inner epoll loop.
        if let Some(tx) = self.worker_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
            let _ = self.worker_stopfd.write(1);
        }
        if let Some(worker) = self.worker_thread.take() {
            if let Err(e) = worker.join() {
                error!("error waiting for gpu worker thread: {e:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy::DummyIrqChip;
    use crate::virtio::{InterruptTransport, Queue};
    use krun_display::{
        DisplayBackendBasicFramebuffer, DisplayBackendError, DisplayBackendNew, IntoDisplayBackend,
        Rect, ResourceFormat,
    };
    use std::sync::Arc;
    use vm_memory::GuestAddress;

    /// Minimal no-op display backend so the reset test can spin a real worker thread
    /// without a window/capture sink. Its frame methods are never reached (no queue
    /// activity); only `new` is invoked, via `VirtioGpu::new`'s `create_instance`.
    struct StubBackend {
        buf: Vec<u8>,
    }
    impl DisplayBackendNew<()> for StubBackend {
        fn new(_userdata: Option<&()>) -> Self {
            StubBackend { buf: vec![0u8; 4] }
        }
    }
    impl DisplayBackendBasicFramebuffer for StubBackend {
        fn configure_scanout(
            &mut self,
            _scanout_id: u32,
            _display_width: u32,
            _display_height: u32,
            _width: u32,
            _height: u32,
            _format: ResourceFormat,
        ) -> Result<(), DisplayBackendError> {
            Ok(())
        }
        fn disable_scanout(&mut self, _scanout_id: u32) -> Result<(), DisplayBackendError> {
            Ok(())
        }
        fn alloc_frame(
            &mut self,
            _scanout_id: u32,
        ) -> Result<(u32, &mut [u8]), DisplayBackendError> {
            Ok((0, &mut self.buf))
        }
        fn present_frame(
            &mut self,
            _scanout_id: u32,
            _frame_id: u32,
            _rect: Option<&Rect>,
        ) -> Result<(), DisplayBackendError> {
            Ok(())
        }
    }

    fn dummy_device_queue() -> DeviceQueue {
        DeviceQueue::new(
            Queue::new(QUEUE_SIZE),
            Arc::new(EventFd::new(EFD_NONBLOCK).unwrap()),
        )
    }

    fn test_gpu() -> Gpu {
        let backend = StubBackend::into_display_backend(None);
        Gpu::new(
            0,
            true, // software_2d -> no rutabaga, no renderer init
            Vec::<DisplayInfo>::new().into_boxed_slice(),
            backend,
            #[cfg(target_os = "macos")]
            crossbeam_channel::unbounded().0,
        )
        .unwrap()
    }

    /// The renderer-singleton fix: `virgl_renderer_init` can't be re-run, so the worker (which
    /// owns the renderer) must be spawned ONCE and **persist** across resets — `activate()`/
    /// `reset()` just bind/unbind the transport. This asserts that lifecycle: the worker thread
    /// is created on first activate, survives a reset (not joined), and re-activates without a
    /// second spawn; `Drop` then joins it cleanly (a deadlock there would hang this test).
    /// (Runs with `software_2d=true` → no real renderer, so it exercises the thread/transport
    /// lifecycle, not the renderer reuse itself — that's `venus_reset.rs`.)
    #[test]
    fn test_worker_persists_across_reset() {
        let mem = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let intc =
            InterruptTransport::new(DummyIrqChip::new().into(), "gpu-test".to_string()).unwrap();

        let mut gpu = test_gpu();
        gpu.set_shm_region(VirtioShmRegion {
            host_addr: 0,
            guest_addr: 0,
            size: 0,
        });

        // First activate -> the single worker thread is spawned.
        gpu.activate(
            mem.clone(),
            intc.clone(),
            vec![dummy_device_queue(), dummy_device_queue()],
        )
        .unwrap();
        assert!(gpu.is_activated());
        assert!(gpu.worker_thread.is_some());
        assert!(gpu.worker_tx.is_some());

        // Reset -> device inactive, but the worker thread PERSISTS (renderer must outlive it).
        assert!(gpu.reset());
        assert!(!gpu.is_activated());
        assert!(gpu.worker_thread.is_some(), "worker must survive reset");

        // Re-activate -> reuses the same worker (no second spawn, no panic).
        gpu.activate(mem, intc, vec![dummy_device_queue(), dummy_device_queue()])
            .unwrap();
        assert!(gpu.is_activated());

        // Drop joins the worker (Shutdown + stop_fd kick); a broken teardown would hang here.
        drop(gpu);
    }
}
