use std::io::Write;
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
use super::worker::Worker;
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
    /// Handle to the running worker thread, present only while activated.
    worker_thread: Option<JoinHandle<()>>,
    /// Signals the worker to exit so it can be joined on reset/re-activation.
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
        if self.worker_thread.is_some() {
            // reset() must have joined the previous worker before re-activation.
            panic!("virtio_gpu: worker thread already exists");
        }

        let [control_q, _cursor_q]: [_; defs::NUM_QUEUES] = queues.try_into().map_err(|_| {
            error!(
                "Cannot perform activate. Expected {} queue(s)",
                defs::NUM_QUEUES
            );
            ActivateError::BadActivate
        })?;

        let shm_region = match self.shm_region.as_ref() {
            Some(s) => s.clone(),
            None => panic!("virtio_gpu: missing SHM region"),
        };

        // cursor queue not used by worker
        let worker = Worker::new(
            control_q,
            mem.clone(),
            interrupt.clone(),
            shm_region,
            self.virgl_flags,
            self.software_2d,
            self.worker_stopfd.try_clone().unwrap(),
            #[cfg(target_os = "macos")]
            self.map_sender.clone(),
            self.export_table.take(),
            self.displays.clone(),
            self.display_backend,
        );
        self.worker_thread = Some(worker.run());

        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        // Stop and join the worker so a stale thread doesn't keep running on the old queue
        // when the guest re-initializes the device (firmware -> kernel hand-off, driver
        // rebind, reboot). Returning true lets the transport recreate the queues and a later
        // activate() spawn a fresh worker.
        if let Some(worker) = self.worker_thread.take() {
            let _ = self.worker_stopfd.write(1);
            if let Err(e) = worker.join() {
                error!("error waiting for gpu worker thread: {e:?}");
            }
        }
        self.device_state = DeviceState::Inactive;
        true
    }

    fn shm_region(&self) -> Option<&VirtioShmRegion> {
        debug!("virtio_gpu: GET_shm_region");
        self.shm_region.as_ref()
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

    /// Re-activating virtio-gpu (firmware -> kernel hand-off, driver rebind, reboot) must not
    /// leave the previous worker thread running on the stale queue. `reset()` must stop+join
    /// the worker and return true so a fresh `activate()` can spawn a new one. If the stop
    /// signal were broken, `reset()`'s `join()` would hang and this test would time out.
    #[test]
    fn test_reset_stops_worker_and_allows_reactivation() {
        let mem = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap();
        let intc =
            InterruptTransport::new(DummyIrqChip::new().into(), "gpu-test".to_string()).unwrap();

        let mut gpu = test_gpu();
        gpu.set_shm_region(VirtioShmRegion {
            host_addr: 0,
            guest_addr: 0,
            size: 0,
        });

        // Activate -> a worker thread is running.
        gpu.activate(
            mem.clone(),
            intc.clone(),
            vec![dummy_device_queue(), dummy_device_queue()],
        )
        .unwrap();
        assert!(gpu.is_activated());
        assert!(gpu.worker_thread.is_some());

        // Reset -> worker stopped+joined, device inactive (join() hangs if stop is broken).
        assert!(gpu.reset());
        assert!(!gpu.is_activated());
        assert!(gpu.worker_thread.is_none());

        // Re-activate -> a fresh worker, no "already exists" panic.
        gpu.activate(mem, intc, vec![dummy_device_queue(), dummy_device_queue()])
            .unwrap();
        assert!(gpu.is_activated());

        // Clean up the second worker.
        assert!(gpu.reset());
    }
}
