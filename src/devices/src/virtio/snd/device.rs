// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::Arc;
use std::thread::JoinHandle;

use utils::eventfd::EventFd;
use vm_memory::{ByteValued, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, QueueConfig, VirtioDevice,
};
use super::protocol::*;
use super::worker::{SndWorker, SndWorkerConfig};
use super::{SndError, defs, defs::uapi};
use crate::virtio::InterruptTransport;

// Supported features: virtio 1.0 only (no VIRTIO_SND_F_CTLS — we expose no control
// elements yet; guest/host volume stack instead).
pub(crate) const AVAIL_FEATURES: u64 = 1 << uapi::VIRTIO_F_VERSION_1 as u64;

/// The virtio-snd device. Its queues are serviced by an [`SndWorker`] on a thread of its own,
/// started on activation and joined on reset; this half only holds configuration and the
/// embedder's hooks between activations.
pub struct Snd {
    avail_features: u64,
    acked_features: u64,
    device_state: DeviceState,
    /// Whether the mic-capture input stream is advertised (opt-in; default off for
    /// privacy). When false the device exposes playback only and never touches the mic.
    capture_enabled: bool,
    /// Kicked by the CoreAudio callbacks; handed to each worker.
    #[cfg(target_os = "macos")]
    completion_evt: Arc<EventFd>,
    /// Audibility reporting the embedder asked for: the pause threshold and the hook.
    #[cfg(target_os = "macos")]
    audibility: Option<(std::time::Duration, super::PcmAudibilityFn)>,
    /// Optional embedder hook, told about every accepted PCM lifecycle request. The device
    /// itself does nothing with stream state beyond driving the host sink; this exists so a
    /// VMM can know when the guest is holding its audio device open.
    pcm_state_cb: Option<super::PcmStateFn>,
    worker_stop_fd: EventFd,
    worker: Option<JoinHandle<()>>,
}

impl Snd {
    pub fn new(capture_enabled: bool) -> super::Result<Snd> {
        Ok(Snd {
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            device_state: DeviceState::Inactive,
            capture_enabled,
            #[cfg(target_os = "macos")]
            completion_evt: Arc::new(
                EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(SndError::EventFd)?,
            ),
            #[cfg(target_os = "macos")]
            audibility: None,
            pcm_state_cb: None,
            worker_stop_fd: EventFd::new(utils::eventfd::EFD_NONBLOCK)
                .map_err(SndError::EventFd)?,
            worker: None,
        })
    }

    /// Report every accepted PCM lifecycle request to `cb`. Mechanism only: the device does
    /// not filter by stream, debounce, or interpret. Set before the device is activated.
    pub fn set_pcm_state_callback(&mut self, cb: super::PcmStateFn) {
        self.pcm_state_cb = Some(cb);
    }

    /// Report to `cb` when the playback stream crosses between sound and silence, `silence`
    /// being how long a run of bit-exact zero frames has to last before it counts as a pause.
    /// Set before the device is activated.
    #[cfg(target_os = "macos")]
    pub fn set_pcm_audibility_callback(
        &mut self,
        silence: std::time::Duration,
        cb: super::PcmAudibilityFn,
    ) {
        self.audibility = Some((silence, cb));
    }

    pub fn id(&self) -> &str {
        defs::SND_DEV_ID
    }

    fn config(&self) -> VirtioSndConfig {
        // Playback stream 0 always; capture stream 1 only when the mic is enabled.
        let n = if self.capture_enabled { 2 } else { 1 };
        VirtioSndConfig {
            jacks: 0,
            streams: n,
            chmaps: n,
            controls: 0,
        }
    }

    fn stop_worker(&mut self) {
        let Some(worker) = self.worker.take() else {
            return;
        };
        if let Err(e) = self.worker_stop_fd.write(1) {
            error!("snd: cannot signal the worker to stop: {e:?}");
            return;
        }
        if worker.join().is_err() {
            error!("snd: the worker thread panicked");
        }
    }
}

impl VirtioDevice for Snd {
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
        uapi::VIRTIO_ID_SOUND
    }

    fn device_name(&self) -> &str {
        "snd"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let config = self.config();
        let src = config.as_slice();
        let offset = offset as usize;
        if let Some(end) = offset.checked_add(data.len()) {
            if end <= src.len() {
                data.copy_from_slice(&src[offset..end]);
                return;
            }
        }
        error!(
            "snd: out-of-bounds config read (offset={offset}, len={})",
            data.len()
        );
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "snd: guest attempted to write device config (offset={offset:x}, len={:x})",
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
                "snd: bad activate — expected {} queues, got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }

        let stop_fd = self.worker_stop_fd.try_clone().map_err(|e| {
            error!("snd: cannot clone the worker stop fd: {e:?}");
            ActivateError::BadActivate
        })?;
        let config = SndWorkerConfig {
            capture_enabled: self.capture_enabled,
            stop_fd,
            pcm_state_cb: self.pcm_state_cb.clone(),
            #[cfg(target_os = "macos")]
            completion_evt: self.completion_evt.clone(),
            #[cfg(target_os = "macos")]
            audibility: self.audibility.clone(),
        };
        self.stop_worker();
        self.worker = Some(SndWorker::new(queues, mem.clone(), interrupt.clone(), config).run());
        self.device_state = DeviceState::Activated(mem, interrupt);
        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        self.stop_worker();
        self.device_state = DeviceState::Inactive;
        true
    }
}
