// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

use utils::eventfd::EventFd;
use vm_memory::{ByteValued, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, QueueConfig, VirtioDevice,
};
use super::protocol::*;
use super::{defs, defs::uapi, SndError};
use crate::virtio::descriptor_utils::{Reader, Writer};
use crate::virtio::InterruptTransport;

// Supported features: virtio 1.0 only (no VIRTIO_SND_F_CTLS — we expose no control
// elements yet; guest/host volume stack instead).
pub(crate) const AVAIL_FEATURES: u64 = 1 << uapi::VIRTIO_F_VERSION_1 as u64;

// The single advertised playback stream (stream_id 0) and channel map (chmap_id 0).
const NUM_STREAMS: u32 = 1;
const NUM_CHMAPS: u32 = 1;

/// Parameters the guest selected for a stream via SET_PARAMS. Retained so Phase B's
/// CoreAudio sink can match its render format; unused in Phase A (null sink).
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)] // fields consumed by the Phase B CoreAudio sink.
pub(crate) struct StreamParams {
    pub buffer_bytes: u32,
    pub period_bytes: u32,
    pub channels: u8,
    pub format: u8,
    pub rate: u8,
}

pub struct Snd {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    pub(crate) params: StreamParams,
}

impl Snd {
    pub(crate) fn queue_event(&self, idx: usize) -> &std::sync::Arc<utils::eventfd::EventFd> {
        &self.queues.as_ref().expect("queues should exist")[idx].event
    }

    pub fn new() -> super::Result<Snd> {
        Ok(Snd {
            queues: None,
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(SndError::EventFd)?,
            device_state: DeviceState::Inactive,
            params: StreamParams::default(),
        })
    }

    pub fn id(&self) -> &str {
        defs::SND_DEV_ID
    }

    fn config(&self) -> VirtioSndConfig {
        VirtioSndConfig {
            jacks: 0,
            streams: NUM_STREAMS,
            chmaps: NUM_CHMAPS,
            controls: 0,
        }
    }

    /// Service the control queue: parse each request, write its response, mark used.
    /// Returns true if any descriptor was consumed (caller signals the guest).
    pub fn process_control(&mut self) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };
        let queues = self.queues.as_mut().expect("queues exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[defs::CONTROL_INDEX].queue.pop(&mem) {
            let index = head.index;
            let written = process_control_req(&mut self.params, &mem, &head);
            have_used = true;
            if let Err(e) = queues[defs::CONTROL_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used control descriptor: {e:?}");
            }
        }
        have_used
    }

    /// Phase A null sink: accept tx (playback) frames and complete them immediately.
    /// Phase B replaces this with a CoreAudio-paced worker (see snd/mod.rs).
    pub fn process_tx(&mut self) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };
        let queues = self.queues.as_mut().expect("queues exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[defs::TX_INDEX].queue.pop(&mem) {
            let index = head.index;
            let mut written = 0u32;
            // The device-writable tail is a single status word; the PCM payload in the
            // readable descriptors is simply dropped (null sink).
            if let Ok(mut writer) = Writer::new(&mem, head.clone()) {
                let status = VirtioSndPcmStatus {
                    status: VIRTIO_SND_S_OK,
                    latency_bytes: 0,
                };
                if writer.write_obj(status).is_ok() {
                    written = writer.bytes_written() as u32;
                }
            }
            have_used = true;
            if let Err(e) = queues[defs::TX_INDEX].queue.add_used(&mem, index, written) {
                error!("snd: failed to add used tx descriptor: {e:?}");
            }
        }
        have_used
    }
}

/// Parse one control request from `head` and write its response. Returns the number
/// of bytes written to the device-writable descriptors (the used length).
fn process_control_req(
    params: &mut StreamParams,
    mem: &GuestMemoryMmap,
    head: &crate::virtio::DescriptorChain,
) -> u32 {
    let mut writer = match Writer::new(mem, head.clone()) {
        Ok(w) => w,
        Err(e) => {
            error!("snd: control response writer error: {e:?}");
            return 0;
        }
    };

    // Every request starts with a 32-bit code; read it from a fresh reader so the
    // full request struct can then be re-read from the top.
    let code = Reader::new(mem, head.clone())
        .ok()
        .and_then(|mut r| r.read_obj::<VirtioSndHdr>().ok())
        .map(|hdr| hdr.code);
    let code = match code {
        Some(c) => c,
        None => {
            error!("snd: could not read control request header");
            let _ = writer.write_obj(VirtioSndHdr {
                code: VIRTIO_SND_S_BAD_MSG,
            });
            return writer.bytes_written() as u32;
        }
    };

    let mut reader = match Reader::new(mem, head.clone()) {
        Ok(r) => r,
        Err(_) => return 0,
    };

    match code {
        VIRTIO_SND_R_PCM_INFO => {
            let q: VirtioSndQueryInfo = reader.read_obj().unwrap_or_default();
            write_status(&mut writer, VIRTIO_SND_S_OK);
            for id in q.start_id..q.start_id.saturating_add(q.count) {
                if id == 0 {
                    let _ = writer.write_obj(pcm_info());
                }
            }
        }
        VIRTIO_SND_R_CHMAP_INFO => {
            let q: VirtioSndQueryInfo = reader.read_obj().unwrap_or_default();
            write_status(&mut writer, VIRTIO_SND_S_OK);
            for id in q.start_id..q.start_id.saturating_add(q.count) {
                if id == 0 {
                    let _ = writer.write_obj(chmap_info());
                }
            }
        }
        VIRTIO_SND_R_JACK_INFO => {
            // No jacks advertised; acknowledge with no items.
            let _q: VirtioSndQueryInfo = reader.read_obj().unwrap_or_default();
            write_status(&mut writer, VIRTIO_SND_S_OK);
        }
        VIRTIO_SND_R_PCM_SET_PARAMS => {
            let p: VirtioSndPcmSetParams = reader.read_obj().unwrap_or_default();
            let status = if p.stream_id == 0 {
                *params = StreamParams {
                    buffer_bytes: p.buffer_bytes,
                    period_bytes: p.period_bytes,
                    channels: p.channels,
                    format: p.format,
                    rate: p.rate,
                };
                VIRTIO_SND_S_OK
            } else {
                VIRTIO_SND_S_BAD_MSG
            };
            write_status(&mut writer, status);
        }
        VIRTIO_SND_R_PCM_PREPARE
        | VIRTIO_SND_R_PCM_START
        | VIRTIO_SND_R_PCM_STOP
        | VIRTIO_SND_R_PCM_RELEASE => {
            let h: VirtioSndPcmHdr = reader.read_obj().unwrap_or_default();
            let status = if h.stream_id == 0 {
                VIRTIO_SND_S_OK
            } else {
                VIRTIO_SND_S_BAD_MSG
            };
            write_status(&mut writer, status);
        }
        other => {
            warn!("snd: unsupported control request code {other:#x}");
            write_status(&mut writer, VIRTIO_SND_S_NOT_SUPP);
        }
    }

    writer.bytes_written() as u32
}

fn write_status(writer: &mut Writer, code: u32) {
    if let Err(e) = writer.write_obj(VirtioSndHdr { code }) {
        error!("snd: failed to write response status: {e:?}");
    }
}

/// The single stereo playback stream we advertise: S16_LE @ 48 kHz, 2 channels.
fn pcm_info() -> VirtioSndPcmInfo {
    VirtioSndPcmInfo {
        hda_fn_nid: 0,
        features: 0,
        formats: 1u64 << VIRTIO_SND_PCM_FMT_S16,
        rates: 1u64 << VIRTIO_SND_PCM_RATE_48000,
        direction: VIRTIO_SND_D_OUTPUT,
        channels_min: 2,
        channels_max: 2,
        padding: [0; 5],
    }
}

/// Front-left / front-right stereo channel map for the playback stream.
fn chmap_info() -> VirtioSndChmapInfo {
    let mut positions = [0u8; VIRTIO_SND_CHMAP_MAX_SIZE];
    positions[0] = VIRTIO_SND_CHMAP_FL;
    positions[1] = VIRTIO_SND_CHMAP_FR;
    VirtioSndChmapInfo {
        hda_fn_nid: 0,
        direction: VIRTIO_SND_D_OUTPUT,
        channels: 2,
        positions,
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

        if self.activate_evt.write(1).is_err() {
            error!("snd: cannot write activate_evt");
            return Err(ActivateError::BadActivate);
        }

        self.queues = Some(queues);
        self.device_state = DeviceState::Activated(mem, interrupt);
        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        self.queues = None;
        self.device_state = DeviceState::Inactive;
        self.params = StreamParams::default();
        true
    }
}
