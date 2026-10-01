// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use utils::eventfd::EventFd;
use vm_memory::{ByteValued, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, QueueConfig, VirtioDevice,
};
use super::protocol::*;
use super::{defs, defs::uapi, SndError};
use crate::virtio::descriptor_utils::{Reader, Writer};
use crate::virtio::{DescriptorChain, InterruptTransport};

// Supported features: virtio 1.0 only (no VIRTIO_SND_F_CTLS — we expose no control
// elements yet; guest/host volume stack instead).
pub(crate) const AVAIL_FEATURES: u64 = 1 << uapi::VIRTIO_F_VERSION_1 as u64;

// Stream / chmap identifiers. Playback (output) is always stream 0 / chmap 0; when
// mic capture is enabled it adds an input stream 1 / chmap 1.
const OUTPUT_STREAM_ID: u32 = 0;
const CAPTURE_STREAM_ID: u32 = 1;

/// Parameters the guest selected for a stream via SET_PARAMS.
#[derive(Debug, Clone, Copy, Default)]
#[allow(dead_code)] // buffer_bytes/period_bytes retained for future latency reporting.
pub(crate) struct StreamParams {
    pub buffer_bytes: u32,
    pub period_bytes: u32,
    pub channels: u8,
    pub format: u8,
    pub rate: u8,
}

impl StreamParams {
    /// Bytes per PCM frame on the tx queue. We only advertise S16 (2 bytes/sample),
    /// so this is `channels * 2`; default to stereo before SET_PARAMS lands.
    fn bytes_per_frame(&self) -> usize {
        let channels = if self.channels == 0 {
            2
        } else {
            self.channels as usize
        };
        channels * 2
    }
}

pub struct Snd {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    pub(crate) params: StreamParams,
    /// Whether the mic-capture input stream is advertised (opt-in; default off for
    /// privacy). When false the device exposes playback only and never touches the mic.
    pub(crate) capture_enabled: bool,
    /// Parameters the guest selected for the capture stream via SET_PARAMS.
    #[cfg(target_os = "macos")]
    pub(crate) capture_params: StreamParams,
    /// Kicked by the CoreAudio render callback when it consumes (playback) or produces
    /// (capture) frames, so the device thread reaps tx completions and/or drains the
    /// capture ring into rx buffers.
    #[cfg(target_os = "macos")]
    pub(crate) completion_evt: Arc<EventFd>,
    /// The host CoreAudio output sink (created on PCM_PREPARE for stream 0).
    #[cfg(target_os = "macos")]
    pub(crate) audio: Option<super::audio_macos::OutputStream>,
    /// The host CoreAudio input source (created on PCM_PREPARE for stream 1).
    #[cfg(target_os = "macos")]
    pub(crate) capture: Option<super::audio_macos::InputStream>,
    /// Popped tx descriptors awaiting completion, each tagged with the cumulative
    /// frame count at which it becomes complete (paced by real playback).
    #[cfg(target_os = "macos")]
    pub(crate) in_flight: std::collections::VecDeque<(u16, u64)>,
    /// Total frames handed to the sink since the last PREPARE.
    #[cfg(target_os = "macos")]
    pub(crate) submitted: u64,
}

impl Snd {
    pub(crate) fn queue_event(&self, idx: usize) -> &std::sync::Arc<utils::eventfd::EventFd> {
        &self.queues.as_ref().expect("queues should exist")[idx].event
    }

    pub fn new(capture_enabled: bool) -> super::Result<Snd> {
        Ok(Snd {
            queues: None,
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(SndError::EventFd)?,
            device_state: DeviceState::Inactive,
            params: StreamParams::default(),
            capture_enabled,
            #[cfg(target_os = "macos")]
            capture_params: StreamParams::default(),
            #[cfg(target_os = "macos")]
            completion_evt: Arc::new(
                EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(SndError::EventFd)?,
            ),
            #[cfg(target_os = "macos")]
            audio: None,
            #[cfg(target_os = "macos")]
            capture: None,
            #[cfg(target_os = "macos")]
            in_flight: std::collections::VecDeque::new(),
            #[cfg(target_os = "macos")]
            submitted: 0,
        })
    }

    pub fn id(&self) -> &str {
        defs::SND_DEV_ID
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn completion_evt_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.completion_evt.as_raw_fd()
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

    /// Service the control queue: parse each request, write its response, mark used.
    /// Returns true if any descriptor was consumed (caller signals the guest).
    pub fn process_control(&mut self) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };
        let mut have_used = false;

        loop {
            // Pop under a short borrow so the handler can take &mut self freely.
            let head = match self.queues.as_mut().expect("queues exist")[defs::CONTROL_INDEX]
                .queue
                .pop(&mem)
            {
                Some(h) => h,
                None => break,
            };
            let index = head.index;
            let written = self.handle_control_req(&mem, &head);
            if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::CONTROL_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used control descriptor: {e:?}");
            }
            have_used = true;
        }
        have_used
    }

    /// Parse one control request and write its response; returns the used length.
    fn handle_control_req(&mut self, mem: &GuestMemoryMmap, head: &DescriptorChain) -> u32 {
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
                write_status(&mut writer, VIRTIO_SND_S_BAD_MSG);
                return writer.bytes_written() as u32;
            }
        };

        let mut reader = match Reader::new(mem, head.clone()) {
            Ok(r) => r,
            Err(_) => return 0,
        };

        debug!("snd: control req code={code:#x}");
        match code {
            VIRTIO_SND_R_PCM_INFO => {
                let q: VirtioSndQueryInfo = reader.read_obj().unwrap_or_default();
                write_status(&mut writer, VIRTIO_SND_S_OK);
                for id in q.start_id..q.start_id.saturating_add(q.count) {
                    if id == OUTPUT_STREAM_ID {
                        let _ = writer.write_obj(output_pcm_info());
                    } else if id == CAPTURE_STREAM_ID && self.capture_enabled {
                        let _ = writer.write_obj(capture_pcm_info());
                    }
                }
            }
            VIRTIO_SND_R_CHMAP_INFO => {
                let q: VirtioSndQueryInfo = reader.read_obj().unwrap_or_default();
                write_status(&mut writer, VIRTIO_SND_S_OK);
                for id in q.start_id..q.start_id.saturating_add(q.count) {
                    if id == OUTPUT_STREAM_ID {
                        let _ = writer.write_obj(output_chmap_info());
                    } else if id == CAPTURE_STREAM_ID && self.capture_enabled {
                        let _ = writer.write_obj(capture_chmap_info());
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
                let params = StreamParams {
                    buffer_bytes: p.buffer_bytes,
                    period_bytes: p.period_bytes,
                    channels: p.channels,
                    format: p.format,
                    rate: p.rate,
                };
                let status = if p.stream_id == OUTPUT_STREAM_ID {
                    self.params = params;
                    VIRTIO_SND_S_OK
                } else if p.stream_id == CAPTURE_STREAM_ID && self.capture_enabled {
                    #[cfg(target_os = "macos")]
                    {
                        self.capture_params = params;
                    }
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
                let status = if h.stream_id == OUTPUT_STREAM_ID {
                    self.handle_pcm_lifecycle(code);
                    VIRTIO_SND_S_OK
                } else if h.stream_id == CAPTURE_STREAM_ID && self.capture_enabled {
                    self.handle_capture_lifecycle(code);
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

    /// Drive the host sink through the PCM lifecycle (macOS). On other hosts this is a
    /// no-op and tx falls back to the immediate null sink.
    #[cfg(target_os = "macos")]
    fn handle_pcm_lifecycle(&mut self, code: u32) {
        match code {
            VIRTIO_SND_R_PCM_PREPARE => {
                // Stop the RT thread, then flush any outstanding tx descriptors back to
                // the guest (a re-prepare during xrun recovery discards unplayed audio)
                // before clearing the frame counters.
                if let Some(a) = self.audio.as_mut() {
                    a.stop();
                }
                self.complete_all_in_flight();
                self.submitted = 0;
                if self.audio.is_none() {
                    let channels = if self.params.channels == 0 {
                        2
                    } else {
                        self.params.channels as usize
                    };
                    match super::audio_macos::OutputStream::new(
                        48_000.0,
                        channels,
                        self.completion_evt.clone(),
                    ) {
                        Ok(s) => self.audio = Some(s),
                        Err(e) => error!("snd: CoreAudio output init failed ({e}); silent sink"),
                    }
                }
                if let Some(a) = self.audio.as_mut() {
                    a.reset();
                }
            }
            VIRTIO_SND_R_PCM_START => {
                if let Some(a) = self.audio.as_mut() {
                    a.start();
                }
            }
            VIRTIO_SND_R_PCM_STOP => {
                if let Some(a) = self.audio.as_mut() {
                    a.stop();
                }
            }
            VIRTIO_SND_R_PCM_RELEASE => {
                self.audio = None; // Drop stops + disposes the unit.
                self.complete_all_in_flight();
                self.submitted = 0;
            }
            _ => {}
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn handle_pcm_lifecycle(&mut self, _code: u32) {}

    /// Drive the host mic source through the capture stream's PCM lifecycle (macOS).
    /// Creating the input unit on PREPARE is what triggers the mic TCC prompt.
    #[cfg(target_os = "macos")]
    fn handle_capture_lifecycle(&mut self, code: u32) {
        debug!("snd: capture lifecycle code={code:#x} enter (have_unit={})", self.capture.is_some());
        match code {
            VIRTIO_SND_R_PCM_PREPARE => {
                if let Some(c) = self.capture.as_mut() {
                    c.stop();
                }
                if self.capture.is_none() {
                    let channels = if self.capture_params.channels == 0 {
                        1
                    } else {
                        self.capture_params.channels as usize
                    };
                    debug!("snd: capture InputStream::new({channels}ch) begin");
                    match super::audio_macos::InputStream::new(
                        48_000.0,
                        channels,
                        self.completion_evt.clone(),
                    ) {
                        Ok(s) => self.capture = Some(s),
                        Err(e) => error!("snd: CoreAudio input init failed ({e}); silent mic"),
                    }
                    debug!("snd: capture InputStream::new end (ok={})", self.capture.is_some());
                }
                if let Some(c) = self.capture.as_mut() {
                    c.reset();
                }
            }
            VIRTIO_SND_R_PCM_START => {
                if let Some(c) = self.capture.as_mut() {
                    c.start();
                }
            }
            VIRTIO_SND_R_PCM_STOP => {
                if let Some(c) = self.capture.as_mut() {
                    c.stop();
                }
                // The guest stops queueing rx buffers after STOP; return any it already
                // posted so its ring drains (the driver waits for them on release).
                self.flush_rx();
            }
            VIRTIO_SND_R_PCM_RELEASE => {
                debug!("snd: capture RELEASE — dropping InputStream begin");
                self.capture = None; // Drop stops + disposes the unit.
                // The Linux virtio_snd driver's PCM release blocks until every posted rx
                // I/O buffer has been returned (msg_count == 0). Return them here, or the
                // guest hangs and the NEXT open times out (device appears wedged).
                self.flush_rx();
                debug!("snd: capture RELEASE — dropped");
            }
            _ => {}
        }
        debug!("snd: capture lifecycle code={code:#x} done");
    }

    #[cfg(not(target_os = "macos"))]
    fn handle_capture_lifecycle(&mut self, _code: u32) {}

    /// Playback path. On macOS: copy the guest frames into the sink's ring and record
    /// the descriptor for paced completion (do NOT complete here). Elsewhere: null sink
    /// that completes immediately. Returns true if any descriptor was completed inline
    /// (only the non-macOS path; macOS completes from `reap_completions`).
    pub fn process_tx(&mut self) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };

        #[cfg(target_os = "macos")]
        {
            let bpf = self.params.bytes_per_frame();
            loop {
                let head = match self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
                    .queue
                    .pop(&mem)
                {
                    Some(h) => h,
                    None => break,
                };
                let index = head.index;
                let frames = self.enqueue_tx(&mem, &head, bpf);
                // Status is written now (it is always S_OK); the buffer is made visible
                // to the guest later, once these frames have actually been played.
                self.submitted += frames as u64;
                self.in_flight.push_back((index, self.submitted));

                // No sink (init failed): fall back to immediate completion.
                if self.audio.is_none() {
                    self.complete_all_in_flight();
                }
            }
            // Completions are signalled from reap_completions, not here.
            false
        }

        #[cfg(not(target_os = "macos"))]
        {
            let mut have_used = false;
            loop {
                let head = match self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
                    .queue
                    .pop(&mem)
                {
                    Some(h) => h,
                    None => break,
                };
                let index = head.index;
                let mut written = 0u32;
                if let Ok(mut writer) = Writer::new(&mem, head.clone()) {
                    let status = VirtioSndPcmStatus {
                        status: VIRTIO_SND_S_OK,
                        latency_bytes: 0,
                    };
                    if writer.write_obj(status).is_ok() {
                        written = writer.bytes_written() as u32;
                    }
                }
                if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
                    .queue
                    .add_used(&mem, index, written)
                {
                    error!("snd: failed to add used tx descriptor: {e:?}");
                }
                have_used = true;
            }
            have_used
        }
    }

    /// Read one tx buffer's PCM payload (S16_LE) into the CoreAudio ring as f32, and
    /// write its status word. Returns the number of frames the buffer carried (whether
    /// or not they fit in the ring — the descriptor is always accounted so it completes
    /// on schedule and the guest never stalls).
    #[cfg(target_os = "macos")]
    fn enqueue_tx(&mut self, mem: &GuestMemoryMmap, head: &DescriptorChain, bpf: usize) -> usize {
        use std::io::Read;

        let mut frames = 0usize;
        if let Ok(mut reader) = Reader::new(mem, head.clone()) {
            // Skip the xfer header (stream_id); the rest of the readable region is PCM.
            let _xfer: VirtioSndPcmXfer = reader.read_obj().unwrap_or_default();
            let pcm_bytes = (reader.available_bytes() / bpf) * bpf;
            frames = pcm_bytes / bpf;
            if pcm_bytes > 0 {
                let mut data = vec![0u8; pcm_bytes];
                if reader.read_exact(&mut data).is_ok() {
                    if let Some(a) = self.audio.as_ref() {
                        let samples: Vec<f32> = data
                            .chunks_exact(2)
                            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                            .collect();
                        a.push_samples(&samples);
                    }
                }
            }
        }
        // The device-writable tail is a single status word.
        if let Ok(mut writer) = Writer::new(mem, head.clone()) {
            let _ = writer.write_obj(VirtioSndPcmStatus {
                status: VIRTIO_SND_S_OK,
                latency_bytes: 0,
            });
        }
        frames
    }

    /// Complete every tx descriptor whose frames the sink has now played, advancing the
    /// guest's `hw_ptr` at the host DAC's real rate. Called on the completion eventfd.
    #[cfg(target_os = "macos")]
    pub fn reap_completions(&mut self) -> bool {
        // Drain the eventfd counter (level is re-kicked by the callback as it consumes).
        let _ = self.completion_evt.read();
        let consumed = match self.audio.as_ref() {
            Some(a) => a.frames_consumed(),
            None => return false,
        };
        self.complete_up_to(consumed)
    }

    /// Complete all in-flight tx descriptors with end-frame <= `consumed`.
    #[cfg(target_os = "macos")]
    fn complete_up_to(&mut self, consumed: u64) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => return false,
        };
        let mut used_any = false;
        while let Some(&(index, end)) = self.in_flight.front() {
            if end > consumed {
                break;
            }
            self.in_flight.pop_front();
            if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
                .queue
                .add_used(
                    &mem,
                    index,
                    std::mem::size_of::<VirtioSndPcmStatus>() as u32,
                )
            {
                error!("snd: failed to add used tx descriptor: {e:?}");
            }
            used_any = true;
        }
        if used_any {
            self.device_state.signal_used_queue();
        }
        used_any
    }

    /// Complete every outstanding tx descriptor immediately (RELEASE, or no sink).
    #[cfg(target_os = "macos")]
    fn complete_all_in_flight(&mut self) {
        let end = self.submitted;
        self.complete_up_to(end);
    }

    /// Capture path (macOS). Fill each posted rx buffer with a full period of captured
    /// mic audio (S16_LE) drained from the CoreAudio input ring, and complete it. A
    /// buffer is filled only once enough frames are queued — otherwise it is left posted
    /// (put back) so the status word lands at the buffer's fixed tail offset and the
    /// guest `hw_ptr` advances by whole periods. Called on the rx kick and the completion
    /// eventfd. Non-macOS never advertises a capture stream, so there is no rx path there.
    #[cfg(target_os = "macos")]
    pub fn process_rx(&mut self) -> bool {
        use std::io::Write;

        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => return false,
        };
        let bpf = self.capture_params.bytes_per_frame();
        let status_sz = std::mem::size_of::<VirtioSndPcmStatus>();
        let mut used_any = false;

        loop {
            // No source yet (not prepared/started): leave rx buffers posted.
            let avail_samples = match self.capture.as_ref() {
                Some(c) => c.available(),
                None => break,
            };
            let head = match self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                .queue
                .pop(&mem)
            {
                Some(h) => h,
                None => break,
            };
            let index = head.index;
            let mut writer = match Writer::new(&mem, head.clone()) {
                Ok(w) => w,
                Err(e) => {
                    error!("snd: capture writer error: {e:?}");
                    if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                        .queue
                        .add_used(&mem, index, 0)
                    {
                        error!("snd: failed to add used rx descriptor: {e:?}");
                    }
                    used_any = true;
                    continue;
                }
            };

            // Writable region is [PCM data (period)] [status]; fill the whole PCM span.
            let pcm_cap = writer.available_bytes().saturating_sub(status_sz);
            let frames = pcm_cap / bpf;
            let need_samples = frames * (bpf / 2); // bpf/2 == channels (S16)
            if frames == 0 || avail_samples < need_samples {
                // Not enough captured yet — put the buffer back and wait for more.
                self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                    .queue
                    .go_to_previous_position();
                break;
            }

            let mut samples = vec![0f32; need_samples];
            let got = self
                .capture
                .as_ref()
                .expect("capture present")
                .pull_samples(&mut samples);
            // Fill the entire PCM region: real frames, zero-padded tail if non-aligned.
            let mut pcm = vec![0u8; pcm_cap];
            for (i, s) in samples[..got].iter().enumerate() {
                let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                pcm[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
            }
            let _ = writer.write_all(&pcm);
            let _ = writer.write_obj(VirtioSndPcmStatus {
                status: VIRTIO_SND_S_OK,
                latency_bytes: 0,
            });
            let written = writer.bytes_written() as u32;
            if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used rx descriptor: {e:?}");
            }
            used_any = true;
        }

        if used_any {
            self.device_state.signal_used_queue();
        }
        used_any
    }

    /// Return every rx buffer the guest has posted, completing each with a silent period.
    /// Called on capture STOP/RELEASE: the Linux virtio_snd driver's release path blocks
    /// until all posted I/O buffers are back in the used ring, so leaving any outstanding
    /// wedges the stream (the next open times out). Idempotent — a no-op if none are posted.
    #[cfg(target_os = "macos")]
    fn flush_rx(&mut self) {
        use std::io::Write;

        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => return,
        };
        let status_sz = std::mem::size_of::<VirtioSndPcmStatus>();
        let mut used_any = false;

        loop {
            let head = match self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                .queue
                .pop(&mem)
            {
                Some(h) => h,
                None => break,
            };
            let index = head.index;
            let mut written = 0u32;
            if let Ok(mut writer) = Writer::new(&mem, head.clone()) {
                // Fill the whole PCM region with silence so the status word lands at the
                // buffer's fixed tail offset, then write it.
                let pcm_cap = writer.available_bytes().saturating_sub(status_sz);
                let zeros = vec![0u8; pcm_cap];
                let _ = writer.write_all(&zeros);
                let _ = writer.write_obj(VirtioSndPcmStatus {
                    status: VIRTIO_SND_S_OK,
                    latency_bytes: 0,
                });
                written = writer.bytes_written() as u32;
            }
            if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::RX_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used rx descriptor on flush: {e:?}");
            }
            used_any = true;
        }

        if used_any {
            self.device_state.signal_used_queue();
            debug!("snd: flushed outstanding rx buffers");
        }
    }
}

fn write_status(writer: &mut Writer, code: u32) {
    if let Err(e) = writer.write_obj(VirtioSndHdr { code }) {
        error!("snd: failed to write response status: {e:?}");
    }
}

/// The stereo playback stream (stream 0): S16_LE @ 48 kHz, 2 channels.
fn output_pcm_info() -> VirtioSndPcmInfo {
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
fn output_chmap_info() -> VirtioSndChmapInfo {
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

/// The mono mic-capture stream (stream 1): S16_LE @ 48 kHz, 1 channel.
fn capture_pcm_info() -> VirtioSndPcmInfo {
    VirtioSndPcmInfo {
        hda_fn_nid: 0,
        features: 0,
        formats: 1u64 << VIRTIO_SND_PCM_FMT_S16,
        rates: 1u64 << VIRTIO_SND_PCM_RATE_48000,
        direction: VIRTIO_SND_D_INPUT,
        channels_min: 1,
        channels_max: 1,
        padding: [0; 5],
    }
}

/// Mono channel map for the capture stream.
fn capture_chmap_info() -> VirtioSndChmapInfo {
    let mut positions = [0u8; VIRTIO_SND_CHMAP_MAX_SIZE];
    positions[0] = VIRTIO_SND_CHMAP_MONO;
    VirtioSndChmapInfo {
        hda_fn_nid: 0,
        direction: VIRTIO_SND_D_INPUT,
        channels: 1,
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
        #[cfg(target_os = "macos")]
        {
            self.audio = None;
            self.capture = None;
            self.capture_params = StreamParams::default();
            self.in_flight.clear();
            self.submitted = 0;
        }
        true
    }
}
