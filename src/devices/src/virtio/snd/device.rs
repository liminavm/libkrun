// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use utils::eventfd::EventFd;
#[cfg(target_os = "macos")]
use vm_memory::Bytes;
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
#[cfg(target_os = "macos")]
static LEAD_RAMP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// How often the starvation counters are summarised. Long enough that a persistent fault
/// does not flood the log, short enough to localise one within a session.
#[cfg(target_os = "macos")]
const REPORT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

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

/// A tx descriptor handed to the sink, waiting for its frames to be accounted played.
///
/// The status word's guest address is resolved at submit time but *written* at
/// completion time: the guest reads `latency_bytes` from it then, and a figure computed
/// at submit time would be stale by the time it is seen. (The address is kept rather
/// than the `DescriptorChain`, which borrows guest memory and cannot outlive the call.)
#[cfg(target_os = "macos")]
pub(crate) struct InFlightTx {
    index: u16,
    end_frame: u64,
    status_addr: Option<vm_memory::GuestAddress>,
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
    pub(crate) in_flight: std::collections::VecDeque<InFlightTx>,
    /// Total frames handed to the sink since the last PREPARE.
    #[cfg(target_os = "macos")]
    pub(crate) submitted: u64,
    /// Highest frame position completed to the guest. Ratchets, so a shrinking lead
    /// (a host device change) stalls `hw_ptr` until real playback catches up rather
    /// than moving it backwards, which the guest would read as a buffer wrap.
    #[cfg(target_os = "macos")]
    pub(crate) completed_to: u64,
    /// When the starvation counters were last summarised (throttles a ~93 Hz path).
    #[cfg(target_os = "macos")]
    pub(crate) last_stats_log: Option<std::time::Instant>,
    /// Starvation counters, owned here so they outlive any single `OutputStream`.
    #[cfg(target_os = "macos")]
    pub(crate) snd_stats: std::sync::Arc<super::audio_macos::AudioStats>,
    /// Counter values at the last report, so each line describes its own interval rather
    /// than a running total nobody can difference by eye.
    #[cfg(target_os = "macos")]
    pub(crate) reported_callbacks: u64,
    #[cfg(target_os = "macos")]
    pub(crate) reported_underruns: u64,
    #[cfg(target_os = "macos")]
    pub(crate) reported_frames_short: u64,
    /// Frames submitted as of the last report, so an interval in which the guest fed the
    /// device nothing can be told apart from one in which it fed it too slowly.
    #[cfg(target_os = "macos")]
    pub(crate) reported_submitted: u64,
    /// Delivery-side sampling: when this thread last ran, the longest it went without
    /// running, the deepest unpicked tx backlog, and how often the DAC was dry while the
    /// guest had already queued audio for us. See `sample_delivery`.
    #[cfg(target_os = "macos")]
    pub(crate) last_pass: Option<std::time::Instant>,
    #[cfg(target_os = "macos")]
    pub(crate) max_pass_gap: std::time::Duration,
    #[cfg(target_os = "macos")]
    pub(crate) max_tx_backlog: u16,
    #[cfg(target_os = "macos")]
    pub(crate) dry_with_backlog: u64,
    /// The output device's realtime workgroup, refreshed on each PREPARE.
    #[cfg(target_os = "macos")]
    pub(crate) snd_workgroup: Option<std::sync::Arc<super::audio_macos::AudioWorkgroup>>,
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
            #[cfg(target_os = "macos")]
            completed_to: 0,
            #[cfg(target_os = "macos")]
            last_stats_log: None,
            #[cfg(target_os = "macos")]
            snd_stats: super::audio_macos::AudioStats::new(),
            #[cfg(target_os = "macos")]
            snd_workgroup: None,
            #[cfg(target_os = "macos")]
            reported_callbacks: 0,
            #[cfg(target_os = "macos")]
            reported_underruns: 0,
            #[cfg(target_os = "macos")]
            reported_frames_short: 0,
            #[cfg(target_os = "macos")]
            reported_submitted: 0,
            #[cfg(target_os = "macos")]
            last_pass: None,
            #[cfg(target_os = "macos")]
            max_pass_gap: std::time::Duration::ZERO,
            #[cfg(target_os = "macos")]
            max_tx_backlog: 0,
            #[cfg(target_os = "macos")]
            dry_with_backlog: 0,
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
        let name = match code {
            VIRTIO_SND_R_PCM_PREPARE => "PREPARE",
            VIRTIO_SND_R_PCM_START => "START",
            VIRTIO_SND_R_PCM_STOP => "STOP",
            VIRTIO_SND_R_PCM_RELEASE => "RELEASE",
            _ => "?",
        };
        log::info!("snd: playback {name}");
        match code {
            VIRTIO_SND_R_PCM_PREPARE => {
                self.snd_workgroup = super::audio_macos::AudioWorkgroup::current();
                // Stop the RT thread, then flush any outstanding tx descriptors back to
                // the guest (a re-prepare during xrun recovery discards unplayed audio)
                // before clearing the frame counters.
                if let Some(a) = self.audio.as_mut() {
                    a.stop();
                }
                self.complete_all_in_flight();
                self.submitted = 0;
                self.completed_to = 0;
                self.reported_submitted = 0;
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
                        self.snd_stats.clone(),
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
                // `virtsnd_pcm_sync_stop` waits for every posted I/O buffer to come back
                // before it will let the stream be reused, and a stopped unit consumes
                // nothing — so paced completion would never return them. Same contract the
                // capture path meets with `flush_rx`. The audio these carry is discarded
                // by the stop, which is what the guest asked for.
                self.complete_all_in_flight();
            }
            VIRTIO_SND_R_PCM_RELEASE => {
                self.audio = None; // Drop stops + disposes the unit.
                self.complete_all_in_flight();
                self.submitted = 0;
                self.completed_to = 0;
                self.reported_submitted = 0;
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
        debug!(
            "snd: capture lifecycle code={code:#x} enter (have_unit={})",
            self.capture.is_some()
        );
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
                    debug!(
                        "snd: capture InputStream::new end (ok={})",
                        self.capture.is_some()
                    );
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
            // Feeding the DAC is part of the audio device's realtime deadline, so run it
            // enrolled in that device's workgroup. Scoped to this call, not to the thread:
            // virtio-snd shares the event-manager thread with the GPU, block and net
            // devices, and their work has no business inside an audio deadline contract.
            let _wg = self.snd_workgroup.as_ref().and_then(|w| w.join());

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
                let status_addr = status_word_addr(&head);
                let frames = self.enqueue_tx(&mem, &head, bpf);
                // The buffer is made visible to the guest later, once these frames are
                // accounted played; the status word is written then, not now.
                self.submitted += frames as u64;
                self.in_flight.push_back(InFlightTx {
                    index,
                    end_frame: self.submitted,
                    status_addr,
                });

                // No sink (init failed): fall back to immediate completion.
                if self.audio.is_none() {
                    self.complete_all_in_flight();
                }
            }
            // Drive the paced completion here too. The RT callback only kicks the
            // completion eventfd when it consumed something, so a ring that ran dry
            // would otherwise wait for the *next* callback before the guest learned it
            // had room — exactly when it is most behind. This also ramps the lead up at
            // stream start, before the DAC has consumed anything at all.
            self.complete_paced()
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
        // The device-writable tail is a single status word, written by `complete_up_to`
        // when the buffer is actually handed back — `latency_bytes` has to be current as
        // of then, not as of now.
        frames
    }

    /// Complete every tx descriptor whose frames the sink has now played, advancing the
    /// guest's `hw_ptr` at the host DAC's real rate. Called on the completion eventfd.
    #[cfg(target_os = "macos")]
    pub fn reap_completions(&mut self) -> bool {
        // Drain the eventfd counter (level is re-kicked by the callback as it consumes).
        let _ = self.completion_evt.read();
        if self.audio.is_none() {
            return false;
        }
        self.sample_delivery();
        self.log_starvation_stats();
        self.complete_paced()
    }

    /// Sample, at the DAC's own cadence, the two things that tell apart *who* is late when
    /// the ring runs dry.
    ///
    /// The guest's ALSA `delay` and our ring occupancy measure the same quantity — frames
    /// submitted but not yet played — so when the guest believes it has audio queued and
    /// our ring is empty, the difference is frames the guest has made available and this
    /// device has not collected. `tx_backlog` counts them. `pass_gap` measures how long
    /// this thread went without running at all: virtio-snd shares the event-manager thread
    /// with the GPU, block and net devices, so a gap much longer than a host callback means
    /// the audio starved waiting on *us*, not on the guest.
    #[cfg(target_os = "macos")]
    fn sample_delivery(&mut self) {
        let now = std::time::Instant::now();
        if let Some(prev) = self.last_pass {
            let gap = now.duration_since(prev);
            if gap > self.max_pass_gap {
                self.max_pass_gap = gap;
            }
        }
        self.last_pass = Some(now);

        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => return,
        };
        let backlog = self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
            .queue
            .len(&mem);
        if backlog > self.max_tx_backlog {
            self.max_tx_backlog = backlog;
        }
        // The decisive combination: the DAC had nothing to play while the guest had
        // already handed us audio we had not picked up.
        let dry = self
            .audio
            .as_ref()
            .is_some_and(|a| a.frames_queued() == 0);
        if dry && backlog > 0 {
            self.dry_with_backlog += 1;
        }
    }

    /// Complete tx descriptors up to one host callback *ahead* of real playback, so the
    /// guest keeps that much extra audio queued in our ring where host scheduling jitter
    /// cannot starve the DAC. See [`OutputStream::lead_frames`] for why this is the only
    /// lever that works and why it is honest.
    #[cfg(target_os = "macos")]
    fn complete_paced(&mut self) -> bool {
        let Some(audio) = self.audio.as_ref() else {
            return false;
        };
        let consumed = audio.frames_consumed();
        // Ramp the lead in rather than stepping it. Applying it all at once makes hw_ptr
        // jump by a whole host period at the first completion after PREPARE, which reads
        // to the guest's clock estimator as a discontinuity: it recovers by re-preparing
        // the stream, and the fresh PREPARE re-does the jump, so the fault sustains
        // itself. Growing the lead by one frame per RAMP_PER_FRAME consumed instead makes
        // the device look very slightly fast for the first couple of seconds, which is
        // exactly what a clock matcher exists to absorb.
        // LIMINA_SND_LEAD_RAMP is that divisor; 0 applies the lead as a step.
        const RAMP_PER_FRAME: u64 = 200; // 0.5% fast; full lead after ~2 s
        let ramp = *LEAD_RAMP.get_or_init(|| {
            std::env::var("LIMINA_SND_LEAD_RAMP")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(RAMP_PER_FRAME)
        });
        let lead = match ramp {
            0 => audio.lead_frames(),
            r => audio.lead_frames().min(consumed / r),
        };
        let target = consumed + lead;
        // Ratchet: hw_ptr must never move backwards.
        self.completed_to = self.completed_to.max(target);
        let target = self.completed_to;
        self.complete_up_to(target)
    }

    /// Report the sink's starvation counters once a second. Nothing else in the stack
    /// can see an underrun: the callback pads silence, and the guest's `hw_ptr` only
    /// advances on frames really consumed, so a dropout reads as a briefly slow device
    /// rather than an xrun. `queued` is the host-side lead a late delivery must survive.
    #[cfg(target_os = "macos")]
    fn log_starvation_stats(&mut self) {
        let Some(audio) = self.audio.as_ref() else {
            return;
        };
        let now = std::time::Instant::now();
        let since = self.last_stats_log.map(|last| now.duration_since(last));
        if since.is_some_and(|d| d < REPORT_INTERVAL) {
            return;
        }
        self.last_stats_log = Some(now);

        let (callbacks, underruns, frames_short, min_avail, dropped) = self.snd_stats.snapshot();
        let min_avail = if min_avail == u64::MAX { 0 } else { min_avail };
        let d_callbacks = callbacks - self.reported_callbacks;
        let d_underruns = underruns - self.reported_underruns;
        let d_short = frames_short - self.reported_frames_short;
        let d_submitted = self.submitted - self.reported_submitted;
        self.reported_callbacks = callbacks;
        self.reported_underruns = underruns;
        self.reported_frames_short = frames_short;
        self.reported_submitted = self.submitted;

        // A stream the guest has left open but is not feeding is not a starving DAC — it
        // is silence, and a game between sounds produces it constantly. Without this the
        // device reports a flawless 100% starvation for every quiet moment.
        //
        // But silence and a deadlock look identical from the callback's side, and telling
        // them apart matters more than either: if descriptors are still in flight, the
        // guest is not quiet, it is *waiting for us*. It cannot submit until we hand those
        // back, and we hand them back as the DAC consumes — so if the DAC is also idle,
        // neither side can move again without help. Never file that as silence.
        if d_submitted == 0 {
            let owed = self.in_flight.len();
            let secs = since.unwrap_or(REPORT_INTERVAL).as_secs_f64();
            if owed > 0 {
                log::warn!(
                    "snd: playback WEDGED — {owed} tx descriptors owed to the guest and nothing \
                     submitted or played in {secs:.1}s; the guest is blocked waiting for buffers \
                     this device has not returned (consumed={}, submitted={})",
                    self.audio.as_ref().map_or(0, |a| a.frames_consumed()),
                    self.submitted,
                );
            } else if d_underruns > 0 {
                log::info!(
                    "snd: stream open but idle — the guest submitted no audio in the last {secs:.1}s"
                );
            }
            self.max_pass_gap = std::time::Duration::ZERO;
            self.max_tx_backlog = 0;
            self.dry_with_backlog = 0;
            return;
        }

        // A starving DAC is a fault, so say so at a level the shipped app actually
        // records: the worker runs at `warn` unless RUST_LOG says otherwise, and a
        // counter that only speaks at `info` is silent in exactly the deployment where
        // someone is trying to find out why the audio is broken.
        if d_underruns > 0 {
            let secs = since.unwrap_or(REPORT_INTERVAL).as_secs_f64();
            log::warn!(
                "snd: DAC starved — {d_underruns} of {d_callbacks} callbacks short in {secs:.1}s \
                 ({d_short} frames of silence, {:.0} ms); queued now {}f, low water {min_avail}f; \
                 guest period={}B buffer={}B; delivery: max thread gap {:.1}ms, max tx backlog {}, \
                 dry-with-backlog {}; (cumulative: {underruns}/{callbacks}, dropped={dropped})",
                d_short as f64 * 1000.0 / 48_000.0,
                audio.frames_queued(),
                self.params.period_bytes,
                self.params.buffer_bytes,
                self.max_pass_gap.as_secs_f64() * 1000.0,
                self.max_tx_backlog,
                self.dry_with_backlog,
            );
            self.max_pass_gap = std::time::Duration::ZERO;
            self.max_tx_backlog = 0;
            self.dry_with_backlog = 0;
        } else {
            log::info!(
                "snd: callbacks={callbacks} underruns={underruns} frames_short={frames_short} \
                 min_queued={min_avail}f queued_now={}f dropped={dropped} period={}B buffer={}B",
                audio.frames_queued(),
                self.params.period_bytes,
                self.params.buffer_bytes,
            );
            self.max_pass_gap = std::time::Duration::ZERO;
            self.max_tx_backlog = 0;
            self.dry_with_backlog = 0;
        }
    }

    /// Complete all in-flight tx descriptors with end-frame <= `consumed`.
    #[cfg(target_os = "macos")]
    fn complete_up_to(&mut self, consumed: u64) -> bool {
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => return false,
        };
        // Delay the guest cannot already see for itself. It accounts for everything
        // between `hw_ptr` and `appl_ptr` on its own, and ALSA adds `runtime->delay` on
        // top — so this must be *only* the part we hid from it, i.e. how far `hw_ptr` has
        // been advanced beyond frames the DAC has really played. Reporting whole-ring
        // occupancy here double-counts the frames that are in the ring but not yet
        // completed, and makes the figure sawtooth with ring depth instead of tracking
        // the (smooth) lead.
        let bpf = self.params.bytes_per_frame() as u64;
        let latency_bytes = self
            .audio
            .as_ref()
            .map(|a| self.completed_to.saturating_sub(a.frames_consumed()) * bpf)
            .unwrap_or(0)
            .min(u32::MAX as u64) as u32;

        let mut used_any = false;
        while let Some(front) = self.in_flight.front() {
            if front.end_frame > consumed {
                break;
            }
            let tx = self.in_flight.pop_front().expect("front exists");
            let mut written = 0u32;
            if let Some(addr) = tx.status_addr {
                let status = VirtioSndPcmStatus {
                    status: VIRTIO_SND_S_OK,
                    latency_bytes,
                };
                if mem.write_obj(status, addr).is_ok() {
                    written = std::mem::size_of::<VirtioSndPcmStatus>() as u32;
                }
            }
            if let Err(e) = self.queues.as_mut().expect("queues exist")[defs::TX_INDEX]
                .queue
                .add_used(&mem, tx.index, written)
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
        self.completed_to = self.completed_to.max(end);
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

/// Guest address of a tx chain's status word: the first device-writable descriptor.
/// A playback chain is `[xfer header][PCM data…][status]`, so the writable tail is where
/// the driver expects `virtio_snd_pcm_status`.
#[cfg(target_os = "macos")]
fn status_word_addr(head: &DescriptorChain) -> Option<vm_memory::GuestAddress> {
    let mut desc = Some(head.clone());
    while let Some(d) = desc {
        if d.is_write_only() && (d.len as usize) >= std::mem::size_of::<VirtioSndPcmStatus>() {
            return Some(d.addr);
        }
        desc = d.next_descriptor();
    }
    None
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
            self.completed_to = 0;
            self.last_stats_log = None;
        }
        true
    }
}
