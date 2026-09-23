// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The virtio-snd device's own thread.
//!
//! Everything the guest's audio depends on runs here rather than on the VMM's shared device
//! event loop: control requests, taking tx buffers into the CoreAudio ring, completing them as
//! the render callback plays them, and filling rx buffers from the mic. On the shared loop a
//! slow neighbour delayed every one of those, and the guest heard it as a gap.
//!
//! This thread is also the only one that creates, starts, stops and disposes the CoreAudio
//! units (`audio_macos`'s `Send` invariant): the control queue that drives their lifecycle is
//! serviced here, and a device reset tears them down here before the thread exits.

use std::os::unix::io::AsRawFd;
#[cfg(target_os = "macos")]
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use utils::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use utils::eventfd::EventFd;
use vm_memory::GuestMemoryMmap;

use super::super::DeviceQueue;
use super::protocol::*;
use super::{PcmStateFn, defs};
use crate::virtio::descriptor_utils::{Reader, Writer};
use crate::virtio::{DescriptorChain, InterruptTransport};

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

pub(crate) struct SndWorker {
    queues: Vec<DeviceQueue>,
    mem: GuestMemoryMmap,
    interrupt: InterruptTransport,
    /// Written by `Snd::reset` to make the thread tear down and exit.
    stop_fd: EventFd,
    params: StreamParams,
    /// Whether the mic-capture input stream is advertised (opt-in; default off for
    /// privacy). When false the device exposes playback only and never touches the mic.
    capture_enabled: bool,
    /// Parameters the guest selected for the capture stream via SET_PARAMS.
    #[cfg(target_os = "macos")]
    capture_params: StreamParams,
    /// Kicked by the CoreAudio render callback when it consumes (playback) or produces
    /// (capture) frames, so this thread reaps tx completions and/or drains the capture ring
    /// into rx buffers.
    #[cfg(target_os = "macos")]
    completion_evt: Arc<EventFd>,
    /// The host CoreAudio output sink (created on PCM_PREPARE for stream 0).
    #[cfg(target_os = "macos")]
    audio: Option<super::audio_macos::OutputStream>,
    /// The host CoreAudio input source (created on PCM_PREPARE for stream 1).
    #[cfg(target_os = "macos")]
    capture: Option<super::audio_macos::InputStream>,
    /// Popped tx descriptors awaiting completion, each tagged with the cumulative
    /// frame count at which it becomes complete (paced by real playback).
    #[cfg(target_os = "macos")]
    in_flight: std::collections::VecDeque<(u16, u64)>,
    /// Total frames handed to the sink since the last PREPARE.
    #[cfg(target_os = "macos")]
    submitted: u64,
    /// Frames the host DAC still owes after our render callback returns, reported to the
    /// guest as `latency_bytes` so its `runtime->delay` covers the whole path. Re-read on
    /// a slow cadence, because the user can switch to a Bluetooth device mid-stream and
    /// change it by an order of magnitude.
    #[cfg(target_os = "macos")]
    host_latency_frames: u32,
    /// When `host_latency_frames` was last read from CoreAudio.
    #[cfg(target_os = "macos")]
    host_latency_read_at: Option<std::time::Instant>,
    /// Per-second tx trace, present only when `LIMINA_SND_TRACE=1`.
    #[cfg(target_os = "macos")]
    tx_trace: Option<TxTrace>,
    /// Audibility edge reporting, present only when an embedder asked for it.
    #[cfg(target_os = "macos")]
    audibility: Option<Audibility>,
    /// Optional embedder hook, told about every accepted PCM lifecycle request.
    pcm_state_cb: Option<PcmStateFn>,
}

/// What `Snd` hands a new worker on activation, beyond the queues themselves.
pub(crate) struct SndWorkerConfig {
    pub capture_enabled: bool,
    pub stop_fd: EventFd,
    pub pcm_state_cb: Option<PcmStateFn>,
    #[cfg(target_os = "macos")]
    pub completion_evt: Arc<EventFd>,
    #[cfg(target_os = "macos")]
    pub audibility: Option<(std::time::Duration, super::PcmAudibilityFn)>,
}

impl SndWorker {
    pub(crate) fn new(
        queues: Vec<DeviceQueue>,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        config: SndWorkerConfig,
    ) -> Self {
        SndWorker {
            queues,
            mem,
            interrupt,
            stop_fd: config.stop_fd,
            params: StreamParams::default(),
            capture_enabled: config.capture_enabled,
            #[cfg(target_os = "macos")]
            capture_params: StreamParams::default(),
            #[cfg(target_os = "macos")]
            completion_evt: config.completion_evt,
            #[cfg(target_os = "macos")]
            audio: None,
            #[cfg(target_os = "macos")]
            capture: None,
            #[cfg(target_os = "macos")]
            in_flight: std::collections::VecDeque::new(),
            #[cfg(target_os = "macos")]
            submitted: 0,
            #[cfg(target_os = "macos")]
            host_latency_frames: 0,
            #[cfg(target_os = "macos")]
            host_latency_read_at: None,
            #[cfg(target_os = "macos")]
            tx_trace: TxTrace::new_if_enabled(),
            #[cfg(target_os = "macos")]
            audibility: config
                .audibility
                .map(|(silence, cb)| Audibility::new(silence, cb)),
            pcm_state_cb: config.pcm_state_cb,
        }
    }

    pub(crate) fn run(self) -> JoinHandle<()> {
        thread::Builder::new()
            .name("snd worker".into())
            .spawn(|| self.work())
            .expect("spawn the snd worker thread")
    }

    fn work(mut self) {
        const CONTROL: u64 = 0;
        const TX: u64 = 1;
        #[cfg(target_os = "macos")]
        const RX: u64 = 2;
        #[cfg(target_os = "macos")]
        const COMPLETION: u64 = 3;
        const STOP: u64 = 4;

        debug!("snd worker: starting");
        let mut epoll = Epoll::new().expect("create the snd worker's epoll");
        // Playback (control+tx) always; capture (rx) only when the mic is enabled.
        #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
        let mut interest = vec![
            (self.queues[defs::CONTROL_INDEX].event.as_raw_fd(), CONTROL),
            (self.queues[defs::TX_INDEX].event.as_raw_fd(), TX),
            (self.stop_fd.as_raw_fd(), STOP),
        ];
        #[cfg(target_os = "macos")]
        {
            if self.capture_enabled {
                interest.push((self.queues[defs::RX_INDEX].event.as_raw_fd(), RX));
            }
            interest.push((self.completion_evt.as_raw_fd(), COMPLETION));
        }
        for (fd, token) in interest {
            epoll
                .ctl(
                    ControlOperation::Add,
                    fd,
                    &EpollEvent::new(EventSet::IN, token),
                )
                .expect("add an fd to the snd worker's epoll");
        }

        let mut events = vec![EpollEvent::default(); 8];
        loop {
            let n = match epoll.wait(events.len(), -1, &mut events) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    error!("snd worker: epoll wait failed: {e:?}");
                    break;
                }
            };
            for event in &events[..n] {
                match event.data() {
                    CONTROL => {
                        let _ = self.queues[defs::CONTROL_INDEX].event.read();
                        if self.process_control() {
                            self.interrupt.signal_used_queue();
                        }
                    }
                    TX => {
                        let _ = self.queues[defs::TX_INDEX].event.read();
                        if self.process_tx() {
                            self.interrupt.signal_used_queue();
                        }
                    }
                    #[cfg(target_os = "macos")]
                    RX => {
                        let _ = self.queues[defs::RX_INDEX].event.read();
                        self.process_rx();
                    }
                    // The shared completion fd is kicked by both the playback (consumed
                    // frames) and capture (produced frames) callbacks.
                    #[cfg(target_os = "macos")]
                    COMPLETION => {
                        self.reap_completions();
                        if self.capture_enabled {
                            self.process_rx();
                        }
                    }
                    STOP => {
                        let _ = self.stop_fd.read();
                        debug!("snd worker: stopping");
                        // Returning drops `self`, and with it the CoreAudio units, here on
                        // the thread that created them.
                        return;
                    }
                    other => warn!("snd worker: unexpected epoll token {other}"),
                }
            }
        }
    }

    /// Service the control queue: parse each request, write its response, mark used.
    /// Returns true if any descriptor was consumed (caller signals the guest).
    pub fn process_control(&mut self) -> bool {
        let mem = self.mem.clone();
        let mut have_used = false;

        loop {
            // Pop under a short borrow so the handler can take &mut self freely.
            let head = match self.queues[defs::CONTROL_INDEX].queue.pop(&mem) {
                Some(h) => h,
                None => break,
            };
            let index = head.index;
            let written = self.handle_control_req(&mem, &head);
            if let Err(e) = self.queues[defs::CONTROL_INDEX]
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
                // Only announce transitions we accepted: a request for a stream that does not
                // exist moved nothing, and reporting it would invent state.
                if status == VIRTIO_SND_S_OK {
                    self.report_pcm_state(h.stream_id, code);
                }
                write_status(&mut writer, status);
            }
            other => {
                warn!("snd: unsupported control request code {other:#x}");
                write_status(&mut writer, VIRTIO_SND_S_NOT_SUPP);
            }
        }

        writer.bytes_written() as u32
    }

    /// Tell the embedder's hook, if any, that a stream moved. Host-independent by design:
    /// the guest's stream lifetime is a virtio fact, not a CoreAudio one.
    fn report_pcm_state(&self, stream_id: u32, code: u32) {
        let Some(cb) = self.pcm_state_cb.as_ref() else {
            return;
        };
        let event = match code {
            VIRTIO_SND_R_PCM_PREPARE => super::PcmEvent::Prepare,
            VIRTIO_SND_R_PCM_START => super::PcmEvent::Start,
            VIRTIO_SND_R_PCM_STOP => super::PcmEvent::Stop,
            VIRTIO_SND_R_PCM_RELEASE => super::PcmEvent::Release,
            _ => return,
        };
        cb(super::PcmStreamState { stream_id, event });
    }

    /// Drive the host sink through the PCM lifecycle (macOS). On other hosts this is a
    /// no-op and tx falls back to the immediate null sink.
    #[cfg(target_os = "macos")]
    fn handle_pcm_lifecycle(&mut self, code: u32) {
        // Whatever the guest just did, the stream it left behind is a different one: a
        // re-prepare discards unplayed audio and a start begins from nothing.
        if let Some(a) = self.audibility.as_mut() {
            a.reset();
        }
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
        let mem = self.mem.clone();

        #[cfg(target_os = "macos")]
        {
            let bpf = self.params.bytes_per_frame();
            self.refresh_host_latency();
            loop {
                let head = match self.queues[defs::TX_INDEX].queue.pop(&mem) {
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
            if let Some(t) = self.tx_trace.as_mut() {
                t.flush();
            }
            // Completions are signalled from reap_completions, not here.
            false
        }

        #[cfg(not(target_os = "macos"))]
        {
            let mut have_used = false;
            loop {
                let head = match self.queues[defs::TX_INDEX].queue.pop(&mem) {
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
                if let Err(e) = self.queues[defs::TX_INDEX]
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
                    if self.tx_trace.is_some() || self.audibility.is_some() {
                        let peak = data
                            .chunks_exact(2)
                            .map(|b| i16::from_le_bytes([b[0], b[1]]).unsigned_abs())
                            .max()
                            .unwrap_or(0);
                        if let Some(t) = self.tx_trace.as_mut() {
                            t.buffer(frames, f32::from(peak) / 32768.0);
                        }
                        if let Some(a) = self.audibility.as_mut() {
                            a.buffer(OUTPUT_STREAM_ID, frames as u64, peak == 0);
                        }
                    }
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
        // The device-writable tail is a single status word. `latency_bytes` is what the
        // host still owes downstream of us; the guest driver feeds it straight into
        // `runtime->delay`, which is how an application learns when its audio is audible.
        if let Ok(mut writer) = Writer::new(mem, head.clone()) {
            let _ = writer.write_obj(VirtioSndPcmStatus {
                status: VIRTIO_SND_S_OK,
                latency_bytes: self.host_latency_frames.saturating_mul(bpf as u32),
            });
        }
        frames
    }

    /// Re-read the host's output latency, at most once a second.
    ///
    /// Frames already in our ring or the virtqueue are deliberately *not* counted: a tx
    /// descriptor is completed only once the render callback has consumed its frames, so
    /// the guest's own `appl_ptr - hw_ptr` covers everything up to that point. Counting
    /// them here as well would double the reported delay.
    #[cfg(target_os = "macos")]
    fn refresh_host_latency(&mut self) {
        const REFRESH: std::time::Duration = std::time::Duration::from_secs(1);
        if self.audio.is_none() {
            self.host_latency_frames = 0;
            return;
        }
        let now = std::time::Instant::now();
        if self
            .host_latency_read_at
            .is_some_and(|t| now.duration_since(t) < REFRESH)
        {
            return;
        }
        self.host_latency_read_at = Some(now);
        // Diagnostic escape hatch: report 0 as the device did before this was filled in, so the
        // effect on an application's own latency figure can be A/B'd inside one boot.
        let frames = if std::env::var("LIMINA_SND_ZERO_LATENCY").as_deref() == Ok("1") {
            0
        } else {
            super::audio_macos::default_output_latency_frames()
        };
        if frames != self.host_latency_frames {
            info!(
                "snd: host output latency {frames} frames ({:.1} ms at 48 kHz)",
                f64::from(frames) * 1000.0 / 48_000.0
            );
            self.host_latency_frames = frames;
        }
    }

    /// Complete every tx descriptor whose frames the sink has now played, advancing the
    /// guest's `hw_ptr` at the host DAC's real rate. Called on the completion eventfd.
    #[cfg(target_os = "macos")]
    pub fn reap_completions(&mut self) -> bool {
        // Drain the eventfd counter (level is re-kicked by the callback as it consumes).
        let _ = self.completion_evt.read();
        if let Some(t) = self.tx_trace.as_mut() {
            t.flush();
        }
        let consumed = match self.audio.as_ref() {
            Some(a) => a.frames_consumed(),
            None => return false,
        };
        self.complete_up_to(consumed)
    }

    /// Complete all in-flight tx descriptors with end-frame <= `consumed`.
    #[cfg(target_os = "macos")]
    fn complete_up_to(&mut self, consumed: u64) -> bool {
        let mem = self.mem.clone();
        let mut used_any = false;
        while let Some(&(index, end)) = self.in_flight.front() {
            if end > consumed {
                break;
            }
            self.in_flight.pop_front();
            if let Err(e) = self.queues[defs::TX_INDEX].queue.add_used(
                &mem,
                index,
                std::mem::size_of::<VirtioSndPcmStatus>() as u32,
            ) {
                error!("snd: failed to add used tx descriptor: {e:?}");
            }
            used_any = true;
        }
        if used_any {
            self.interrupt.signal_used_queue();
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

        let mem = self.mem.clone();
        let bpf = self.capture_params.bytes_per_frame();
        let status_sz = std::mem::size_of::<VirtioSndPcmStatus>();
        let mut used_any = false;

        loop {
            // No source yet (not prepared/started): leave rx buffers posted.
            let avail_samples = match self.capture.as_ref() {
                Some(c) => c.available(),
                None => break,
            };
            let head = match self.queues[defs::RX_INDEX].queue.pop(&mem) {
                Some(h) => h,
                None => break,
            };
            let index = head.index;
            let mut writer = match Writer::new(&mem, head.clone()) {
                Ok(w) => w,
                Err(e) => {
                    error!("snd: capture writer error: {e:?}");
                    if let Err(e) = self.queues[defs::RX_INDEX].queue.add_used(&mem, index, 0) {
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
                self.queues[defs::RX_INDEX].queue.go_to_previous_position();
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
            if let Err(e) = self.queues[defs::RX_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used rx descriptor: {e:?}");
            }
            used_any = true;
        }

        if used_any {
            self.interrupt.signal_used_queue();
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

        let mem = self.mem.clone();
        let status_sz = std::mem::size_of::<VirtioSndPcmStatus>();
        let mut used_any = false;

        loop {
            let head = match self.queues[defs::RX_INDEX].queue.pop(&mem) {
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
            if let Err(e) = self.queues[defs::RX_INDEX]
                .queue
                .add_used(&mem, index, written)
            {
                error!("snd: failed to add used rx descriptor on flush: {e:?}");
            }
            used_any = true;
        }

        if used_any {
            self.interrupt.signal_used_queue();
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

/// Whether the playback stream is carrying sound, and the edges between.
///
/// A pause is a run of bit-exact zero frames; sound returning is a single frame that is not.
/// The run is counted in frames rather than buffers because the guest picks the period size and
/// that is not a unit the host should inherit — the same 500 ms means the same thing whatever
/// the guest asked for.
#[cfg(target_os = "macos")]
struct Audibility {
    cb: super::PcmAudibilityFn,
    /// How many consecutive zero-filled frames make a pause, from the embedder's threshold.
    silent_after_frames: u64,
    /// The zero run in progress.
    zero_frames: u64,
    /// What we last told the embedder. `None` after a lifecycle transition: the stream is a new
    /// thing then, and the next edge has to be reported even if it repeats the last one.
    reported: Option<super::PcmAudibility>,
}

#[cfg(target_os = "macos")]
impl Audibility {
    /// The conversion to frames uses 48 kHz, the only rate this device advertises
    /// (`output_pcm_info`); if that ever grows, this has to follow the negotiated rate.
    fn new(silence: std::time::Duration, cb: super::PcmAudibilityFn) -> Self {
        Audibility {
            cb,
            silent_after_frames: (silence.as_secs_f64() * 48_000.0) as u64,
            zero_frames: 0,
            reported: None,
        }
    }

    /// Account one tx buffer and report an edge if it crossed one.
    fn buffer(&mut self, stream_id: u32, frames: u64, all_zero: bool) {
        let edge = if all_zero {
            self.zero_frames += frames;
            if self.zero_frames < self.silent_after_frames {
                return;
            }
            super::PcmAudibility::Silent
        } else {
            self.zero_frames = 0;
            super::PcmAudibility::Audible
        };
        if self.reported == Some(edge) {
            return;
        }
        self.reported = Some(edge);
        (self.cb)(stream_id, edge);
    }

    /// The guest moved the stream. Forget the run and the reported state: whatever comes next
    /// is news, including a repeat of what we last said.
    fn reset(&mut self) {
        self.zero_frames = 0;
        self.reported = None;
    }
}

/// A per-second trace of the tx queue, gated on `LIMINA_SND_TRACE=1`.
///
/// It exists to answer one question that no other vantage point can: when the guest pauses
/// playback but keeps its PCM stream open, does it stop submitting buffers, or keep submitting
/// silent ones? The host's "is the guest playing" detector has to be built differently for each,
/// and the guest's PCM lifecycle only reports the pause ~5s later, once PipeWire suspends the node.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub(crate) struct TxTrace {
    window_start: std::time::Instant,
    last_buffer: Option<std::time::Instant>,
    buffers: u64,
    frames: u64,
    peak: f32,
    max_gap_us: u128,
    /// Buffers in the window that were entirely zero samples.
    zeros: u64,
    /// The run of consecutive all-zero buffers in progress, carried across windows.
    zero_run: u64,
    /// The longest such run seen in this window.
    zero_run_max: u64,
}

#[cfg(target_os = "macos")]
impl TxTrace {
    fn new_if_enabled() -> Option<Self> {
        if std::env::var("LIMINA_SND_TRACE").as_deref() != Ok("1") {
            return None;
        }
        Some(TxTrace {
            window_start: std::time::Instant::now(),
            last_buffer: None,
            buffers: 0,
            frames: 0,
            peak: 0.0,
            max_gap_us: 0,
            zeros: 0,
            zero_run: 0,
            zero_run_max: 0,
        })
    }

    /// Account one tx buffer: its frame count and the largest absolute sample in it.
    fn buffer(&mut self, frames: usize, peak: f32) {
        let now = std::time::Instant::now();
        if let Some(prev) = self.last_buffer {
            self.max_gap_us = self.max_gap_us.max(now.duration_since(prev).as_micros());
        }
        self.last_buffer = Some(now);
        self.buffers += 1;
        self.frames += frames as u64;
        self.peak = self.peak.max(peak);
        if peak == 0.0 {
            self.zeros += 1;
            self.zero_run += 1;
            self.zero_run_max = self.zero_run_max.max(self.zero_run);
        } else {
            self.zero_run = 0;
        }
    }

    /// Emit a line once a second. Called from the tx path and from the completion reaper, so
    /// the line keeps coming while the sink is alive even when the guest submits nothing —
    /// which is exactly the state under investigation.
    fn flush(&mut self) {
        let now = std::time::Instant::now();
        if now.duration_since(self.window_start) < std::time::Duration::from_secs(1) {
            return;
        }
        let idle_ms = match self.last_buffer {
            Some(t) => now.duration_since(t).as_millis() as i64,
            None => -1,
        };
        info!(
            "snd trace: buffers={} frames={} peak={:.5} zeros={} zero_run_max={} max_gap_ms={:.1} idle_ms={}",
            self.buffers,
            self.frames,
            self.peak,
            self.zeros,
            self.zero_run_max,
            self.max_gap_us as f64 / 1000.0,
            idle_ms
        );
        self.window_start = now;
        self.buffers = 0;
        self.frames = 0;
        self.peak = 0.0;
        self.max_gap_us = 0;
        self.zeros = 0;
        self.zero_run_max = self.zero_run;
    }
}
