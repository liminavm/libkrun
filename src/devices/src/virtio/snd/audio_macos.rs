// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! macOS CoreAudio output sink for the virtio-snd device.
//!
//! A single AUHAL DefaultOutput AudioUnit renders guest playback. The guest's PCM
//! frames (converted S16_LE → f32) flow through a lock-free SPSC ring to the render
//! callback, which runs on CoreAudio's realtime thread and therefore takes no locks
//! and does no allocation. The callback publishes how many *real* guest frames it has
//! consumed (`frames_consumed`) and kicks a completion eventfd; the device's
//! event-manager thread reaps and completes the tx descriptors those frames belong to,
//! so the guest `hw_ptr` advances at the host DAC's real rate (no xrun/clock runaway).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use utils::eventfd::EventFd;

// ---- minimal AudioToolbox / AudioUnit FFI (validated in spikes/snd-coreaudio) ----

#[repr(C)]
struct AudioComponentDescription {
    component_type: u32,
    component_sub_type: u32,
    component_manufacturer: u32,
    component_flags: u32,
    component_flags_mask: u32,
}

#[repr(C)]
struct AudioStreamBasicDescription {
    sample_rate: f64,
    format_id: u32,
    format_flags: u32,
    bytes_per_packet: u32,
    frames_per_packet: u32,
    bytes_per_frame: u32,
    channels_per_frame: u32,
    bits_per_channel: u32,
    reserved: u32,
}

#[repr(C)]
struct AudioBuffer {
    number_channels: u32,
    data_byte_size: u32,
    data: *mut c_void,
}

#[repr(C)]
struct AudioBufferList {
    number_buffers: u32,
    buffers: [AudioBuffer; 1],
}

type AuRenderCallback = extern "C" fn(
    in_ref_con: *mut c_void,
    io_action_flags: *mut u32,
    in_time_stamp: *const c_void,
    in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> i32;

#[repr(C)]
struct AuRenderCallbackStruct {
    input_proc: AuRenderCallback,
    input_proc_ref_con: *mut c_void,
}

type AudioComponent = *mut c_void;
type AudioComponentInstance = *mut c_void;

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioComponentFindNext(
        in_component: AudioComponent,
        in_desc: *const AudioComponentDescription,
    ) -> AudioComponent;
    fn AudioComponentInstanceNew(
        in_component: AudioComponent,
        out_instance: *mut AudioComponentInstance,
    ) -> i32;
    fn AudioComponentInstanceDispose(in_instance: AudioComponentInstance) -> i32;
    fn AudioUnitSetProperty(
        in_unit: AudioComponentInstance,
        in_id: u32,
        in_scope: u32,
        in_element: u32,
        in_data: *const c_void,
        in_data_size: u32,
    ) -> i32;
    fn AudioUnitInitialize(in_unit: AudioComponentInstance) -> i32;
    fn AudioUnitUninitialize(in_unit: AudioComponentInstance) -> i32;
    fn AudioOutputUnitStart(in_unit: AudioComponentInstance) -> i32;
    fn AudioOutputUnitStop(in_unit: AudioComponentInstance) -> i32;
}

const K_AUDIO_UNIT_TYPE_OUTPUT: u32 = u32::from_be_bytes(*b"auou");
const K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT: u32 = u32::from_be_bytes(*b"def ");
const K_AUDIO_UNIT_MANUFACTURER_APPLE: u32 = u32::from_be_bytes(*b"appl");
const K_AUDIO_FORMAT_LINEAR_PCM: u32 = u32::from_be_bytes(*b"lpcm");
const K_AUDIO_FORMAT_FLAG_IS_FLOAT: u32 = 1 << 0;
const K_AUDIO_FORMAT_FLAG_IS_PACKED: u32 = 1 << 3;
const K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT: u32 = 8;
const K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK: u32 = 23;
const K_AUDIO_UNIT_SCOPE_INPUT: u32 = 1;

// ---- lock-free single-producer / single-consumer f32 ring -------------------

/// Producer = the device's event-manager thread (push in `process_tx`); consumer =
/// the CoreAudio render callback. Monotonic indices modulo capacity; the atomics
/// publish visibility of the cells the other side may read. Fixed capacity so the
/// callback's pointer to it never dangles across a re-prepare.
struct SpscRing {
    buf: Box<[std::cell::UnsafeCell<f32>]>,
    cap: usize,
    head: AtomicUsize, // consumer position (frames*channels units)
    tail: AtomicUsize, // producer position
}

// SAFETY: producer touches only [tail, tail+n); consumer only [head, head+n); the two
// ranges never overlap (bounded by computed space/avail), and the atomics order the
// publication of each side's writes.
unsafe impl Sync for SpscRing {}
unsafe impl Send for SpscRing {}

impl SpscRing {
    fn new(cap: usize) -> Self {
        let mut v = Vec::with_capacity(cap);
        v.resize_with(cap, || std::cell::UnsafeCell::new(0.0));
        SpscRing {
            buf: v.into_boxed_slice(),
            cap,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Producer: append up to `samples.len()`; returns the number actually written
    /// (fewer than requested only if the ring is full).
    fn push(&self, samples: &[f32]) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let space = self.cap - (tail - head);
        let n = space.min(samples.len());
        for (i, &s) in samples.iter().take(n).enumerate() {
            // SAFETY: (tail+i)%cap is in the producer-owned range, unread by the consumer.
            unsafe { *self.buf[(tail + i) % self.cap].get() = s };
        }
        self.tail.store(tail + n, Ordering::Release);
        n
    }

    /// Consumer: fill `out` from the ring; returns the number of samples read (fewer
    /// than `out.len()` on underrun).
    fn pull(&self, out: &mut [f32]) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Relaxed);
        let avail = tail - head;
        let n = avail.min(out.len());
        for (i, o) in out.iter_mut().take(n).enumerate() {
            // SAFETY: (head+i)%cap is in the consumer-owned range, no longer written.
            *o = unsafe { *self.buf[(head + i) % self.cap].get() };
        }
        self.head.store(head + n, Ordering::Release);
        n
    }

    fn reset(&self) {
        self.head.store(0, Ordering::Relaxed);
        self.tail.store(0, Ordering::Relaxed);
    }
}

/// State shared between the device thread and the RT render callback.
struct AudioShared {
    ring: SpscRing,
    /// Real guest frames drained by the callback (monotonic per prepare).
    frames_consumed: AtomicU64,
    /// Kicked by the callback after it consumes frames, so the device thread reaps
    /// and completes the corresponding tx descriptors.
    completion_evt: Arc<EventFd>,
    channels: usize,
}

/// One second of stereo f32 at 48 kHz. The guest only keeps `buffer_bytes` in flight,
/// so real occupancy stays far below this; the slack just avoids overflow.
const RING_FRAMES: usize = 48_000;

extern "C" fn render(
    in_ref_con: *mut c_void,
    _io_action_flags: *mut u32,
    _in_time_stamp: *const c_void,
    _in_bus_number: u32,
    in_number_frames: u32,
    io_data: *mut AudioBufferList,
) -> i32 {
    // SAFETY: refCon is the Arc<AudioShared> inner, kept alive by the OutputStream for
    // as long as the unit is running; io_data is CoreAudio's output buffer list.
    let shared = unsafe { &*(in_ref_con as *const AudioShared) };
    let list = unsafe { &mut *io_data };
    if list.number_buffers == 0 {
        return 0;
    }
    let buf = &mut list.buffers[0];
    let want = in_number_frames as usize * shared.channels;
    let out = unsafe { std::slice::from_raw_parts_mut(buf.data as *mut f32, want) };

    let got = shared.ring.pull(out);
    if got < want {
        // Underrun: pad with silence so the DAC never plays stale samples.
        for s in out.iter_mut().skip(got) {
            *s = 0.0;
        }
    }
    if got > 0 {
        shared
            .frames_consumed
            .fetch_add((got / shared.channels) as u64, Ordering::Release);
        // Wake the device thread to reap completed tx descriptors. A single
        // non-blocking pipe write; harmless if it coalesces with earlier kicks.
        let _ = shared.completion_evt.write(1);
    }
    0
}

/// A running (or paused) CoreAudio output unit plus the ring it renders from.
pub(crate) struct OutputStream {
    unit: AudioComponentInstance,
    shared: Arc<AudioShared>,
    started: bool,
}

// SAFETY: the AudioComponentInstance is only ever touched from the device thread
// (create/start/stop/dispose); the RT callback touches only `shared` (Sync).
unsafe impl Send for OutputStream {}

impl OutputStream {
    /// Create an output unit for `channels` @ `sample_rate` and wire the render
    /// callback. Not started until [`start`]. `completion_evt` is kicked from the RT
    /// thread when frames are consumed.
    pub(crate) fn new(
        sample_rate: f64,
        channels: usize,
        completion_evt: Arc<EventFd>,
    ) -> Result<OutputStream, i32> {
        let shared = Arc::new(AudioShared {
            ring: SpscRing::new(RING_FRAMES * channels),
            frames_consumed: AtomicU64::new(0),
            completion_evt,
            channels,
        });

        unsafe {
            let desc = AudioComponentDescription {
                component_type: K_AUDIO_UNIT_TYPE_OUTPUT,
                component_sub_type: K_AUDIO_UNIT_SUB_TYPE_DEFAULT_OUTPUT,
                component_manufacturer: K_AUDIO_UNIT_MANUFACTURER_APPLE,
                component_flags: 0,
                component_flags_mask: 0,
            };
            let comp = AudioComponentFindNext(std::ptr::null_mut(), &desc);
            if comp.is_null() {
                return Err(-1);
            }
            let mut unit: AudioComponentInstance = std::ptr::null_mut();
            let r = AudioComponentInstanceNew(comp, &mut unit);
            if r != 0 {
                return Err(r);
            }

            let bytes_per_frame = 4 * channels as u32;
            let asbd = AudioStreamBasicDescription {
                sample_rate,
                format_id: K_AUDIO_FORMAT_LINEAR_PCM,
                format_flags: K_AUDIO_FORMAT_FLAG_IS_FLOAT | K_AUDIO_FORMAT_FLAG_IS_PACKED,
                bytes_per_packet: bytes_per_frame,
                frames_per_packet: 1,
                bytes_per_frame,
                channels_per_frame: channels as u32,
                bits_per_channel: 32,
                reserved: 0,
            };
            let r = AudioUnitSetProperty(
                unit,
                K_AUDIO_UNIT_PROPERTY_STREAM_FORMAT,
                K_AUDIO_UNIT_SCOPE_INPUT,
                0,
                &asbd as *const _ as *const c_void,
                std::mem::size_of::<AudioStreamBasicDescription>() as u32,
            );
            if r != 0 {
                AudioComponentInstanceDispose(unit);
                return Err(r);
            }

            let cb = AuRenderCallbackStruct {
                input_proc: render,
                input_proc_ref_con: Arc::as_ptr(&shared) as *mut c_void,
            };
            let r = AudioUnitSetProperty(
                unit,
                K_AUDIO_UNIT_PROPERTY_SET_RENDER_CALLBACK,
                K_AUDIO_UNIT_SCOPE_INPUT,
                0,
                &cb as *const _ as *const c_void,
                std::mem::size_of::<AuRenderCallbackStruct>() as u32,
            );
            if r != 0 {
                AudioComponentInstanceDispose(unit);
                return Err(r);
            }

            let r = AudioUnitInitialize(unit);
            if r != 0 {
                AudioComponentInstanceDispose(unit);
                return Err(r);
            }

            Ok(OutputStream {
                unit,
                shared,
                started: false,
            })
        }
    }

    pub(crate) fn start(&mut self) {
        if !self.started {
            let r = unsafe { AudioOutputUnitStart(self.unit) };
            if r == 0 {
                self.started = true;
            } else {
                log::error!("snd: AudioOutputUnitStart failed: {r}");
            }
        }
    }

    pub(crate) fn stop(&mut self) {
        if self.started {
            unsafe { AudioOutputUnitStop(self.unit) };
            self.started = false;
        }
    }

    /// Clear the ring and the consumed-frame counter for a fresh PREPARE. Must be
    /// called while stopped (the ALSA prepare precondition), so the RT thread is idle.
    pub(crate) fn reset(&mut self) {
        self.shared.ring.reset();
        self.shared.frames_consumed.store(0, Ordering::Relaxed);
    }

    /// Producer side: append interleaved f32 frames; returns samples actually queued.
    pub(crate) fn push_samples(&self, samples: &[f32]) -> usize {
        self.shared.ring.push(samples)
    }

    /// Total real guest frames the DAC has consumed since the last reset.
    pub(crate) fn frames_consumed(&self) -> u64 {
        self.shared.frames_consumed.load(Ordering::Acquire)
    }
}

impl Drop for OutputStream {
    fn drop(&mut self) {
        self.stop();
        unsafe {
            AudioUnitUninitialize(self.unit);
            AudioComponentInstanceDispose(self.unit);
        }
    }
}
