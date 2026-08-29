// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Native virtio-sound device (device ID 25, virtio spec 5.14).
//!
//! Implemented in-VMM so it works on macOS (the vhost-user path libkrun would
//! otherwise use for sound is Linux-host-only). The guest's stock `virtio_snd`
//! driver binds it and exposes an ALSA card; PipeWire/PulseAudio route to it with
//! no guest-side limina components — so a stock Fedora guest gets sound too.
//!
//! Playback (stream 0) is always advertised: guest tx frames feed a CoreAudio
//! output unit with tx-completion pacing (S16→f32, paced by the host DAC). Mic
//! capture (stream 1, `snd_capture`) is opt-in and default-off for privacy: when
//! enabled it advertises a mono input stream whose rx buffers are filled from a
//! CoreAudio input unit — creating that unit on PCM_PREPARE is what triggers the
//! macOS mic TCC prompt. Capture is macOS-only; non-macOS keeps the null tx sink.

#[cfg(target_os = "macos")]
mod audio_macos;
mod device;
mod event_handler;
mod protocol;

pub use self::defs::uapi::VIRTIO_ID_SOUND as TYPE_SND;
pub use self::device::Snd;

/// Which PCM lifecycle transition the guest just asked for.
///
/// These are the four `VIRTIO_SND_R_PCM_*` control requests that move a stream, reported
/// verbatim: the device says what the guest did and takes no view on what it means. An
/// embedder that wants "is the VM playing" builds that from these, including any hold-off it
/// needs — a driver reopening a stream between tracks is indistinguishable from one closing
/// it for good at this layer, and the device is the wrong place to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmEvent {
    Prepare,
    Start,
    Stop,
    Release,
}

/// A PCM stream state change, as reported to an embedder-supplied callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmStreamState {
    /// The guest's stream id, unfiltered. Stream 0 is playback and stream 1 is capture in
    /// this device, but the callback reports whichever the guest addressed.
    pub stream_id: u32,
    pub event: PcmEvent,
}

/// Callback invoked on every accepted PCM lifecycle request. Runs on the device thread, so
/// it must not block: hand the state somewhere else and return.
pub type PcmStateFn = std::sync::Arc<dyn Fn(PcmStreamState) + Send + Sync>;

mod defs {
    use crate::virtio::QueueConfig;

    pub const SND_DEV_ID: &str = "virtio_snd";

    // Queue layout (virtio_snd.h): control(0), event(1), tx/playback(2), rx/capture(3).
    pub const CONTROL_INDEX: usize = 0;
    pub const _EVENT_INDEX: usize = 1;
    pub const TX_INDEX: usize = 2;
    pub const RX_INDEX: usize = 3;
    pub const NUM_QUEUES: usize = 4;

    const QUEUE_SIZE: u16 = 64;
    pub static QUEUE_CONFIG: [QueueConfig; NUM_QUEUES] = [QueueConfig::new(QUEUE_SIZE); NUM_QUEUES];

    pub mod uapi {
        pub const VIRTIO_F_VERSION_1: u32 = 32;
        pub const VIRTIO_ID_SOUND: u32 = 25;
    }
}

#[derive(Debug)]
pub enum SndError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
}

type Result<T> = std::result::Result<T, SndError>;
