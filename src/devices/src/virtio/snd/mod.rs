// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Native virtio-sound device (device ID 25, virtio spec 5.14).
//!
//! Implemented in-VMM so it works on macOS (the vhost-user path libkrun would
//! otherwise use for sound is Linux-host-only). The guest's stock `virtio_snd`
//! driver binds it and exposes an ALSA card; PipeWire/PulseAudio route to it with
//! no guest-side limina components — so a stock Fedora guest gets sound too.
//!
//! Phase A (this commit) enumerates a single stereo playback stream and answers
//! the full control-queue handshake, discarding tx frames into a null sink. Phase
//! B replaces the null sink with a CoreAudio output unit driven from a dedicated
//! worker thread, with tx-completion pacing.

mod device;
mod event_handler;
mod protocol;

pub use self::defs::uapi::VIRTIO_ID_SOUND as TYPE_SND;
pub use self::device::Snd;

mod defs {
    use crate::virtio::QueueConfig;

    pub const SND_DEV_ID: &str = "virtio_snd";

    // Queue layout (virtio_snd.h): control(0), event(1), tx/playback(2), rx/capture(3).
    pub const CONTROL_INDEX: usize = 0;
    pub const _EVENT_INDEX: usize = 1;
    pub const TX_INDEX: usize = 2;
    pub const _RX_INDEX: usize = 3;
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
