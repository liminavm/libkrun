// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! virtio-sound wire format (virtio spec 5.14, structs from the Linux/QEMU
//! `virtio_snd.h` UAPI, © 2021 OpenSynergy, BSD-3-Clause).
//!
//! Only the subset the device needs is transcribed here. All structs are
//! `#[repr(C)]` little-endian POD (host and guest are both aarch64 LE), flattened
//! (the C nested `virtio_snd_hdr`/`virtio_snd_pcm_hdr` headers are inlined as a
//! leading `code`/`stream_id`) so `Reader::read_obj` / `Writer::write_obj` see a
//! single contiguous object with no implicit padding.

// A faithful transcription of the wire format: some codes/formats are only used by
// the Phase B CoreAudio sink and Phase C capture path, so allow the currently-unused
// constants rather than transcribe the spec piecemeal.
#![allow(dead_code)]

use vm_memory::ByteValued;

// ---- request / status codes (virtio_snd.h) ----------------------------------

// Jack control.
pub const VIRTIO_SND_R_JACK_INFO: u32 = 1;
// PCM control.
pub const VIRTIO_SND_R_PCM_INFO: u32 = 0x0100;
pub const VIRTIO_SND_R_PCM_SET_PARAMS: u32 = 0x0101;
pub const VIRTIO_SND_R_PCM_PREPARE: u32 = 0x0102;
pub const VIRTIO_SND_R_PCM_RELEASE: u32 = 0x0103;
pub const VIRTIO_SND_R_PCM_START: u32 = 0x0104;
pub const VIRTIO_SND_R_PCM_STOP: u32 = 0x0105;
// Channel-map control.
pub const VIRTIO_SND_R_CHMAP_INFO: u32 = 0x0200;

// Status codes returned in the response header.
pub const VIRTIO_SND_S_OK: u32 = 0x8000;
pub const VIRTIO_SND_S_BAD_MSG: u32 = 0x8001;
pub const VIRTIO_SND_S_NOT_SUPP: u32 = 0x8002;
pub const VIRTIO_SND_S_IO_ERR: u32 = 0x8003;

// Dataflow directions.
pub const VIRTIO_SND_D_OUTPUT: u8 = 0;
pub const VIRTIO_SND_D_INPUT: u8 = 1;

// Sample formats (bit index into virtio_snd_pcm_info::formats).
pub const VIRTIO_SND_PCM_FMT_S16: u8 = 5;
pub const VIRTIO_SND_PCM_FMT_FLOAT: u8 = 19;

// Frame rates (bit index into virtio_snd_pcm_info::rates).
pub const VIRTIO_SND_PCM_RATE_44100: u8 = 6;
pub const VIRTIO_SND_PCM_RATE_48000: u8 = 7;

// Channel-map positions.
pub const VIRTIO_SND_CHMAP_MONO: u8 = 2;
pub const VIRTIO_SND_CHMAP_FL: u8 = 3;
pub const VIRTIO_SND_CHMAP_FR: u8 = 4;
pub const VIRTIO_SND_CHMAP_MAX_SIZE: usize = 18;

// ---- configuration space ----------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndConfig {
    pub jacks: u32,
    pub streams: u32,
    pub chmaps: u32,
    pub controls: u32,
}
unsafe impl ByteValued for VirtioSndConfig {}

// ---- control-queue requests -------------------------------------------------

/// Common header — every control request begins with a 32-bit `code`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndHdr {
    pub code: u32,
}
unsafe impl ByteValued for VirtioSndHdr {}

/// `VIRTIO_SND_R_*_INFO`: query `count` items starting at `start_id`, each `size`
/// bytes wide (the guest's `sizeof` of the item struct).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndQueryInfo {
    pub code: u32,
    pub start_id: u32,
    pub count: u32,
    pub size: u32,
}
unsafe impl ByteValued for VirtioSndQueryInfo {}

/// `VIRTIO_SND_R_PCM_{PREPARE,START,STOP,RELEASE}`: header + target stream.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndPcmHdr {
    pub code: u32,
    pub stream_id: u32,
}
unsafe impl ByteValued for VirtioSndPcmHdr {}

/// `VIRTIO_SND_R_PCM_SET_PARAMS` request body.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndPcmSetParams {
    pub code: u32,
    pub stream_id: u32,
    pub buffer_bytes: u32,
    pub period_bytes: u32,
    pub features: u32,
    pub channels: u8,
    pub format: u8,
    pub rate: u8,
    pub padding: u8,
}
unsafe impl ByteValued for VirtioSndPcmSetParams {}

// ---- control-queue info responses -------------------------------------------

/// `virtio_snd_pcm_info` (32 bytes): describes one PCM stream.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndPcmInfo {
    pub hda_fn_nid: u32,
    pub features: u32,
    pub formats: u64,
    pub rates: u64,
    pub direction: u8,
    pub channels_min: u8,
    pub channels_max: u8,
    pub padding: [u8; 5],
}
unsafe impl ByteValued for VirtioSndPcmInfo {}

/// `virtio_snd_chmap_info` (24 bytes): one channel-map layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VirtioSndChmapInfo {
    pub hda_fn_nid: u32,
    pub direction: u8,
    pub channels: u8,
    pub positions: [u8; VIRTIO_SND_CHMAP_MAX_SIZE],
}
unsafe impl ByteValued for VirtioSndChmapInfo {}

// ---- PCM I/O (tx/rx) messages -----------------------------------------------

/// I/O request header on the tx/rx queues — precedes the PCM payload.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndPcmXfer {
    pub stream_id: u32,
}
unsafe impl ByteValued for VirtioSndPcmXfer {}

/// I/O completion status written back on the tx/rx queues.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VirtioSndPcmStatus {
    pub status: u32,
    pub latency_bytes: u32,
}
unsafe impl ByteValued for VirtioSndPcmStatus {}
