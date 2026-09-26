// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The GPU section of a snapshot file, read as a restore reads it. It sits behind the head CRC,
//! but a writer bug or a version skew reaches it all the same, and its counts drive allocations.
//! A payload must be read or refused, never panic or exhaust memory.
//!
//! Whatever reads must also survive a round trip: written back, read again and written again, the
//! bytes must not change.

#![no_main]

use devices::virtio::gpu_snapshot::GpuSnapshotPayload;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some(payload) = GpuSnapshotPayload::from_bytes(data) else {
        return;
    };
    let once = payload.to_bytes();
    let again = GpuSnapshotPayload::from_bytes(&once)
        .expect("a payload as the writer writes it must read back");
    assert!(
        again.to_bytes() == once,
        "a payload read back from its own encoding encodes differently"
    );
});
