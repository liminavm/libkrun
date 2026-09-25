// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A whole snapshot file, head and RAM frames, applied into guest memory as a resume applies it:
//! the frame walk, the zero, raw and lz4 frames, and the writes into RAM. A corrupt file must be
//! refused, never panic, hang or allocate without bound, whatever its region and chunk sizes say.
//!
//! Built with `--cfg fuzzing` neither the head CRC nor a frame's CRC is enforced, so the fuzzer
//! reaches the decompressor and the writes. The guest memory is one region at the GPA the seeds
//! are written for (`snapshot-seeds`).

#![no_main]

use krun_fuzz::snapshot::{RAM_GPA, RAM_LEN};
use libfuzzer_sys::fuzz_target;
use vm_memory::{GuestAddress, GuestMemoryMmap};
use vmm::snapshot::read_bytes;

fuzz_target!(|data: &[u8]| {
    let Ok(file) = read_bytes(data.to_vec()) else {
        return;
    };
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(RAM_GPA), RAM_LEN)])
        .expect("guest memory for the frames");
    let _ = file.apply_ram(&mem);
});
