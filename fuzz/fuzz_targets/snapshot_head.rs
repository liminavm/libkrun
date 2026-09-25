// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A snapshot file's head, read as a resume reads it. The file is not guest input, but it is read
//! back after a crash, a partial write or a version skew, and the head's counts drive allocations
//! before its CRC is checked. A head must be read or refused, never panic or exhaust memory.
//!
//! Whatever reads must also survive a round trip: written back as the writer writes it, read
//! again, and written again, the bytes must not change. (The first write is not compared with the
//! input, which may be an older version or carry the GPU section uncompressed.)
//!
//! Built with `--cfg fuzzing` the reader does not enforce the head CRC, so the fuzzer can reach
//! past it; `snapshot-seeds` writes the seeds from a real snapshot.

#![no_main]

use libfuzzer_sys::fuzz_target;
use vmm::snapshot::{encode_head_for_fuzzing, read_bytes};

fuzz_target!(|data: &[u8]| {
    let Ok(file) = read_bytes(data.to_vec()) else {
        return;
    };
    let once = encode_head_for_fuzzing(&file.head);
    let again = read_bytes(once.clone()).expect("a head as the writer writes it must read back");
    assert!(
        encode_head_for_fuzzing(&again.head) == once,
        "a head read back from its own encoding encodes differently"
    );
});
