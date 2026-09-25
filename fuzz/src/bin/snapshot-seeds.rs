// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Seed the snapshot fuzz targets from a real snapshot.
//!
//!     cargo run --release --bin snapshot-seeds -- <file.snap> corpus
//!
//! (with `RUSTFLAGS="--cfg fuzzing"`, like `xhci-depth`.) It reads the snapshot's head only, and
//! writes it back as the writer does, without the GPU section, which runs to hundreds of MB on a
//! lived-in guest: once as it is, once with a small raw GPU section and once with a compressible
//! one (so both GPU encodings are seeded), and once with no vCPUs, devices or USB state. Then it
//! writes a whole small snapshot, the first head with RAM from `krun_fuzz::snapshot`'s guest
//! memory holding a zero, an lz4 and a raw frame, into `corpus/snapshot_ram`.

use std::path::Path;

use krun_fuzz::snapshot::{RAM_GPA, RAM_LEN};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};
use vmm::snapshot::{encode_head_for_fuzzing, read, write_streaming};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, snap, corpus] = &args[..] else {
        eprintln!("usage: snapshot-seeds <file.snap> <corpus dir>");
        std::process::exit(2);
    };
    let head_dir = Path::new(corpus).join("snapshot_head");
    let ram_dir = Path::new(corpus).join("snapshot_ram");
    std::fs::create_dir_all(&head_dir).unwrap();
    std::fs::create_dir_all(&ram_dir).unwrap();

    let mut file = read(Path::new(snap)).expect("the snapshot's head");
    let head = &mut file.head;
    head.gpu = None;
    let seed = |name: &str, bytes: Vec<u8>| {
        println!("{name}: {} bytes", bytes.len());
        std::fs::write(head_dir.join(name), bytes).unwrap();
    };
    seed("real-no-gpu", encode_head_for_fuzzing(head));
    head.gpu = Some((0..64u8).collect());
    seed("real-gpu-raw", encode_head_for_fuzzing(head));
    head.gpu = Some(b"LGPU".repeat(1024));
    seed("real-gpu-lz4", encode_head_for_fuzzing(head));
    head.gpu = None;

    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(RAM_GPA), RAM_LEN)]).unwrap();
    let pattern: Vec<u8> = (0..4u32 << 20).map(|i| (i / 4096) as u8).collect();
    mem.write_slice(&pattern, GuestAddress(RAM_GPA + (4 << 20)))
        .unwrap();
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let noise: Vec<u8> = (0..64 << 10)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();
    mem.write_slice(&noise, GuestAddress(RAM_GPA + (8 << 20)))
        .unwrap();
    let ram = ram_dir.join("real-head-three-frames");
    let stats = write_streaming(&ram, head, &mem, &[(RAM_GPA, RAM_LEN as u64)]).unwrap();
    println!("snapshot_ram: {} bytes", stats.written_bytes);

    head.vcpus.clear();
    head.devices.clear();
    head.usb = None;
    head.slots = Some(Vec::new());
    seed("empty", encode_head_for_fuzzing(head));
}
