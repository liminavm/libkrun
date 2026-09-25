// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! How far into the controller a corpus reaches, read from the state each input leaves behind.
//!
//!     RUSTFLAGS="--cfg fuzzing" CARGO_TARGET_DIR=target/depth \
//!         cargo run --release --bin xhci-depth -- corpus/xhci_guest
//!
//! `--cfg fuzzing` is what `cargo fuzz` sets, and what exposes the controller's `run_pass`.
//!
//! A fuzz target that never crashes says little until its inputs are known to get past controller
//! reset. This replays every input through the same `drive` the target uses and counts how many
//! reached each stage a guest driver walks through on its way to talking to a device.

use std::path::Path;

use krun_fuzz::{drive, parse};

/// USBCMD.RS.
const CMD_RUN: u32 = 1;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpus/xhci_guest".into());
    let stages: [(&str, fn(&devices::usb_state::XhciState) -> bool); 7] = [
        ("controller running", |s| s.usbcmd & CMD_RUN != 0),
        ("event ring built", |s| s.event_ring.is_some()),
        ("an event posted", |s| {
            s.event_ring.as_ref().is_some_and(|e| e.enqueue_idx > 0)
        }),
        ("command ring set", |s| s.cmd_ring.is_some()),
        ("a slot enabled", |s| !s.slots.is_empty()),
        ("a device addressed", |s| {
            s.slots.iter().any(|sl| sl.address != 0)
        }),
        ("an endpoint configured", |s| {
            s.slots.iter().any(|sl| !sl.eps.is_empty())
        }),
    ];
    let mut hits = [0usize; 7];
    let mut total = 0usize;
    let mut entries: Vec<_> = std::fs::read_dir(Path::new(&dir))
        .expect("read the corpus directory")
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.path());
    for e in entries {
        let bytes = std::fs::read(e.path()).expect("read a corpus entry");
        let state = drive(&parse(&bytes));
        total += 1;
        for (i, (_, reached)) in stages.iter().enumerate() {
            if reached(&state) {
                hits[i] += 1;
            }
        }
    }
    println!("{total} inputs");
    for ((name, _), n) in stages.iter().zip(hits) {
        println!("  {n:>6}  {name}");
    }
}
