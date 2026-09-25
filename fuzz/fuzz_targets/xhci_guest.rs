// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The emulated xHCI controller driven the only ways a guest can drive it: register reads and
//! writes through its MMIO window, and whatever it leaves in its own memory for the rings to walk.
//!
//! The controller runs with the two hardware-free gadgets plugged in, and every worker pass runs
//! synchronously through `run_pass`, the same function the worker thread calls. Beyond not
//! panicking and not hanging (libFuzzer's `-timeout` stands for the second), a snapshot taken at
//! any point must restore onto a controller with the same gadgets and save back to the same state,
//! and the harness then carries on driving the restored controller.

#![no_main]

use krun_fuzz::{drive, parse};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    drive(&parse(data));
});
