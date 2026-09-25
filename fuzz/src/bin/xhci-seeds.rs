// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Seed the xHCI fuzz target with what a guest driver actually does.
//!
//!     cargo run --release --bin xhci-seeds -- corpus/xhci_guest
//!
//! (with `RUSTFLAGS="--cfg fuzzing"`, like `xhci-depth`.) Random bytes never get past controller
//! reset: a session only reaches the rings once it has pointed DCBAAP, CRCR, ERSTBA and ERDP at
//! structures it has laid out in RAM, and only reaches a device once it has enabled a slot and
//! addressed it through a well-formed input context. This writes that bring-up, following the
//! controller's own unit tests and register map, as every prefix of one session, so the fuzzer
//! can start mutating from each stage: reset, run, enable, address, fetch a descriptor on the
//! mock, configure the HID gadget's interrupt IN endpoint and queue a transfer on it, then
//! snapshot, stop and reposition the endpoint, and tear the slots down.

use krun_fuzz::{Op, drive, encode};

// Register offsets from the MMIO base (see `usb/xhci/device.rs`).
const USBCMD: u16 = 0x20;
const CRCR: u16 = 0x20 + 0x18;
const DCBAAP: u16 = 0x20 + 0x30;
const CONFIG: u16 = 0x20 + 0x38;
const IMAN: u16 = 0x1020;
const ERSTSZ: u16 = 0x1028;
const ERSTBA: u16 = 0x1030;
const ERDP: u16 = 0x1038;
const DOORBELL: u16 = 0x2000;

const CMD_RS: u64 = 1;
const CMD_HCRST: u64 = 1 << 1;
const CMD_INTE: u64 = 1 << 2;

// TRB types and control bits (see `usb/xhci/trb.rs`).
const NORMAL: u32 = 1;
const SETUP_STAGE: u32 = 2;
const DATA_STAGE: u32 = 3;
const STATUS_STAGE: u32 = 4;
const ENABLE_SLOT: u32 = 9;
const DISABLE_SLOT: u32 = 10;
const ADDRESS_DEVICE: u32 = 11;
const CONFIGURE_ENDPOINT: u32 = 12;
const STOP_ENDPOINT: u32 = 15;
const SET_TR_DEQUEUE: u32 = 16;
const CYCLE: u32 = 1;
const IOC: u32 = 1 << 5;
const IDT: u32 = 1 << 6;
const DIR_IN: u32 = 1 << 16;

// Where the session lays things out in RAM.
const DCBAA: u16 = 0x1000;
const OUT_CTX: [u16; 2] = [0x2000, 0x2400];
const CMD_RING: u16 = 0x3000;
const ERST: u16 = 0x4000;
const EVENT_SEG: u16 = 0x5000;
const INPUT_CTX: [u16; 2] = [0x6000, 0x6400];
const CONFIG_CTX: u16 = 0x6800;
const EP0_RING: [u16; 2] = [0x7000, 0x7800];
const DATA_BUF: u16 = 0x8000;
const INT_RING: u16 = 0x9000;

fn trb(parameter: u64, status: u32, control: u32) -> [u8; 16] {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&parameter.to_le_bytes());
    b[8..12].copy_from_slice(&status.to_le_bytes());
    b[12..].copy_from_slice(&control.to_le_bytes());
    b
}

fn cmd(ty: u32, slot: u8, extra: u32) -> u32 {
    (ty << 10) | CYCLE | (u32::from(slot) << 24) | extra
}

/// A 32-byte context from its eight dwords, as two plants.
fn context(at: u16, dwords: [u32; 8]) -> [Op; 2] {
    let mut b = [0u8; 32];
    for (i, d) in dwords.iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&d.to_le_bytes());
    }
    [
        Op::Plant {
            at,
            bytes: b[..16].try_into().unwrap(),
        },
        Op::Plant {
            at: at + 16,
            bytes: b[16..].try_into().unwrap(),
        },
    ]
}

/// An endpoint context: type, max packet size, error count 3, and its ring (dequeue cycle 1).
fn endpoint(ty: u32, mps: u32, ring: u16) -> [u32; 8] {
    [
        0,
        (3 << 1) | (ty << 3) | (mps << 16),
        u32::from(ring) | 1,
        0,
        0,
        0,
        0,
        0,
    ]
}

fn write(off: u16, val: u64) -> Op {
    Op::Write { off, width: 2, val }
}

fn session() -> Vec<Op> {
    let mut s = vec![
        // Reset, then the structures the controller reads from RAM.
        write(USBCMD, CMD_HCRST),
        write(CONFIG, 8),
        Op::Point {
            off: DCBAAP,
            to: DCBAA,
            low: 0,
        },
        Op::Point {
            off: CRCR,
            to: CMD_RING,
            low: 1,
        },
        Op::Plant {
            at: ERST,
            bytes: trb(u64::from(EVENT_SEG), 16, 0),
        },
        Op::Plant {
            at: DCBAA,
            bytes: [[0u8; 8], (u64::from(OUT_CTX[0])).to_le_bytes()]
                .concat()
                .try_into()
                .unwrap(),
        },
        Op::Plant {
            at: DCBAA + 16,
            bytes: [(u64::from(OUT_CTX[1])).to_le_bytes(), [0u8; 8]]
                .concat()
                .try_into()
                .unwrap(),
        },
        write(ERSTSZ, 1),
        Op::Point {
            off: ERSTBA,
            to: ERST,
            low: 0,
        },
        Op::Point {
            off: ERDP,
            to: EVENT_SEG,
            low: 0,
        },
        write(IMAN, 2),
        write(USBCMD, CMD_RS | CMD_INTE),
        Op::Pass,
        // Enable two slots and address the mock (port 1) and the HID gadget (port 2).
        Op::Plant {
            at: CMD_RING,
            bytes: trb(0, 0, cmd(ENABLE_SLOT, 0, 0)),
        },
        Op::Plant {
            at: CMD_RING + 0x10,
            bytes: trb(0, 0, cmd(ENABLE_SLOT, 0, 0)),
        },
        write(DOORBELL, 0),
        Op::Pass,
    ];
    for i in 0..2 {
        s.extend(context(INPUT_CTX[i], [0, 0b11, 0, 0, 0, 0, 0, 0]));
        s.extend(context(
            INPUT_CTX[i] + 32,
            [1 << 27, (i as u32 + 1) << 16, 0, 0, 0, 0, 0, 0],
        ));
        s.extend(context(INPUT_CTX[i] + 64, endpoint(4, 64, EP0_RING[i])));
    }
    s.extend([
        Op::Plant {
            at: CMD_RING + 0x20,
            bytes: trb(u64::from(INPUT_CTX[0]), 0, cmd(ADDRESS_DEVICE, 1, 0)),
        },
        Op::Plant {
            at: CMD_RING + 0x30,
            bytes: trb(u64::from(INPUT_CTX[1]), 0, cmd(ADDRESS_DEVICE, 2, 0)),
        },
        write(DOORBELL, 0),
        Op::Pass,
        // GET_DESCRIPTOR(DEVICE) on the mock's EP0.
        Op::Plant {
            at: EP0_RING[0],
            bytes: trb(
                u64::from_le_bytes([0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 18, 0]),
                8,
                (SETUP_STAGE << 10) | CYCLE | IDT | (3 << 16),
            ),
        },
        Op::Plant {
            at: EP0_RING[0] + 0x10,
            bytes: trb(u64::from(DATA_BUF), 18, (DATA_STAGE << 10) | CYCLE | DIR_IN),
        },
        Op::Plant {
            at: EP0_RING[0] + 0x20,
            bytes: trb(0, 0, (STATUS_STAGE << 10) | CYCLE | IOC),
        },
        write(DOORBELL + 4, 1),
        Op::Pass,
    ]);
    // Configure the HID gadget's interrupt IN endpoint (EP1 IN, DCI 3) and queue a transfer.
    s.extend(context(CONFIG_CTX, [0, 1 | (1 << 3), 0, 0, 0, 0, 0, 0]));
    s.extend(context(
        CONFIG_CTX + 32,
        [3 << 27, 2 << 16, 0, 0, 0, 0, 0, 0],
    ));
    s.extend(context(CONFIG_CTX + 32 * 4, endpoint(7, 64, INT_RING)));
    s.extend([
        Op::Plant {
            at: CMD_RING + 0x40,
            bytes: trb(u64::from(CONFIG_CTX), 0, cmd(CONFIGURE_ENDPOINT, 2, 0)),
        },
        write(DOORBELL, 0),
        Op::Pass,
        Op::Plant {
            at: INT_RING,
            bytes: trb(
                u64::from(DATA_BUF + 0x100),
                64,
                (NORMAL << 10) | CYCLE | IOC,
            ),
        },
        write(DOORBELL + 8, 3),
        Op::Pass,
        Op::Snapshot,
        // Stop the endpoint, reposition it, and tear both slots down.
        Op::Plant {
            at: CMD_RING + 0x50,
            bytes: trb(0, 0, cmd(STOP_ENDPOINT, 2, 3 << 16)),
        },
        Op::Plant {
            at: CMD_RING + 0x60,
            bytes: trb(u64::from(INT_RING) | 1, 0, cmd(SET_TR_DEQUEUE, 2, 3 << 16)),
        },
        write(DOORBELL, 0),
        Op::Pass,
        Op::Snapshot,
        Op::Plant {
            at: CMD_RING + 0x70,
            bytes: trb(0, 0, cmd(DISABLE_SLOT, 1, 0)),
        },
        Op::Plant {
            at: CMD_RING + 0x80,
            bytes: trb(0, 0, cmd(DISABLE_SLOT, 2, 0)),
        },
        write(DOORBELL, 0),
        Op::Pass,
    ]);
    s
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpus/xhci_guest".into());
    std::fs::create_dir_all(&dir).expect("create the corpus directory");
    let s = session();
    // The whole session must run clean before it seeds anything.
    drive(&s);
    let mut n = 0;
    for end in 1..=s.len() {
        if matches!(s[end - 1], Op::Pass | Op::Snapshot) {
            let path = format!("{dir}/seed-{end:03}");
            std::fs::write(&path, encode(&s[..end])).expect("write a seed");
            n += 1;
        }
    }
    println!("{n} seeds from a {}-operation session into {dir}", s.len());
}
