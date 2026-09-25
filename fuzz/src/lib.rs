// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The guest the xHCI fuzz target plays, shared with `xhci-depth` (how far a corpus reaches) and
//! `xhci-seeds` (a corpus that starts where a real driver does).
//!
//! An input is a small byte format of its own rather than an `Arbitrary` derive, so that seeds
//! can be written: a guest session that gets past controller reset has to point several registers
//! at rings it has laid out in RAM, and random bytes almost never do. The format is a sequence of
//! operations, each a tag byte and fixed-size fields; a truncated operation ends the input.

use std::sync::{Arc, Mutex};

use devices::BusDevice;
use devices::usb::xhci::run_pass;
use devices::usb::{HidMockDevice, MockUsbDevice, UsbDeviceModel, XhciDevice};
use devices::usb_state::XhciState;
use utils::eventfd::{EFD_NONBLOCK, EventFd};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

/// Guest RAM the harness gives the controller, at guest physical 0.
pub const RAM: u64 = 0x1_0000;
/// The registers live in the first 16 KiB of the 64 KiB window: capability, operational, runtime,
/// doorbell and extended-capability blocks.
pub const REGS: u64 = 0x4000;
/// USBSTS.CNR, which a restore clears on purpose (see `XhciDevice::restore_state`).
const STS_CNR: u32 = 1 << 11;
/// Operations per input: enough for a full enumeration and then some.
const MAX_OPS: usize = 512;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// A register write of 1, 2, 4 or 8 bytes.
    Write { off: u16, width: u8, val: u64 },
    /// A register read of 1, 2, 4 or 8 bytes.
    Read { off: u16, width: u8 },
    /// An 8-byte register write of a pointer into guest RAM, with the low four bits (cycle and
    /// flags) chosen separately: random 64-bit values would almost never point anywhere a ring
    /// can live.
    Point { off: u16, to: u16, low: u8 },
    /// Sixteen bytes into guest RAM: one TRB, or half a context.
    Plant { at: u16, bytes: [u8; 16] },
    /// One worker pass, as a doorbell or run edge would schedule.
    Pass,
    /// Snapshot, restore onto a fresh controller, and continue on the restored one.
    Snapshot,
}

/// Read `N` bytes, or `None` at the end of the input.
fn take<const N: usize>(b: &mut &[u8]) -> Option<[u8; N]> {
    let (head, rest) = b.split_first_chunk::<N>()?;
    *b = rest;
    Some(*head)
}

impl Op {
    fn parse(b: &mut &[u8]) -> Option<Op> {
        let [tag] = take::<1>(b)?;
        let u16_ = |b: &mut &[u8]| take::<2>(b).map(u16::from_le_bytes);
        Some(match tag % 6 {
            0 => Op::Write {
                off: u16_(b)?,
                width: take::<1>(b)?[0],
                val: u64::from_le_bytes(take::<8>(b)?),
            },
            1 => Op::Read {
                off: u16_(b)?,
                width: take::<1>(b)?[0],
            },
            2 => Op::Point {
                off: u16_(b)?,
                to: u16_(b)?,
                low: take::<1>(b)?[0],
            },
            3 => Op::Plant {
                at: u16_(b)?,
                bytes: take::<16>(b)?,
            },
            4 => Op::Pass,
            _ => Op::Snapshot,
        })
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        match *self {
            Op::Write { off, width, val } => {
                out.push(0);
                out.extend_from_slice(&off.to_le_bytes());
                out.push(width);
                out.extend_from_slice(&val.to_le_bytes());
            }
            Op::Read { off, width } => {
                out.push(1);
                out.extend_from_slice(&off.to_le_bytes());
                out.push(width);
            }
            Op::Point { off, to, low } => {
                out.push(2);
                out.extend_from_slice(&off.to_le_bytes());
                out.extend_from_slice(&to.to_le_bytes());
                out.push(low);
            }
            Op::Plant { at, bytes } => {
                out.push(3);
                out.extend_from_slice(&at.to_le_bytes());
                out.extend_from_slice(&bytes);
            }
            Op::Pass => out.push(4),
            Op::Snapshot => out.push(5),
        }
    }
}

/// Parse an input into its operations. Never fails: whatever parses is the session.
pub fn parse(mut b: &[u8]) -> Vec<Op> {
    let mut ops = Vec::new();
    while ops.len() < MAX_OPS
        && let Some(op) = Op::parse(&mut b)
    {
        ops.push(op);
    }
    ops
}

pub fn encode(ops: &[Op]) -> Vec<u8> {
    let mut out = Vec::new();
    for op in ops {
        op.encode(&mut out);
    }
    out
}

fn width(w: u8) -> usize {
    [1, 2, 4, 8][w as usize % 4]
}

/// The gadgets, in port order. One set per session, shared by the original and any restored
/// controller, as the same process's models would be across a suspend.
fn models() -> Vec<Arc<dyn UsbDeviceModel>> {
    vec![
        Arc::new(MockUsbDevice::new()),
        Arc::new(HidMockDevice::new()),
    ]
}

fn controller(models: &[Arc<dyn UsbDeviceModel>]) -> Arc<Mutex<XhciDevice>> {
    let mut d = XhciDevice::new(EventFd::new(EFD_NONBLOCK).unwrap());
    d.attach_devices(models.to_vec(), EventFd::new(EFD_NONBLOCK).unwrap());
    Arc::new(Mutex::new(d))
}

/// Play one guest session against a fresh controller, with zeroed RAM, and hand back its final
/// state. Panics where the controller breaks a property: a snapshot that does not restore to
/// itself.
pub fn drive(ops: &[Op]) -> XhciState {
    let mem = GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), RAM as usize)]).unwrap();
    let models = models();
    let mut dev = controller(&models);
    for op in ops {
        match *op {
            Op::Write { off, width: w, val } => {
                let bytes = val.to_le_bytes();
                dev.lock()
                    .unwrap()
                    .write(0, off as u64 % REGS, &bytes[..width(w)]);
            }
            Op::Read { off, width: w } => {
                let mut buf = [0u8; 8];
                dev.lock()
                    .unwrap()
                    .read(0, off as u64 % REGS, &mut buf[..width(w)]);
            }
            Op::Point { off, to, low } => {
                let val = (to as u64 & !0xf) | (low as u64 & 0xf);
                dev.lock()
                    .unwrap()
                    .write(0, off as u64 % REGS & !0x7, &val.to_le_bytes());
            }
            Op::Plant { at, bytes } => {
                let at = (at as u64).min(RAM - 16);
                mem.write_slice(&bytes, GuestAddress(at)).unwrap();
            }
            Op::Pass => run_pass(&dev, &mem),
            Op::Snapshot => {
                let saved = dev.lock().unwrap().save_state();
                let restored = controller(&models);
                restored.lock().unwrap().restore_state(&saved);
                let mut want = saved.clone();
                want.usbsts &= !STS_CNR;
                assert_eq!(
                    restored.lock().unwrap().save_state(),
                    want,
                    "a restored controller saves a different state"
                );
                dev = restored;
            }
        }
    }
    dev.lock().unwrap().save_state()
}
