// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! xHCI Transfer Request Block (TRB) ABI and the ring machinery: the command /
//! transfer ring *walker* (cycle-bit + Link-TRB traversal) and the event ring
//! *enqueuer* (producer cycle + single-segment ERST + ring-full handling).
//!
//! These are pure functions over guest memory. **Ring pointers are hostile
//! input** (this is where xHCI's guest-triggerable CVEs live): every walk is
//! bounded — a malformed or self-referential Link chain errors out, it never
//! hangs or panics.
//!
//! Some TRB types / completion codes below are declared for ABI completeness and
//! only used once B2 wires interrupt/bulk data flow and richer error paths.
#![allow(dead_code)]

use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

/// A raw TRB (16 bytes = two dwords of parameter, one status, one control).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Trb {
    pub parameter: u64,
    pub status: u32,
    pub control: u32,
}

// ---- TRB types (control bits 15:10) -----------------------------------------

pub mod trb_type {
    pub const NORMAL: u32 = 1;
    pub const SETUP_STAGE: u32 = 2;
    pub const DATA_STAGE: u32 = 3;
    pub const STATUS_STAGE: u32 = 4;
    pub const LINK: u32 = 6;
    pub const EVENT_DATA: u32 = 7;
    pub const NO_OP: u32 = 8;
    pub const ENABLE_SLOT: u32 = 9;
    pub const DISABLE_SLOT: u32 = 10;
    pub const ADDRESS_DEVICE: u32 = 11;
    pub const CONFIGURE_ENDPOINT: u32 = 12;
    pub const EVALUATE_CONTEXT: u32 = 13;
    pub const RESET_ENDPOINT: u32 = 14;
    pub const STOP_ENDPOINT: u32 = 15;
    pub const SET_TR_DEQUEUE: u32 = 16;
    pub const RESET_DEVICE: u32 = 17;
    pub const NO_OP_CMD: u32 = 23;
    pub const TRANSFER_EVENT: u32 = 32;
    pub const COMMAND_COMPLETION: u32 = 33;
    pub const PORT_STATUS_CHANGE: u32 = 34;
}

// ---- completion codes (event status bits 31:24) -----------------------------

pub mod cc {
    pub const SUCCESS: u32 = 1;
    pub const DATA_BUFFER_ERROR: u32 = 2;
    pub const USB_TRANSACTION_ERROR: u32 = 4;
    pub const TRB_ERROR: u32 = 5;
    pub const STALL_ERROR: u32 = 6;
    pub const RESOURCE_ERROR: u32 = 7;
    pub const NO_SLOTS_AVAILABLE: u32 = 9;
    pub const SLOT_NOT_ENABLED: u32 = 11;
    pub const SHORT_PACKET: u32 = 13;
    pub const PARAMETER_ERROR: u32 = 17;
    pub const CONTEXT_STATE_ERROR: u32 = 19;
}

// ---- control-word flag bits -------------------------------------------------

pub const CTRL_CYCLE: u32 = 1 << 0;
/// Link TRB: Toggle Cycle.
pub const CTRL_TOGGLE_CYCLE: u32 = 1 << 1;
/// Transfer TRB: Evaluate Next TRB (bit 1, same position as Toggle Cycle).
pub const CTRL_ENT: u32 = 1 << 1;
/// Interrupt on Short Packet.
pub const CTRL_ISP: u32 = 1 << 2;
/// Chain bit.
pub const CTRL_CHAIN: u32 = 1 << 4;
/// Interrupt On Completion.
pub const CTRL_IOC: u32 = 1 << 5;
/// Immediate Data.
pub const CTRL_IDT: u32 = 1 << 6;
/// Data/Status stage direction (1 = IN).
pub const CTRL_DIR_IN: u32 = 1 << 16;
/// Address Device: Block Set Address Request.
pub const CTRL_BSR: u32 = 1 << 9;
/// Configure Endpoint: Deconfigure.
pub const CTRL_DC: u32 = 1 << 9;

impl Trb {
    /// Read a TRB from guest memory at `addr` (16-byte aligned by construction).
    pub fn read(mem: &GuestMemoryMmap, addr: u64) -> Result<Trb, RingError> {
        let mut b = [0u8; 16];
        mem.read_slice(&mut b, GuestAddress(addr))
            .map_err(|_| RingError::BadAccess(addr))?;
        Ok(Trb {
            parameter: u64::from_le_bytes(b[0..8].try_into().unwrap()),
            status: u32::from_le_bytes(b[8..12].try_into().unwrap()),
            control: u32::from_le_bytes(b[12..16].try_into().unwrap()),
        })
    }

    /// Write this TRB to guest memory at `addr`.
    pub fn write(&self, mem: &GuestMemoryMmap, addr: u64) -> Result<(), RingError> {
        let mut b = [0u8; 16];
        b[0..8].copy_from_slice(&self.parameter.to_le_bytes());
        b[8..12].copy_from_slice(&self.status.to_le_bytes());
        b[12..16].copy_from_slice(&self.control.to_le_bytes());
        mem.write_slice(&b, GuestAddress(addr))
            .map_err(|_| RingError::BadAccess(addr))
    }

    pub fn cycle(&self) -> bool {
        self.control & CTRL_CYCLE != 0
    }

    pub fn trb_type(&self) -> u32 {
        (self.control >> 10) & 0x3f
    }

    /// Link TRB target pointer (bits 63:4 of the parameter).
    pub fn link_target(&self) -> u64 {
        self.parameter & !0xf
    }

    pub fn toggle_cycle(&self) -> bool {
        self.control & CTRL_TOGGLE_CYCLE != 0
    }

    /// Command TRB: Slot ID (control bits 31:24).
    pub fn slot_id(&self) -> u8 {
        ((self.control >> 24) & 0xff) as u8
    }

    /// Command/Transfer TRB: Endpoint ID / DCI (control bits 20:16).
    pub fn endpoint_id(&self) -> u8 {
        ((self.control >> 16) & 0x1f) as u8
    }

    /// Transfer TRB length (status bits 16:0).
    pub fn transfer_len(&self) -> u32 {
        self.status & 0x1_ffff
    }

    /// The 8 SETUP bytes carried in a Setup Stage TRB's parameter field.
    pub fn setup_bytes(&self) -> [u8; 8] {
        self.parameter.to_le_bytes()
    }
}

/// Errors from walking a guest-programmed ring. All are non-fatal to the VMM:
/// the controller logs and stops processing the offending ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RingError {
    /// A guest-memory read/write at this address failed (out of range).
    BadAccess(u64),
    /// The Link-TRB chain exceeded the hop bound (a loop or a pathological ring).
    LinkLoop,
    /// A single TD chained more work TRBs than [`MAX_TD_TRBS`] — a hostile ring whose
    /// Chain bit never clears (unbounded data-buffer / immediate accumulation).
    TdTooLong,
}

/// The maximum number of work (Normal/Event-Data) TRBs to accept in a single TD before
/// declaring it malformed. A TD whose Chain bit never clears (hostile input) would otherwise
/// grow the collected segment / immediate buffers without bound and hang the worker. Real TDs
/// are a handful of TRBs; this bound is generous.
pub const MAX_TD_TRBS: usize = 1024;

/// The maximum number of consecutive Link TRBs to follow before declaring the
/// ring malformed. A self-referential or looping Link chain (hostile input)
/// is bounded here rather than hanging the worker.
const MAX_LINK_HOPS: usize = 64;

/// A consumer walker over a command or transfer ring: tracks the dequeue pointer
/// and the Consumer Cycle State, follows Link TRBs, and stops at the producer's
/// boundary (a TRB whose cycle bit != CCS).
#[derive(Clone, Copy, Debug)]
pub struct RingWalker {
    ptr: u64,
    ccs: bool,
}

impl RingWalker {
    pub fn new(ptr: u64, ccs: bool) -> Self {
        RingWalker {
            ptr: ptr & !0xf,
            ccs,
        }
    }

    /// The current dequeue pointer (for persisting an endpoint's TR dequeue).
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    pub fn ccs(&self) -> bool {
        self.ccs
    }

    /// Fetch the next work TRB, following (and consuming) Link TRBs. Returns:
    /// - `Ok(Some((addr, trb)))` — a work TRB at `addr`; the walker has advanced
    ///   past it.
    /// - `Ok(None)` — the ring is empty (producer cycle boundary reached).
    /// - `Err(_)` — a malformed ring (bounded loop / bad pointer).
    pub fn next(&mut self, mem: &GuestMemoryMmap) -> Result<Option<(u64, Trb)>, RingError> {
        for _ in 0..MAX_LINK_HOPS {
            let addr = self.ptr;
            let trb = Trb::read(mem, addr)?;
            if trb.cycle() != self.ccs {
                // Producer hasn't published this slot yet: ring empty.
                return Ok(None);
            }
            if trb.trb_type() == trb_type::LINK {
                // Follow the link; toggle CCS if the TC bit is set.
                self.ptr = trb.link_target();
                if trb.toggle_cycle() {
                    self.ccs = !self.ccs;
                }
                continue;
            }
            // A work TRB: advance past it and return it.
            self.ptr = addr.wrapping_add(16);
            return Ok(Some((addr, trb)));
        }
        Err(RingError::LinkLoop)
    }
}

/// A single-segment event ring enqueuer (we advertise ERST Max = 0 ⇒ 1 segment).
/// Tracks the enqueue index and Producer Cycle State; detects ring-full against
/// the guest's ERDP so unread events are never overwritten.
#[derive(Clone, Copy, Debug)]
pub struct EventRing {
    seg_base: u64,
    seg_size: u32,
    enqueue_idx: u32,
    pcs: bool,
}

impl EventRing {
    /// Build from the single ERST entry (segment base + size). The producer
    /// cycle starts at 1 (matching the guest's initial CCS after reset).
    pub fn new(seg_base: u64, seg_size: u32) -> Self {
        EventRing {
            seg_base: seg_base & !0x3f,
            seg_size: seg_size.max(1),
            enqueue_idx: 0,
            pcs: true,
        }
    }

    /// Guest-physical address the next event will be written to.
    pub fn enqueue_addr(&self) -> u64 {
        self.seg_base + u64::from(self.enqueue_idx) * 16
    }

    /// Enqueue one event TRB, stamping the producer cycle bit. `erdp` is the
    /// guest's current Event Ring Dequeue Pointer (for ring-full detection).
    /// Returns `Ok(())`, or `Err` if the ring is full (the event is dropped —
    /// never at HID rates, but we never corrupt an unread slot).
    pub fn enqueue(
        &mut self,
        mem: &GuestMemoryMmap,
        mut trb: Trb,
        erdp: u64,
    ) -> Result<(), RingError> {
        let dequeue_idx = self.dequeue_idx(erdp);
        let next_idx = (self.enqueue_idx + 1) % self.seg_size;
        if next_idx == dequeue_idx {
            // One-slot-reserved full convention: refuse rather than clobber the
            // slot the consumer is about to read.
            return Err(RingError::LinkLoop);
        }
        // Stamp the cycle bit into the control word.
        trb.control = (trb.control & !CTRL_CYCLE) | if self.pcs { CTRL_CYCLE } else { 0 };
        trb.write(mem, self.enqueue_addr())?;

        self.enqueue_idx += 1;
        if self.enqueue_idx == self.seg_size {
            self.enqueue_idx = 0;
            self.pcs = !self.pcs;
        }
        Ok(())
    }

    fn dequeue_idx(&self, erdp: u64) -> u32 {
        let ptr = erdp & !0xf;
        let end = self.seg_base + u64::from(self.seg_size) * 16;
        if ptr >= self.seg_base && ptr < end {
            ((ptr - self.seg_base) / 16) as u32
        } else {
            0
        }
    }
}

/// Build a Command Completion Event TRB.
pub fn command_completion_event(cmd_trb_ptr: u64, completion_code: u32, slot_id: u8) -> Trb {
    Trb {
        parameter: cmd_trb_ptr,
        status: (completion_code & 0xff) << 24,
        control: (trb_type::COMMAND_COMPLETION << 10) | ((u32::from(slot_id)) << 24),
    }
}

/// Build a Transfer Event TRB. `residue` is the number of bytes *not*
/// transferred (reported in status bits 23:0).
pub fn transfer_event(
    trb_ptr: u64,
    residue: u32,
    completion_code: u32,
    slot_id: u8,
    endpoint_id: u8,
    event_data: bool,
) -> Trb {
    let mut control = (trb_type::TRANSFER_EVENT << 10)
        | ((u32::from(endpoint_id) & 0x1f) << 16)
        | ((u32::from(slot_id)) << 24);
    if event_data {
        control |= 1 << 2; // ED bit
    }
    Trb {
        parameter: trb_ptr,
        status: ((residue & 0xff_ffff) | ((completion_code & 0xff) << 24)),
        control,
    }
}

/// Build a Port Status Change Event TRB for `port_id` (1-based).
pub fn port_status_change_event(port_id: u8) -> Trb {
    Trb {
        parameter: u64::from(port_id) << 24,
        status: cc::SUCCESS << 24,
        control: trb_type::PORT_STATUS_CHANGE << 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vm_memory::GuestMemoryMmap;

    fn mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap()
    }

    /// Write a work TRB (type NORMAL) with the given cycle bit at `addr`.
    fn put_normal(mem: &GuestMemoryMmap, addr: u64, cycle: bool) {
        let control = (trb_type::NORMAL << 10) | if cycle { CTRL_CYCLE } else { 0 };
        Trb {
            parameter: addr,
            status: 0,
            control,
        }
        .write(mem, addr)
        .unwrap();
    }

    fn put_link(mem: &GuestMemoryMmap, addr: u64, target: u64, cycle: bool, toggle: bool) {
        let mut control = (trb_type::LINK << 10) | if cycle { CTRL_CYCLE } else { 0 };
        if toggle {
            control |= CTRL_TOGGLE_CYCLE;
        }
        Trb {
            parameter: target,
            status: 0,
            control,
        }
        .write(mem, addr)
        .unwrap();
    }

    #[test]
    fn walker_stops_at_producer_cycle_boundary() {
        let m = mem();
        put_normal(&m, 0x1000, true);
        put_normal(&m, 0x1010, true);
        // 0x1020 left zeroed: cycle bit 0 != CCS(1) -> ring empty.
        let mut w = RingWalker::new(0x1000, true);
        assert_eq!(w.next(&m).unwrap().unwrap().0, 0x1000);
        assert_eq!(w.next(&m).unwrap().unwrap().0, 0x1010);
        assert!(w.next(&m).unwrap().is_none(), "boundary reached");
    }

    #[test]
    fn walker_follows_link_and_toggles_cycle() {
        let m = mem();
        // Segment A at 0x2000: one TRB then a Link to 0x3000 with Toggle Cycle.
        put_normal(&m, 0x2000, true);
        put_link(&m, 0x2010, 0x3000, true, true);
        // Segment B at 0x3000: the producer wrote it with cycle now 0 (toggled).
        put_normal(&m, 0x3000, false);
        let mut w = RingWalker::new(0x2000, true);
        assert_eq!(w.next(&m).unwrap().unwrap().0, 0x2000);
        // Crossing the toggling Link flips CCS to 0, so the false-cycle TRB matches.
        assert_eq!(w.next(&m).unwrap().unwrap().0, 0x3000);
        assert!(!w.ccs(), "CCS toggled across the Link");
    }

    #[test]
    fn self_referential_link_is_bounded() {
        let m = mem();
        // A Link that points at itself and never toggles: an infinite loop of
        // links with no work TRB. Must error, not hang.
        put_link(&m, 0x4000, 0x4000, true, false);
        let mut w = RingWalker::new(0x4000, true);
        assert_eq!(w.next(&m), Err(RingError::LinkLoop));
    }

    #[test]
    fn event_ring_stamps_cycle_and_wraps() {
        let m = mem();
        // 4-slot segment at 0x8000.
        let mut er = EventRing::new(0x8000, 4);
        // ERDP at the segment base: dequeue_idx 0.
        let erdp = 0x8000;
        // Fill 3 slots (the 4th is the reserved full-guard slot).
        for i in 0..3 {
            er.enqueue(&m, port_status_change_event(1), erdp).unwrap();
            let ctrl = Trb::read(&m, 0x8000 + i * 16).unwrap().control;
            assert_ne!(ctrl & CTRL_CYCLE, 0, "PCS=1 stamped on slot {i}");
        }
        // The next enqueue would collide with the (reserved) dequeue slot -> full.
        assert!(er.enqueue(&m, port_status_change_event(1), erdp).is_err());
    }

    #[test]
    fn event_ring_full_tracks_the_guest_dequeue_pointer() {
        let m = mem();
        let mut er = EventRing::new(0x9000, 4);
        // Guest dequeue at index 2: with the one-slot-reserved convention, only a
        // single enqueue fits before the producer would collide with the consumer.
        let erdp = 0x9000 + 2 * 16;
        er.enqueue(&m, port_status_change_event(1), erdp).unwrap(); // idx0 -> next idx1
                                                                    // Next enqueue's next index (2) equals the dequeue index -> full.
        assert!(er.enqueue(&m, port_status_change_event(1), erdp).is_err());
    }

    #[test]
    fn setup_bytes_round_trip() {
        let raw = [0x80u8, 0x06, 0x00, 0x01, 0x00, 0x00, 0x40, 0x00];
        let trb = Trb {
            parameter: u64::from_le_bytes(raw),
            status: 8,
            control: (trb_type::SETUP_STAGE << 10) | CTRL_CYCLE | CTRL_IDT,
        };
        assert_eq!(trb.setup_bytes(), raw);
        assert_eq!(trb.trb_type(), trb_type::SETUP_STAGE);
    }
}
