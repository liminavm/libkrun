// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! xHCI device/input/endpoint context access (32-byte contexts, CSZ = 0).
//!
//! Input Context layout: Input Control Context (dwords 0-7), then the Device
//! Context — Slot Context at offset 32, Endpoint Context for DCI `d` at offset
//! `32 * (d + 1)`. Output Device Context: Slot Context at offset 0, Endpoint
//! DCI `d` at offset `32 * d`. All context reads are bounds-checked through
//! `GuestMemoryMmap`; a bad DCBAA/context pointer errors rather than panics.
//!
//! A few accessors (endpoint type/state getters, TR-dequeue setter, drop flags)
//! are ABI-complete and only exercised once B2 adds data endpoints.
#![allow(dead_code)]

use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

use super::trb::RingError;

/// One 32-byte context (Slot / Endpoint / Input Control), as eight dwords.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ctx32(pub [u32; 8]);

impl Ctx32 {
    pub fn read(mem: &GuestMemoryMmap, addr: u64) -> Result<Ctx32, RingError> {
        let mut d = [0u32; 8];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = mem
                .read_obj::<u32>(GuestAddress(addr + (i as u64) * 4))
                .map_err(|_| RingError::BadAccess(addr))?;
        }
        Ok(Ctx32(d))
    }

    pub fn write(&self, mem: &GuestMemoryMmap, addr: u64) -> Result<(), RingError> {
        for (i, v) in self.0.iter().enumerate() {
            mem.write_obj::<u32>(*v, GuestAddress(addr + (i as u64) * 4))
                .map_err(|_| RingError::BadAccess(addr))?;
        }
        Ok(())
    }
}

// ---- slot states -------------------------------------------------------------

pub mod slot_state {
    pub const DISABLED_ENABLED: u32 = 0;
    pub const DEFAULT: u32 = 1;
    pub const ADDRESSED: u32 = 2;
    pub const CONFIGURED: u32 = 3;
}

// ---- endpoint states ---------------------------------------------------------

pub mod ep_state {
    pub const DISABLED: u32 = 0;
    pub const RUNNING: u32 = 1;
    pub const HALTED: u32 = 2;
    pub const STOPPED: u32 = 3;
}

/// Byte offset of the Slot Context inside an Input Context.
pub const INPUT_SLOT_OFFSET: u64 = 32;
/// Byte offset of the Endpoint Context for `dci` inside an Input Context.
pub fn input_ep_offset(dci: u8) -> u64 {
    32 * (u64::from(dci) + 1)
}
/// Byte offset of the Endpoint Context for `dci` inside an Output Device Context.
pub fn output_ep_offset(dci: u8) -> u64 {
    32 * u64::from(dci)
}

/// The Input Control Context's Add flags (dword1): bit `d` = add DCI `d`
/// (bit 0 = slot context).
pub fn input_add_flags(input: &Ctx32) -> u32 {
    input.0[1]
}

/// The Input Control Context's Drop flags (dword0).
pub fn input_drop_flags(input: &Ctx32) -> u32 {
    input.0[0]
}

// ---- Slot Context accessors --------------------------------------------------

/// Root Hub Port Number (slot dword1, bits 23:16), 1-based.
pub fn slot_root_hub_port(slot: &Ctx32) -> u8 {
    ((slot.0[1] >> 16) & 0xff) as u8
}

/// Speed field (slot dword0, bits 23:20).
pub fn slot_speed(slot: &Ctx32) -> u32 {
    (slot.0[0] >> 20) & 0xf
}

/// Context Entries (slot dword0, bits 31:27) — index of the last valid EP DCI.
pub fn slot_context_entries(slot: &Ctx32) -> u8 {
    ((slot.0[0] >> 27) & 0x1f) as u8
}

pub fn set_slot_state(slot: &mut Ctx32, state: u32) {
    slot.0[3] = (slot.0[3] & !(0x1f << 27)) | ((state & 0x1f) << 27);
}

pub fn set_slot_address(slot: &mut Ctx32, address: u8) {
    slot.0[3] = (slot.0[3] & !0xff) | u32::from(address);
}

pub fn slot_state(slot: &Ctx32) -> u32 {
    (slot.0[3] >> 27) & 0x1f
}

// ---- Endpoint Context accessors ---------------------------------------------

/// Endpoint type (ep dword1, bits 5:3): 0=invalid, 4=Control, others bulk/int/isoch.
pub fn ep_type(ep: &Ctx32) -> u32 {
    (ep.0[1] >> 3) & 0x7
}

/// TR Dequeue Pointer (ep dwords 2-3, bits 63:4) and Dequeue Cycle State (bit 0).
pub fn ep_tr_dequeue(ep: &Ctx32) -> (u64, bool) {
    let lo = u64::from(ep.0[2]);
    let hi = u64::from(ep.0[3]);
    let raw = lo | (hi << 32);
    (raw & !0xf, raw & 1 != 0)
}

pub fn set_ep_tr_dequeue(ep: &mut Ctx32, ptr: u64, dcs: bool) {
    let raw = (ptr & !0xf) | if dcs { 1 } else { 0 };
    ep.0[2] = raw as u32;
    ep.0[3] = (raw >> 32) as u32;
}

pub fn set_ep_state(ep: &mut Ctx32, state: u32) {
    ep.0[0] = (ep.0[0] & !0x7) | (state & 0x7);
}

pub fn ep_state(ep: &Ctx32) -> u32 {
    ep.0[0] & 0x7
}

/// Read the Output Device Context base pointer for `slot_id` from the DCBAA.
/// Entry 0 is the scratchpad array (unused here); slots are 1-based.
pub fn dcbaa_entry(mem: &GuestMemoryMmap, dcbaap: u64, slot_id: u8) -> Result<u64, RingError> {
    let addr = (dcbaap & !0x3f) + u64::from(slot_id) * 8;
    let raw = mem
        .read_obj::<u64>(GuestAddress(addr))
        .map_err(|_| RingError::BadAccess(addr))?;
    Ok(raw & !0x3f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x10000)]).unwrap()
    }

    #[test]
    fn slot_context_field_round_trip() {
        let mut slot = Ctx32::default();
        // Speed 1 (full), context entries 1, root hub port 1.
        slot.0[0] = (1 << 20) | (1 << 27);
        slot.0[1] = 1 << 16;
        assert_eq!(slot_speed(&slot), 1);
        assert_eq!(slot_context_entries(&slot), 1);
        assert_eq!(slot_root_hub_port(&slot), 1);

        set_slot_state(&mut slot, slot_state::ADDRESSED);
        set_slot_address(&mut slot, 7);
        assert_eq!(slot_state(&slot), slot_state::ADDRESSED);
        assert_eq!(slot.0[3] & 0xff, 7);
    }

    #[test]
    fn ep_tr_dequeue_round_trip() {
        let mut ep = Ctx32::default();
        set_ep_tr_dequeue(&mut ep, 0x1_2340, true);
        let (ptr, dcs) = ep_tr_dequeue(&ep);
        assert_eq!(ptr, 0x1_2340);
        assert!(dcs);
    }

    #[test]
    fn input_context_parse_from_guest_memory() {
        let m = mem();
        // Input Control Context at 0x1000: add flags = A0 | A1.
        let mut input_ctrl = Ctx32::default();
        input_ctrl.0[1] = 0b11;
        input_ctrl.write(&m, 0x1000).unwrap();
        // Slot context at +32.
        let mut slot = Ctx32::default();
        slot.0[1] = 1 << 16; // root hub port 1
        slot.write(&m, 0x1000 + INPUT_SLOT_OFFSET).unwrap();

        let ic = Ctx32::read(&m, 0x1000).unwrap();
        assert_eq!(input_add_flags(&ic), 0b11);
        let s = Ctx32::read(&m, 0x1000 + INPUT_SLOT_OFFSET).unwrap();
        assert_eq!(slot_root_hub_port(&s), 1);
    }

    #[test]
    fn dcbaa_entry_reads_output_context_pointer() {
        let m = mem();
        // DCBAA at 0x2000; slot 3 -> output context @ 0x5000.
        m.write_obj::<u64>(0x5000, GuestAddress(0x2000 + 3 * 8))
            .unwrap();
        assert_eq!(dcbaa_entry(&m, 0x2000, 3).unwrap(), 0x5000);
    }
}
