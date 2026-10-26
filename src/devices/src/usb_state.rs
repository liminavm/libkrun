// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The xHCI controller's **host-side** state, as carried across a VM snapshot.
//!
//! A snapshot-suspend tears the whole VMM worker down and a fresh one restores the guest, so any
//! controller state that lives only in the host is lost — while the guest, which used xHCI's own
//! `USBCMD.CSS`/`CRS` save/restore feature, resumes believing the controller kept it. Carrying
//! this struct in the snapshot closes that gap. The design invariant is:
//!
//! > **After restore, the fresh controller must be indistinguishable from the in-place one.**
//!
//! …so this covers *every* field of `XhciDevice` that is neither reconstructible from guest RAM
//! nor re-established by the fresh worker (its interrupt plumbing, worker kick eventfd, and
//! cold-plugged device models). Adding a field to `XhciDevice` without adding it here is a bug;
//! `save_state`/`restore_state` in `usb::xhci::device` are written field-by-field against the
//! struct definition for exactly that reason.
//!
//! Plain data, no logic, and deliberately **outside the `usb` cargo feature gate**: the snapshot
//! file format must not depend on which features the VMM was built with.

/// One ring consumer position: dequeue pointer + Consumer Cycle State.
pub type XhciRingPos = (u64, bool);

/// The single-segment event-ring **producer** — host-only state (the consumer half lives in the
/// guest's ERDP). All four fields are needed: re-reading `seg_size` from the guest's ERST would
/// be the guest-RAM parse this design exists to avoid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct XhciEventRingState {
    pub seg_base: u64,
    pub seg_size: u32,
    pub enqueue_idx: u32,
    pub pcs: bool,
}

/// A live data endpoint (DCI 2..=31).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct XhciEpState {
    pub dci: u8,
    pub ring: XhciRingPos,
    pub dir_in: bool,
    pub ep_type: u32,
    pub state: u32,
    pub generation: u64,
}

/// One device slot. `config_value` is the reason a DCBAA re-derivation could not replace this
/// struct: the SET_CONFIGURATION value exists nowhere in guest RAM.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct XhciSlotState {
    pub slot_id: u8,
    pub port: u8,
    pub address: u8,
    pub state: u32,
    pub config_value: u8,
    pub ep0_ring: Option<XhciRingPos>,
    pub eps: Vec<XhciEpState>,
}

/// One root-hub port: its `PORTSC` plus the identity of whatever gadget was attached.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct XhciPortState {
    pub portsc: u32,
    /// `(idVendor, idProduct)` of the attached model, or `None` for an empty port. The restore
    /// worker cold-plugs gadgets in a fixed order but each one can fail to build independently,
    /// which would silently *shift* the rest onto different ports — so the identity is checked,
    /// not assumed.
    pub model_id: Option<(u16, u16)>,
}

/// The complete carried state (see the module docs).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct XhciState {
    // Operational registers.
    pub usbcmd: u32,
    pub usbsts: u32,
    pub dnctrl: u32,
    pub crcr: u64,
    pub dcbaap: u64,
    pub config: u32,
    pub ports: Vec<XhciPortState>,
    // Runtime registers (interrupter 0).
    pub iman: u32,
    pub imod: u32,
    pub erstsz: u32,
    pub erstba: u64,
    pub erdp: u64,
    // Derived controller state.
    pub cmd_ring: Option<XhciRingPos>,
    pub event_ring: Option<XhciEventRingState>,
    pub slots: Vec<XhciSlotState>,
    pub next_address: u8,
    /// Mid-64-bit-write latch for a `CRCR` stop/abort (see `write_crcr_lo`). Carried only so the
    /// "every host-only field is carried" invariant is literally true — a snapshot landing between
    /// the two halves of one guest store is vanishingly unlikely.
    pub crcr_stop_write: bool,
    // Worker work pending at capture. Normally all empty: the guest quiesced its USB stack before
    // the freeze and the worker drained. Carried so a doorbell that raced the freeze is not lost.
    pub work_run_started: bool,
    pub work_cmd_doorbell: bool,
    pub work_cmd_abort: bool,
    pub work_ep_doorbells: Vec<(u8, u8)>,
    pub work_port_events: Vec<u8>,
}
