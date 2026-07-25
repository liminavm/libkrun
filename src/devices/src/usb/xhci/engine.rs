// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The xHCI ring engine: the command set, EP0 control-transfer walker, port
//! enumeration and event posting. These run on the worker thread while holding
//! the single controller lock — **except** the actual gadget call, which the
//! worker makes with the lock released (see the `device.rs` module docs). The
//! completion each forwarded transfer carries re-locks the controller to post
//! its Transfer Event, so a gadget may complete now (the mock) or seconds later
//! (the FIDO gadget's Touch ID wait) from any thread.

use std::sync::{Arc, Mutex};

use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

use super::super::model::{
    Completion, ControlTransfer, EpAddr, SetupPacket, Transfer, UsbDeviceModel, XferOutcome,
};
use super::context::{
    dcbaa_entry, ep_state, ep_tr_dequeue, ep_type, input_add_flags, input_drop_flags,
    input_ep_offset, output_ep_offset, set_ep_state, set_slot_address, set_slot_state,
    slot_root_hub_port, Ctx32, INPUT_SLOT_OFFSET,
};
use super::device::{EpRing, SlotCtx, XhciDevice, NUM_PORTS};
use super::trb::{
    cc, command_completion_event, port_status_change_event, transfer_event, trb_type, EventRing,
    RingError, RingWalker, Trb, CTRL_BSR, CTRL_CHAIN, CTRL_DC, CTRL_DIR_IN, CTRL_IDT, CTRL_IOC,
    MAX_TD_TRBS,
};

/// A gadget call the worker makes with the controller lock **released**.
pub(super) enum DeferredCall {
    /// Forward a class/vendor EP0 control request to the gadget.
    Control(Arc<dyn UsbDeviceModel>, ControlTransfer),
    /// Forward an interrupt/bulk transfer to the gadget (it may hold the TRBs).
    Transfer(Arc<dyn UsbDeviceModel>, EpAddr, Transfer),
    /// Reset the device model (Disable Slot / Reset Device).
    Reset(Arc<dyn UsbDeviceModel>),
}

/// Everything a non-EP0 transfer's completion needs to post its Transfer Event —
/// captured while walking the ring, replayed (possibly seconds later, from any
/// thread) on completion. `generation` is the staleness token (see [`EpRing`]).
#[derive(Clone)]
pub(super) struct EpEvents {
    slot_id: u8,
    ep_id: u8,
    /// Whether this is an IN (device-to-host) transfer.
    dir_in: bool,
    /// Data-buffer guest segments `(addr, len)` (IN scatter target / OUT gather source).
    data_segs: Vec<(u64, u32)>,
    /// The TRB the Transfer Event points at (the TD's last / IOC-carrying TRB).
    event_trb: u64,
    /// The endpoint generation captured at dispatch; a completion whose endpoint has
    /// since been re-programmed (this value stale) is dropped.
    generation: u64,
    /// (ptr, dcs) of the TRB after this TD — the output Endpoint Context TR Dequeue to
    /// publish once the transfer completes, keeping `hw_deq` truthful for Stop/stall.
    td_next: (u64, bool),
}

/// Everything a control transfer's completion needs to post its Transfer
/// Event(s) — captured while walking the ring, replayed on completion.
#[derive(Clone)]
pub(super) struct ControlEvents {
    slot_id: u8,
    ep_id: u8,
    /// Data-stage guest buffer segments `(addr, len)` (for IN scatter / OUT gather).
    data_segs: Vec<(u64, u32)>,
    /// Address of the last data-stage TRB (the short-packet event target).
    data_trb: Option<u64>,
    /// Address of the Status-stage TRB.
    status_trb: u64,
    /// Whether the status stage requested Interrupt On Completion (Linux always sets it).
    status_ioc: bool,
    /// wLength from the SETUP packet.
    wlength: u32,
    /// (ptr, dcs) of this TD's first TRB — the Endpoint Context TR Dequeue value to
    /// leave behind if the transfer *halts* (the guest's stall recovery reads it via
    /// `xhci_get_hw_deq` and walks forward from here to reposition the ring).
    td_start: (u64, bool),
    /// (ptr, dcs) of the TRB *after* this TD — the TR Dequeue value to publish once
    /// the transfer completes successfully (keeps the hardware dequeue current so a
    /// later Stop Endpoint / stall recovery reads a truthful position).
    td_next: (u64, bool),
}

/// A parsed control-transfer descriptor (one Setup/Data/Status TD).
struct ControlTd {
    setup: [u8; 8],
    events: ControlEvents,
}

impl XhciDevice {
    /// Do one worker pass: build the event ring if needed, scan for cold-plug
    /// connects, post queued port-change events, run the command ring, and
    /// collect EP0 control-transfer work. Returns gadget calls to make with the
    /// lock released.
    pub(super) fn run_worker_pass(
        &mut self,
        mem: &GuestMemoryMmap,
        dev_arc: &Arc<Mutex<XhciDevice>>,
    ) -> Vec<DeferredCall> {
        let mut deferred = Vec::new();
        self.ensure_event_ring(mem);

        if std::mem::take(&mut self.work.run_started) {
            self.scan_ports_on_run(mem);
        }
        let ports = std::mem::take(&mut self.work.port_events);
        for pid in ports {
            self.post_port_event(mem, pid);
        }
        if std::mem::take(&mut self.work.cmd_abort) {
            self.post_command_ring_stopped(mem);
        }
        if std::mem::take(&mut self.work.cmd_doorbell) {
            self.process_command_ring(mem, &mut deferred);
        }
        let doorbells = std::mem::take(&mut self.work.ep_doorbells);
        if !doorbells.is_empty() {
            debug!("xhci: worker draining ep doorbells {doorbells:?}");
        }
        for (slot, dci) in doorbells {
            if dci == 1 {
                self.collect_ep0_work(mem, slot, dev_arc, &mut deferred);
            } else {
                self.collect_ep_work(mem, slot, dci, dev_arc, &mut deferred);
            }
        }
        deferred
    }

    /// Build the single-segment event ring from the ERST if it isn't up yet.
    fn ensure_event_ring(&mut self, mem: &GuestMemoryMmap) {
        if self.event_ring.is_some() {
            return;
        }
        let erstba = self.erstba();
        if erstba == 0 {
            return;
        }
        // ERST entry: segment base (dwords 0-1), segment size (dword2 [15:0]).
        let base = match mem.read_obj::<u64>(GuestAddress(erstba)) {
            Ok(v) => v & !0x3f,
            Err(_) => {
                warn!("xhci: bad ERSTBA {erstba:#x}");
                return;
            }
        };
        let size = mem
            .read_obj::<u32>(GuestAddress(erstba + 8))
            .map(|v| v & 0xffff)
            .unwrap_or(0);
        if base == 0 || size == 0 {
            return;
        }
        self.event_ring = Some(EventRing::new(base, size));
    }

    /// Enqueue one event TRB and (unconditionally-latching) assert the interrupt.
    fn post_event(&mut self, mem: &GuestMemoryMmap, trb: Trb) {
        let erdp = self.erdp_ptr();
        match self.event_ring.as_mut() {
            Some(er) => {
                debug!(
                    "xhci: post event type={} at {:#x} (erdp {:#x})",
                    trb.trb_type(),
                    er.enqueue_addr(),
                    erdp
                );
                if let Err(e) = er.enqueue(mem, trb, erdp) {
                    warn!("xhci: event ring full / bad ({e:?}); event dropped");
                    return;
                }
            }
            None => {
                warn!("xhci: event posted before the event ring is up; dropped");
                return;
            }
        }
        self.assert_interrupt();
    }

    // ---- ports ---------------------------------------------------------------

    /// On USBCMD.RS 0→1: latch a connect on every port that has a cold-plugged
    /// device and post a Port Status Change Event so the guest's hub thread starts
    /// enumeration.
    /// **Idempotent**, because the run edge fires on every resume too (`xhci_resume` sets
    /// `USBCMD.RS` again): a port already reporting `CCS` is left alone. Re-latching `CSC` and
    /// forcing PLS back to Polling there made the guest's `xhci_bus_resume` see a port that was no
    /// longer in U3 — so it abandoned the resume — while the fresh `CSC` made its hub thread treat
    /// the still-present device as a re-plug and tear it down. Devices blinked out and back on
    /// every single suspend/resume cycle.
    ///
    /// Skipping by `CCS` (rather than "did we scan before") is what still lets a genuinely-changed
    /// port re-enumerate: a restore that reconciles a port to disconnected leaves `CCS` clear, so
    /// this scan latches a fresh connect for whatever model is there now.
    fn scan_ports_on_run(&mut self, mem: &GuestMemoryMmap) {
        for idx in 0..NUM_PORTS {
            if self.port_populated(idx) && !self.port_connected(idx) {
                self.set_port_connected(idx);
                self.post_port_event(mem, (idx + 1) as u8);
            }
        }
    }

    /// Acknowledge a `CRCR.CA` (Command Abort) write with a **Command Ring Stopped** Command
    /// Completion Event pointing at the current dequeue position.
    ///
    /// This is what lets the guest recover. Linux's command watchdog aborts the ring on a 5 s
    /// timeout and then waits for exactly this event: `xhci_handle_stopped_cmd_ring` reads the
    /// dequeue pointer out of it, restarts the ring and re-queues the pending commands. With no
    /// event its `cmd_ring_state` stays `CMD_RING_STATE_ABORTED` and every subsequent command is
    /// refused outright — the ring is intact but USB is dead all the same.
    fn post_command_ring_stopped(&mut self, mem: &GuestMemoryMmap) {
        let deq = self
            .cmd_ring
            .map(|w| w.ptr())
            .unwrap_or_else(|| self.crcr_ptr());
        debug!("xhci: command ring aborted; posting Command Ring Stopped at {deq:#x}");
        self.post_event(
            mem,
            command_completion_event(deq, cc::COMMAND_RING_STOPPED, 0),
        );
    }

    fn post_port_event(&mut self, mem: &GuestMemoryMmap, port_id: u8) {
        self.post_event(mem, port_status_change_event(port_id));
    }

    // ---- command ring --------------------------------------------------------

    fn process_command_ring(&mut self, mem: &GuestMemoryMmap, deferred: &mut Vec<DeferredCall>) {
        // Never walk address 0. A doorbell can arrive with no ring programmed (a stray write, or
        // the tail end of a guest tearing a dead HCD down), and building a walker at `crcr_ptr() ==
        // 0` turns that into an unbounded `BadAccess(0)` log loop. Belt-and-braces alongside the
        // CRCR stop/abort handling in `device.rs`.
        if self.cmd_ring.is_none() && self.crcr_ptr() == 0 {
            debug!("xhci: command doorbell with no command ring programmed; ignored");
            return;
        }
        let mut walker = self
            .cmd_ring
            .take()
            .unwrap_or_else(|| RingWalker::new(self.crcr_ptr(), self.crcr_rcs()));
        loop {
            match walker.next(mem) {
                Ok(Some((addr, trb))) => {
                    let (code, slot) = self.run_command(mem, &trb, deferred);
                    debug!(
                        "xhci: command type={} slot={} -> cc={}",
                        trb.trb_type(),
                        slot,
                        code
                    );
                    self.post_event(mem, command_completion_event(addr, code, slot));
                }
                Ok(None) => break,
                Err(e) => {
                    warn!("xhci: command ring walk error: {e:?}");
                    break;
                }
            }
        }
        self.cmd_ring = Some(walker);
    }

    /// Execute one command TRB. Returns `(completion_code, slot_id_for_event)`.
    /// Unimplemented/failed commands return an error completion code (never a
    /// silent drop).
    fn run_command(
        &mut self,
        mem: &GuestMemoryMmap,
        trb: &Trb,
        deferred: &mut Vec<DeferredCall>,
    ) -> (u32, u8) {
        use super::context::ep_state as es;
        use super::context::slot_state as ss;
        match trb.trb_type() {
            trb_type::NO_OP_CMD => (cc::SUCCESS, 0),
            trb_type::ENABLE_SLOT => match self.alloc_slot() {
                Some(slot) => (cc::SUCCESS, slot),
                None => (cc::NO_SLOTS_AVAILABLE, 0),
            },
            trb_type::DISABLE_SLOT => {
                let slot = trb.slot_id();
                self.defer_slot_reset(slot, deferred);
                if let Some(s) = self.slots.get_mut(slot as usize) {
                    *s = None;
                }
                (cc::SUCCESS, slot)
            }
            trb_type::ADDRESS_DEVICE => self.cmd_address_device(mem, trb),
            trb_type::CONFIGURE_ENDPOINT => {
                let slot = trb.slot_id();
                let deconfig = trb.control & CTRL_DC != 0;
                let input = trb.parameter & !0xf;
                match self.cmd_configure_endpoint(mem, slot, input, deconfig) {
                    Ok(()) => (cc::SUCCESS, slot),
                    Err(code) => (code, slot),
                }
            }
            trb_type::EVALUATE_CONTEXT => {
                // We don't gate on the evaluated fields (max packet size, exit
                // latency) in B1; accept and report success.
                (cc::SUCCESS, trb.slot_id())
            }
            trb_type::RESET_ENDPOINT => {
                let slot = trb.slot_id();
                let dci = trb.endpoint_id();
                // Clear the halt: back to Running, and bump the generation so any
                // outstanding (pre-halt) completion is dropped rather than posting onto
                // the ring the guest is about to reposition with Set TR Dequeue.
                self.set_output_ep_state(mem, slot, dci, es::RUNNING);
                if let Some(e) = self.ep_mut(slot, dci) {
                    e.state = es::RUNNING;
                    e.generation = e.generation.wrapping_add(1);
                }
                (cc::SUCCESS, slot)
            }
            trb_type::STOP_ENDPOINT => {
                let slot = trb.slot_id();
                let dci = trb.endpoint_id();
                // Quiesce the endpoint: mark it Stopped and bump the generation, so a
                // late gadget completion (a still-held IN transfer) is dropped — the
                // command completes only after this bump, i.e. after the endpoint can no
                // longer post onto the ring. Publish the current dequeue as `hw_deq`.
                self.set_output_ep_state(mem, slot, dci, es::STOPPED);
                if let Some(e) = self.ep_mut(slot, dci) {
                    e.state = es::STOPPED;
                    e.generation = e.generation.wrapping_add(1);
                    let (p, c) = (e.ring.ptr(), e.ring.ccs());
                    self.write_output_ep_dequeue(mem, slot, dci, p, c, None);
                }
                (cc::SUCCESS, slot)
            }
            trb_type::SET_TR_DEQUEUE => {
                let slot = trb.slot_id();
                let dci = trb.endpoint_id();
                let ptr = trb.parameter & !0xf;
                let dcs = trb.parameter & 1 != 0;
                debug!("xhci: SET_TR_DEQUEUE slot={slot} dci={dci} ptr={ptr:#x} dcs={dcs}");
                if dci == 1 {
                    if let Some(s) = self.slots.get_mut(slot as usize).and_then(|x| x.as_mut()) {
                        s.ep0_ring = Some(RingWalker::new(ptr, dcs));
                    }
                } else if let Some(e) = self.ep_mut(slot, dci) {
                    // Reposition the ring and bump the generation: a completion still in
                    // flight against the old ring is now stale and must not corrupt the
                    // re-programmed dequeue. Endpoint stays Stopped until the next doorbell.
                    e.ring = RingWalker::new(ptr, dcs);
                    e.state = ep_state::STOPPED;
                    e.generation = e.generation.wrapping_add(1);
                }
                // Mirror the repositioned dequeue into the output Endpoint Context so
                // a subsequent xhci_get_hw_deq reads the value the guest just set.
                self.write_output_ep_dequeue(mem, slot, dci, ptr, dcs, None);
                (cc::SUCCESS, slot)
            }
            trb_type::RESET_DEVICE => {
                let slot = trb.slot_id();
                self.defer_slot_reset(slot, deferred);
                if let Some(s) = self.slots.get_mut(slot as usize).and_then(|x| x.as_mut()) {
                    s.state = ss::DEFAULT;
                    s.config_value = 0;
                    // Data endpoints are torn down; any late completion finds no ring.
                    s.eps.clear();
                }
                (cc::SUCCESS, slot)
            }
            other => {
                warn!("xhci: unimplemented command TRB type {other}");
                (cc::TRB_ERROR, trb.slot_id())
            }
        }
    }

    fn alloc_slot(&mut self) -> Option<u8> {
        for id in 1..self.slots.len() {
            if self.slots[id].is_none() {
                self.slots[id] = Some(SlotCtx {
                    port: 0,
                    address: 0,
                    state: super::context::slot_state::DISABLED_ENABLED,
                    config_value: 0,
                    ep0_ring: None,
                    eps: Default::default(),
                });
                return Some(id as u8);
            }
        }
        None
    }

    fn defer_slot_reset(&self, slot: u8, deferred: &mut Vec<DeferredCall>) {
        if let Some(s) = self.slots.get(slot as usize).and_then(|x| x.as_ref()) {
            if s.port != 0 {
                if let Some(Some(m)) = self.port_models.get((s.port - 1) as usize) {
                    deferred.push(DeferredCall::Reset(m.clone()));
                }
            }
        }
    }

    /// Address Device: parse the input context, write the output device context
    /// (slot + EP0), and bind the slot to its root-hub port + model.
    fn cmd_address_device(&mut self, mem: &GuestMemoryMmap, trb: &Trb) -> (u32, u8) {
        use super::context::slot_state as ss;
        let slot_id = trb.slot_id();
        let bsr = trb.control & CTRL_BSR != 0;
        let input = trb.parameter & !0xf;

        let result = (|| -> Result<u8, u32> {
            let ictl = Ctx32::read(mem, input).map_err(|_| cc::TRB_ERROR)?;
            // A0 (slot) and A1 (EP0) must be added.
            if input_add_flags(&ictl) & 0b11 != 0b11 {
                return Err(cc::PARAMETER_ERROR);
            }
            let islot = Ctx32::read(mem, input + INPUT_SLOT_OFFSET).map_err(|_| cc::TRB_ERROR)?;
            let iep0 = Ctx32::read(mem, input + input_ep_offset(1)).map_err(|_| cc::TRB_ERROR)?;
            let port = slot_root_hub_port(&islot);
            if port == 0 || port as usize > NUM_PORTS {
                return Err(cc::PARAMETER_ERROR);
            }
            let out = dcbaa_entry(mem, self.dcbaap(), slot_id).map_err(|_| cc::TRB_ERROR)?;

            let address = if bsr {
                0
            } else {
                let a = self.next_address;
                self.next_address = self.next_address.wrapping_add(1);
                if self.next_address == 0 {
                    self.next_address = 1;
                }
                a
            };
            let new_state = if bsr { ss::DEFAULT } else { ss::ADDRESSED };

            let mut oslot = islot;
            set_slot_state(&mut oslot, new_state);
            set_slot_address(&mut oslot, address);
            oslot.write(mem, out).map_err(|_| cc::TRB_ERROR)?;

            let mut oep0 = iep0;
            {
                use super::context::ep_state as es;
                set_ep_state(&mut oep0, es::RUNNING);
            }
            oep0.write(mem, out + output_ep_offset(1))
                .map_err(|_| cc::TRB_ERROR)?;

            let (dq, dcs) = ep_tr_dequeue(&iep0);
            self.slots[slot_id as usize] = Some(SlotCtx {
                port,
                address,
                state: new_state,
                config_value: 0,
                ep0_ring: Some(RingWalker::new(dq, dcs)),
                eps: Default::default(),
            });
            Ok(slot_id)
        })();

        match result {
            Ok(_) => (cc::SUCCESS, slot_id),
            Err(code) => (code, slot_id),
        }
    }

    /// Configure Endpoint: apply the input context's Add / Drop flags. For every
    /// added data endpoint (DCI 2..=31) read its Endpoint Context (type + TR Dequeue),
    /// copy it into the output Device Context in the Running state, and stand up a live
    /// [`EpRing`] the worker walks on that endpoint's doorbell. Dropped endpoints are
    /// torn down. Deconfigure (DC bit) drops all data endpoints and returns the slot to
    /// the Addressed state.
    fn cmd_configure_endpoint(
        &mut self,
        mem: &GuestMemoryMmap,
        slot_id: u8,
        input: u64,
        deconfig: bool,
    ) -> Result<(), u32> {
        use super::context::slot_state as ss;
        let out = dcbaa_entry(mem, self.dcbaap(), slot_id).map_err(|_| cc::TRB_ERROR)?;

        if deconfig {
            let mut oslot = Ctx32::read(mem, out).map_err(|_| cc::TRB_ERROR)?;
            set_slot_state(&mut oslot, ss::ADDRESSED);
            oslot.write(mem, out).map_err(|_| cc::TRB_ERROR)?;
            if let Some(s) = self.slots.get_mut(slot_id as usize).and_then(|x| x.as_mut()) {
                s.state = ss::ADDRESSED;
                s.eps.clear();
            }
            return Ok(());
        }

        let ictl = Ctx32::read(mem, input).map_err(|_| cc::TRB_ERROR)?;
        let add = input_add_flags(&ictl);
        let drop = input_drop_flags(&ictl);

        // Slot context: if A0 is set the guest supplies an updated Slot Context (Context
        // Entries grown for the new endpoints); otherwise keep the current one. Either
        // way the slot ends Configured.
        let mut oslot = if add & 1 != 0 {
            Ctx32::read(mem, input + INPUT_SLOT_OFFSET).map_err(|_| cc::TRB_ERROR)?
        } else {
            Ctx32::read(mem, out).map_err(|_| cc::TRB_ERROR)?
        };
        set_slot_state(&mut oslot, ss::CONFIGURED);
        oslot.write(mem, out).map_err(|_| cc::TRB_ERROR)?;

        for dci in 2u8..=31 {
            let bit = 1u32 << dci;
            if drop & bit != 0 {
                if let Some(s) = self.slots.get_mut(slot_id as usize).and_then(|x| x.as_mut()) {
                    s.eps.remove(&dci);
                }
                let addr = out + output_ep_offset(dci);
                if let Ok(mut ep) = Ctx32::read(mem, addr) {
                    set_ep_state(&mut ep, ep_state::DISABLED);
                    let _ = ep.write(mem, addr);
                }
            }
            if add & bit != 0 {
                let iep = Ctx32::read(mem, input + input_ep_offset(dci)).map_err(|_| cc::TRB_ERROR)?;
                let etype = ep_type(&iep);
                let (dq, dcs) = ep_tr_dequeue(&iep);
                // DCI parity is the authoritative direction (IN = odd DCI); the type
                // field is kept for bookkeeping and agrees with it.
                let dir_in = dci & 1 == 1;
                let mut oep = iep;
                set_ep_state(&mut oep, ep_state::RUNNING);
                oep.write(mem, out + output_ep_offset(dci))
                    .map_err(|_| cc::TRB_ERROR)?;
                if let Some(s) = self.slots.get_mut(slot_id as usize).and_then(|x| x.as_mut()) {
                    // Re-adding an existing endpoint bumps its generation so a completion
                    // still in flight against the old ring is dropped.
                    let generation = s.eps.get(&dci).map(|e| e.generation.wrapping_add(1)).unwrap_or(0);
                    s.eps.insert(
                        dci,
                        EpRing {
                            ring: RingWalker::new(dq, dcs),
                            dir_in,
                            ep_type: etype,
                            state: ep_state::RUNNING,
                            generation,
                        },
                    );
                }
            }
        }

        if let Some(s) = self.slots.get_mut(slot_id as usize).and_then(|x| x.as_mut()) {
            s.state = ss::CONFIGURED;
        }
        Ok(())
    }

    /// Mutable access to a slot's data endpoint (DCI 2..=31), if it exists.
    fn ep_mut(&mut self, slot_id: u8, dci: u8) -> Option<&mut EpRing> {
        self.slots
            .get_mut(slot_id as usize)
            .and_then(|x| x.as_mut())
            .and_then(|s| s.eps.get_mut(&dci))
    }

    fn set_output_ep_state(&self, mem: &GuestMemoryMmap, slot_id: u8, dci: u8, state: u32) {
        let out = match dcbaa_entry(mem, self.dcbaap(), slot_id) {
            Ok(v) => v,
            Err(_) => return,
        };
        let addr = out + output_ep_offset(dci);
        if let Ok(mut ep) = Ctx32::read(mem, addr) {
            set_ep_state(&mut ep, state);
            let _ = ep.write(mem, addr);
        }
    }

    /// Publish the endpoint's TR Dequeue Pointer + Dequeue Cycle State into the
    /// output Endpoint Context (dwords 2-3), optionally updating the EP State.
    ///
    /// A real xHC keeps this field current so that, on a Stop Endpoint or a halt,
    /// the guest can read where the controller stopped (`xhci_get_hw_deq`). We had
    /// left it frozen at the Address-Device value, so after a control-transfer stall
    /// the guest's `xhci_move_dequeue_past_td` read a bogus dequeue and computed the
    /// wrong cycle state — leaving the ring wedged. Keeping it truthful fixes that.
    fn write_output_ep_dequeue(
        &self,
        mem: &GuestMemoryMmap,
        slot_id: u8,
        dci: u8,
        ptr: u64,
        dcs: bool,
        state: Option<u32>,
    ) {
        let out = match dcbaa_entry(mem, self.dcbaap(), slot_id) {
            Ok(v) => v,
            Err(_) => return,
        };
        let addr = out + output_ep_offset(dci);
        if let Ok(mut ep) = Ctx32::read(mem, addr) {
            super::context::set_ep_tr_dequeue(&mut ep, ptr, dcs);
            if let Some(s) = state {
                set_ep_state(&mut ep, s);
            }
            let _ = ep.write(mem, addr);
        }
    }

    // ---- EP0 control transfers ----------------------------------------------

    /// Walk the doorbelled slot's EP0 transfer ring, completing standard requests
    /// inline (from the model's descriptors) and forwarding class/vendor requests
    /// to the gadget with a deferred completion.
    fn collect_ep0_work(
        &mut self,
        mem: &GuestMemoryMmap,
        slot_id: u8,
        dev_arc: &Arc<Mutex<XhciDevice>>,
        deferred: &mut Vec<DeferredCall>,
    ) {
        // Resolve the slot's port + model.
        let model = match self.slots.get(slot_id as usize).and_then(|x| x.as_ref()) {
            Some(s) if s.port != 0 => self
                .port_models
                .get((s.port - 1) as usize)
                .and_then(|m| m.clone()),
            _ => None,
        };
        let Some(model) = model else {
            debug!("xhci: EP0 doorbell for unaddressed slot {slot_id}");
            return;
        };

        loop {
            let Some(mut walker) = self
                .slots
                .get(slot_id as usize)
                .and_then(|x| x.as_ref())
                .and_then(|s| s.ep0_ring)
            else {
                debug!("xhci: EP0 slot {slot_id} has no ring");
                return;
            };
            let saved = walker;
            debug!(
                "xhci: EP0 walk slot={slot_id} dq={:#x} ccs={}",
                walker.ptr(),
                walker.ccs()
            );
            match read_control_td(mem, &mut walker, slot_id) {
                Ok(TdRead::Complete(mut td)) => {
                    // Record where this TD begins (for a halt) and where the next TD
                    // begins (for a success), so post_control_result can keep the
                    // Endpoint Context TR Dequeue Pointer truthful.
                    td.events.td_start = (saved.ptr(), saved.ccs());
                    td.events.td_next = (walker.ptr(), walker.ccs());
                    debug!(
                        "xhci: EP0 TD setup={:02x?} segs={:?} status_trb={:#x}",
                        td.setup, td.events.data_segs, td.events.status_trb
                    );
                    // Consume the TD.
                    if let Some(s) = self
                        .slots
                        .get_mut(slot_id as usize)
                        .and_then(|x| x.as_mut())
                    {
                        s.ep0_ring = Some(walker);
                    }
                    let setup = SetupPacket::from_bytes(td.setup);
                    // Standard requests are answered by the controller itself.
                    if setup.kind() == 0 {
                        if let Some(outcome) = self.answer_standard(slot_id, &setup, &model) {
                            self.post_control_result(mem, &td.events, outcome);
                            continue;
                        }
                    }
                    // Forward class/vendor (and unhandled standard) to the gadget.
                    let data_out = if !setup.is_in() {
                        read_out_data(mem, &td.events.data_segs)
                    } else {
                        Vec::new()
                    };
                    let dev2 = dev_arc.clone();
                    let mem2 = mem.clone();
                    let events = td.events.clone();
                    let completion = Completion::new(move |outcome| {
                        let mut d = dev2.lock().unwrap();
                        d.post_control_result(&mem2, &events, outcome);
                    });
                    deferred.push(DeferredCall::Control(
                        model.clone(),
                        ControlTransfer::new(setup, data_out, completion),
                    ));
                }
                Ok(TdRead::Empty) => {
                    // Clean producer boundary: COMMIT the walker (it followed any
                    // Link TRBs to get here — restoring would undo that and desync
                    // the cycle state, wedging every transfer after a ring wrap).
                    if let Some(s) = self
                        .slots
                        .get_mut(slot_id as usize)
                        .and_then(|x| x.as_mut())
                    {
                        s.ep0_ring = Some(walker);
                    }
                    return;
                }
                Ok(TdRead::Incomplete) => {
                    // The guest is still writing this TD — restore and wait for the
                    // next doorbell.
                    if let Some(s) = self
                        .slots
                        .get_mut(slot_id as usize)
                        .and_then(|x| x.as_mut())
                    {
                        s.ep0_ring = Some(saved);
                    }
                    return;
                }
                Err(e) => {
                    warn!("xhci: EP0 ring error on slot {slot_id}: {e:?}");
                    return;
                }
            }
        }
    }

    /// Answer a standard EP0 request from the model's descriptors, or `None` to
    /// forward it to the gadget (which stalls unknown requests by default).
    fn answer_standard(
        &mut self,
        slot_id: u8,
        s: &SetupPacket,
        model: &Arc<dyn UsbDeviceModel>,
    ) -> Option<XferOutcome> {
        use super::context::slot_state as ss;
        if s.kind() != 0 {
            return None;
        }
        match s.request {
            0x06 => {
                // GET_DESCRIPTOR
                let desc = model.descriptors();
                let dt = (s.value >> 8) as u8;
                let idx = (s.value & 0xff) as usize;
                let full = match dt {
                    1 => desc.device,
                    2 => desc.configs.get(idx).cloned()?,
                    3 => desc.strings.get(idx).cloned()?,
                    // Device qualifier / other-speed config: a full-speed-only device
                    // must stall these (the guest uses the stall to conclude "no
                    // high-speed alternate"). Answered inline so the mock needs no code.
                    6 | 7 => return Some(XferOutcome::Stall),
                    // Class/vendor descriptor types (HID 0x21, Report 0x22, Physical
                    // 0x23, …) are the gadget's to serve — forward rather than stall.
                    _ => return None,
                };
                let n = (s.length as usize).min(full.len());
                Some(XferOutcome::In(full[..n].to_vec()))
            }
            0x05 => Some(XferOutcome::Ack), // SET_ADDRESS (xHCI-side; benign no-op)
            0x09 => {
                // SET_CONFIGURATION
                if let Some(sl) = self
                    .slots
                    .get_mut(slot_id as usize)
                    .and_then(|x| x.as_mut())
                {
                    sl.config_value = s.value as u8;
                    sl.state = ss::CONFIGURED;
                }
                Some(XferOutcome::Ack)
            }
            0x08 => {
                // GET_CONFIGURATION
                let cv = self
                    .slots
                    .get(slot_id as usize)
                    .and_then(|x| x.as_ref())
                    .map(|s| s.config_value)
                    .unwrap_or(0);
                Some(XferOutcome::In(vec![cv]))
            }
            0x00 => Some(XferOutcome::In(vec![0, 0])), // GET_STATUS (bus-powered, no wakeup)
            0x0b => Some(XferOutcome::Ack),            // SET_INTERFACE
            0x0a => Some(XferOutcome::In(vec![0])),    // GET_INTERFACE
            0x01 | 0x03 => Some(XferOutcome::Ack),     // CLEAR_FEATURE / SET_FEATURE
            _ => None,                                 // unknown standard: forward → stall
        }
    }

    /// Post the Transfer Event(s) for a completed control transfer.
    ///
    /// Called inline (under the lock) for controller-answered standard requests,
    /// and from a gadget completion closure (which re-locks) for forwarded ones.
    ///
    /// One Transfer Event is posted per stage TRB that carries a completion flag:
    /// the Data TRB on a short read (ISP — conveys the actual length via the residue)
    /// and always the Status TRB (IOC — the stage that retires the TD). Because the
    /// guest leaves the Chain bit clear, the stages are independent TDs, so a short
    /// Data stage does not cancel the Status stage — the guest waits for the Status
    /// completion and would otherwise hit its 5 s control-transfer timeout.
    pub(super) fn post_control_result(
        &mut self,
        mem: &GuestMemoryMmap,
        ev: &ControlEvents,
        outcome: XferOutcome,
    ) {
        // Halt vs. advance: a stall leaves the hardware dequeue *at* the failing TD
        // (HALTED), so the guest's Reset Endpoint + Set TR Dequeue can walk forward
        // from a truthful position; success advances it past the TD.
        let halted = matches!(outcome, XferOutcome::Stall);

        // Emit the Transfer Event(s). The Setup / Data / Status TRBs are *separate*
        // TDs (the guest leaves the Chain bit clear on control rings), so — matching
        // real hardware and QEMU — a short Data stage does **not** cancel the Status
        // stage: we post the Short Packet event *and* the Status completion, one event
        // per TRB that carries IOC/ISP. Posting only the Data event leaves the guest
        // waiting on the Status stage forever (a 5 s control-transfer timeout).
        match outcome {
            XferOutcome::In(bytes) => {
                let capacity: u32 = ev.data_segs.iter().map(|(_, l)| *l).sum();
                let written = scatter_in(mem, &ev.data_segs, &bytes);
                let residue = capacity.saturating_sub(written);
                if residue > 0 {
                    // Short read: report the residue on the Data TRB (this is where the
                    // guest learns the actual length), then complete at the Status TRB.
                    let dtrb = ev.data_trb.unwrap_or(ev.status_trb);
                    self.post_event(
                        mem,
                        transfer_event(
                            dtrb,
                            residue,
                            cc::SHORT_PACKET,
                            ev.slot_id,
                            ev.ep_id,
                            false,
                        ),
                    );
                }
                self.post_event(
                    mem,
                    transfer_event(ev.status_trb, 0, cc::SUCCESS, ev.slot_id, ev.ep_id, false),
                );
            }
            // OUT / no-data: the Status stage completes the TD.
            XferOutcome::Ack => {
                self.post_event(
                    mem,
                    transfer_event(ev.status_trb, 0, cc::SUCCESS, ev.slot_id, ev.ep_id, false),
                );
            }
            // Endpoint stall: one error event on the offending (data or status) TRB.
            XferOutcome::Stall => {
                let ptr = ev.data_trb.unwrap_or(ev.status_trb);
                self.post_event(
                    mem,
                    transfer_event(
                        ptr,
                        ev.wlength,
                        cc::STALL_ERROR,
                        ev.slot_id,
                        ev.ep_id,
                        false,
                    ),
                );
            }
        }

        // Keep the Endpoint Context TR Dequeue Pointer truthful for the guest's
        // stop/stall bookkeeping (only meaningful once the slot has a device context;
        // best-effort — a bad DCBAA/context pointer is simply skipped).
        if halted {
            let (p, c) = ev.td_start;
            self.write_output_ep_dequeue(
                mem,
                ev.slot_id,
                ev.ep_id,
                p,
                c,
                Some(super::context::ep_state::HALTED),
            );
        } else {
            let (p, c) = ev.td_next;
            self.write_output_ep_dequeue(mem, ev.slot_id, ev.ep_id, p, c, None);
        }
    }

    // ---- non-EP0 (interrupt/bulk) transfers ---------------------------------

    /// Walk a doorbelled data endpoint's transfer ring, forwarding each TD to the
    /// gadget as a [`Transfer`] with a deferred completion. An **IN** transfer the
    /// gadget can't satisfy yet is *held* (the TRBs stay outstanding, the walker
    /// stays committed past them); an **OUT** transfer carries the guest's bytes.
    ///
    /// A ringing doorbell on a Stopped endpoint restarts it (Running); a Halted
    /// endpoint is left alone until Reset Endpoint clears it.
    fn collect_ep_work(
        &mut self,
        mem: &GuestMemoryMmap,
        slot_id: u8,
        dci: u8,
        dev_arc: &Arc<Mutex<XhciDevice>>,
        deferred: &mut Vec<DeferredCall>,
    ) {
        let model = match self.slots.get(slot_id as usize).and_then(|x| x.as_ref()) {
            Some(s) if s.port != 0 => self
                .port_models
                .get((s.port - 1) as usize)
                .and_then(|m| m.clone()),
            _ => None,
        };
        let Some(model) = model else {
            debug!("xhci: data doorbell for unaddressed slot {slot_id}");
            return;
        };

        loop {
            // Snapshot the endpoint (ring walker, generation, direction, state).
            let Some((mut walker, generation, dir_in, state)) = self
                .slots
                .get(slot_id as usize)
                .and_then(|x| x.as_ref())
                .and_then(|s| s.eps.get(&dci))
                .map(|e| (e.ring, e.generation, e.dir_in, e.state))
            else {
                debug!("xhci: data doorbell slot {slot_id} dci {dci} has no endpoint");
                return;
            };
            if state == ep_state::HALTED {
                // A halted endpoint stays wedged until the guest issues Reset Endpoint.
                return;
            }
            if state == ep_state::STOPPED {
                // A doorbell restarts a stopped endpoint.
                if let Some(e) = self.ep_mut(slot_id, dci) {
                    e.state = ep_state::RUNNING;
                }
                self.set_output_ep_state(mem, slot_id, dci, ep_state::RUNNING);
            }

            let saved = walker;
            match read_data_td(mem, &mut walker) {
                Ok(DataTdRead::Complete {
                    segs,
                    immediate,
                    event_trb,
                }) => {
                    let td_next = (walker.ptr(), walker.ccs());
                    // Commit the walker past this TD (a held IN must not be re-fetched).
                    if let Some(e) = self.ep_mut(slot_id, dci) {
                        e.ring = walker;
                    }
                    let capacity: usize = segs.iter().map(|(_, l)| *l as usize).sum();
                    // OUT data comes from inline immediate bytes (IDT=1) when present, else gathered
                    // from the guest DMA segments. IN transfers never carry immediate data.
                    let (data_out, in_len) = if dir_in {
                        (Vec::new(), capacity)
                    } else if let Some(imm) = immediate {
                        (imm, 0)
                    } else {
                        (read_out_data(mem, &segs), 0)
                    };
                    let events = EpEvents {
                        slot_id,
                        ep_id: dci,
                        dir_in,
                        data_segs: segs,
                        event_trb,
                        generation,
                        td_next,
                    };
                    let dev2 = dev_arc.clone();
                    let mem2 = mem.clone();
                    // NOTE: the Completion must not be dropped while the controller lock
                    // is held (its Drop fires the stall closure, which re-locks). We hand
                    // it straight to the DeferredCall below with no intervening early
                    // return, exactly as the EP0 path does.
                    let completion = Completion::new(move |outcome| {
                        let mut d = dev2.lock().unwrap();
                        d.post_transfer_result(&mem2, &events, outcome);
                    });
                    deferred.push(DeferredCall::Transfer(
                        model.clone(),
                        EpAddr {
                            num: dci >> 1,
                            dir_in,
                        },
                        Transfer::new(data_out, in_len, completion),
                    ));
                }
                Ok(DataTdRead::Empty) => {
                    // Commit the walker (it followed Link TRBs to reach the boundary).
                    if let Some(e) = self.ep_mut(slot_id, dci) {
                        e.ring = walker;
                    }
                    return;
                }
                Ok(DataTdRead::Incomplete) => {
                    // The guest is still writing this TD — restore and await the next doorbell.
                    if let Some(e) = self.ep_mut(slot_id, dci) {
                        e.ring = saved;
                    }
                    return;
                }
                Err(e) => {
                    warn!("xhci: data ring error slot {slot_id} dci {dci}: {e:?}");
                    return;
                }
            }
        }
    }

    /// Post the Transfer Event for a completed non-EP0 transfer — called from a gadget
    /// completion closure (which re-locks the controller), possibly long after dispatch.
    ///
    /// The **generation guard** is the staleness fix: if the endpoint has been
    /// re-programmed (Stop / Reset Endpoint, Set TR Dequeue, re-Configure) or torn down
    /// (Disable Slot / Reset Device) since this transfer was dispatched, the captured
    /// generation no longer matches (or the endpoint is gone) and the completion is
    /// dropped — its TRBs are gone, so posting its event could corrupt a re-programmed
    /// ring or a re-used slot.
    pub(super) fn post_transfer_result(
        &mut self,
        mem: &GuestMemoryMmap,
        ev: &EpEvents,
        outcome: XferOutcome,
    ) {
        let live = self
            .slots
            .get(ev.slot_id as usize)
            .and_then(|x| x.as_ref())
            .and_then(|s| s.eps.get(&ev.ep_id))
            .map(|e| e.generation == ev.generation)
            .unwrap_or(false);
        if !live {
            debug!(
                "xhci: dropping stale completion slot {} dci {} (endpoint re-programmed)",
                ev.slot_id, ev.ep_id
            );
            return;
        }

        match outcome {
            XferOutcome::In(bytes) => {
                let capacity: u32 = ev.data_segs.iter().map(|(_, l)| *l).sum();
                let written = scatter_in(mem, &ev.data_segs, &bytes);
                let residue = capacity.saturating_sub(written);
                let code = if residue > 0 {
                    cc::SHORT_PACKET
                } else {
                    cc::SUCCESS
                };
                self.post_event(
                    mem,
                    transfer_event(ev.event_trb, residue, code, ev.slot_id, ev.ep_id, false),
                );
                let (p, c) = ev.td_next;
                self.write_output_ep_dequeue(mem, ev.slot_id, ev.ep_id, p, c, None);
            }
            XferOutcome::Ack => {
                // OUT / zero-length: success with no residue.
                let residue = if ev.dir_in {
                    ev.data_segs.iter().map(|(_, l)| *l).sum()
                } else {
                    0
                };
                let code = if residue > 0 {
                    cc::SHORT_PACKET
                } else {
                    cc::SUCCESS
                };
                self.post_event(
                    mem,
                    transfer_event(ev.event_trb, residue, code, ev.slot_id, ev.ep_id, false),
                );
                let (p, c) = ev.td_next;
                self.write_output_ep_dequeue(mem, ev.slot_id, ev.ep_id, p, c, None);
            }
            XferOutcome::Stall => {
                self.post_event(
                    mem,
                    transfer_event(ev.event_trb, 0, cc::STALL_ERROR, ev.slot_id, ev.ep_id, false),
                );
                // Halt the endpoint: leave the dequeue at the TD start and wait for the
                // guest's Reset Endpoint + Set TR Dequeue recovery.
                if let Some(e) = self.ep_mut(ev.slot_id, ev.ep_id) {
                    e.state = ep_state::HALTED;
                }
                self.set_output_ep_state(mem, ev.slot_id, ev.ep_id, ep_state::HALTED);
            }
        }
    }
}

// ---- pure ring/memory helpers -----------------------------------------------

/// The outcome of trying to read one control TD off the EP0 ring.
enum TdRead {
    /// A full Setup/[Data]/Status TD — process it; the walker is committed past it.
    Complete(ControlTd),
    /// The ring is empty at a TD boundary. The walker followed any Link TRBs to
    /// reach the producer boundary and must be **committed** (not restored) so the
    /// next doorbell resumes at the right position + cycle.
    Empty,
    /// A Setup was read but the Status stage isn't written yet (the guest is still
    /// filling the TD). Restore the walker and wait.
    Incomplete,
}

/// Read one control TD (Setup / [Data...] / Status) off the EP0 ring.
fn read_control_td(
    mem: &GuestMemoryMmap,
    walker: &mut RingWalker,
    slot_id: u8,
) -> Result<TdRead, RingError> {
    // Setup Stage.
    let (saddr, setup_trb) = match walker.next(mem)? {
        Some(v) => v,
        None => {
            // Ring empty at a producer boundary (cycle mismatch): the guest hasn't
            // enqueued the next TD yet. Normal — wait for the next doorbell.
            trace!(
                "xhci:   EP0 ring empty @{:#x} ccs={}",
                walker.ptr(),
                walker.ccs()
            );
            return Ok(TdRead::Empty);
        }
    };
    debug!(
        "xhci:   TD setup @{saddr:#x} ctrl={:#x} -> walker now dq={:#x} ccs={}",
        setup_trb.control,
        walker.ptr(),
        walker.ccs()
    );
    if setup_trb.trb_type() != trb_type::SETUP_STAGE {
        // A control ring must start each TD with a Setup Stage TRB. Treat as an
        // (empty) boundary and commit — never spin re-reading it.
        warn!(
            "xhci: EP0 TD did not start with a Setup Stage TRB (type {})",
            setup_trb.trb_type()
        );
        return Ok(TdRead::Empty);
    }
    let setup = setup_trb.setup_bytes();
    let wlength = u16::from_le_bytes([setup[6], setup[7]]) as u32;

    let mut data_segs: Vec<(u64, u32)> = Vec::new();
    let mut data_trb: Option<u64> = None;

    // Data / Status stages.
    loop {
        let (addr, trb) = match walker.next(mem)? {
            Some(v) => v,
            // No Status stage yet — the guest hasn't finished writing the TD.
            None => return Ok(TdRead::Incomplete),
        };
        debug!(
            "xhci:   TD trb @{addr:#x} type={} ctrl={:#x} (ioc={} isp={} ch={} ent={})",
            trb.trb_type(),
            trb.control,
            trb.control & CTRL_IOC != 0,
            trb.control & super::trb::CTRL_ISP != 0,
            trb.control & super::trb::CTRL_CHAIN != 0,
            trb.control & super::trb::CTRL_ENT != 0,
        );
        match trb.trb_type() {
            trb_type::DATA_STAGE | trb_type::NORMAL => {
                let len = trb.transfer_len();
                if len > 0 {
                    data_segs.push((trb.parameter, len));
                }
                data_trb = Some(addr);
                let _ = trb.control & CTRL_DIR_IN; // direction taken from SETUP
            }
            trb_type::STATUS_STAGE => {
                return Ok(TdRead::Complete(ControlTd {
                    setup,
                    events: ControlEvents {
                        slot_id,
                        ep_id: 1,
                        data_segs,
                        data_trb,
                        status_trb: addr,
                        status_ioc: trb.control & CTRL_IOC != 0,
                        wlength,
                        // Filled in by the caller (which holds the pre-read walker).
                        td_start: (0, false),
                        td_next: (0, false),
                    },
                }));
            }
            trb_type::EVENT_DATA => {
                // Treat an Event Data TRB as the transfer's completion point.
                return Ok(TdRead::Complete(ControlTd {
                    setup,
                    events: ControlEvents {
                        slot_id,
                        ep_id: 1,
                        data_segs,
                        data_trb,
                        status_trb: addr,
                        status_ioc: true,
                        wlength,
                        td_start: (0, false),
                        td_next: (0, false),
                    },
                }));
            }
            other => {
                warn!("xhci: unexpected TRB type {other} in EP0 TD");
                return Ok(TdRead::Empty);
            }
        }
    }
}

/// The outcome of trying to read one interrupt/bulk TD off a data endpoint ring.
enum DataTdRead {
    /// A full TD (chain-terminated): its data-buffer `segs`, any inline `immediate` data
    /// (IDT=1, OUT only — carried in the TRB parameter instead of guest memory), and the TRB the
    /// Transfer Event should point at (the last / IOC-carrying TRB). The walker is committed past it.
    Complete {
        segs: Vec<(u64, u32)>,
        immediate: Option<Vec<u8>>,
        event_trb: u64,
    },
    /// The ring is empty at a TD boundary (nothing queued). Commit the walker.
    Empty,
    /// A chained TD is only partially written (the guest is mid-enqueue). Restore + wait.
    Incomplete,
}

/// Read one interrupt/bulk TD off a data endpoint ring: consecutive Normal TRBs, each
/// chained to the next except the last (Chain bit clear). Collects each TRB's data
/// buffer `(addr, len)` and returns the address of the terminating TRB as the Transfer
/// Event target. Bounded/hostile-input-safe via the shared [`RingWalker`].
fn read_data_td(mem: &GuestMemoryMmap, walker: &mut RingWalker) -> Result<DataTdRead, RingError> {
    let mut segs: Vec<(u64, u32)> = Vec::new();
    // Immediate data (IDT=1) accumulated inline from the TRB parameter fields instead of DMA'd
    // from guest memory (xHCI §4.11.7). OUT direction only; a real guest uses a single small TRB.
    let mut immediate: Vec<u8> = Vec::new();
    let mut first = true;
    // Bound the work TRBs in one TD: a hostile ring whose Chain bit never clears would grow
    // `segs` / `immediate` without limit and hang the worker (see MAX_TD_TRBS).
    let mut work_trbs = 0usize;
    loop {
        let (addr, trb) = match walker.next(mem)? {
            Some(v) => v,
            // No (further) TRB published: an empty ring at the start, else a half-written
            // chained TD.
            None => {
                return Ok(if first {
                    DataTdRead::Empty
                } else {
                    DataTdRead::Incomplete
                });
            }
        };
        first = false;
        let ttype = trb.trb_type();
        // Count work TRBs (Normal / Event Data) and refuse a TD that never terminates.
        if matches!(ttype, trb_type::NORMAL | trb_type::EVENT_DATA) {
            work_trbs += 1;
            if work_trbs > MAX_TD_TRBS {
                return Err(RingError::TdTooLong);
            }
        }
        match ttype {
            trb_type::NORMAL => {
                let len = trb.transfer_len();
                if trb.control & CTRL_IDT != 0 {
                    // Immediate Data (xHCI §4.11.7): up to 8 data bytes live in the TRB's
                    // parameter field itself, not at a DMA pointer. Linux xhci-hcd uses this for
                    // small bulk-OUT transfers (e.g. the 3-byte elanmoc commands). `len` (≤ 8) is
                    // the byte count; the bytes are little-endian in the parameter dword pair.
                    let n = (len as usize).min(8);
                    immediate.extend_from_slice(&trb.parameter.to_le_bytes()[..n]);
                } else if len > 0 {
                    segs.push((trb.parameter, len));
                }
                if trb.control & CTRL_CHAIN == 0 {
                    return Ok(DataTdRead::Complete {
                        segs,
                        immediate: (!immediate.is_empty()).then_some(immediate),
                        event_trb: addr,
                    });
                }
            }
            trb_type::EVENT_DATA => {
                // An Event Data TRB terminates the TD; it carries the completion.
                if trb.control & CTRL_CHAIN == 0 {
                    return Ok(DataTdRead::Complete {
                        segs,
                        immediate: (!immediate.is_empty()).then_some(immediate),
                        event_trb: addr,
                    });
                }
            }
            other => {
                warn!("xhci: unexpected TRB type {other} in data TD; treating as boundary");
                return Ok(DataTdRead::Empty);
            }
        }
    }
}

/// Scatter `bytes` into the guest IN data-stage segments, returning how many
/// bytes were written (bounded by the segments' total capacity).
fn scatter_in(mem: &GuestMemoryMmap, segs: &[(u64, u32)], bytes: &[u8]) -> u32 {
    let mut off = 0usize;
    for (addr, len) in segs {
        if off >= bytes.len() {
            break;
        }
        let n = (*len as usize).min(bytes.len() - off);
        if mem
            .write_slice(&bytes[off..off + n], GuestAddress(*addr))
            .is_err()
        {
            break;
        }
        off += n;
    }
    off as u32
}

/// Gather the guest OUT data-stage bytes into a host buffer.
fn read_out_data(mem: &GuestMemoryMmap, segs: &[(u64, u32)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (addr, len) in segs {
        let mut buf = vec![0u8; *len as usize];
        if mem.read_slice(&mut buf, GuestAddress(*addr)).is_ok() {
            out.extend_from_slice(&buf);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::usb::mock::MockUsbDevice;
    use crate::usb::xhci::context::slot_state;
    use crate::usb::xhci::trb::{CTRL_CYCLE, CTRL_IDT};
    use crate::BusDevice;
    use utils::eventfd::EventFd;

    fn mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x100000)]).unwrap()
    }

    fn new_dev() -> Arc<Mutex<XhciDevice>> {
        Arc::new(Mutex::new(XhciDevice::new(
            EventFd::new(utils::eventfd::EFD_NONBLOCK).unwrap(),
        )))
    }

    /// Cold-plug the mock on port 1, stand up an event ring at `seg`, and build an
    /// addressed slot 1 whose EP0 ring dequeues at `ep0`.
    fn prime(d: &mut XhciDevice, ep0: u64, seg: u64, seg_size: u32) {
        d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
        d.event_ring = Some(EventRing::new(seg, seg_size));
        d.slots[1] = Some(SlotCtx {
            port: 1,
            address: 1,
            state: slot_state::ADDRESSED,
            config_value: 0,
            ep0_ring: Some(RingWalker::new(ep0, true)),
            eps: Default::default(),
        });
    }

    /// Lay a Setup/Data-IN/Status control TD at `ep0` (data buffer `buf`, buffer
    /// length `blen`, wLength `wlen`).
    fn lay_control_in(m: &GuestMemoryMmap, ep0: u64, buf: u64, blen: u32, wlen: u16) {
        let mut setup = [0x80u8, 0x06, 0x00, 0x01, 0x00, 0x00, 0, 0];
        setup[6] = wlen as u8;
        setup[7] = (wlen >> 8) as u8;
        Trb {
            parameter: u64::from_le_bytes(setup),
            status: 8,
            control: (trb_type::SETUP_STAGE << 10) | CTRL_CYCLE | CTRL_IDT | (3 << 16),
        }
        .write(m, ep0)
        .unwrap();
        Trb {
            parameter: buf,
            status: blen,
            control: (trb_type::DATA_STAGE << 10) | CTRL_CYCLE | CTRL_DIR_IN,
        }
        .write(m, ep0 + 0x10)
        .unwrap();
        Trb {
            parameter: 0,
            status: 0,
            control: (trb_type::STATUS_STAGE << 10) | CTRL_CYCLE | CTRL_IOC,
        }
        .write(m, ep0 + 0x20)
        .unwrap();
    }

    #[test]
    fn ep0_get_descriptor_returns_device_descriptor() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
        }
        lay_control_in(&m, 0x4000, 0x5000, 18, 18);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep0_work(&m, 1, &dev, &mut deferred);
        }
        assert!(deferred.is_empty(), "standard request answered inline");

        let mut got = [0u8; 18];
        m.read_slice(&mut got, GuestAddress(0x5000)).unwrap();
        assert_eq!(got[0], 18, "bLength");
        assert_eq!(u16::from_le_bytes([got[8], got[9]]), 0x1d6b, "idVendor");

        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
    }

    #[test]
    fn ep0_short_in_reports_residue() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
        }
        // 64-byte buffer, 18-byte device descriptor -> residue 46.
        lay_control_in(&m, 0x4000, 0x5000, 64, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep0_work(&m, 1, &dev, &mut deferred);
        }
        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(ev.status >> 24, cc::SHORT_PACKET, "short packet code");
        assert_eq!(ev.status & 0xff_ffff, 46, "residue = 64 - 18");
    }

    /// A short control-IN must post BOTH a Short Packet event (on the Data TRB) and
    /// a Success event (on the Status TRB). The stages are independent TDs (Chain
    /// bit clear), so omitting the Status event leaves the guest waiting for the
    /// Status stage until its 5 s control-transfer timeout. RED before the two-event
    /// fix (only the Data event was posted).
    #[test]
    fn ep0_short_in_posts_data_then_status_events() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
        }
        lay_control_in(&m, 0x4000, 0x5000, 64, 64); // 18-byte descriptor into 64-byte buf

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep0_work(&m, 1, &dev, &mut deferred);
        }

        // Event 0: Short Packet on the Data TRB (0x4010), residue 46.
        let e0 = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(e0.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(e0.status >> 24, cc::SHORT_PACKET);
        assert_eq!(
            e0.parameter, 0x4010,
            "short-packet event points at the Data TRB"
        );
        // Event 1: Success on the Status TRB (0x4020) — the stage that retires the TD.
        let e1 = Trb::read(&m, 0x3010).unwrap();
        assert_eq!(e1.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(e1.status >> 24, cc::SUCCESS);
        assert_eq!(
            e1.parameter, 0x4020,
            "completion event points at the Status TRB"
        );
    }

    /// Build DCBAA[1] -> output device context at `out`, so tests can read back the
    /// Endpoint Context the controller maintains. Returns the EP0 (DCI 1) offset.
    fn prime_with_ctx(d: &mut XhciDevice, m: &GuestMemoryMmap, ep0: u64, dcbaap: u64, out: u64) {
        prime(d, ep0, 0x3000, 16);
        m.write_obj::<u64>(out, GuestAddress(dcbaap + 8)).unwrap();
        d.set_dcbaap_for_test(dcbaap);
        // As Address Device would: an EP0 context in the Running state.
        let mut oep0 = Ctx32::default();
        crate::usb::xhci::context::set_ep_state(
            &mut oep0,
            crate::usb::xhci::context::ep_state::RUNNING,
        );
        oep0.write(m, out + output_ep_offset(1)).unwrap();
    }

    /// On a successful control transfer, the output Endpoint Context TR Dequeue
    /// Pointer must advance to the TRB *after* the TD (so a later Stop Endpoint /
    /// stall recovery reads a truthful `hw_deq`).
    #[test]
    fn ep0_success_advances_output_ep_dequeue() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime_with_ctx(&mut d, &m, 0x4000, 0x9000, 0x8000);
        }
        lay_control_in(&m, 0x4000, 0x5000, 18, 18); // full 18-byte read

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep0_work(&m, 1, &dev, &mut deferred);
        }
        let oep0 = Ctx32::read(&m, 0x8000 + output_ep_offset(1)).unwrap();
        let (ptr, _dcs) = crate::usb::xhci::context::ep_tr_dequeue(&oep0);
        assert_eq!(
            ptr, 0x4030,
            "dequeue advanced past the Setup/Data/Status TD"
        );
        assert_eq!(
            oep0.0[0] & 0x7,
            crate::usb::xhci::context::ep_state::RUNNING,
            "endpoint stays Running after success"
        );
    }

    /// On a stall, the output Endpoint Context TR Dequeue Pointer must be left *at*
    /// the failing TD's first TRB with the endpoint HALTED — this is the `hw_deq`
    /// the guest's `xhci_move_dequeue_past_td` reads to reposition the ring. Leaving
    /// it stale (or advanced past the TD) made the guest compute the wrong cycle
    /// state and wedged the ring after a device-qualifier stall. RED before the fix.
    #[test]
    fn ep0_stall_holds_output_ep_dequeue_and_halts() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime_with_ctx(&mut d, &m, 0x4000, 0x9000, 0x8000);
        }
        // GET_DESCRIPTOR(DEVICE_QUALIFIER) — a full-speed device stalls it.
        Trb {
            parameter: u64::from_le_bytes([0x80, 0x06, 0x00, 0x06, 0x00, 0x00, 0x0a, 0x00]),
            status: 8,
            control: (trb_type::SETUP_STAGE << 10) | CTRL_CYCLE | CTRL_IDT | (3 << 16),
        }
        .write(&m, 0x4000)
        .unwrap();
        Trb {
            parameter: 0x5000,
            status: 10,
            control: (trb_type::DATA_STAGE << 10) | CTRL_CYCLE | CTRL_DIR_IN,
        }
        .write(&m, 0x4010)
        .unwrap();
        Trb {
            parameter: 0,
            status: 0,
            control: (trb_type::STATUS_STAGE << 10) | CTRL_CYCLE | CTRL_IOC,
        }
        .write(&m, 0x4020)
        .unwrap();

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep0_work(&m, 1, &dev, &mut deferred);
        }

        // A Stall Error event was posted.
        let e0 = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(e0.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(e0.status >> 24, cc::STALL_ERROR);

        // hw_deq is left AT the TD start (the Setup TRB), endpoint HALTED.
        let oep0 = Ctx32::read(&m, 0x8000 + output_ep_offset(1)).unwrap();
        let (ptr, _dcs) = crate::usb::xhci::context::ep_tr_dequeue(&oep0);
        assert_eq!(ptr, 0x4000, "dequeue held at the failing TD's first TRB");
        assert_eq!(
            oep0.0[0] & 0x7,
            crate::usb::xhci::context::ep_state::HALTED,
            "endpoint halted on stall"
        );
    }

    #[test]
    fn enable_slot_then_address_device_binds_port() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
            d.event_ring = Some(EventRing::new(0x3000, 16));
        }
        // Enable Slot.
        let mut deferred = Vec::new();
        let (code, slot) = {
            let mut d = dev.lock().unwrap();
            let trb = Trb {
                parameter: 0,
                status: 0,
                control: trb_type::ENABLE_SLOT << 10,
            };
            d.run_command(&m, &trb, &mut deferred)
        };
        assert_eq!(code, cc::SUCCESS);
        assert_eq!(slot, 1, "first slot handed out");

        // Build an input context at 0x6000: add A0|A1, slot root-hub port 1,
        // EP0 TR dequeue 0x7001 (DCS=1). DCBAA[1] -> output ctx 0x8000.
        let dcbaap = 0x9000u64;
        m.write_obj::<u64>(0x8000, GuestAddress(dcbaap + 8))
            .unwrap();
        Ctx32([0, 0b11, 0, 0, 0, 0, 0, 0])
            .write(&m, 0x6000)
            .unwrap(); // input control
        Ctx32([0, 1 << 16, 0, 0, 0, 0, 0, 0])
            .write(&m, 0x6000 + INPUT_SLOT_OFFSET)
            .unwrap(); // slot: port 1
        let mut iep0 = Ctx32::default();
        iep0.0[2] = 0x7001; // TR dequeue 0x7000, DCS=1
        iep0.write(&m, 0x6000 + input_ep_offset(1)).unwrap();

        let (code, _) = {
            let mut d = dev.lock().unwrap();
            // Point DCBAAP at our table.
            d.set_dcbaap_for_test(dcbaap);
            let trb = Trb {
                parameter: 0x6000,
                status: 0,
                control: (trb_type::ADDRESS_DEVICE << 10) | (1 << 24),
            };
            d.run_command(&m, &trb, &mut deferred)
        };
        assert_eq!(code, cc::SUCCESS, "address device");

        // The output slot context is Addressed with a nonzero address, and EP0 is Running.
        let oslot = Ctx32::read(&m, 0x8000).unwrap();
        assert_eq!(slot_state(&oslot), slot_state::ADDRESSED);
        assert_ne!(oslot.0[3] & 0xff, 0, "device address assigned");
        let oep0 = Ctx32::read(&m, 0x8000 + output_ep_offset(1)).unwrap();
        assert_eq!(
            oep0.0[0] & 0x7,
            crate::usb::xhci::context::ep_state::RUNNING
        );
    }

    #[test]
    fn unimplemented_command_returns_trb_error() {
        let m = mem();
        let dev = new_dev();
        let mut deferred = Vec::new();
        let mut d = dev.lock().unwrap();
        d.event_ring = Some(EventRing::new(0x3000, 16));
        // TRB type 40 is not a command we implement.
        let trb = Trb {
            parameter: 0,
            status: 0,
            control: 40 << 10,
        };
        let (code, _) = d.run_command(&m, &trb, &mut deferred);
        assert_eq!(code, cc::TRB_ERROR);
    }

    // ---- B2: non-EP0 (interrupt/bulk) data flow ----------------------------

    use super::super::context::ep_state;
    use crate::usb::model::Transfer;

    /// Insert a live data endpoint (DCI `dci`) whose ring dequeues at `ring`.
    fn add_ep(d: &mut XhciDevice, dci: u8, ring: u64, dir_in: bool) {
        let s = d.slots[1].as_mut().unwrap();
        s.eps.insert(
            dci,
            EpRing {
                ring: RingWalker::new(ring, true),
                dir_in,
                ep_type: if dir_in { 7 } else { 3 },
                state: ep_state::RUNNING,
                generation: 0,
            },
        );
    }

    /// Lay a single Normal TRB (one-TRB TD, Chain clear, IOC set) at `addr` pointing at
    /// data buffer `buf` of length `len`.
    fn put_normal(m: &GuestMemoryMmap, addr: u64, buf: u64, len: u32) {
        Trb {
            parameter: buf,
            status: len,
            control: (trb_type::NORMAL << 10) | CTRL_CYCLE | CTRL_IOC,
        }
        .write(m, addr)
        .unwrap();
    }

    /// Extract the single dispatched Transfer from a deferred batch.
    fn take_transfer(deferred: Vec<DeferredCall>) -> Transfer {
        for c in deferred {
            if let DeferredCall::Transfer(_, _, x) = c {
                return x;
            }
        }
        panic!("no transfer was dispatched");
    }

    /// Configure Endpoint stands up a live per-endpoint transfer ring from the input
    /// Endpoint Context and writes the output context Running.
    #[test]
    fn configure_endpoint_creates_data_ring() {
        let m = mem();
        let dev = new_dev();
        let dcbaap = 0x9000u64;
        let out = 0x8000u64;
        m.write_obj::<u64>(out, GuestAddress(dcbaap + 8)).unwrap();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            d.set_dcbaap_for_test(dcbaap);
        }
        // Input context at 0x6000: add A0 (slot) + A3 (DCI 3, interrupt IN).
        let input = 0x6000u64;
        Ctx32([0, (1 << 0) | (1 << 3), 0, 0, 0, 0, 0, 0])
            .write(&m, input)
            .unwrap();
        Ctx32::default().write(&m, input + INPUT_SLOT_OFFSET).unwrap();
        // EP context DCI 3: type 7 (interrupt IN), TR dequeue 0xA001 (DCS=1).
        let mut iep = Ctx32::default();
        iep.0[1] = 7 << 3;
        iep.0[2] = 0xA001;
        iep.write(&m, input + input_ep_offset(3)).unwrap();

        let mut deferred = Vec::new();
        let (code, _) = {
            let mut d = dev.lock().unwrap();
            let trb = Trb {
                parameter: input,
                status: 0,
                control: (trb_type::CONFIGURE_ENDPOINT << 10) | (1 << 24),
            };
            d.run_command(&m, &trb, &mut deferred)
        };
        assert_eq!(code, cc::SUCCESS);

        let d = dev.lock().unwrap();
        let ep = d.slots[1].as_ref().unwrap().eps.get(&3).expect("ep created");
        assert_eq!(ep.ring.ptr(), 0xA000, "ring dequeue from input context");
        assert!(ep.dir_in, "DCI 3 is IN");
        assert_eq!(ep.state, ep_state::RUNNING);
        // Output Endpoint Context reflects Running.
        let oep = Ctx32::read(&m, out + output_ep_offset(3)).unwrap();
        assert_eq!(oep.0[0] & 0x7, ep_state::RUNNING);
    }

    /// A doorbell on a data endpoint walks the ring and dispatches one Transfer per TD.
    #[test]
    fn data_doorbell_dispatches_one_transfer_per_td() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 3, 0xA000, true);
        }
        // Two IN TDs (each a single Normal TRB, 64-byte buffer).
        put_normal(&m, 0xA000, 0x5000, 64);
        put_normal(&m, 0xA010, 0x5100, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 3, &dev, &mut deferred);
        }
        let n = deferred
            .iter()
            .filter(|c| matches!(c, DeferredCall::Transfer(..)))
            .count();
        assert_eq!(n, 2, "one Transfer dispatched per TD");
    }

    /// A held interrupt-IN transfer, completed later, scatters its data into the guest
    /// buffer and posts a SUCCESS Transfer Event on the endpoint.
    #[test]
    fn held_in_completes_and_posts_event() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 3, 0xA000, true);
        }
        put_normal(&m, 0xA000, 0x5000, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 3, &dev, &mut deferred);
        }
        let xfer = take_transfer(deferred);
        // Complete it now (as a gadget would, from any thread): 64 bytes.
        let payload: Vec<u8> = (0..64).collect();
        xfer.complete_in(payload.clone());

        // Guest buffer got the bytes.
        let mut got = [0u8; 64];
        m.read_slice(&mut got, GuestAddress(0x5000)).unwrap();
        assert_eq!(&got[..], &payload[..]);
        // A SUCCESS Transfer Event on the Normal TRB, DCI 3.
        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(ev.status >> 24, cc::SUCCESS);
        assert_eq!(ev.parameter, 0xA000, "event points at the data TRB");
        assert_eq!((ev.control >> 16) & 0x1f, 3, "endpoint id = DCI 3");
    }

    /// A short interrupt-IN read reports the residue with a SHORT_PACKET completion.
    #[test]
    fn short_packet_residue_on_interrupt_in() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 3, 0xA000, true);
        }
        put_normal(&m, 0xA000, 0x5000, 64); // buffer 64

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 3, &dev, &mut deferred);
        }
        take_transfer(deferred).complete_in(vec![0xEE; 20]); // only 20 bytes

        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(ev.status >> 24, cc::SHORT_PACKET);
        assert_eq!(ev.status & 0xff_ffff, 44, "residue = 64 - 20");
    }

    /// The staleness guard: a completion firing after Set TR Dequeue re-programmed the
    /// endpoint is dropped — no event posted, the re-programmed ring untouched.
    #[test]
    fn stale_completion_after_set_tr_dequeue_is_dropped() {
        let m = mem();
        let dev = new_dev();
        let dcbaap = 0x9000u64;
        m.write_obj::<u64>(0x8000, GuestAddress(dcbaap + 8)).unwrap();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            d.set_dcbaap_for_test(dcbaap);
            add_ep(&mut d, 3, 0xA000, true);
        }
        put_normal(&m, 0xA000, 0x5000, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 3, &dev, &mut deferred);
        }
        let xfer = take_transfer(deferred);

        // Guest re-programs the ring with Set TR Dequeue (bumps the generation).
        {
            let mut cmd_deferred = Vec::new();
            let mut d = dev.lock().unwrap();
            let trb = Trb {
                parameter: 0xB001, // new dequeue 0xB000, DCS=1
                status: 0,
                control: (trb_type::SET_TR_DEQUEUE << 10) | (3 << 16) | (1 << 24),
            };
            d.run_command(&m, &trb, &mut cmd_deferred);
        }

        // The late completion must be dropped: no Transfer Event, and the ring stays at
        // the re-programmed dequeue.
        xfer.complete_in(vec![0u8; 64]);
        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_ne!(
            ev.trb_type(),
            trb_type::TRANSFER_EVENT,
            "stale completion posted no event"
        );
        let d = dev.lock().unwrap();
        assert_eq!(
            d.slots[1].as_ref().unwrap().eps.get(&3).unwrap().ring.ptr(),
            0xB000,
            "ring left at the re-programmed dequeue"
        );
    }

    /// Stop Endpoint quiesces: it marks the endpoint Stopped and bumps the generation so
    /// a still-held IN transfer's later completion is dropped (the command completes only
    /// after the endpoint can no longer post onto the ring).
    #[test]
    fn stop_endpoint_quiesces_outstanding_transfer() {
        let m = mem();
        let dev = new_dev();
        let dcbaap = 0x9000u64;
        m.write_obj::<u64>(0x8000, GuestAddress(dcbaap + 8)).unwrap();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            d.set_dcbaap_for_test(dcbaap);
            add_ep(&mut d, 3, 0xA000, true);
        }
        put_normal(&m, 0xA000, 0x5000, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 3, &dev, &mut deferred);
        }
        let xfer = take_transfer(deferred);

        // Stop Endpoint (DCI 3).
        let (code, _) = {
            let mut cmd_deferred = Vec::new();
            let mut d = dev.lock().unwrap();
            let trb = Trb {
                parameter: 0,
                status: 0,
                control: (trb_type::STOP_ENDPOINT << 10) | (3 << 16) | (1 << 24),
            };
            d.run_command(&m, &trb, &mut cmd_deferred)
        };
        assert_eq!(code, cc::SUCCESS);
        {
            let d = dev.lock().unwrap();
            assert_eq!(
                d.slots[1].as_ref().unwrap().eps.get(&3).unwrap().state,
                ep_state::STOPPED,
                "endpoint marked Stopped"
            );
        }
        // The late completion is dropped — no event on the ring.
        xfer.complete_in(vec![0u8; 64]);
        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_ne!(ev.trb_type(), trb_type::TRANSFER_EVENT);
    }

    /// An interrupt-OUT transfer delivers the guest's bytes to the gadget (via the
    /// Transfer's data_out), and an Ack posts a SUCCESS event.
    #[test]
    fn interrupt_out_delivers_bytes_and_acks() {
        let m = mem();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 2, 0xA000, false); // DCI 2 = interrupt OUT
        }
        // Guest writes 64 bytes into 0x5000, then a Normal OUT TRB referencing it.
        let payload: Vec<u8> = (0..64u8).map(|b| b ^ 0x5a).collect();
        m.write_slice(&payload, GuestAddress(0x5000)).unwrap();
        put_normal(&m, 0xA000, 0x5000, 64);

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 2, &dev, &mut deferred);
        }
        let xfer = take_transfer(deferred);
        assert_eq!(xfer.data_out(), &payload[..], "OUT bytes delivered to gadget");
        xfer.ack();

        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(ev.status >> 24, cc::SUCCESS);
        assert_eq!((ev.control >> 16) & 0x1f, 2, "endpoint id = DCI 2");
    }

    /// Immediate Data (IDT=1) OUT: the guest carries a small payload inline in the TRB parameter
    /// field, not at a DMA pointer (xHCI §4.11.7 — Linux xhci-hcd does this for small bulk-OUT
    /// commands, e.g. the 3-byte elanmoc fingerprint-reader commands). The gadget must receive the
    /// immediate bytes, NOT bytes read from `parameter` misinterpreted as an address. This is the
    /// regression guard for the fingerprint reader's open sequence (was read as garbage → stall).
    #[test]
    fn immediate_data_out_delivers_inline_bytes_not_a_dma_read() {
        // A 32 MiB guest so the address the immediate bytes would be *misread* as
        // (`0x00ff_0040`, ~16 MiB) is in range and can be poisoned — proving the delivered payload
        // comes from the TRB parameter's immediate bytes, not a DMA read of that address.
        let m = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x2000000)]).unwrap();
        let dev = new_dev();
        {
            let mut d = dev.lock().unwrap();
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 2, 0xA000, false); // DCI 2 = bulk/interrupt OUT
        }
        // The 3-byte elanmoc cal-status command `40 ff 00`, packed little-endian into the TRB
        // parameter (IDT=1). Read as an address this is `0x0000_0000_00ff_0040`.
        let cmd = [0x40u8, 0xff, 0x00];
        let mut param = [0u8; 8];
        param[..3].copy_from_slice(&cmd);
        let param = u64::from_le_bytes(param);
        // Poison that misread address: a regressed reader would DMA these bytes and the assert
        // below would see `de ad be` instead of the command.
        m.write_slice(&[0xde, 0xad, 0xbe], GuestAddress(param)).unwrap();
        Trb {
            parameter: param,
            status: cmd.len() as u32, // transfer length = 3 (≤ 8)
            control: (trb_type::NORMAL << 10) | CTRL_CYCLE | CTRL_IOC | CTRL_IDT,
        }
        .write(&m, 0xA000)
        .unwrap();

        let mut deferred = Vec::new();
        {
            let mut d = dev.lock().unwrap();
            d.collect_ep_work(&m, 1, 2, &dev, &mut deferred);
        }
        let xfer = take_transfer(deferred);
        assert_eq!(
            xfer.data_out(),
            &cmd[..],
            "IDT immediate bytes delivered verbatim, not a DMA read of the parameter"
        );
        xfer.ack();

        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::TRANSFER_EVENT);
        assert_eq!(ev.status >> 24, cc::SUCCESS, "OUT ack, no residue");
    }

    /// A hostile TD whose Chain bit never clears must not grow the collected buffers without
    /// bound: `read_data_td` refuses it after `MAX_TD_TRBS` work TRBs (a DoS guard).
    #[test]
    fn a_td_that_never_terminates_is_refused() {
        let m = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x2000000)]).unwrap();
        // Lay MAX_TD_TRBS + 1 Normal TRBs, all Chain-set (never terminating), contiguously.
        let ring = 0x10000u64;
        for i in 0..(MAX_TD_TRBS as u64 + 1) {
            Trb {
                parameter: 0x5000,
                status: 1,
                control: (trb_type::NORMAL << 10) | CTRL_CYCLE | CTRL_CHAIN,
            }
            .write(&m, ring + i * 16)
            .unwrap();
        }
        let mut walker = RingWalker::new(ring, true);
        assert!(matches!(
            read_data_td(&m, &mut walker),
            Err(RingError::TdTooLong)
        ));
    }

    // ---- suspend/resume behaviour (docs/design/usb-xhci-snapshot/) -----------------------

    /// The `USBCMD.RS` 0→1 edge fires on every **resume** too (`xhci_resume` step 4), not just at
    /// boot. The port scan it triggers must therefore be idempotent: re-latching `CCS | CSC` and
    /// forcing PLS back to Polling made the guest's `xhci_bus_resume` see a port no longer in U3
    /// (so it abandoned the resume) *and* made its hub thread treat the still-present device as a
    /// re-plug — devices blinked out and back on every suspend/resume cycle.
    #[test]
    fn run_edge_does_not_relatch_an_already_connected_port() {
        let m = mem();
        let dev = new_dev();
        let mut d = dev.lock().unwrap();
        d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
        d.event_ring = Some(EventRing::new(0x3000, 16));

        // First run edge: the cold-plug is latched and announced (the boot path, unchanged).
        d.work.run_started = true;
        d.run_worker_pass(&m, &dev);
        assert!(
            d.port_connected(0),
            "cold plug latched on the first run edge"
        );
        let first_event = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(first_event.trb_type(), trb_type::PORT_STATUS_CHANGE);

        // The guest enumerates: it clears CSC and the port ends up in U0.
        d.clear_port_change_for_test(0);
        let before = d.portsc_for_test(0);

        // A resume's run edge must leave that port completely alone — no re-latched CSC, no PLS
        // rewrite, and no second Port Status Change Event.
        d.work.run_started = true;
        d.run_worker_pass(&m, &dev);
        assert_eq!(
            d.portsc_for_test(0),
            before,
            "PORTSC must be untouched by a resume's run edge"
        );
        let second = Trb::read(&m, 0x3010).unwrap();
        assert_eq!(
            second.trb_type(),
            0,
            "no second port event may be posted (slot is still an empty TRB)"
        );
    }

    /// A command doorbell with no command ring programmed must be ignored, not turned into a walk
    /// of guest address 0 (`BadAccess(0)` forever). Reachable whenever a guest rings DB0 while
    /// `CRCR` is zero — e.g. the tail of a dead-HCD teardown.
    #[test]
    fn command_doorbell_with_no_ring_is_ignored() {
        let m = mem();
        let dev = new_dev();
        let mut d = dev.lock().unwrap();
        d.event_ring = Some(EventRing::new(0x3000, 16));
        d.work.cmd_doorbell = true;
        d.run_worker_pass(&m, &dev);
        // Nothing posted: the event ring's first slot is still an untouched TRB.
        assert_eq!(
            Trb::read(&m, 0x3000).unwrap(),
            Trb::default(),
            "a doorbell with crcr == 0 must post nothing at all"
        );
    }

    /// The snapshot invariant: a controller rebuilt from `save_state` must be
    /// **indistinguishable** from the one it was captured from. Uses a genuinely lived-in
    /// controller (enumerated slot, configured data endpoints, an advanced event-ring producer,
    /// a mid-ring command walker) so a forgotten field shows up as a difference.
    #[test]
    fn save_state_round_trips_a_lived_in_controller() {
        let m = mem();
        let dev = new_dev();
        let captured = {
            let mut d = dev.lock().unwrap();
            d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
            // Bring the controller up the way a guest does: reset (clears CNR), program the
            // rings, run.
            // Every register gets a DISTINCTIVE value, so that dropping any one of them from
            // the carry shows up as a round-trip difference rather than matching a default.
            d.write(0, 0x20, &2u32.to_le_bytes()); // USBCMD.HCRST
            d.write(0, 0x20 + 0x18, &(0x4000u64 | 1).to_le_bytes()); // CRCR
            d.write(0, 0x20 + 0x30, &0x6000u64.to_le_bytes()); // DCBAAP
            d.write(0, 0x20 + 0x14, &0x5u32.to_le_bytes()); // DNCTRL
            d.write(0, 0x20 + 0x38, &0x8u32.to_le_bytes()); // CONFIG (MaxSlotsEn)
            d.write(0, 0x1000 + 0x20 + 0x10, &0x2000u64.to_le_bytes()); // ERSTBA
            d.write(0, 0x1000 + 0x20 + 0x08, &16u32.to_le_bytes()); // ERSTSZ
            d.write(0, 0x1000 + 0x20 + 0x04, &0x40u32.to_le_bytes()); // IMOD
            d.write(0, 0x1000 + 0x20 + 0x00, &0x2u32.to_le_bytes()); // IMAN.IE
            d.write(0, 0x1000 + 0x20 + 0x18, &0x3010u64.to_le_bytes()); // ERDP, mid-ring
            d.next_address = 3;
            d.event_ring = Some(EventRing::new(0x3000, 16));
            d.work.run_started = true;
            d.run_worker_pass(&m, &dev);
            // An addressed, configured slot with one IN and one OUT data endpoint.
            prime(&mut d, 0x4000, 0x3000, 16);
            add_ep(&mut d, 3, 0x7000, true);
            add_ep(&mut d, 2, 0x8000, false);
            if let Some(s) = d.slots[1].as_mut() {
                s.state = slot_state::CONFIGURED;
                s.config_value = 1;
                // A non-default generation and a stopped endpoint, so a restore that resets
                // either is visible.
                let ep = s.eps.get_mut(&3).unwrap();
                ep.generation = 7;
                ep.state = ep_state::STOPPED;
            }
            // Advance the command walker off its base, so a "rebuild from CRCR" would differ...
            d.cmd_ring = Some(RingWalker::new(0x4020, true));
            // ...and the event-ring producer off ITS base (prime() reset it), so a "rebuild from
            // ERSTBA" would differ too.
            let ev = super::super::trb::port_status_change_event(1);
            for _ in 0..2 {
                d.event_ring
                    .as_mut()
                    .unwrap()
                    .enqueue(&m, ev, 0x3000)
                    .unwrap();
            }
            // Finally the state a running guest leaves USBCMD in.
            d.write(0, 0x20, &(1u32 | 4).to_le_bytes()); // RS | INTE
            d.save_state()
        };
        assert!(!captured.slots.is_empty(), "the capture has a live slot");
        assert_eq!(
            captured.slots[0].eps.len(),
            2,
            "both data endpoints carried"
        );
        let er = captured.event_ring.expect("the producer is carried");
        assert_eq!(er.enqueue_idx, 2, "the producer sits mid-segment");
        assert_eq!(captured.ports[0].model_id, Some((0x1d6b, 0x0f10)));

        // A brand-new controller with the same gadget cold-plugged — the restore worker's shape.
        let fresh = new_dev();
        let restored = {
            let mut d = fresh.lock().unwrap();
            d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
            d.restore_state(&captured);
            d.save_state()
        };
        assert_eq!(
            restored, captured,
            "a restored controller must capture identically to the original"
        );
    }

    /// If the gadget on a port is not the one that was there at suspend (a cold-plug that failed
    /// to build shifts the rest onto different ports), the restore must NOT bind the slot to the
    /// wrong device: it presents an unplug and drops the slot.
    #[test]
    fn restore_reconciles_a_port_whose_gadget_changed() {
        let m = mem();
        let dev = new_dev();
        let captured = {
            let mut d = dev.lock().unwrap();
            d.port_models[0] = Some(Arc::new(MockUsbDevice::new()));
            d.event_ring = Some(EventRing::new(0x3000, 16));
            d.work.run_started = true;
            d.run_worker_pass(&m, &dev);
            prime(&mut d, 0x4000, 0x3000, 16);
            d.save_state()
        };
        assert!(captured.ports[0].model_id.is_some());
        assert_eq!(captured.slots.len(), 1);

        // Restore into a worker where that port came up EMPTY.
        let fresh = new_dev();
        let mut d = fresh.lock().unwrap();
        d.restore_state(&captured);
        assert!(
            !d.port_connected(0),
            "the vanished device must not be reported as still connected"
        );
        assert_eq!(
            d.save_state().slots.len(),
            0,
            "the slot bound to the vanished device is dropped"
        );
        assert_eq!(
            d.work.port_events,
            vec![1],
            "the guest is told the port changed"
        );
    }

    /// `CNR` must never survive into a restored controller: `xhci_resume` handshakes it for ten
    /// seconds and then declares the HCD dead for good (nothing on a light-resume path issues the
    /// `HCRST` that would clear it).
    #[test]
    fn restore_never_leaves_the_controller_not_ready() {
        let dev = new_dev();
        let mut state = dev.lock().unwrap().save_state();
        state.usbsts |= 1 << 11; // CNR, as a fresh controller reports at power-on
        let fresh = new_dev();
        let mut d = fresh.lock().unwrap();
        d.restore_state(&state);
        assert_eq!(
            d.save_state().usbsts & (1 << 11),
            0,
            "the restored controller must report itself ready"
        );
    }

    /// A `CRCR.CA` (Command Abort) write must be acknowledged with a **Command Ring Stopped**
    /// Command Completion Event. Linux's command watchdog waits for exactly that event to restart
    /// the ring (`xhci_handle_stopped_cmd_ring`); without it `cmd_ring_state` stays ABORTED and
    /// every later command is refused, so the ring survives but USB is dead all the same.
    #[test]
    fn command_ring_abort_posts_a_command_ring_stopped_event() {
        let m = mem();
        let dev = new_dev();
        let mut d = dev.lock().unwrap();
        d.event_ring = Some(EventRing::new(0x3000, 16));
        // A programmed command ring, then the guest's abort write.
        d.write(0, 0x20 + 0x18, &(0x4000u64 | 1).to_le_bytes());
        d.write(0, 0x20 + 0x18, &(0x4u64).to_le_bytes()); // CRCR.CA
        assert!(d.work.cmd_abort, "the abort was queued for the worker");

        d.run_worker_pass(&m, &dev);
        let ev = Trb::read(&m, 0x3000).unwrap();
        assert_eq!(ev.trb_type(), trb_type::COMMAND_COMPLETION);
        assert_eq!(
            ev.status >> 24,
            cc::COMMAND_RING_STOPPED,
            "completion code must be Command Ring Stopped (24)"
        );
        assert_eq!(
            ev.parameter & !0xf,
            0x4000,
            "the event carries the dequeue pointer the guest restarts from"
        );
    }
}
