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

use super::super::model::{Completion, ControlTransfer, SetupPacket, UsbDeviceModel, XferOutcome};
use super::context::{
    dcbaa_entry, ep_tr_dequeue, input_add_flags, input_ep_offset, output_ep_offset, set_ep_state,
    set_slot_address, set_slot_state, slot_root_hub_port, Ctx32, INPUT_SLOT_OFFSET,
};
use super::device::{SlotCtx, XhciDevice, NUM_PORTS};
use super::trb::{
    cc, command_completion_event, port_status_change_event, transfer_event, trb_type, EventRing,
    RingError, RingWalker, Trb, CTRL_BSR, CTRL_DC, CTRL_DIR_IN, CTRL_IOC,
};

/// A gadget call the worker makes with the controller lock **released**.
pub(super) enum DeferredCall {
    /// Forward a class/vendor EP0 control request to the gadget.
    Control(Arc<dyn UsbDeviceModel>, ControlTransfer),
    /// Reset the device model (Disable Slot / Reset Device).
    Reset(Arc<dyn UsbDeviceModel>),
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
                // B1 exercises EP0 only; non-EP0 endpoints stall until B2 wires
                // interrupt/bulk data flow. The mock has no such endpoints, so a
                // stock guest never rings these.
                debug!("xhci: non-EP0 doorbell slot {slot} dci {dci} ignored (B1: EP0 only)");
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
    fn scan_ports_on_run(&mut self, mem: &GuestMemoryMmap) {
        for idx in 0..NUM_PORTS {
            if self.port_populated(idx) {
                self.set_port_connected(idx);
                self.post_port_event(mem, (idx + 1) as u8);
            }
        }
    }

    fn post_port_event(&mut self, mem: &GuestMemoryMmap, port_id: u8) {
        self.post_event(mem, port_status_change_event(port_id));
    }

    // ---- command ring --------------------------------------------------------

    fn process_command_ring(&mut self, mem: &GuestMemoryMmap, deferred: &mut Vec<DeferredCall>) {
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
                match self.cmd_configure_endpoint(mem, slot, deconfig) {
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
                self.set_output_ep_state(mem, slot, dci, es::RUNNING);
                (cc::SUCCESS, slot)
            }
            trb_type::STOP_ENDPOINT => {
                let slot = trb.slot_id();
                let dci = trb.endpoint_id();
                self.set_output_ep_state(mem, slot, dci, es::STOPPED);
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
            });
            Ok(slot_id)
        })();

        match result {
            Ok(_) => (cc::SUCCESS, slot_id),
            Err(code) => (code, slot_id),
        }
    }

    /// Configure Endpoint: apply the input context's add/drop flags. Our
    /// gadgets carry no non-EP0 endpoints in B1, so this only advances the slot
    /// state; the machinery to add EP contexts lands with B2 data endpoints.
    fn cmd_configure_endpoint(
        &mut self,
        mem: &GuestMemoryMmap,
        slot_id: u8,
        deconfig: bool,
    ) -> Result<(), u32> {
        use super::context::slot_state as ss;
        let out = dcbaa_entry(mem, self.dcbaap(), slot_id).map_err(|_| cc::TRB_ERROR)?;
        let mut oslot = Ctx32::read(mem, out).map_err(|_| cc::TRB_ERROR)?;
        let new_state = if deconfig {
            ss::ADDRESSED
        } else {
            ss::CONFIGURED
        };
        set_slot_state(&mut oslot, new_state);
        oslot.write(mem, out).map_err(|_| cc::TRB_ERROR)?;
        if let Some(s) = self
            .slots
            .get_mut(slot_id as usize)
            .and_then(|x| x.as_mut())
        {
            s.state = new_state;
        }
        Ok(())
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
                    // Device qualifier / BOS / other: not a full-speed feature — stall.
                    _ => return Some(XferOutcome::Stall),
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
}
