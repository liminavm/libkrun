// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The xHCI controller register file (a `BusDevice`) plus the shared controller
//! state the ring worker drives.
//!
//! Stage A brought up a *functional register file with a stub data path*: enough
//! of the capability / operational / runtime / doorbell register semantics for a
//! stock `xhci-plat` guest to reset the controller, read its parameters, program
//! the DCBAA / command ring / event ring, and declare the HCD running with the
//! root hub up. **Stage B1** adds the data path: the command/event/transfer ring
//! walkers, the command set, EP0 control transfers, port enumeration and the
//! [`UsbDeviceModel`] gadgets, so a stock guest **enumerates** an attached device.
//!
//! ## Threading & locking (the #1 structural hazard)
//!
//! The whole controller is one `Arc<Mutex<XhciDevice>>`:
//! - the **vcpu thread** locks it for each MMIO register access (short; register
//!   mutation only — a doorbell/RS write enqueues work and kicks the worker);
//! - the **worker thread** locks it to walk the rings, run the command set and
//!   collect EP0 transfer work, then **releases the lock before calling any
//!   gadget** (a gadget may block for seconds on Touch ID);
//! - a **completion** (invoked from any thread when a gadget finishes) re-locks it
//!   to post the Transfer Event + interrupt.
//!
//! There is exactly one lock and it is never held across a gadget call, so a
//! synchronous gadget completion (the mock) cannot re-enter it and no lock-order
//! cycle exists — the design cannot deadlock. See `engine.rs` / `worker.rs`.

use std::collections::HashMap;
use std::sync::Arc;

use utils::eventfd::EventFd;

use super::super::model::UsbDeviceModel;
use super::trb::{EventRing, RingWalker};
use crate::legacy::IrqChip;
use crate::BusDevice;

// ---- controller geometry ----------------------------------------------------

/// Root ports (USB 2.0 only — one Supported Protocol extended cap covers all).
pub(crate) const NUM_PORTS: usize = 4;
/// Device slots advertised in HCSPARAMS1.
const NUM_SLOTS: u32 = 8;
/// Interrupters (interrupter 0 only).
const NUM_INTRS: u32 = 1;
/// The MMIO window the controller claims (64 KiB); mirrors the FDT `reg` size.
pub const XHCI_MMIO_LEN: u64 = 0x1_0000;

// ---- register-region base offsets (from the device's MMIO base) -------------

/// Length of the capability register block == operational-register base (CAPLENGTH).
const CAP_LENGTH: u32 = 0x20;
/// Operational registers start here (== CAPLENGTH).
const OP_BASE: u64 = CAP_LENGTH as u64;
/// Runtime registers (RTSOFF).
const RUNTIME_BASE: u64 = 0x1000;
/// Doorbell array (DBOFF).
const DB_BASE: u64 = 0x2000;
/// Extended capabilities list (HCCPARAMS1 xECP points here, in dword units).
const XECP_BASE: u64 = 0x3000;

/// HCIVERSION (xHCI 1.0.0).
const HCI_VERSION: u32 = 0x0100;

// ---- capability-register offsets (from CAP base 0) --------------------------

const CAP_CAPLENGTH_HCIVERSION: u64 = 0x00; // CAPLENGTH (byte 0) | HCIVERSION (bytes 2-3)
const CAP_HCSPARAMS1: u64 = 0x04;
const CAP_HCSPARAMS2: u64 = 0x08;
const CAP_HCSPARAMS3: u64 = 0x0c;
const CAP_HCCPARAMS1: u64 = 0x10;
const CAP_DBOFF: u64 = 0x14;
const CAP_RTSOFF: u64 = 0x18;
const CAP_HCCPARAMS2: u64 = 0x1c;

// ---- operational-register offsets (from OP_BASE) ----------------------------

const OP_USBCMD: u64 = 0x00;
const OP_USBSTS: u64 = 0x04;
const OP_PAGESIZE: u64 = 0x08;
const OP_DNCTRL: u64 = 0x14;
const OP_CRCR_LO: u64 = 0x18;
const OP_CRCR_HI: u64 = 0x1c;
const OP_DCBAAP_LO: u64 = 0x30;
const OP_DCBAAP_HI: u64 = 0x34;
const OP_CONFIG: u64 = 0x38;
/// PORTSC register array begins 0x400 into the operational region; each port
/// occupies 0x10 bytes (PORTSC, PORTPMSC, PORTLI, PORTHLPMC).
const OP_PORTSC_BASE: u64 = 0x400;
const PORT_REGS_STRIDE: u64 = 0x10;

// ---- runtime-register offsets (from RUNTIME_BASE) ---------------------------

const RT_MFINDEX: u64 = 0x00;
/// Interrupter register sets start 0x20 into the runtime region; interrupter 0 only.
const RT_IR0_BASE: u64 = 0x20;
const IR_IMAN: u64 = 0x00;
const IR_IMOD: u64 = 0x04;
const IR_ERSTSZ: u64 = 0x08;
const IR_ERSTBA_LO: u64 = 0x10;
const IR_ERSTBA_HI: u64 = 0x14;
const IR_ERDP_LO: u64 = 0x18;
const IR_ERDP_HI: u64 = 0x1c;

// ---- USBCMD bits ------------------------------------------------------------

const CMD_RS: u32 = 1 << 0; // Run/Stop
const CMD_HCRST: u32 = 1 << 1; // Host Controller Reset (RW1S, self-clearing)
const CMD_INTE: u32 = 1 << 2; // Interrupter Enable
const CMD_LHCRST: u32 = 1 << 7; // Light HC Reset (RW1S, self-clearing)
/// Bits USBCMD holds as plain RW state (everything but the self-clearing resets).
const CMD_STORE_MASK: u32 = !(CMD_HCRST | CMD_LHCRST);

// ---- USBSTS bits ------------------------------------------------------------

const STS_HCH: u32 = 1 << 0; // HCHalted (RO, tracks !RS)
const STS_HSE: u32 = 1 << 2; // Host System Error (RW1C)
const STS_EINT: u32 = 1 << 3; // Event Interrupt (RW1C)
const STS_PCD: u32 = 1 << 4; // Port Change Detect (RW1C)
const STS_SRE: u32 = 1 << 10; // Save/Restore Error (RW1C)
const STS_CNR: u32 = 1 << 11; // Controller Not Ready (RO)
/// The write-1-to-clear status bits.
const STS_RW1C_MASK: u32 = STS_HSE | STS_EINT | STS_PCD | STS_SRE;

// ---- interrupter IMAN bits --------------------------------------------------

const IMAN_IP: u32 = 1 << 0; // Interrupt Pending (RW1C)
const IMAN_IE: u32 = 1 << 1; // Interrupt Enable (RW)

// ---- PORTSC bits ------------------------------------------------------------

const PORTSC_CCS: u32 = 1 << 0; // Current Connect Status (RO)
const PORTSC_PED: u32 = 1 << 1; // Port Enabled/Disabled (RW1C to disable)
const PORTSC_PR: u32 = 1 << 4; // Port Reset (RW1S, self-clearing)
const PORTSC_PLS_SHIFT: u32 = 5;
const PORTSC_PLS_MASK: u32 = 0xf << PORTSC_PLS_SHIFT; // Port Link State (RWS via LWS)
const PORTSC_PP: u32 = 1 << 9; // Port Power (RW)
const PORTSC_SPEED_SHIFT: u32 = 10;
const PORTSC_SPEED_MASK: u32 = 0xf << PORTSC_SPEED_SHIFT; // Port Speed (RO)
const PORTSC_PIC_MASK: u32 = 0x3 << 14; // Port Indicator Control (RW)
const PORTSC_LWS: u32 = 1 << 16; // Link State Write Strobe (gates PLS write)
const PORTSC_CSC: u32 = 1 << 17; // Connect Status Change (RW1C)
const PORTSC_PRC: u32 = 1 << 21; // Port Reset Change (RW1C)
const PORTSC_WAKE_MASK: u32 = 0x7 << 25; // WCE/WDE/WOE (RW)
const PORTSC_WPR: u32 = 1 << 31; // Warm Port Reset (RW1S, self-clearing)
/// PORTSC change bits (CSC, PEC, WRC, OCC, PRC, PLC, CEC) — all write-1-to-clear.
const PORTSC_RW1C_MASK: u32 = 0x7f << 17;
/// Port Link State value for a powered but disconnected port (RxDetect).
const PLS_RX_DETECT: u32 = 5;
/// Port Link State: Polling (USB2 on connect, pre-reset).
const PLS_POLLING: u32 = 7;
/// Port Link State: U0 (enabled, post-reset).
const PLS_U0: u32 = 0;
/// A freshly powered, nothing-connected USB2 port.
const PORTSC_DEFAULT: u32 = PORTSC_PP | (PLS_RX_DETECT << PORTSC_PLS_SHIFT);
/// Port Speed ID for a full-speed device (`XDEV_FS`, Linux `xhci.h`).
const SPEED_FULL: u32 = 1;
const SPEED_LOW: u32 = 2;
const SPEED_HIGH: u32 = 3;

// ---- extended capability: Supported Protocol (USB 2.0, ports 1-4) -----------

const XECP_CAP_ID_SUPPORTED_PROTOCOL: u32 = 0x02;
// dword0: CapID=0x02, Next=0 (end of list), MinorRev=0x00, MajorRev=0x02. The
// zero-shifted fields are kept explicit to mirror the spec's field layout.
#[allow(clippy::identity_op)]
const XECP_DW0: u32 = XECP_CAP_ID_SUPPORTED_PROTOCOL | (0x00 << 8) | (0x00 << 16) | (0x02 << 24);
// dword1: name string "USB " (little-endian bytes 'U','S','B',' ').
const XECP_DW1: u32 = u32::from_le_bytes(*b"USB ");
// dword2: Compatible Port Offset = 1, Compatible Port Count = 4, PSIC = 0.
const XECP_DW2: u32 = 0x01 | ((NUM_PORTS as u32) << 8);
// dword3: Protocol Slot Type = 0.
const XECP_DW3: u32 = 0x0000_0000;

/// A live data (interrupt/bulk) endpoint: its transfer-ring consumer position plus
/// the state a held transfer's completion needs to stay correct across cancellation.
///
/// The **generation counter** is the staleness guard (the QEMU "endpoint stopped/reset
/// with transfers in flight" rough edge, done right): it is captured into every
/// forwarded transfer's completion closure and bumped whenever the endpoint is
/// re-programmed (Stop / Reset Endpoint, Set TR Dequeue, re-Configure). A gadget
/// completion firing against a stale generation is silently dropped — the TRBs it
/// referenced are gone, so posting its event could corrupt a re-programmed ring.
pub(super) struct EpRing {
    /// The transfer-ring consumer position.
    pub ring: RingWalker,
    /// Transfer direction (true = device-to-host / IN).
    pub dir_in: bool,
    /// Endpoint type (EP Context dword1 bits 5:3; bookkeeping / future gating).
    #[allow(dead_code)]
    pub ep_type: u32,
    /// Endpoint state (see `context::ep_state`): RUNNING fetches TDs, STOPPED waits for a
    /// doorbell, HALTED waits for Reset Endpoint.
    pub state: u32,
    /// Bumped on every re-program; a completion carrying an older value is dropped.
    pub generation: u64,
}

/// Per-slot controller state, keyed by Slot ID (1-based). Created at Enable Slot
/// (port unknown), bound to a port + model at Address Device.
pub(super) struct SlotCtx {
    /// Root-hub port number (1-based), set at Address Device.
    pub port: u8,
    /// Assigned USB device address (bookkeeping; xHCI addressing is internal).
    #[allow(dead_code)]
    pub address: u8,
    /// Slot state (see `context::slot_state`).
    pub state: u32,
    /// Current configuration value (SET_CONFIGURATION).
    pub config_value: u8,
    /// The EP0 (control) transfer-ring consumer position.
    pub ep0_ring: Option<RingWalker>,
    /// Live data endpoints, keyed by DCI (2..=31); EP0 (DCI 1) stays in `ep0_ring`.
    pub eps: HashMap<u8, EpRing>,
}

/// Work the vcpu-thread register writes hand to the worker, drained each pass.
#[derive(Default)]
pub(super) struct WorkQueue {
    /// USBCMD.RS went 0→1: scan ports for cold-plugged devices and post connects.
    pub run_started: bool,
    /// The command-ring doorbell (DB[0]) was rung.
    pub cmd_doorbell: bool,
    /// Slot transfer doorbells: (slot_id, endpoint DCI).
    pub ep_doorbells: Vec<(u8, u8)>,
    /// Ports needing a Port Status Change Event (1-based ids).
    pub port_events: Vec<u8>,
}

/// The emulated xHCI controller: the register file (serviced on the vcpu thread)
/// plus the ring/slot/port state the worker drives. One `Arc<Mutex<_>>` shared by
/// the bus, the worker and transfer completions (see the module docs).
pub struct XhciDevice {
    // Operational registers.
    usbcmd: u32,
    usbsts: u32,
    dnctrl: u32,
    crcr: u64,
    dcbaap: u64,
    config: u32,
    ports: [u32; NUM_PORTS],

    // Runtime registers (interrupter 0).
    iman: u32,
    imod: u32,
    erstsz: u32,
    erstba: u64,
    erdp: u64,

    // Interrupt plumbing (HVF: intc + irq_line; KVM: interrupt_evt irqfd).
    intc: Option<IrqChip>,
    irq_line: Option<u32>,
    interrupt_evt: EventFd,

    // ---- Stage B controller state ----
    /// The command-ring walker (rebuilt when CRCR is (re)programmed).
    pub(super) cmd_ring: Option<RingWalker>,
    /// The single-segment event ring (built from ERSTBA on first use).
    pub(super) event_ring: Option<EventRing>,
    /// Per-slot state (index 0 unused; 1..=NUM_SLOTS).
    pub(super) slots: Vec<Option<SlotCtx>>,
    /// Device models per root port (index 0 = port 1).
    pub(super) port_models: Vec<Option<Arc<dyn UsbDeviceModel>>>,
    /// Next USB device address to hand out at Address Device (BSR=0).
    pub(super) next_address: u8,
    /// Pending worker work, set by register writes.
    pub(super) work: WorkQueue,
    /// Nudges the worker thread; `None` in unit tests (work stays queued).
    worker_kick: Option<EventFd>,
}

impl XhciDevice {
    /// Build the controller in its post-power-on state: halted (`HCH`), not-ready
    /// (`CNR`), interrupts disabled, ports powered but disconnected.
    pub fn new(interrupt_evt: EventFd) -> Self {
        XhciDevice {
            usbcmd: 0,
            // Powered on: halted and not-yet-ready. `xhci_reset` clears CNR.
            usbsts: STS_HCH | STS_CNR,
            dnctrl: 0,
            crcr: 0,
            dcbaap: 0,
            config: 0,
            ports: [PORTSC_DEFAULT; NUM_PORTS],
            iman: 0,
            imod: 0,
            erstsz: 0,
            erstba: 0,
            erdp: 0,
            intc: None,
            irq_line: None,
            interrupt_evt,
            cmd_ring: None,
            event_ring: None,
            slots: (0..=NUM_SLOTS).map(|_| None).collect(),
            port_models: (0..NUM_PORTS).map(|_| None).collect(),
            next_address: 1,
            work: WorkQueue::default(),
            worker_kick: None,
        }
    }

    /// Wire the controller to the interrupt controller (macOS/HVF path), mirroring
    /// the PL031 RTC. On KVM `intc` stays `None` and the worker pokes the irqfd.
    pub fn set_intc(&mut self, intc: IrqChip) {
        self.intc = Some(intc);
    }

    /// The SPI line this controller's interrupter raises (edge-triggered).
    pub fn set_irq_line(&mut self, irq: u32) {
        self.irq_line = Some(irq);
    }

    /// Attach the cold-plugged device models (one per root port, in order) and the
    /// worker kick eventfd. Called once at registration, before boot.
    pub fn attach_devices(&mut self, models: Vec<Arc<dyn UsbDeviceModel>>, kick: EventFd) {
        for (i, m) in models.into_iter().enumerate() {
            if i < NUM_PORTS {
                self.port_models[i] = Some(m);
            } else {
                warn!("xhci: more USB devices than root ports ({NUM_PORTS}); dropping extras");
                break;
            }
        }
        self.worker_kick = Some(kick);
    }

    /// A clone of the guest-visible interrupt eventfd (for the worker's irqfd path
    /// bookkeeping / KVM). Not otherwise used on HVF.
    pub fn interrupt_evt(&self) -> &EventFd {
        &self.interrupt_evt
    }

    /// Nudge the worker to process queued work (no-op in unit tests).
    fn kick_worker(&self) {
        if let Some(k) = &self.worker_kick {
            if let Err(e) = k.write(1) {
                warn!("xhci: failed to kick worker: {e:?}");
            }
        }
    }

    /// Assert the interrupter-0 SPI (HVF) or poke the irqfd (KVM).
    ///
    /// Per spec (and the Stage A reviewer's mandate): **latch IMAN.IP and
    /// USBSTS.EINT unconditionally** when an event is posted — a guest that polls,
    /// or that enables interrupts *after* an event, must still observe them. Only
    /// the physical SPI pulse is gated by IMAN.IE && USBCMD.INTE.
    pub fn assert_interrupt(&mut self) {
        self.iman |= IMAN_IP;
        self.usbsts |= STS_EINT;
        if self.iman & IMAN_IE == 0 || self.usbcmd & CMD_INTE == 0 {
            return; // latched, but no edge delivered while masked
        }
        self.pulse_spi();
    }

    /// If an interrupt is latched (IMAN.IP) and the guest has just armed delivery
    /// (IMAN.IE && USBCMD.INTE), pulse the edge now. We are edge-triggered, so a
    /// guest that posts an event while masked and *then* enables interrupts would
    /// otherwise never see the edge — this synthesizes the level-triggered
    /// behavior QEMU/crosvm get for free from INTx.
    fn deliver_latched_on_enable(&mut self) {
        if self.iman & IMAN_IP != 0 && self.iman & IMAN_IE != 0 && self.usbcmd & CMD_INTE != 0 {
            self.pulse_spi();
        }
    }

    fn pulse_spi(&self) {
        log::trace!(
            "xhci: SPI pulse irq={:?} iman={:#x} usbcmd={:#x} erdp={:#x} enq={:#x}",
            self.irq_line,
            self.iman,
            self.usbcmd,
            self.erdp & !0xf,
            self.event_ring.map(|e| e.enqueue_addr()).unwrap_or(0)
        );
        if let Some(intc) = &self.intc {
            if let Err(e) = intc
                .lock()
                .unwrap()
                .set_irq(self.irq_line, Some(&self.interrupt_evt))
            {
                warn!("xhci: failed to assert interrupter SPI: {e:?}");
            }
        } else if let Err(e) = self.interrupt_evt.write(1) {
            warn!("xhci: failed to poke interrupter irqfd: {e:?}");
        }
    }

    /// True while the event ring holds events the guest hasn't dequeued (its
    /// enqueue pointer is ahead of ERDP). Used by the lost-edge re-pulse guard.
    fn event_ring_has_pending(&self) -> bool {
        self.event_ring
            .map(|er| er.enqueue_addr() != (self.erdp & !0xf))
            .unwrap_or(false)
    }

    /// A guest-visible host controller reset (`USBCMD.HCRST`): back to post-power-on
    /// state with `CNR` cleared (ready) — the transition `xhci_reset` waits on — and
    /// all Stage B ring/slot state discarded (the attached models persist).
    fn reset_controller(&mut self) {
        self.usbcmd = 0;
        self.usbsts = STS_HCH; // halted, ready (CNR clear)
        self.dnctrl = 0;
        self.crcr = 0;
        self.dcbaap = 0;
        self.config = 0;
        self.ports = [PORTSC_DEFAULT; NUM_PORTS];
        self.iman = 0;
        self.imod = 0;
        self.erstsz = 0;
        self.erstba = 0;
        self.erdp = 0;
        self.cmd_ring = None;
        self.event_ring = None;
        for s in self.slots.iter_mut() {
            *s = None;
        }
        self.next_address = 1;
        self.work = WorkQueue::default();
    }

    // ---- register read/write dispatch ---------------------------------------

    /// Read the 32-bit register at the 4-byte-aligned `off`.
    fn read_reg32(&self, off: u64) -> u32 {
        match off {
            // Capability registers.
            CAP_CAPLENGTH_HCIVERSION => CAP_LENGTH | (HCI_VERSION << 16),
            CAP_HCSPARAMS1 => {
                // MaxSlots [7:0], MaxIntrs [18:8], MaxPorts [31:24].
                NUM_SLOTS | (NUM_INTRS << 8) | ((NUM_PORTS as u32) << 24)
            }
            // IST=0, ERST Max=0 (=> 1 segment), 0 scratchpads.
            CAP_HCSPARAMS2 => 0,
            CAP_HCSPARAMS3 => 0,
            // AC64 [0]=1 (64-bit addressing; guest RAM starts at 0x8000_0000),
            // CSZ [2]=0 (32-byte contexts), xECP [31:16] = dword offset of the caps list.
            CAP_HCCPARAMS1 => 0x1 | (((XECP_BASE >> 2) as u32) << 16),
            CAP_DBOFF => DB_BASE as u32,
            CAP_RTSOFF => RUNTIME_BASE as u32,
            CAP_HCCPARAMS2 => 0,

            // Operational registers.
            _ if off == OP_BASE + OP_USBCMD => self.usbcmd,
            _ if off == OP_BASE + OP_USBSTS => self.usbsts,
            _ if off == OP_BASE + OP_PAGESIZE => 0x1, // 4 KiB page size supported
            _ if off == OP_BASE + OP_DNCTRL => self.dnctrl,
            // Per spec the Command Ring Pointer field reads as 0; CRR is modelled
            // minimally as 0 (we never leave a command mid-execution).
            _ if off == OP_BASE + OP_CRCR_LO => 0,
            _ if off == OP_BASE + OP_CRCR_HI => 0,
            _ if off == OP_BASE + OP_DCBAAP_LO => self.dcbaap as u32,
            _ if off == OP_BASE + OP_DCBAAP_HI => (self.dcbaap >> 32) as u32,
            _ if off == OP_BASE + OP_CONFIG => self.config,

            // Runtime registers.
            _ if off == RUNTIME_BASE + RT_MFINDEX => 0,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMAN => self.iman,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMOD => self.imod,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTSZ => self.erstsz,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_LO => self.erstba as u32,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_HI => (self.erstba >> 32) as u32,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERDP_LO => self.erdp as u32,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERDP_HI => (self.erdp >> 32) as u32,

            // Extended capabilities: the single Supported Protocol cap (USB 2.0).
            XECP_BASE => XECP_DW0,
            _ if off == XECP_BASE + 0x4 => XECP_DW1,
            _ if off == XECP_BASE + 0x8 => XECP_DW2,
            _ if off == XECP_BASE + 0xc => XECP_DW3,

            // PORTSC array (only PORTSC itself is modelled; the other three
            // per-port registers read as zero).
            _ if self.port_index(off).is_some() => {
                let (idx, sub) = self.port_index(off).unwrap();
                if sub == 0 {
                    self.ports[idx]
                } else {
                    0
                }
            }

            // Doorbell array and everything else read as zero.
            _ => 0,
        }
    }

    /// Write the 32-bit `val` to the register at the 4-byte-aligned `off`.
    fn write_reg32(&mut self, off: u64, val: u32) {
        match off {
            // Operational registers.
            _ if off == OP_BASE + OP_USBCMD => self.handle_usbcmd(val),
            _ if off == OP_BASE + OP_USBSTS => {
                // Write-1-to-clear the status bits; the rest are read-only.
                self.usbsts &= !(val & STS_RW1C_MASK);
            }
            _ if off == OP_BASE + OP_PAGESIZE => {} // read-only
            _ if off == OP_BASE + OP_DNCTRL => self.dnctrl = val,
            _ if off == OP_BASE + OP_CRCR_LO => {
                self.crcr = (self.crcr & 0xffff_ffff_0000_0000) | u64::from(val);
                // A CRCR (re)program rebases the command ring.
                self.cmd_ring = None;
            }
            _ if off == OP_BASE + OP_CRCR_HI => {
                self.crcr = (self.crcr & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32);
                self.cmd_ring = None;
            }
            _ if off == OP_BASE + OP_DCBAAP_LO => {
                self.dcbaap = (self.dcbaap & 0xffff_ffff_0000_0000) | u64::from(val)
            }
            _ if off == OP_BASE + OP_DCBAAP_HI => {
                self.dcbaap = (self.dcbaap & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32)
            }
            _ if off == OP_BASE + OP_CONFIG => self.config = val,

            // Runtime registers (interrupter 0).
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMAN => self.handle_iman(val),
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMOD => self.imod = val,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTSZ => self.erstsz = val,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_LO => {
                self.erstba = (self.erstba & 0xffff_ffff_0000_0000) | u64::from(val);
                // A new ERST rebuilds the event ring.
                self.event_ring = None;
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_HI => {
                self.erstba = (self.erstba & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32);
                self.event_ring = None;
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERDP_LO => {
                // ERDP bit 3 (EHB) is write-1-to-clear; store the dequeue pointer
                // (with EHB forced low) for ring-full detection.
                let ptr = val & !(1 << 3);
                self.erdp = (self.erdp & 0xffff_ffff_0000_0000) | u64::from(ptr);
                log::trace!("xhci: ERDP write ptr={ptr:#x} (ehb-clear)");
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERDP_HI => {
                self.erdp = (self.erdp & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32)
            }

            // PORTSC array.
            _ if self.port_index(off).is_some() => {
                let (idx, sub) = self.port_index(off).unwrap();
                if sub == 0 {
                    self.write_portsc(idx, val);
                }
                // PORTPMSC/PORTLI/PORTHLPMC are accepted and dropped for now.
            }

            // Doorbell array: enqueue worker work and kick.
            _ if (DB_BASE..DB_BASE + (NUM_SLOTS as u64 + 1) * 4).contains(&off) => {
                self.handle_doorbell(((off - DB_BASE) / 4) as u8, val);
            }

            // Read-only capability registers and reserved space: ignore writes.
            _ => {}
        }
    }

    /// IMAN write: IE is RW, IP is RW1C. Reviewer-mandated lost-edge guard — after
    /// a write clears IP, if the event ring still holds undelivered events we are
    /// edge-triggered, so re-latch IP and pulse again (QEMU/crosvm get this free
    /// from level INTx; we must synthesize the missed edge).
    fn handle_iman(&mut self, val: u32) {
        let cleared_ip = val & IMAN_IP != 0 && self.iman & IMAN_IP != 0;
        let ie_was_off = self.iman & IMAN_IE == 0;
        let mut iman = self.iman;
        if val & IMAN_IP != 0 {
            iman &= !IMAN_IP;
        }
        iman = (iman & !IMAN_IE) | (val & IMAN_IE);
        self.iman = iman;
        log::trace!(
            "xhci: IMAN write val={:#x} cleared_ip={} pending={} iman_now={:#x}",
            val,
            cleared_ip,
            self.event_ring_has_pending(),
            self.iman
        );
        if cleared_ip && self.event_ring_has_pending() {
            // Lost-edge guard: undelivered events remain — re-latch and re-pulse.
            self.assert_interrupt();
        } else if ie_was_off && iman & IMAN_IE != 0 {
            // Interrupts just enabled: deliver anything latched while masked.
            self.deliver_latched_on_enable();
        }
    }

    /// Handle a doorbell write: DB[0] is the command ring; DB[slot] is a device
    /// slot with the endpoint DCI in the low byte. Queue the work and kick.
    fn handle_doorbell(&mut self, index: u8, val: u32) {
        if index == 0 {
            self.work.cmd_doorbell = true;
        } else {
            let dci = (val & 0xff) as u8;
            self.work.ep_doorbells.push((index, dci));
        }
        self.kick_worker();
    }

    /// USBCMD write: a HCRST triggers a full reset; otherwise store the RW bits,
    /// keep `USBSTS.HCH` in sync with `RS`, and flag the run edge for the worker
    /// (the port connect scan happens there, with guest memory in hand).
    fn handle_usbcmd(&mut self, val: u32) {
        if val & CMD_HCRST != 0 {
            self.reset_controller();
            return;
        }
        // Light HC reset (bit 7) self-clears with no other effect here.
        let was_running = self.usbcmd & CMD_RS != 0;
        let inte_was_off = self.usbcmd & CMD_INTE == 0;
        let stored = val & CMD_STORE_MASK;
        self.usbcmd = stored;
        if stored & CMD_RS != 0 {
            self.usbsts &= !STS_HCH; // running
            if !was_running {
                self.work.run_started = true;
                self.kick_worker();
            }
        } else {
            self.usbsts |= STS_HCH; // halted
        }
        // Interrupts just globally enabled: deliver anything latched while masked.
        if inte_was_off && stored & CMD_INTE != 0 {
            self.deliver_latched_on_enable();
        }
    }

    /// PORTSC write with the spec's mixed RW / RW1C / RW1S semantics. A Port Reset
    /// on a connected port completes immediately (USB2 emulated): PR self-clears,
    /// PED sets, PLS→U0, the speed field latches, PRC latches, and a Port Status
    /// Change Event is queued for the worker.
    fn write_portsc(&mut self, idx: usize, val: u32) {
        let mut p = self.ports[idx];
        // Write-1-to-clear the change bits.
        p &= !(val & PORTSC_RW1C_MASK);
        // PED is write-1-to-clear (writing 1 disables the port).
        if val & PORTSC_PED != 0 {
            p &= !PORTSC_PED;
        }
        // Port Power (RW).
        p = (p & !PORTSC_PP) | (val & PORTSC_PP);
        // Port Indicator Control (RW).
        p = (p & !PORTSC_PIC_MASK) | (val & PORTSC_PIC_MASK);
        // Port Link State is writable only when the Link State Write Strobe is set.
        if val & PORTSC_LWS != 0 {
            p = (p & !PORTSC_PLS_MASK) | (val & PORTSC_PLS_MASK);
        }
        // Wake-enable bits (RW).
        p = (p & !PORTSC_WAKE_MASK) | (val & PORTSC_WAKE_MASK);

        // PR / WPR initiate a reset.
        let reset = val & (PORTSC_PR | PORTSC_WPR) != 0;
        self.ports[idx] = p;
        if reset && p & PORTSC_CCS != 0 {
            self.complete_port_reset(idx);
        }
    }

    /// Complete an emulated USB2 port reset: enable the port, drop it to U0, latch
    /// the device speed and the Port Reset Change bit, and queue the change event.
    fn complete_port_reset(&mut self, idx: usize) {
        let speed = match self.port_models[idx].as_ref().map(|m| m.speed()) {
            Some(super::super::model::UsbSpeed::Low) => SPEED_LOW,
            Some(super::super::model::UsbSpeed::High) => SPEED_HIGH,
            _ => SPEED_FULL,
        };
        let mut p = self.ports[idx];
        p &= !PORTSC_PR; // reset done (self-clearing)
        p |= PORTSC_PED; // port enabled
        p = (p & !PORTSC_PLS_MASK) | (PLS_U0 << PORTSC_PLS_SHIFT);
        p = (p & !PORTSC_SPEED_MASK) | (speed << PORTSC_SPEED_SHIFT);
        p |= PORTSC_PRC; // reset-change latched
        self.ports[idx] = p;
        self.usbsts |= STS_PCD;
        self.work.port_events.push((idx + 1) as u8);
        self.kick_worker();
    }

    /// If `off` lands in the PORTSC array, return `(port index, sub-register offset)`.
    fn port_index(&self, off: u64) -> Option<(usize, u64)> {
        let base = OP_BASE + OP_PORTSC_BASE;
        let end = base + (NUM_PORTS as u64) * PORT_REGS_STRIDE;
        if (base..end).contains(&off) {
            let rel = off - base;
            Some(((rel / PORT_REGS_STRIDE) as usize, rel % PORT_REGS_STRIDE))
        } else {
            None
        }
    }

    // ---- accessors used by the engine (engine.rs) ---------------------------

    pub(super) fn crcr_ptr(&self) -> u64 {
        self.crcr & !0x3f
    }
    pub(super) fn crcr_rcs(&self) -> bool {
        self.crcr & 1 != 0
    }
    pub(super) fn dcbaap(&self) -> u64 {
        self.dcbaap & !0x3f
    }
    pub(super) fn erstba(&self) -> u64 {
        self.erstba & !0x3f
    }
    pub(super) fn erdp_ptr(&self) -> u64 {
        self.erdp & !0xf
    }
    pub(super) fn set_port_connected(&mut self, idx: usize) {
        let mut p = self.ports[idx];
        p |= PORTSC_CCS | PORTSC_CSC;
        p = (p & !PORTSC_PLS_MASK) | (PLS_POLLING << PORTSC_PLS_SHIFT);
        self.ports[idx] = p;
        self.usbsts |= STS_PCD;
    }
    pub(super) fn port_populated(&self, idx: usize) -> bool {
        self.port_models[idx].is_some()
    }

    #[cfg(test)]
    pub(super) fn set_dcbaap_for_test(&mut self, v: u64) {
        self.dcbaap = v;
    }
}

impl BusDevice for XhciDevice {
    fn read(&mut self, _vcpuid: u64, offset: u64, data: &mut [u8]) {
        match data.len() {
            8 => {
                // A single 64-bit access reads two consecutive dwords.
                let lo = self.read_reg32(offset & !0x3);
                let hi = self.read_reg32((offset & !0x3) + 4);
                data[..4].copy_from_slice(&lo.to_le_bytes());
                data[4..].copy_from_slice(&hi.to_le_bytes());
            }
            len @ 1..=4 => {
                let dword = self.read_reg32(offset & !0x3).to_le_bytes();
                let start = (offset & 0x3) as usize;
                if start + len <= 4 {
                    data.copy_from_slice(&dword[start..start + len]);
                } else {
                    warn!("xhci: unaligned {len}-byte read at offset {offset:#x}");
                    data.fill(0);
                }
            }
            other => {
                warn!("xhci: unsupported {other}-byte read at offset {offset:#x}");
                data.fill(0);
            }
        }
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, data: &[u8]) {
        match data.len() {
            8 => {
                let v = u64::from_le_bytes(data.try_into().unwrap());
                // 64-bit registers arrive either as one 8-byte access or two 4-byte
                // halves; handle the first by splitting into low/high dword writes.
                self.write_reg32(offset & !0x3, v as u32);
                self.write_reg32((offset & !0x3) + 4, (v >> 32) as u32);
            }
            4 => {
                let v = u32::from_le_bytes(data.try_into().unwrap());
                self.write_reg32(offset & !0x3, v);
            }
            len @ 1..=2 => {
                // Sub-dword writes are not used by the driver for these registers;
                // read-modify-write the covering dword so we never corrupt neighbours.
                let base = offset & !0x3;
                let start = (offset & 0x3) as usize;
                if start + len <= 4 {
                    let mut dword = self.read_reg32(base).to_le_bytes();
                    dword[start..start + len].copy_from_slice(data);
                    self.write_reg32(base, u32::from_le_bytes(dword));
                } else {
                    warn!("xhci: unaligned {len}-byte write at offset {offset:#x}");
                }
            }
            other => warn!("xhci: unsupported {other}-byte write at offset {offset:#x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> XhciDevice {
        XhciDevice::new(EventFd::new(utils::eventfd::EFD_NONBLOCK).unwrap())
    }

    fn r32(d: &mut XhciDevice, off: u64) -> u32 {
        let mut b = [0u8; 4];
        d.read(0, off, &mut b);
        u32::from_le_bytes(b)
    }

    fn w32(d: &mut XhciDevice, off: u64, v: u32) {
        d.write(0, off, &v.to_le_bytes());
    }

    fn r64(d: &mut XhciDevice, off: u64) -> u64 {
        let mut b = [0u8; 8];
        d.read(0, off, &mut b);
        u64::from_le_bytes(b)
    }

    // Register-region absolute offsets used by the tests.
    const USBCMD: u64 = OP_BASE + OP_USBCMD;
    const USBSTS: u64 = OP_BASE + OP_USBSTS;
    const PAGESIZE: u64 = OP_BASE + OP_PAGESIZE;
    const CONFIG: u64 = OP_BASE + OP_CONFIG;
    const DCBAAP: u64 = OP_BASE + OP_DCBAAP_LO;
    const CRCR: u64 = OP_BASE + OP_CRCR_LO;
    const PORTSC0: u64 = OP_BASE + OP_PORTSC_BASE;
    const IMAN0: u64 = RUNTIME_BASE + RT_IR0_BASE + IR_IMAN;

    #[test]
    fn caplength_and_hciversion_pack_into_dword0() {
        let mut d = dev();
        let cap = r32(&mut d, CAP_CAPLENGTH_HCIVERSION);
        assert_eq!(cap & 0xff, CAP_LENGTH, "CAPLENGTH in byte 0");
        assert_eq!(cap >> 16, HCI_VERSION, "HCIVERSION in bytes 2-3");
    }

    #[test]
    fn hcsparams1_reports_slots_ports_interrupters() {
        let mut d = dev();
        let hcs1 = r32(&mut d, CAP_HCSPARAMS1);
        assert_eq!(hcs1 & 0xff, NUM_SLOTS, "MaxSlots");
        assert_eq!((hcs1 >> 8) & 0x7ff, NUM_INTRS, "MaxIntrs");
        assert_eq!((hcs1 >> 24) & 0xff, NUM_PORTS as u32, "MaxPorts");
    }

    #[test]
    fn hccparams1_advertises_ac64_csz0_and_xecp() {
        let mut d = dev();
        let hcc = r32(&mut d, CAP_HCCPARAMS1);
        assert_eq!(hcc & 0x1, 0x1, "AC64 = 1");
        assert_eq!(hcc & 0x4, 0x0, "CSZ = 0 (32-byte contexts)");
        assert_eq!((hcc >> 16) << 2, XECP_BASE as u32, "xECP dword offset");
    }

    #[test]
    fn pagesize_reports_4k() {
        let mut d = dev();
        assert_eq!(r32(&mut d, PAGESIZE) & 0x1, 0x1, "4 KiB page size bit");
    }

    #[test]
    fn extended_caps_walk_reaches_usb2_supported_protocol() {
        let mut d = dev();
        let hcc = r32(&mut d, CAP_HCCPARAMS1);
        let xecp = u64::from((hcc >> 16) << 2);
        let dw0 = r32(&mut d, xecp);
        assert_eq!(
            dw0 & 0xff,
            XECP_CAP_ID_SUPPORTED_PROTOCOL,
            "Supported Protocol cap id"
        );
        assert_eq!((dw0 >> 8) & 0xff, 0, "Next pointer = 0 (end of list)");
        assert_eq!((dw0 >> 24) & 0xff, 0x02, "Major revision 2 (USB 2.0)");
        assert_eq!(r32(&mut d, xecp + 4), u32::from_le_bytes(*b"USB "));
        let dw2 = r32(&mut d, xecp + 8);
        assert_eq!(dw2 & 0xff, 1, "compatible port offset");
        assert_eq!((dw2 >> 8) & 0xff, NUM_PORTS as u32, "compatible port count");
    }

    #[test]
    fn hcrst_self_clears_and_clears_cnr() {
        let mut d = dev();
        assert_ne!(r32(&mut d, USBSTS) & STS_CNR, 0, "CNR set at power-on");
        w32(&mut d, USBCMD, CMD_HCRST);
        assert_eq!(r32(&mut d, USBCMD) & CMD_HCRST, 0, "HCRST self-cleared");
        assert_eq!(r32(&mut d, USBSTS) & STS_CNR, 0, "CNR cleared after reset");
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "still halted after reset");
    }

    #[test]
    fn hch_tracks_run_stop() {
        let mut d = dev();
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "halted before run");
        w32(&mut d, USBCMD, CMD_RS);
        assert_eq!(r32(&mut d, USBSTS) & STS_HCH, 0, "HCH clears when running");
        w32(&mut d, USBCMD, 0);
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "HCH sets when stopped");
    }

    #[test]
    fn crcr_pointer_reads_as_zero() {
        let mut d = dev();
        // Program a command ring pointer (RCS=1).
        let addr: u64 = 0x0000_0001_2345_6041;
        d.write(0, CRCR, &addr.to_le_bytes());
        // Per spec the pointer field reads back as 0.
        assert_eq!(r64(&mut d, CRCR), 0, "CRCR pointer reads as 0");
        // But the internal value is retained for the command ring walker.
        assert_eq!(d.crcr_ptr(), addr & !0x3f);
        assert!(d.crcr_rcs());
    }

    #[test]
    fn usbsts_eint_is_write_one_to_clear() {
        let mut d = dev();
        d.usbsts |= STS_EINT;
        assert_ne!(r32(&mut d, USBSTS) & STS_EINT, 0);
        w32(&mut d, USBSTS, 0);
        assert_ne!(r32(&mut d, USBSTS) & STS_EINT, 0, "0 must not clear EINT");
        w32(&mut d, USBSTS, STS_EINT);
        assert_eq!(r32(&mut d, USBSTS) & STS_EINT, 0, "EINT is RW1C");
    }

    #[test]
    fn iman_ip_is_rw1c_and_ie_is_rw() {
        let mut d = dev();
        w32(&mut d, IMAN0, IMAN_IE);
        d.iman |= IMAN_IP;
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IP, 0);
        w32(&mut d, IMAN0, IMAN_IE);
        assert_ne!(
            r32(&mut d, IMAN0) & IMAN_IP,
            0,
            "IP not cleared by writing 0"
        );
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IE, 0, "IE stays enabled");
        w32(&mut d, IMAN0, IMAN_IE | IMAN_IP);
        assert_eq!(r32(&mut d, IMAN0) & IMAN_IP, 0, "IP is RW1C");
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IE, 0, "IE preserved");
    }

    #[test]
    fn assert_interrupt_latches_even_while_masked() {
        let mut d = dev();
        // Interrupts globally and locally disabled.
        assert_eq!(d.usbcmd & CMD_INTE, 0);
        assert_eq!(d.iman & IMAN_IE, 0);
        // A posted event must still latch IP and EINT (guest may poll / enable later).
        d.assert_interrupt();
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IP, 0, "IP latched while masked");
        assert_ne!(
            r32(&mut d, USBSTS) & STS_EINT,
            0,
            "EINT latched while masked"
        );
    }

    #[test]
    fn portsc_is_powered_and_disconnected() {
        let mut d = dev();
        let p = r32(&mut d, PORTSC0);
        assert_ne!(p & PORTSC_PP, 0, "port powered");
        assert_eq!(p & PORTSC_CCS, 0, "nothing connected");
    }

    #[test]
    fn portsc_change_bits_are_rw1c_and_preserve_power() {
        let mut d = dev();
        d.ports[0] |= PORTSC_CSC;
        assert_ne!(r32(&mut d, PORTSC0) & PORTSC_CSC, 0);
        w32(&mut d, PORTSC0, PORTSC_PP);
        assert_ne!(
            r32(&mut d, PORTSC0) & PORTSC_CSC,
            0,
            "0 in change bit must not clear CSC"
        );
        assert_ne!(r32(&mut d, PORTSC0) & PORTSC_PP, 0, "PP preserved");
        w32(&mut d, PORTSC0, PORTSC_PP | PORTSC_CSC);
        assert_eq!(r32(&mut d, PORTSC0) & PORTSC_CSC, 0, "CSC is RW1C");
        assert_ne!(r32(&mut d, PORTSC0) & PORTSC_PP, 0, "PP still set");
    }

    #[test]
    fn port_reset_completes_and_latches_full_speed() {
        let mut d = dev();
        // Cold-plug a full-speed mock on port 1 and mark it connected (as the run
        // scan would), then drive a port reset.
        d.port_models[0] = Some(Arc::new(super::super::super::mock::MockUsbDevice::new()));
        d.set_port_connected(0);
        w32(&mut d, PORTSC0, PORTSC_PR | PORTSC_PP);
        let p = r32(&mut d, PORTSC0);
        assert_eq!(p & PORTSC_PR, 0, "PR self-cleared");
        assert_ne!(p & PORTSC_PED, 0, "port enabled after reset");
        assert_ne!(p & PORTSC_PRC, 0, "reset-change latched");
        assert_eq!(
            (p & PORTSC_SPEED_MASK) >> PORTSC_SPEED_SHIFT,
            SPEED_FULL,
            "full speed"
        );
        assert_eq!(d.work.port_events, vec![1], "port change event queued");
    }

    #[test]
    fn config_and_dcbaap_store_and_read_back() {
        let mut d = dev();
        w32(&mut d, CONFIG, NUM_SLOTS);
        assert_eq!(r32(&mut d, CONFIG), NUM_SLOTS, "CONFIG.MaxSlotsEn stored");

        let addr: u64 = 0x0000_0001_2345_6000;
        d.write(0, DCBAAP, &addr.to_le_bytes());
        assert_eq!(r64(&mut d, DCBAAP), addr, "64-bit store/readback");
        assert_eq!(r32(&mut d, DCBAAP), addr as u32, "low dword");
        assert_eq!(r32(&mut d, DCBAAP + 4), (addr >> 32) as u32, "high dword");
    }

    #[test]
    fn sixtyfour_bit_register_written_in_two_halves() {
        let mut d = dev();
        let addr: u64 = 0x0000_00ab_cdef_0000;
        w32(&mut d, DCBAAP, addr as u32);
        w32(&mut d, DCBAAP + 4, (addr >> 32) as u32);
        assert_eq!(
            r64(&mut d, DCBAAP),
            addr,
            "halves reassemble to the 64-bit value"
        );
    }

    #[test]
    fn doorbell_queues_worker_work() {
        let mut d = dev();
        // Command ring doorbell (DB[0]).
        w32(&mut d, DB_BASE, 0);
        assert!(d.work.cmd_doorbell, "command doorbell queued");
        // Slot 1 doorbell, EP0 (DCI 1).
        w32(&mut d, DB_BASE + 4, 1);
        assert_eq!(
            d.work.ep_doorbells,
            vec![(1u8, 1u8)],
            "slot doorbell queued"
        );
    }

    #[test]
    fn run_start_flags_worker() {
        let mut d = dev();
        w32(&mut d, USBCMD, CMD_RS);
        assert!(d.work.run_started, "run edge flagged for the port scan");
    }
}
