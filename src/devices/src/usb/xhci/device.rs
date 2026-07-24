// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The xHCI controller register file (a `BusDevice`).
//!
//! Stage A (controller bring-up) implements a *functional register file with a
//! stub data path*: enough of the capability / operational / runtime / doorbell
//! register semantics that the guest's stock `xhci-plat` driver resets the
//! controller, reads its parameters, programs the DCBAA / command ring / event
//! ring, and declares the HCD running with the root hub up — all with **no
//! device connected and no ring processing**. The command/event/transfer ring
//! walkers and the `UsbDeviceModel` gadgets land in Stage B.
//!
//! The premise (verified against Linux `drivers/usb/host/xhci.c`, v6.12): a
//! `generic-xhci` platform controller with no NEC quirk issues **zero xHCI
//! commands** during `xhci_run`/`xhci_start` — it only sets `USBCMD.RS` and
//! waits for `USBSTS.HCH` to clear. `xhci_reset` waits for `USBCMD.HCRST` to
//! self-clear and then for `USBSTS.CNR` to clear; `xhci_mem_init` reads
//! `PAGESIZE` (bit 0 = 4 KiB), `HCSPARAMS1/2`, `HCCPARAMS1` (walking the
//! extended-capabilities list for the Supported Protocol cap), and stores
//! `DCBAAP`/`CRCR`/`CONFIG` and the interrupter-0 event-ring registers. So a
//! register file alone brings the HCD up; no command completions are needed
//! until a device connects.

use utils::eventfd::EventFd;

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

// Current Connect Status (RO) — Stage B sets it on connect; Stage A asserts it absent.
#[allow(dead_code)]
const PORTSC_CCS: u32 = 1 << 0;
const PORTSC_PED: u32 = 1 << 1; // Port Enabled/Disabled (RW1C to disable)
const PORTSC_PR: u32 = 1 << 4; // Port Reset (RW1S, self-clearing)
const PORTSC_PLS_SHIFT: u32 = 5;
const PORTSC_PLS_MASK: u32 = 0xf << PORTSC_PLS_SHIFT; // Port Link State (RWS via LWS)
const PORTSC_PP: u32 = 1 << 9; // Port Power (RW)
const PORTSC_PIC_MASK: u32 = 0x3 << 14; // Port Indicator Control (RW)
const PORTSC_LWS: u32 = 1 << 16; // Link State Write Strobe (gates PLS write)
const PORTSC_WAKE_MASK: u32 = 0x7 << 25; // WCE/WDE/WOE (RW)
const PORTSC_WPR: u32 = 1 << 31; // Warm Port Reset (RW1S, self-clearing)
/// PORTSC change bits (CSC, PEC, WRC, OCC, PRC, PLC, CEC) — all write-1-to-clear.
const PORTSC_RW1C_MASK: u32 = 0x7f << 17;
/// Port Link State value for a powered but disconnected port (RxDetect).
const PLS_RX_DETECT: u32 = 5;
/// A freshly powered, nothing-connected USB2 port.
const PORTSC_DEFAULT: u32 = PORTSC_PP | (PLS_RX_DETECT << PORTSC_PLS_SHIFT);

// ---- extended capability: Supported Protocol (USB 2.0, ports 1-4) -----------

const XECP_CAP_ID_SUPPORTED_PROTOCOL: u32 = 0x02;
// dword0: CapID=0x02, Next=0 (end of list), MinorRev=0x00, MajorRev=0x02.
const XECP_DW0: u32 = XECP_CAP_ID_SUPPORTED_PROTOCOL | (0x00 << 8) | (0x00 << 16) | (0x02 << 24);
// dword1: name string "USB " (little-endian bytes 'U','S','B',' ').
const XECP_DW1: u32 = u32::from_le_bytes(*b"USB ");
// dword2: Compatible Port Offset = 1, Compatible Port Count = 4, PSIC = 0.
const XECP_DW2: u32 = 0x01 | ((NUM_PORTS as u32) << 8);
// dword3: Protocol Slot Type = 0.
const XECP_DW3: u32 = 0x0000_0000;

/// The emulated xHCI controller: a synchronous register file on the vcpu thread.
///
/// Stage A carries the interrupt plumbing (`intc` + `irq_line` + `interrupt_evt`,
/// wired exactly like the PL031 RTC) but never asserts — with no device attached
/// there are no transfer/event-ring writes to signal. Stage B's worker thread
/// uses [`XhciDevice::assert_interrupt`] on each event-ring batch.
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

    // Interrupt plumbing (unused until Stage B raises event-ring interrupts).
    #[allow(dead_code)]
    intc: Option<IrqChip>,
    #[allow(dead_code)]
    irq_line: Option<u32>,
    #[allow(dead_code)]
    interrupt_evt: EventFd,
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
        }
    }

    /// Wire the controller to the interrupt controller (macOS/HVF path), mirroring
    /// the PL031 RTC. On KVM `intc` stays `None` and the event-ring worker pokes the
    /// registered irqfd directly. Unused in Stage A (no events are generated).
    pub fn set_intc(&mut self, intc: IrqChip) {
        self.intc = Some(intc);
    }

    /// The SPI line this controller's interrupter raises (edge-triggered; the FDT
    /// declares it so, matching the RTC/GPIO one-shot `hv_gic_set_spi` pulse).
    pub fn set_irq_line(&mut self, irq: u32) {
        self.irq_line = Some(irq);
    }

    /// Assert the interrupter-0 SPI (HVF) or poke the irqfd (KVM). Stage B calls this
    /// on each event-ring batch while `IMAN.IE` is set; unused in Stage A.
    #[allow(dead_code)]
    pub fn assert_interrupt(&mut self) {
        // Only signal when the interrupter is enabled (IMAN.IE) and the guest has
        // globally enabled interrupts (USBCMD.INTE).
        if self.iman & IMAN_IE == 0 || self.usbcmd & CMD_INTE == 0 {
            return;
        }
        self.iman |= IMAN_IP;
        self.usbsts |= STS_EINT;
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

    /// A guest-visible host controller reset (`USBCMD.HCRST`): clear the whole
    /// register file back to the post-power-on state, but with `CNR` **cleared**
    /// (the controller is ready) — this is the transition `xhci_reset` waits on.
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
            _ if off == OP_BASE + OP_CRCR_LO => self.crcr as u32,
            _ if off == OP_BASE + OP_CRCR_HI => (self.crcr >> 32) as u32,
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
                self.crcr = (self.crcr & 0xffff_ffff_0000_0000) | u64::from(val)
            }
            _ if off == OP_BASE + OP_CRCR_HI => {
                self.crcr = (self.crcr & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32)
            }
            _ if off == OP_BASE + OP_DCBAAP_LO => {
                self.dcbaap = (self.dcbaap & 0xffff_ffff_0000_0000) | u64::from(val)
            }
            _ if off == OP_BASE + OP_DCBAAP_HI => {
                self.dcbaap = (self.dcbaap & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32)
            }
            _ if off == OP_BASE + OP_CONFIG => self.config = val,

            // Runtime registers (interrupter 0).
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMAN => {
                let mut iman = self.iman;
                if val & IMAN_IP != 0 {
                    iman &= !IMAN_IP; // write-1-to-clear
                }
                iman = (iman & !IMAN_IE) | (val & IMAN_IE);
                self.iman = iman;
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_IMOD => self.imod = val,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTSZ => self.erstsz = val,
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_LO => {
                self.erstba = (self.erstba & 0xffff_ffff_0000_0000) | u64::from(val)
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERSTBA_HI => {
                self.erstba = (self.erstba & 0x0000_0000_ffff_ffff) | (u64::from(val) << 32)
            }
            _ if off == RUNTIME_BASE + RT_IR0_BASE + IR_ERDP_LO => {
                // ERDP bit 3 (EHB) is write-1-to-clear; we never set it (no events),
                // so keep the dequeue pointer and force EHB low.
                let ptr = val & !(1 << 3);
                self.erdp = (self.erdp & 0xffff_ffff_0000_0000) | u64::from(ptr)
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

            // Doorbell array: accepted, no-op until Stage B walks the rings.
            _ if (DB_BASE..DB_BASE + (NUM_SLOTS as u64 + 1) * 4).contains(&off) => {}

            // Read-only capability registers and reserved space: ignore writes.
            _ => {}
        }
    }

    /// USBCMD write: a HCRST triggers a full reset; otherwise store the RW bits and
    /// keep `USBSTS.HCH` in sync with `RS`.
    fn handle_usbcmd(&mut self, val: u32) {
        if val & CMD_HCRST != 0 {
            self.reset_controller();
            return;
        }
        // Light HC reset (bit 7) self-clears with no other effect here.
        let stored = val & CMD_STORE_MASK;
        self.usbcmd = stored;
        if stored & CMD_RS != 0 {
            self.usbsts &= !STS_HCH; // running
        } else {
            self.usbsts |= STS_HCH; // halted
        }
    }

    /// PORTSC write with the spec's mixed RW / RW1C / RW1S semantics; read-only bits
    /// (CCS, OCA, speed, CAS, DR) are preserved.
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
        // PR / WPR initiate a reset. With nothing connected the reset completes
        // instantly and the bit self-clears — leave it low, raise no change event.
        let _ = (PORTSC_PR, PORTSC_WPR);
        self.ports[idx] = p;
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
        // Follow HCCPARAMS1.xECP to the caps list.
        let hcc = r32(&mut d, CAP_HCCPARAMS1);
        let xecp = u64::from((hcc >> 16) << 2);
        let dw0 = r32(&mut d, xecp);
        assert_eq!(dw0 & 0xff, XECP_CAP_ID_SUPPORTED_PROTOCOL, "Supported Protocol cap id");
        assert_eq!((dw0 >> 8) & 0xff, 0, "Next pointer = 0 (end of list)");
        assert_eq!((dw0 >> 24) & 0xff, 0x02, "Major revision 2 (USB 2.0)");
        // Name string "USB ".
        assert_eq!(r32(&mut d, xecp + 4), u32::from_le_bytes(*b"USB "));
        // Compatible port offset 1, count 4.
        let dw2 = r32(&mut d, xecp + 8);
        assert_eq!(dw2 & 0xff, 1, "compatible port offset");
        assert_eq!((dw2 >> 8) & 0xff, NUM_PORTS as u32, "compatible port count");
    }

    #[test]
    fn hcrst_self_clears_and_clears_cnr() {
        let mut d = dev();
        // Fresh controller: not-ready (CNR) and halted (HCH).
        assert_ne!(r32(&mut d, USBSTS) & STS_CNR, 0, "CNR set at power-on");
        // Issue a host controller reset.
        w32(&mut d, USBCMD, CMD_HCRST);
        // HCRST reads back as 0 (self-clearing) ...
        assert_eq!(r32(&mut d, USBCMD) & CMD_HCRST, 0, "HCRST self-cleared");
        // ... and the controller is now ready (CNR clear) and still halted.
        assert_eq!(r32(&mut d, USBSTS) & STS_CNR, 0, "CNR cleared after reset");
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "still halted after reset");
    }

    #[test]
    fn hch_tracks_run_stop() {
        let mut d = dev();
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "halted before run");
        // Run.
        w32(&mut d, USBCMD, CMD_RS);
        assert_eq!(r32(&mut d, USBSTS) & STS_HCH, 0, "HCH clears when running");
        // Stop.
        w32(&mut d, USBCMD, 0);
        assert_ne!(r32(&mut d, USBSTS) & STS_HCH, 0, "HCH sets when stopped");
    }

    #[test]
    fn usbsts_eint_is_write_one_to_clear() {
        let mut d = dev();
        // Seed an event-interrupt status (Stage B would set this on a real event).
        d.usbsts |= STS_EINT;
        assert_ne!(r32(&mut d, USBSTS) & STS_EINT, 0);
        // Writing 0 to the bit must NOT clear it.
        w32(&mut d, USBSTS, 0);
        assert_ne!(r32(&mut d, USBSTS) & STS_EINT, 0, "0 must not clear EINT");
        // Writing 1 clears it.
        w32(&mut d, USBSTS, STS_EINT);
        assert_eq!(r32(&mut d, USBSTS) & STS_EINT, 0, "EINT is RW1C");
    }

    #[test]
    fn iman_ip_is_rw1c_and_ie_is_rw() {
        let mut d = dev();
        // Enable interrupts (IE) and seed a pending interrupt (IP).
        w32(&mut d, IMAN0, IMAN_IE);
        d.iman |= IMAN_IP;
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IP, 0);
        // Writing IE=1 without IP=1 leaves IP set ...
        w32(&mut d, IMAN0, IMAN_IE);
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IP, 0, "IP not cleared by writing 0");
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IE, 0, "IE stays enabled");
        // ... writing IP=1 clears it.
        w32(&mut d, IMAN0, IMAN_IE | IMAN_IP);
        assert_eq!(r32(&mut d, IMAN0) & IMAN_IP, 0, "IP is RW1C");
        assert_ne!(r32(&mut d, IMAN0) & IMAN_IE, 0, "IE preserved");
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
        // Seed a connect-status-change (bit 17) as a hotplug would. PP is set by default.
        const CSC: u32 = 1 << 17;
        d.ports[0] |= CSC;
        assert_ne!(r32(&mut d, PORTSC0) & CSC, 0);
        // The driver clears change bits with a read-modify-write, always writing PP
        // back. Writing PP with the change bit 0 must NOT clear the change bit.
        w32(&mut d, PORTSC0, PORTSC_PP);
        assert_ne!(r32(&mut d, PORTSC0) & CSC, 0, "0 in the change bit must not clear CSC");
        assert_ne!(r32(&mut d, PORTSC0) & PORTSC_PP, 0, "PP preserved");
        // Writing 1 to the change bit (PP still set) clears just that bit.
        w32(&mut d, PORTSC0, PORTSC_PP | CSC);
        assert_eq!(r32(&mut d, PORTSC0) & CSC, 0, "CSC is RW1C");
        assert_ne!(r32(&mut d, PORTSC0) & PORTSC_PP, 0, "PP still set");
    }

    #[test]
    fn config_and_dcbaap_store_and_read_back() {
        let mut d = dev();
        w32(&mut d, CONFIG, NUM_SLOTS);
        assert_eq!(r32(&mut d, CONFIG), NUM_SLOTS, "CONFIG.MaxSlotsEn stored");

        // DCBAAP written as a single 64-bit access, read back whole and in halves.
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
        // Low dword then high dword, as the driver's xhci_write_64 does on a
        // 32-bit-only accessor.
        w32(&mut d, DCBAAP, addr as u32);
        w32(&mut d, DCBAAP + 4, (addr >> 32) as u32);
        assert_eq!(r64(&mut d, DCBAAP), addr, "halves reassemble to the 64-bit value");
    }
}
