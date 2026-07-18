// Copyright 2021 Arm Limited (or its affiliates). All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

//! ARM PrimeCell General Purpose Input/Output(PL061)
//!
//! This module implements an ARM PrimeCell General Purpose Input/Output(PL061) to support gracefully poweroff microvm from external.
//!

use std::fmt;
use std::os::fd::AsRawFd;
use std::result;

use polly::event_manager::{EventManager, Subscriber};
use utils::byte_order::{read_le_u32, write_le_u32};
use utils::epoll::{EpollEvent, EventSet};
use utils::eventfd::EventFd;

use crate::bus::BusDevice;
use crate::legacy::IrqChip;

const OFS_DATA: u64 = 0x400; // Data Register
const GPIODIR: u64 = 0x400; // Direction Register
const GPIOIS: u64 = 0x404; // Interrupt Sense Register
const GPIOIBE: u64 = 0x408; // Interrupt Both Edges Register
const GPIOIEV: u64 = 0x40c; // Interrupt Event Register
const GPIOIE: u64 = 0x410; // Interrupt Mask Register
const GPIORIE: u64 = 0x414; // Raw Interrupt Status Register
const GPIOMIS: u64 = 0x418; // Masked Interrupt Status Register
const GPIOIC: u64 = 0x41c; // Interrupt Clear Register
const GPIOAFSEL: u64 = 0x420; // Mode Control Select Register
// From 0x424 to 0xFDC => reserved space.
// From 0xFE0 to 0xFFC => Peripheral and PrimeCell Identification Registers which are Read Only registers.
// These registers can conceptually be treated as a 32-bit register, and PartNumber[11:0] is used to identify the peripheral.
// We are putting the expected values (look at 'Reset value' column from above mentioned document) in an array.
const GPIO_ID: [u8; 8] = [0x61, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];
// ID Margins
const GPIO_ID_LOW: u64 = 0xfe0;
const GPIO_ID_HIGH: u64 = 0x1000;

// GPIO lines wired to `gpio-keys` buttons in the FDT (see `create_gpio_node`). Each is a
// distinct key so the host can ask a cooperative guest to power off, suspend, or reboot:
//   line 3 (0x8)  → KEY_POWER   (logind HandlePowerKey   → poweroff)   ← shutdown eventfd
//   line 4 (0x10) → KEY_SLEEP   (logind HandleSuspendKey → suspend)    ← suspend eventfd
//   line 5 (0x20) → KEY_RESTART (logind HandleRebootKey  → reboot)     ← restart eventfd
//   line 6 (0x40) → KEY_WAKEUP  (a `wakeup-source` button)            ← wake eventfd
// The wake button's FDT node carries `wakeup-source`, so the guest arms its irq for wake during
// s2idle (`enable_irq_wake`); pulsing it is the only line that brings the guest OUT of suspend-to-idle
// (M9 restore: the worker injects it after reloading a quiesced snapshot). The other buttons are not
// wake sources — their edges are masked while suspended, so they do not wake the guest.
const POWER_BIT: u32 = 0x8;
const SUSPEND_BIT: u32 = 0x10;
const RESTART_BIT: u32 = 0x20;
const WAKE_BIT: u32 = 0x40;

#[derive(Debug)]
pub enum Error {
    BadWriteOffset(u64),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::BadWriteOffset(offset) => write!(f, "Bad Write Offset: {offset}"),
        }
    }
}

type Result<T> = result::Result<T, Error>;

/// The PL061 register file, for M9 snapshot save/restore. The GPIO registers are the one piece of
/// device state that does NOT ride the guest RAM snapshot; restoring them into a fresh device
/// before the guest resumes is what lets the injected wake demux correctly (`GPIOMIS = istate & im`
/// — a fresh device's `im=0` would swallow the wake) and lets the guest's `pl061_resume` see the
/// register state it left. ~32 bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpioState {
    pub data: u32,
    pub dir: u32,
    pub isense: u32,
    pub ibe: u32,
    pub iev: u32,
    pub im: u32,
    pub istate: u32,
    pub afsel: u32,
}

/// A GPIO device following the PL061 specification.
pub struct Gpio {
    // Data Register
    data: u32,
    // Direction Register
    dir: u32,
    // Interrupt Sense Register
    isense: u32,
    // Interrupt Both Edges Register
    ibe: u32,
    // Interrupt Event Register
    iev: u32,
    // Interrupt Mask Register
    im: u32,
    // Raw Interrupt Status Register
    istate: u32,
    // Mode Control Select Register
    afsel: u32,
    // GPIO irq_field
    interrupt_evt: EventFd,
    intc: Option<IrqChip>,
    irq_line: Option<u32>,
    shutdown_efd: EventFd,
    suspend_efd: EventFd,
    restart_efd: EventFd,
    wake_efd: EventFd,
}

impl Gpio {
    /// Constructs an PL061 GPIO device.
    pub fn new(
        shutdown_efd: EventFd,
        suspend_efd: EventFd,
        restart_efd: EventFd,
        wake_efd: EventFd,
        interrupt_evt: EventFd,
    ) -> Self {
        Self {
            data: 0,
            dir: 0,
            isense: 0,
            ibe: 0,
            iev: 0,
            im: 0,
            istate: 0,
            afsel: 0,
            interrupt_evt,
            intc: None,
            irq_line: None,
            shutdown_efd,
            suspend_efd,
            restart_efd,
            wake_efd,
        }
    }

    pub fn set_intc(&mut self, intc: IrqChip) {
        self.intc = Some(intc);
    }

    /// Capture the PL061 register file for a snapshot (M9 save).
    pub fn save_state(&self) -> GpioState {
        GpioState {
            data: self.data,
            dir: self.dir,
            isense: self.isense,
            ibe: self.ibe,
            iev: self.iev,
            im: self.im,
            istate: self.istate,
            afsel: self.afsel,
        }
    }

    /// Restore the PL061 register file from a snapshot (M9 restore), before the guest resumes — so
    /// the injected wake demuxes (`GPIOMIS = istate & im`) and `pl061_resume` sees its own state.
    pub fn restore_state(&mut self, s: &GpioState) {
        self.data = s.data;
        self.dir = s.dir;
        self.isense = s.isense;
        self.ibe = s.ibe;
        self.iev = s.iev;
        self.im = s.im;
        self.istate = s.istate;
        self.afsel = s.afsel;
    }

    pub fn set_irq_line(&mut self, irq: u32) {
        debug!("SET_IRQ_LINE (GPIO)={irq}");
        self.irq_line = Some(irq);
    }

    fn handle_write(&mut self, offset: u64, val: u32) -> Result<()> {
        if offset < OFS_DATA {
            // In order to write to data register, the corresponding bits in the mask, resulting
            // from the offsite[9:2], must be HIGH. otherwise the bit values remain unchanged.
            let mask = (offset >> 2) as u32 & self.dir;
            self.data = (self.data & !mask) | (val & mask);
        } else {
            match offset {
                GPIODIR => {
                    /* Direction Register */
                    self.dir = val & 0xff;
                }
                GPIOIS => {
                    /* Interrupt Sense Register */
                    self.isense = val & 0xff;
                }
                GPIOIBE => {
                    /* Interrupt Both Edges Register */
                    self.ibe = val & 0xff;
                }
                GPIOIEV => {
                    /* Interrupt Event Register */
                    self.iev = val & 0xff;
                }
                GPIOIE => {
                    /* Interrupt Mask Register */
                    self.im = val & 0xff;
                }
                GPIOIC => {
                    /* Interrupt Clear Register */
                    self.istate &= !val;
                }
                GPIOAFSEL => {
                    /* Mode Control Select Register */
                    self.afsel = val & 0xff;
                }
                o => {
                    return Err(Error::BadWriteOffset(o));
                }
            }
        }
        Ok(())
    }

    pub fn trigger_power_key(&mut self, press: bool) {
        debug!(
            "Generate a power key {} event",
            if press { "press" } else { "release" }
        );
        self.trigger_key(POWER_BIT, press);
    }

    pub fn trigger_suspend_key(&mut self, press: bool) {
        debug!(
            "Generate a suspend key {} event",
            if press { "press" } else { "release" }
        );
        self.trigger_key(SUSPEND_BIT, press);
    }

    pub fn trigger_restart_key(&mut self, press: bool) {
        debug!(
            "Generate a restart key {} event",
            if press { "press" } else { "release" }
        );
        self.trigger_key(RESTART_BIT, press);
    }

    pub fn trigger_wake_key(&mut self, press: bool) {
        debug!(
            "Generate a wake key {} event",
            if press { "press" } else { "release" }
        );
        self.trigger_key(WAKE_BIT, press);
    }

    /// Drive a single GPIO line (`bit`) high (press) or low (release) and raise the
    /// shared interrupt. Per-line so multiple `gpio-keys` buttons (poweroff + suspend)
    /// can coexist on the one PL061 without clobbering each other's data-register state.
    fn trigger_key(&mut self, bit: u32, press: bool) {
        self.istate |= bit;
        if press {
            self.data |= bit;
        } else {
            self.data &= !bit;
        }
        self.trigger_gpio_interrupt();
    }

    fn trigger_gpio_interrupt(&self) {
        if let Some(intc) = &self.intc
            && let Err(e) = intc
                .lock()
                .unwrap()
                .set_irq(self.irq_line, Some(&self.interrupt_evt))
        {
            warn!("Error signalling irq: {e:?}");
        }
    }
}

impl BusDevice for Gpio {
    fn read(&mut self, _base: u64, offset: u64, data: &mut [u8]) {
        let value;
        let mut read_ok = true;

        if (GPIO_ID_LOW..GPIO_ID_HIGH).contains(&offset) {
            let index = ((offset - GPIO_ID_LOW) >> 2) as usize;
            value = u32::from(GPIO_ID[index]);
        } else if offset < OFS_DATA {
            value = self.data & ((offset >> 2) as u32);
            if value != 0 {
                // Now that the guest has read it, send a key release event for exactly
                // the line(s) it just read (so poweroff and suspend release independently).
                self.trigger_key(value, false);
            }
        } else {
            value = match offset {
                GPIODIR => self.dir,
                GPIOIS => self.isense,
                GPIOIBE => self.ibe,
                GPIOIEV => self.iev,
                GPIOIE => self.im,
                GPIORIE => self.istate,
                GPIOMIS => self.istate & self.im,
                GPIOAFSEL => self.afsel,
                _ => {
                    read_ok = false;
                    0
                }
            };
        }

        if read_ok && data.len() <= 4 {
            write_le_u32(data, value);
        } else {
            warn!(
                "Invalid GPIO PL061 read: offset {}, data length {}",
                offset,
                data.len()
            );
        }
    }

    fn write(&mut self, _base: u64, offset: u64, data: &[u8]) {
        if data.len() <= 4 {
            let value = read_le_u32(data);
            if let Err(e) = self.handle_write(offset, value) {
                warn!("Failed to write to GPIO PL061 device: {e}");
            }
        } else {
            warn!(
                "Invalid GPIO PL061 write: offset {}, data length {}",
                offset,
                data.len()
            );
        }
    }
}

impl Subscriber for Gpio {
    fn process(&mut self, event: &EpollEvent, _event_manager: &mut EventManager) {
        let source = event.fd();

        match source {
            _ if source == self.shutdown_efd.as_raw_fd() => {
                _ = self.shutdown_efd.read();
                // Send a poweroff (KEY_POWER) key press event.
                self.trigger_power_key(true);
            }
            _ if source == self.suspend_efd.as_raw_fd() => {
                _ = self.suspend_efd.read();
                // Send a suspend (KEY_SLEEP) key press event.
                self.trigger_suspend_key(true);
            }
            _ if source == self.restart_efd.as_raw_fd() => {
                _ = self.restart_efd.read();
                // Send a reboot (KEY_RESTART) key press event.
                self.trigger_restart_key(true);
            }
            _ if source == self.wake_efd.as_raw_fd() => {
                _ = self.wake_efd.read();
                // Send a wake (KEY_WAKEUP) key press event — brings the guest out of s2idle.
                self.trigger_wake_key(true);
            }
            _ => warn!("Unexpected gpio event received: {source:?}"),
        }
    }

    fn interest_list(&self) -> Vec<EpollEvent> {
        vec![
            EpollEvent::new(EventSet::IN, self.shutdown_efd.as_raw_fd() as u64),
            EpollEvent::new(EventSet::IN, self.suspend_efd.as_raw_fd() as u64),
            EpollEvent::new(EventSet::IN, self.restart_efd.as_raw_fd() as u64),
            EpollEvent::new(EventSet::IN, self.wake_efd.as_raw_fd() as u64),
        ]
    }
}
