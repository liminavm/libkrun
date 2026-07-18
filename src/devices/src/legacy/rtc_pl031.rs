// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! ARM PL031 Real Time Clock
//!
//! This module implements a PL031 Real Time Clock (RTC) that provides to provides long time base counter.
//! This is achieved by generating an interrupt signal after counting for a programmed number of cycles of
//! a real-time clock input.
//!

use std::fmt;
use std::os::fd::AsRawFd;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, Instant};
use std::{io, result};

use polly::event_manager::{EventManager, Subscriber};
use utils::byte_order;
use utils::epoll::{EpollEvent, EventSet};
use utils::eventfd::EventFd;

use crate::legacy::IrqChip;
use crate::BusDevice;
//use bus::Error;

// As you can see in https://static.docs.arm.com/ddi0224/c/real_time_clock_pl031_r1p3_technical_reference_manual_DDI0224C.pdf
// at section 3.2 Summary of RTC registers, the total size occupied by this device is 0x000 -> 0xFFC + 4 = 0x1000.
// From 0x0 to 0x1C we have following registers:
const RTCDR: u64 = 0x0; // Data Register.
const RTCMR: u64 = 0x4; // Match Register.
const RTCLR: u64 = 0x8; // Load Regiser.
const RTCCR: u64 = 0xc; // Control Register.
const RTCIMSC: u64 = 0x10; // Interrupt Mask Set or Clear Register.
const RTCRIS: u64 = 0x14; // Raw Interrupt Status.
const RTCMIS: u64 = 0x18; // Masked Interrupt Status.
const RTCICR: u64 = 0x1c; // Interrupt Clear Register.
// From 0x020 to 0xFDC => reserved space.
// From 0xFE0 to 0x1000 => Peripheral and PrimeCell Identification Registers which are Read Only registers.
// AMBA standard devices have CIDs (Cell IDs) and PIDs (Peripheral IDs). The linux kernel will look for these in order to assert the identity
// of these devices (i.e look at the `amba_device_try_add` function).
// We are putting the expected values (look at 'Reset value' column from above mentioned document) in an array.
const PL031_ID: [u8; 8] = [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];
// We are only interested in the margins.
const AMBA_ID_LOW: u64 = 0xFE0;
const AMBA_ID_HIGH: u64 = 0x1000;

#[derive(Debug)]
pub enum Error {
    BadWriteOffset(u64),
    InterruptFailure(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::BadWriteOffset(offset) => write!(f, "Bad Write Offset: {offset}"),
            Error::InterruptFailure(e) => write!(f, "Failed to trigger interrupt: {e}"),
        }
    }
}
type Result<T> = result::Result<T, Error>;

/// A command to the alarm timer thread.
enum AlarmCmd {
    /// Fire `alarm_wake` after this duration from now (cancels any prior arm).
    Arm(Duration),
    /// Cancel a pending alarm.
    Disarm,
}

/// A RTC device following the PL031 specification..
pub struct RTC {
    previous_now: Instant,
    tick_offset: i64,
    // The Match Register: the guest programs the alarm time here (see the timer thread + `Subscriber`
    // impl, which turn a match into a real wakeup IRQ — needed so a guest can wake from
    // suspend-to-idle via `rtcwake`).
    match_value: u32,
    // Writes to this register load an update value into the RTC.
    load: u32,
    imsc: u32,
    ris: u32,
    interrupt_evt: EventFd,
    // Alarm plumbing. `alarm_wake` is signalled by the timer thread when the programmed match time
    // elapses; the event manager then calls `process`, which raises the SPI via `intc`. `alarm_tx`
    // arms/disarms that thread from the MMIO write path.
    intc: Option<IrqChip>,
    irq_line: Option<u32>,
    alarm_wake: EventFd,
    alarm_tx: Sender<AlarmCmd>,
}

impl RTC {
    /// Constructs an AMBA PL031 RTC device.
    pub fn new(interrupt_evt: EventFd) -> RTC {
        let alarm_wake = EventFd::new(utils::eventfd::EFD_NONBLOCK)
            .expect("failed to create PL031 alarm eventfd");
        let alarm_tx = Self::spawn_alarm_thread(
            alarm_wake
                .try_clone()
                .expect("failed to clone PL031 alarm eventfd"),
        );
        RTC {
            // This is used only for duration measuring purposes.
            previous_now: Instant::now(),
            tick_offset: utils::time::get_time(utils::time::ClockType::Real) as i64,
            match_value: 0,
            load: 0,
            imsc: 0,
            ris: 0,
            interrupt_evt,
            intc: None,
            irq_line: None,
            alarm_wake,
            alarm_tx,
        }
    }

    /// Wire the RTC to the interrupt controller so a fired alarm can raise its SPI (macOS/HVF path,
    /// mirroring the GPIO device). On the KVM path `intc` stays `None` and the alarm signals the
    /// `interrupt_evt` irqfd directly.
    pub fn set_intc(&mut self, intc: IrqChip) {
        self.intc = Some(intc);
    }

    /// The SPI line this RTC's alarm raises.
    pub fn set_irq_line(&mut self, irq: u32) {
        self.irq_line = Some(irq);
    }

    /// The alarm timer: a single long-lived thread that fires `alarm_wake` when the guest-programmed
    /// match time elapses. Kept portable (a thread + eventfd, not a Linux `timerfd`) so it works on
    /// macOS too. Exits when the RTC — and thus `alarm_tx` — is dropped.
    fn spawn_alarm_thread(waker: EventFd) -> Sender<AlarmCmd> {
        let (tx, rx) = mpsc::channel::<AlarmCmd>();
        thread::Builder::new()
            .name("pl031-alarm".into())
            .spawn(move || {
                let mut deadline: Option<Instant> = None;
                loop {
                    let timeout = match deadline {
                        Some(d) => {
                            let now = Instant::now();
                            if now >= d {
                                deadline = None;
                                let _ = waker.write(1);
                                continue;
                            }
                            Some(d - now)
                        }
                        None => None,
                    };
                    let cmd = match timeout {
                        Some(t) => match rx.recv_timeout(t) {
                            Ok(c) => c,
                            Err(mpsc::RecvTimeoutError::Timeout) => {
                                deadline = None;
                                let _ = waker.write(1);
                                continue;
                            }
                            Err(mpsc::RecvTimeoutError::Disconnected) => return,
                        },
                        None => match rx.recv() {
                            Ok(c) => c,
                            Err(_) => return, // device dropped
                        },
                    };
                    match cmd {
                        AlarmCmd::Arm(dur) => deadline = Some(Instant::now() + dur),
                        AlarmCmd::Disarm => deadline = None,
                    }
                }
            })
            .expect("failed to spawn PL031 alarm thread");
        tx
    }

    /// (Re)arm or cancel the alarm timer to reflect the current match/mask registers. Called after
    /// any write that can change when — or whether — the alarm should fire.
    fn update_alarm(&self) {
        if self.imsc & 1 == 0 {
            let _ = self.alarm_tx.send(AlarmCmd::Disarm);
            return;
        }
        // PL031 counts whole seconds; fire just past the match second (+100 ms) so `get_time()` has
        // ticked to/over `match_value` when we raise the IRQ.
        let secs = self.match_value.saturating_sub(self.get_time());
        let delay = Duration::from_secs(u64::from(secs)) + Duration::from_millis(100);
        let _ = self.alarm_tx.send(AlarmCmd::Arm(delay));
    }

    /// Raise the alarm interrupt (called from `process` once the timer fires). Sets the raw
    /// interrupt status and asserts the SPI (HVF) or pokes the irqfd (KVM). No-op if the guest
    /// masked the interrupt in the meantime.
    fn fire_alarm(&mut self) {
        if self.imsc & 1 == 0 {
            return;
        }
        self.ris = 1;
        if let Some(intc) = &self.intc {
            if let Err(e) = intc
                .lock()
                .unwrap()
                .set_irq(self.irq_line, Some(&self.interrupt_evt))
            {
                warn!("RTC PL031: failed to signal alarm IRQ: {e:?}");
            }
        } else if let Err(e) = self.interrupt_evt.write(1) {
            warn!("RTC PL031: failed to signal alarm irqfd: {e:?}");
        }
    }

    fn trigger_interrupt(&mut self) -> Result<()> {
        self.interrupt_evt.write(1).map_err(Error::InterruptFailure)
    }

    fn get_time(&self) -> u32 {
        let ts = (self.tick_offset as i128)
            + (Instant::now().duration_since(self.previous_now).as_nanos() as i128);
        (ts / utils::time::NANOS_PER_SECOND as i128) as u32
    }

    fn handle_write(&mut self, offset: u64, val: u32) -> Result<()> {
        match offset {
            RTCMR => {
                // The Match Register programs the alarm time. A guest uses it (with the interrupt
                // unmasked, below) to wake from suspend-to-idle via `rtcwake`; (re)arm the timer to
                // the new match.
                self.match_value = val;
                self.update_alarm();
            }
            RTCLR => {
                self.load = val;
                self.previous_now = Instant::now();
                // If the unwrap fails, then the internal value of the clock has been corrupted and
                // we want to terminate the execution of the process.
                self.tick_offset = utils::time::seconds_to_nanoseconds(i64::from(val)).unwrap();
            }
            RTCIMSC => {
                self.imsc = val & 1;
                self.trigger_interrupt()?;
                // Enabling the interrupt arms the alarm; masking it disarms.
                self.update_alarm();
            }
            RTCICR => {
                // As per above mentioned doc, the interrupt is cleared by writing any data value to
                // the Interrupt Clear Register.
                self.ris = 0;
                self.trigger_interrupt()?;
            }
            RTCCR => (), // ignore attempts to turn off the timer.
            o => {
                return Err(Error::BadWriteOffset(o));
            }
        }
        Ok(())
    }
}

impl BusDevice for RTC {
    fn read(&mut self, _vcpuid: u64, offset: u64, data: &mut [u8]) {
        let mut read_ok = true;

        let v = if (AMBA_ID_LOW..AMBA_ID_HIGH).contains(&offset) {
            let index = ((offset - AMBA_ID_LOW) >> 2) as usize;
            u32::from(PL031_ID[index])
        } else {
            match offset {
                RTCDR => self.get_time(),
                RTCMR => {
                    // Even though we are not implementing RTC alarm we return the last value
                    self.match_value
                }
                RTCLR => self.load,
                RTCCR => 1, // RTC is always enabled.
                RTCIMSC => self.imsc,
                RTCRIS => self.ris,
                RTCMIS => self.ris & self.imsc,
                _ => {
                    read_ok = false;
                    0
                }
            }
        };
        if read_ok && data.len() <= 4 {
            byte_order::write_le_u32(data, v);
        } else {
            warn!(
                "Invalid RTC PL031 read: offset {}, data length {}",
                offset,
                data.len()
            );
        }
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, data: &[u8]) {
        if data.len() <= 4 {
            let v = byte_order::read_le_u32(data);
            if let Err(e) = self.handle_write(offset, v) {
                warn!("Failed to write to RTC PL031 device: {e}");
            }
        } else {
            warn!(
                "Invalid RTC PL031 write: offset {}, data length {}",
                offset,
                data.len()
            );
        }
    }
}

impl Subscriber for RTC {
    fn process(&mut self, event: &EpollEvent, _event_manager: &mut EventManager) {
        let source = event.fd();
        if source == self.alarm_wake.as_raw_fd() {
            // Drain the timer signal, then raise the alarm interrupt.
            let _ = self.alarm_wake.read();
            self.fire_alarm();
        } else {
            warn!("Unexpected RTC PL031 event received: {source:?}");
        }
    }

    fn interest_list(&self) -> Vec<EpollEvent> {
        vec![EpollEvent::new(
            EventSet::IN,
            self.alarm_wake.as_raw_fd() as u64,
        )]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rtc_read_write_and_event() {
        let mut rtc = RTC::new(EventFd::new(utils::eventfd::EFD_NONBLOCK).unwrap());
        let mut data = [0; 4];

        // Read and write to the MR register.
        byte_order::write_le_u32(&mut data, 123);
        rtc.write(0, RTCMR, &mut data);
        rtc.read(0, RTCMR, &mut data);
        let v = byte_order::read_le_u32(&data[..]);
        assert_eq!(v, 123);

        // Read and write to the LR register.
        let v = utils::time::get_time(utils::time::ClockType::Real);
        byte_order::write_le_u32(&mut data, (v / utils::time::NANOS_PER_SECOND) as u32);
        let previous_now_before = rtc.previous_now;
        rtc.write(0, RTCLR, &mut data);

        assert!(rtc.previous_now > previous_now_before);

        rtc.read(0, RTCLR, &mut data);
        let v_read = byte_order::read_le_u32(&data[..]);
        assert_eq!((v / utils::time::NANOS_PER_SECOND) as u32, v_read);

        // Read and write to IMSC register.
        // Test with non zero value.
        let non_zero = 1;
        byte_order::write_le_u32(&mut data, non_zero);
        rtc.write(0, RTCIMSC, &mut data);
        // The interrupt line should be on.
        assert!(rtc.interrupt_evt.read().unwrap() == 1);
        rtc.read(0, RTCIMSC, &mut data);
        let v = byte_order::read_le_u32(&data[..]);
        assert_eq!(non_zero & 1, v);

        // Now test with 0.
        byte_order::write_le_u32(&mut data, 0);
        rtc.write(0, RTCIMSC, &mut data);
        rtc.read(0, RTCIMSC, &mut data);
        let v = byte_order::read_le_u32(&data[..]);
        assert_eq!(0, v);

        // Attempts to turn off the RTC should not go through.
        byte_order::write_le_u32(&mut data, 0);
        rtc.write(0, RTCCR, &mut data);
        rtc.read(0, RTCCR, &mut data);
        let v = byte_order::read_le_u32(&data[..]);
        assert_eq!(v, 1);

        let mut data = [0; 4];
        rtc.read(0, AMBA_ID_LOW, &mut data);
        let index = AMBA_ID_LOW + 3;
        assert_eq!(data[0], PL031_ID[((index - AMBA_ID_LOW) >> 2) as usize]);
    }

    /// The alarm path (the reason this device carries a timer at all): programming the Match
    /// Register + unmasking the interrupt must, one second later, signal the timer eventfd — which
    /// the event manager turns into a fired interrupt (`RTCMIS`). Without the timer thread the
    /// eventfd never signals and the guest can never wake from `rtcwake`. (Drives the device the way
    /// the event manager would: wait for `alarm_wake`, then `fire_alarm`.)
    #[test]
    fn test_rtc_alarm_fires() {
        let mut rtc = RTC::new(EventFd::new(utils::eventfd::EFD_NONBLOCK).unwrap());
        let mut data = [0; 4];

        // Program the alarm for "now + 1s" and unmask the interrupt (this arms the timer).
        rtc.read(0, RTCDR, &mut data);
        let now = byte_order::read_le_u32(&data);
        byte_order::write_le_u32(&mut data, now + 1);
        rtc.write(0, RTCMR, &mut data);
        byte_order::write_le_u32(&mut data, 1);
        rtc.write(0, RTCIMSC, &mut data);

        // The timer thread should signal `alarm_wake` at ~1.1s; poll it (as the event manager does).
        let mut fired = false;
        for _ in 0..30 {
            if rtc.alarm_wake.read().is_ok() {
                fired = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        assert!(fired, "RTC alarm timer never signalled within the window");

        // Delivering that event must raise the masked interrupt status the guest driver reads.
        rtc.fire_alarm();
        rtc.read(0, RTCMIS, &mut data);
        assert_eq!(
            byte_order::read_le_u32(&data),
            1,
            "RTCMIS should show the fired alarm interrupt"
        );

        // Clearing it (RTCICR) drops the raw status, as the driver expects.
        byte_order::write_le_u32(&mut data, 1);
        rtc.write(0, RTCICR, &mut data);
        rtc.read(0, RTCRIS, &mut data);
        assert_eq!(
            byte_order::read_le_u32(&data),
            0,
            "RTCICR should clear RTCRIS"
        );
    }
}
