// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use utils::eventfd::EventFd;
use vm_memory::{Bytes, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, I2cError, QueueConfig, VirtioDevice,
};
use super::{defs, defs::uapi};
use crate::virtio::InterruptTransport;

// Request queue ("msg" in the Linux driver).
pub(crate) const REQ_INDEX: usize = 0;

pub(crate) const AVAIL_FEATURES: u64 = (1 << uapi::VIRTIO_F_VERSION_1 as u64)
    | (1 << uapi::VIRTIO_I2C_F_ZERO_LENGTH_REQUEST as u64);

/// The 7-bit I2C address of the emulated SBS smart battery (the SBS standard slave
/// address; the DT node `battery@b` must match).
pub const SBS_BATTERY_ADDR: u16 = 0x0b;

/// A snapshot of the host battery, fetched from the VMM on every guest register
/// read (the guest's own polling cadence drives freshness).
#[derive(Debug, Clone, Copy, Default)]
pub struct BatteryState {
    /// 0..=100.
    pub percent: u8,
    /// Actively charging.
    pub charging: bool,
    /// External power connected (charging or not).
    pub ac_online: bool,
    /// Estimated minutes to empty while discharging, if known.
    pub time_to_empty_min: Option<u16>,
    /// Estimated minutes to full while charging, if known.
    pub time_to_full_min: Option<u16>,
    /// Battery cycle count, if known.
    pub cycle_count: Option<u16>,
}

/// Host callback answering "what does the battery look like right now".
pub type BatteryProvider = Arc<dyn Fn() -> BatteryState + Send + Sync>;

/// Synthetic pack constants: the host only reliably exposes percent/state/times,
/// so present a plausible fixed-size pack and scale the charge registers from
/// percent. GNOME/UPower key off percent + status; the absolute mAh numbers only
/// show in details.
const DESIGN_CAPACITY_MAH: u32 = 5000;
const DESIGN_VOLTAGE_MV: u16 = 11400;

// SBS BatteryStatus() bits (sbs-battery.c: battery status value bits).
const BATTERY_INITIALIZED: u16 = 0x80;
const BATTERY_DISCHARGING: u16 = 0x40;
const BATTERY_FULL_CHARGED: u16 = 0x20;
const BATTERY_FULL_DISCHARGED: u16 = 0x10;

/// The emulated SBS smart battery: a register file addressed by an SMBus command
/// byte. SMBus word/block reads arrive as a command-byte write followed by a read
/// (two chained virtio requests), so the last written command is sticky state.
pub(crate) struct SbsBattery {
    provider: BatteryProvider,
    command: u8,
    /// BatteryMode(0x03) — the driver flips bit 15 (mAh vs 10mWh reporting) around
    /// capacity reads; we store it so the round trip is coherent. Charge values are
    /// reported unscaled either way (see above).
    mode: u16,
}

impl SbsBattery {
    fn new(provider: BatteryProvider) -> Self {
        Self {
            provider,
            command: 0,
            mode: 0,
        }
    }

    /// A guest write: byte 0 selects the register; an optional LE word writes it.
    fn write(&mut self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            // Zero-length request: address probe, ack it.
            return true;
        }
        self.command = bytes[0];
        if bytes.len() >= 3 && self.command == 0x03 {
            self.mode = u16::from_le_bytes([bytes[1], bytes[2]]);
        }
        true
    }

    /// A guest read of `len` bytes from the last selected register. `None` = NAK.
    fn read(&mut self, len: usize) -> Option<Vec<u8>> {
        let response = self.register_bytes()?;
        let mut out = response;
        out.resize(len, 0);
        Some(out)
    }

    /// The full byte contents of the selected register (word LE, or a
    /// length-prefixed block for the string registers).
    fn register_bytes(&self) -> Option<Vec<u8>> {
        let state = (self.provider)();
        let percent = u32::from(state.percent.min(100));
        // "Full" as presented to the guest: on external power, not taking charge,
        // and essentially topped up. A charge-limited hold (e.g. macOS optimized
        // charging at 80%) shows as plugged-in/not-charging instead.
        let full = state.ac_online && !state.charging && state.percent >= 95;
        let discharging = !state.ac_online;

        let word = |v: u16| Some(v.to_le_bytes().to_vec());
        let block = |s: &str| {
            let bytes = s.as_bytes();
            let mut out = Vec::with_capacity(bytes.len() + 1);
            out.push(bytes.len() as u8);
            out.extend_from_slice(bytes);
            Some(out)
        };

        match self.command {
            // ManufacturerAccess.
            0x00 => word(0),
            // BatteryMode.
            0x03 => word(self.mode),
            // Temperature, 0.1 K units: a fixed pleasant 25.1 C.
            0x08 => word(2982),
            // Voltage (mV).
            0x09 => word(DESIGN_VOLTAGE_MV),
            // Current / AverageCurrent (mA, signed; sign drives the driver's
            // status correction: 0 while not full reads as plugged-in/not-charging).
            0x0a | 0x0b => {
                let ma: i16 = if state.charging {
                    1500
                } else if discharging {
                    -1200
                } else {
                    0
                };
                word(ma as u16)
            }
            // MaxError (%).
            0x0c => word(1),
            // RelativeStateOfCharge (%).
            0x0d => word(percent as u16),
            // RemainingCapacity, scaled from percent.
            0x0f => word((DESIGN_CAPACITY_MAH * percent / 100) as u16),
            // FullChargeCapacity.
            0x10 => word(DESIGN_CAPACITY_MAH as u16),
            // RunTimeToEmpty / AverageTimeToEmpty (min): only meaningful while
            // discharging; NAK when unknown so the property reads as absent
            // rather than a bogus number.
            0x11 | 0x12 => {
                if discharging {
                    state.time_to_empty_min.and_then(word)
                } else {
                    None
                }
            }
            // AverageTimeToFull (min).
            0x13 => {
                if state.charging {
                    state.time_to_full_min.and_then(word)
                } else {
                    None
                }
            }
            // ChargingCurrent / ChargingVoltage maxima.
            0x14 => word(2000),
            0x15 => word(12600),
            // BatteryStatus.
            0x16 => {
                let mut status = BATTERY_INITIALIZED;
                if full {
                    status |= BATTERY_FULL_CHARGED;
                }
                if discharging {
                    status |= BATTERY_DISCHARGING;
                    if state.percent <= 5 {
                        status |= BATTERY_FULL_DISCHARGED;
                    }
                }
                word(status)
            }
            // CycleCount.
            0x17 => word(state.cycle_count.unwrap_or(0)),
            // DesignCapacity / DesignVoltage.
            0x18 => word(DESIGN_CAPACITY_MAH as u16),
            0x19 => word(DESIGN_VOLTAGE_MV),
            // SpecificationInfo: SBS v1.1, revision 1, NO packet-error-checking
            // (the driver enables PEC for v1.1+PEC and our SMBus emulation-over-i2c
            // path doesn't carry it).
            0x1a => word(0x0021),
            // ManufactureDate: (year-1980) << 9 | month << 5 | day.
            0x1b => word((46 << 9) | (7 << 5) | 2),
            // SerialNumber.
            0x1c => word(1),
            // ManufacturerName / DeviceName / DeviceChemistry (length-prefixed
            // blocks; the driver string-matches "LION" for Li-ion).
            0x20 => block("Limina"),
            0x21 => block("Host Battery"),
            0x22 => block("LION"),
            _ => None,
        }
    }
}

pub struct I2c {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    battery: SbsBattery,
    /// virtio-i2c FAIL_NEXT chaining: a failed request whose FAIL_NEXT flag was
    /// set forces the following request to fail too (transitively).
    fail_next: bool,
}

impl I2c {
    pub(crate) fn queue_event(&self, idx: usize) -> &std::sync::Arc<utils::eventfd::EventFd> {
        &self.queues.as_ref().expect("queues should exist")[idx].event
    }

    pub fn new(provider: BatteryProvider) -> super::Result<I2c> {
        Ok(I2c {
            queues: None,
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(I2cError::EventFd)?,
            device_state: DeviceState::Inactive,
            battery: SbsBattery::new(provider),
            fail_next: false,
        })
    }

    pub fn id(&self) -> &str {
        defs::I2C_DEV_ID
    }

    /// One virtio request = one i2c_msg: out_hdr (8 bytes, device-readable),
    /// an optional data buffer (readable = write, writable = read), and a 1-byte
    /// device-writable in_hdr for the status.
    fn process_chain(
        &mut self,
        mem: &GuestMemoryMmap,
        head: crate::virtio::DescriptorChain,
    ) -> u32 {
        // Collect the (few) descriptors up front; chains are at most 3 long.
        let mut descs = Vec::with_capacity(3);
        let mut cur = Some(head);
        while let Some(d) = cur {
            let next = d.next_descriptor();
            descs.push(d);
            cur = next;
        }

        // Layout sanity: out_hdr first (readable, 8 bytes), in_hdr last
        // (writable, 1 byte). Anything between is the single data buffer.
        if descs.len() < 2 || descs.len() > 3 {
            error!("i2c: malformed request chain ({} descriptors)", descs.len());
            return 0;
        }
        let out_hdr_desc = &descs[0];
        let in_hdr_desc = &descs[descs.len() - 1];
        if !out_hdr_desc.is_read_only()
            || out_hdr_desc.len < 8
            || !in_hdr_desc.is_write_only()
            || in_hdr_desc.len < 1
        {
            error!("i2c: malformed request chain (bad header descriptors)");
            return 0;
        }

        let mut hdr = [0u8; 8];
        if let Err(e) = mem.read_slice(&mut hdr, out_hdr_desc.addr) {
            error!("i2c: failed to read out_hdr: {e:?}");
            return 0;
        }
        let addr = u16::from_le_bytes([hdr[0], hdr[1]]);
        let flags = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        let is_read = flags & uapi::VIRTIO_I2C_FLAGS_M_RD != 0;

        let mut written: u32 = 0;
        let ok = if self.fail_next {
            // A failed predecessor with FAIL_NEXT poisons this request.
            false
        } else if (addr >> 1) != SBS_BATTERY_ADDR {
            // Only the battery lives on this bus.
            false
        } else if descs.len() == 2 {
            // Zero-length request: an address probe.
            self.battery.write(&[])
        } else if is_read {
            let buf_desc = &descs[1];
            if !buf_desc.is_write_only() {
                error!("i2c: M_RD request with a read-only buffer");
                false
            } else {
                match self.battery.read(buf_desc.len as usize) {
                    Some(bytes) => match mem.write_slice(&bytes, buf_desc.addr) {
                        Ok(()) => {
                            written = buf_desc.len;
                            true
                        }
                        Err(e) => {
                            error!("i2c: failed to write read buffer: {e:?}");
                            false
                        }
                    },
                    None => false,
                }
            }
        } else {
            let buf_desc = &descs[1];
            if !buf_desc.is_read_only() {
                error!("i2c: write request with a writable buffer");
                false
            } else {
                let mut bytes = vec![0u8; buf_desc.len as usize];
                match mem.read_slice(&mut bytes, buf_desc.addr) {
                    Ok(()) => self.battery.write(&bytes),
                    Err(e) => {
                        error!("i2c: failed to read write buffer: {e:?}");
                        false
                    }
                }
            }
        };

        self.fail_next = !ok && (flags & uapi::VIRTIO_I2C_FLAGS_FAIL_NEXT != 0);

        let status = if ok {
            uapi::VIRTIO_I2C_MSG_OK
        } else {
            uapi::VIRTIO_I2C_MSG_ERR
        };
        if let Err(e) = mem.write_slice(&[status], in_hdr_desc.addr) {
            error!("i2c: failed to write in_hdr: {e:?}");
            return 0;
        }
        written + 1
    }

    pub fn process_req(&mut self) -> bool {
        debug!("i2c: process_req()");
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            // Validated by the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let mut have_used = false;
        loop {
            let head = {
                let queues = self
                    .queues
                    .as_mut()
                    .expect("queues should exist when activated");
                queues[REQ_INDEX].queue.pop(&mem)
            };
            let Some(head) = head else { break };
            let index = head.index;
            let written = self.process_chain(&mem, head);

            have_used = true;
            let queues = self
                .queues
                .as_mut()
                .expect("queues should exist when activated");
            if let Err(e) = queues[REQ_INDEX].queue.add_used(&mem, index, written) {
                error!("i2c: failed to add used element to the queue: {e:?}");
            }
        }

        have_used
    }
}

impl VirtioDevice for I2c {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features
    }

    fn device_type(&self) -> u32 {
        uapi::VIRTIO_ID_I2C
    }

    fn device_name(&self) -> &str {
        "i2c"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, _offset: u64, _data: &mut [u8]) {
        error!("i2c: invalid request to read config space");
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "i2c: guest driver attempted to write device config (offset={:x}, len={:x})",
            offset,
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if queues.len() != defs::NUM_QUEUES {
            error!(
                "Cannot perform activate. Expected {} queue(s), got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }

        if self.activate_evt.write(1).is_err() {
            error!("Cannot write to activate_evt");
            return Err(ActivateError::BadActivate);
        }

        self.queues = Some(queues);
        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        self.queues = None;
        self.device_state = DeviceState::Inactive;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn battery(state: BatteryState) -> SbsBattery {
        SbsBattery::new(Arc::new(move || state))
    }

    fn read_word(bat: &mut SbsBattery, reg: u8) -> Option<u16> {
        assert!(bat.write(&[reg]));
        bat.read(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
    }

    #[test]
    fn status_word_tracks_the_host_state() {
        // Discharging at 42%.
        let mut bat = battery(BatteryState {
            percent: 42,
            charging: false,
            ac_online: false,
            ..Default::default()
        });
        let status = read_word(&mut bat, 0x16).unwrap();
        assert_ne!(status & BATTERY_DISCHARGING, 0);
        assert_eq!(status & BATTERY_FULL_CHARGED, 0);
        assert_eq!(read_word(&mut bat, 0x0d).unwrap(), 42);
        // Negative current while discharging.
        assert!((read_word(&mut bat, 0x0a).unwrap() as i16) < 0);

        // Charging.
        let mut bat = battery(BatteryState {
            percent: 42,
            charging: true,
            ac_online: true,
            ..Default::default()
        });
        let status = read_word(&mut bat, 0x16).unwrap();
        assert_eq!(status & BATTERY_DISCHARGING, 0);
        assert!((read_word(&mut bat, 0x0a).unwrap() as i16) > 0);

        // Full: plugged in, not taking charge, topped up.
        let mut bat = battery(BatteryState {
            percent: 100,
            charging: false,
            ac_online: true,
            ..Default::default()
        });
        let status = read_word(&mut bat, 0x16).unwrap();
        assert_ne!(status & BATTERY_FULL_CHARGED, 0);
        assert_eq!(read_word(&mut bat, 0x0a).unwrap(), 0);
    }

    #[test]
    fn time_estimates_nak_when_unknown_or_inapplicable() {
        let mut bat = battery(BatteryState {
            percent: 50,
            charging: false,
            ac_online: false,
            time_to_empty_min: Some(123),
            ..Default::default()
        });
        assert_eq!(read_word(&mut bat, 0x12).unwrap(), 123);
        // Time-to-full while discharging: NAK.
        assert!(bat.write(&[0x13]));
        assert!(bat.read(2).is_none());

        // Unknown estimate: NAK instead of a bogus number.
        let mut bat = battery(BatteryState {
            percent: 50,
            charging: false,
            ac_online: false,
            time_to_empty_min: None,
            ..Default::default()
        });
        assert!(bat.write(&[0x12]));
        assert!(bat.read(2).is_none());
    }

    #[test]
    fn string_registers_are_length_prefixed_blocks() {
        let mut bat = battery(BatteryState::default());
        assert!(bat.write(&[0x22]));
        // The driver's fallback path first reads 1 byte (the length)…
        assert_eq!(bat.read(1).unwrap(), vec![4]);
        // …then length+1 bytes.
        assert_eq!(bat.read(5).unwrap(), b"\x04LION".to_vec());
    }

    #[test]
    fn spec_info_disables_pec_and_probe_register_works() {
        let mut bat = battery(BatteryState::default());
        // Presence probe = a successful BatteryStatus read.
        assert!(read_word(&mut bat, 0x16).is_some());
        // v1.1 without PEC.
        assert_eq!(read_word(&mut bat, 0x1a).unwrap(), 0x0021);
        // Unknown registers NAK.
        assert!(bat.write(&[0x7f]));
        assert!(bat.read(2).is_none());
        // Zero-length probe acks.
        assert!(bat.write(&[]));
    }
}
