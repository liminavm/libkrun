// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{fmt, io};

use devices::fdt::DeviceInfoForFDT;
use devices::legacy::IrqChip;
use devices::{BusDevice, DeviceType};
use kernel::cmdline as kernel_cmdline;
use polly::event_manager::EventManager;
#[cfg(target_arch = "aarch64")]
use utils::eventfd::EventFd;

use crate::vstate::Vm;

/// Errors for MMIO device manager.
#[allow(clippy::enum_variant_names)]
#[derive(Debug)]
pub enum Error {
    /// Failed to create MmioTransport
    CreateMmioTransport(devices::virtio::CreateMmioTransportError),
    /// Failed to perform an operation on the bus.
    BusError(devices::BusError),
    /// Appending to kernel command line failed.
    Cmdline(kernel_cmdline::Error),
    /// Failure in creating or cloning an event fd.
    EventFd(io::Error),
    /// No more IRQs are available.
    IrqsExhausted,
    /// Registering an IO Event failed.
    RegisterIoEvent,
    /// Registering an IRQ FD failed.
    RegisterIrqFd,
    /// The device couldn't be found
    DeviceNotFound,
    /// Failed to update the mmio device.
    UpdateFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Error::CreateMmioTransport(ref e) => {
                write!(f, "failed to create mmio transport for the device {e}")
            }
            Error::BusError(ref e) => write!(f, "failed to perform bus operation: {e}"),
            Error::Cmdline(ref e) => {
                write!(f, "unable to add device to kernel command line: {e}")
            }
            Error::EventFd(ref e) => write!(f, "failed to create or clone event descriptor: {e}"),
            Error::IrqsExhausted => write!(f, "no more IRQs are available"),
            Error::RegisterIoEvent => write!(f, "failed to register IO event"),
            Error::RegisterIrqFd => write!(f, "failed to register irqfd"),
            Error::DeviceNotFound => write!(f, "the device couldn't be found"),
            Error::UpdateFailed => write!(f, "failed to update the mmio device"),
        }
    }
}

impl From<devices::virtio::CreateMmioTransportError> for crate::device_manager::mmio::Error {
    fn from(e: devices::virtio::CreateMmioTransportError) -> Self {
        Self::CreateMmioTransport(e)
    }
}

type Result<T> = ::std::result::Result<T, Error>;

/// This represents the size of the mmio device specified to the kernel as a cmdline option
/// It has to be larger than 0x100 (the offset where the configuration space starts from
/// the beginning of the memory mapped device registers) + the size of the configuration space
/// Currently hardcoded to 4K.
const MMIO_LEN: u64 = 0x1000;

/// Manages the complexities of registering a MMIO device.
pub struct MMIODeviceManager {
    pub bus: devices::Bus,
    mmio_base: u64,
    irq: u32,
    last_irq: u32,
    id_to_dev_info: HashMap<(DeviceType, String), MMIODeviceInfo>,
    /// Handle to the live PL061 GPIO device (M9 snapshot save/restore of its register file, and
    /// wake injection). `None` until `register_mmio_gpio` runs (macOS/aarch64 only).
    pub gpio: Option<Arc<Mutex<devices::legacy::Gpio>>>,
}

impl MMIODeviceManager {
    /// Create a new DeviceManager handling mmio devices (virtio net, block).
    pub fn new(mmio_base: &mut u64, irq_interval: (u32, u32)) -> MMIODeviceManager {
        if cfg!(target_arch = "aarch64") {
            *mmio_base += MMIO_LEN;
        }

        MMIODeviceManager {
            mmio_base: *mmio_base,
            irq: irq_interval.0,
            last_irq: irq_interval.1,
            bus: devices::Bus::new(),
            id_to_dev_info: HashMap::new(),
            gpio: None,
        }
    }

    /// Register an already created MMIO device to be used via MMIO transport.
    pub fn register_mmio_device(
        &mut self,
        mut mmio_device: devices::virtio::MmioTransport,
        type_id: u32,
        device_id: String,
    ) -> Result<(u64, u32)> {
        if self.irq > self.last_irq {
            return Err(Error::IrqsExhausted);
        }

        mmio_device.set_irq_line(self.irq);

        self.bus
            .insert(Arc::new(Mutex::new(mmio_device)), self.mmio_base, MMIO_LEN)
            .map_err(Error::BusError)?;
        let ret = (self.mmio_base, self.irq);
        self.id_to_dev_info.insert(
            (DeviceType::Virtio(type_id), device_id),
            MMIODeviceInfo {
                addr: self.mmio_base,
                len: MMIO_LEN,
                irq: self.irq,
            },
        );
        self.mmio_base += MMIO_LEN;
        self.irq += 1;

        Ok(ret)
    }

    #[cfg(target_arch = "aarch64")]
    /// Register an early console at some MMIO address.
    pub fn register_mmio_serial(
        &mut self,
        _vm: &Vm,
        cmdline: &mut kernel_cmdline::Cmdline,
        intc: IrqChip,
        serial: Arc<Mutex<devices::legacy::Serial>>,
    ) -> Result<()> {
        if self.irq > self.last_irq {
            return Err(Error::IrqsExhausted);
        }

        {
            let mut serial = serial.lock().unwrap();
            serial.set_intc(intc);
            serial.set_irq_line(self.irq);
        }

        self.bus
            .insert(serial, self.mmio_base, MMIO_LEN)
            .map_err(Error::BusError)?;

        cmdline
            .insert(
                "earlycon",
                &format!("pl011,mmio32,0x{:08x}", self.mmio_base),
            )
            .map_err(Error::Cmdline)?;

        let ret = self.mmio_base;
        self.id_to_dev_info.insert(
            (DeviceType::Serial, DeviceType::Serial.to_string()),
            MMIODeviceInfo {
                addr: ret,
                len: MMIO_LEN,
                irq: self.irq,
            },
        );

        self.mmio_base += MMIO_LEN;
        self.irq += 1;

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    /// Register a MMIO RTC device.
    pub fn register_mmio_rtc(
        &mut self,
        _vm: &Vm,
        intc: IrqChip,
        event_manager: &mut EventManager,
    ) -> Result<()> {
        if self.irq > self.last_irq {
            return Err(Error::IrqsExhausted);
        }

        // Attaching the RTC device. Wire its alarm interrupt to the GIC and subscribe to its timer
        // eventfd (mirroring the GPIO device) so a programmed match actually raises an SPI — the
        // mechanism a guest needs to wake from suspend-to-idle via `rtcwake`.
        let rtc_evt = EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(Error::EventFd)?;
        let rtc = Arc::new(Mutex::new(devices::legacy::RTC::new(
            rtc_evt.try_clone().map_err(Error::EventFd)?,
        )));

        event_manager.add_subscriber(rtc.clone()).unwrap();
        {
            let mut rtc = rtc.lock().unwrap();
            rtc.set_intc(intc);
            rtc.set_irq_line(self.irq);
        }

        self.bus
            .insert(rtc, self.mmio_base, MMIO_LEN)
            .map_err(Error::BusError)?;

        let ret = self.mmio_base;
        self.id_to_dev_info.insert(
            (DeviceType::RTC, "rtc".to_string()),
            MMIODeviceInfo {
                addr: ret,
                len: MMIO_LEN,
                irq: self.irq,
            },
        );

        self.mmio_base += MMIO_LEN;
        self.irq += 1;

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    /// Register a GPIO
    pub fn register_mmio_gpio(
        &mut self,
        _vm: &Vm,
        intc: IrqChip,
        event_manager: &mut EventManager,
        shutdown_efd: EventFd,
        suspend_efd: EventFd,
        restart_efd: EventFd,
        wake_efd: EventFd,
    ) -> Result<()> {
        // Attaching the GPIO device.
        let gpio_evt = EventFd::new(utils::eventfd::EFD_NONBLOCK).map_err(Error::EventFd)?;
        let gpio = Arc::new(Mutex::new(devices::legacy::Gpio::new(
            shutdown_efd,
            suspend_efd,
            restart_efd,
            wake_efd,
            gpio_evt.try_clone().map_err(Error::EventFd)?,
        )));

        event_manager.add_subscriber(gpio.clone()).unwrap();

        if self.irq > self.last_irq {
            return Err(Error::IrqsExhausted);
        }

        {
            let mut gpio = gpio.lock().unwrap();
            gpio.set_intc(intc);
            gpio.set_irq_line(self.irq);
        }

        // Keep a typed handle for M9 snapshot save/restore of the register file + wake injection,
        // before the Arc is coerced to `dyn BusDevice` and moved into the bus.
        self.gpio = Some(gpio.clone());

        self.bus
            .insert(gpio, self.mmio_base, MMIO_LEN)
            .map_err(Error::BusError)?;

        let ret = self.mmio_base;
        self.id_to_dev_info.insert(
            (DeviceType::Gpio, DeviceType::Gpio.to_string()),
            MMIODeviceInfo {
                addr: ret,
                len: MMIO_LEN,
                irq: self.irq,
            },
        );

        self.mmio_base += MMIO_LEN;
        self.irq += 1;

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    /// Register a MMIO GIC device.
    pub fn register_mmio_gic(&mut self, _vm: &Vm, intc: IrqChip) -> Result<()> {
        let (mmio_addr, mmio_size) = {
            let intc = intc.lock().unwrap();
            (intc.get_mmio_addr(), intc.get_mmio_size())
        };

        // The in-kernel GIC reports a size of 0 to tell us we don't need to map
        // anything in the guest.
        if mmio_size != 0 {
            self.bus
                .insert(intc, mmio_addr, mmio_size)
                .map_err(Error::BusError)?;
        }

        Ok(())
    }

    #[cfg(target_arch = "aarch64")]
    /// Gets the information of the devices registered up to some point in time.
    pub fn get_device_info(&self) -> &HashMap<(DeviceType, String), MMIODeviceInfo> {
        &self.id_to_dev_info
    }

    /// Gets the specified device.
    pub fn get_device(
        &self,
        device_type: DeviceType,
        device_id: &str,
    ) -> Option<&Mutex<dyn BusDevice>> {
        if let Some(dev_info) = self
            .id_to_dev_info
            .get(&(device_type, device_id.to_string()))
            && let Some((_, device)) = self.bus.get_device(dev_info.addr)
        {
            return Some(device);
        }
        None
    }

    /// limina M9.2: snapshot every virtio-mmio device as `(virtio type id, id, device_status)`.
    /// The quiesce oracle / INIT-invariant assert reads this at suspend time. A device quiesced by
    /// its driver's s2idle `.suspend` callback has been reset to `INIT` (0); virtio-gpu is the known
    /// exception (no PM ops → stays `DRIVER_OK`). Only virtio-mmio transports are inspected (the
    /// legacy PL0xx devices carry no virtio status).
    pub fn virtio_statuses(&self) -> Vec<(u32, String, u32)> {
        use devices::virtio::MmioTransport;
        use devices::BusDevice;
        let mut out = Vec::new();
        for (dtype, id) in self.id_to_dev_info.keys() {
            if let DeviceType::Virtio(type_id) = *dtype {
                if let Some(dev) = self.get_device(*dtype, id) {
                    let guard = dev.lock().unwrap();
                    // Deref to `&dyn BusDevice` so `as_any` dispatches via the supertrait vtable
                    // (not the blanket `impl<T: Any> AsAny for T` on the non-'static MutexGuard).
                    let dev_ref: &dyn BusDevice = &*guard;
                    if let Some(mmio) = dev_ref.as_any().downcast_ref::<MmioTransport>() {
                        out.push((type_id, id.clone(), mmio.device_status()));
                    }
                }
            }
        }
        out
    }

    /// limina M9.3: capture the transport state of every virtio-mmio device the guest left
    /// `device_status != 0` at quiesce (today exactly virtio-gpu — it has no s2idle PM ops so it never
    /// resets/re-negotiates, unlike every other device which the guest reset to INIT). Those are the
    /// only devices that need transport restore; a reset device re-drives its own `DRIVER_OK` on resume
    /// and the fresh worker activates it then. Identity (type_id/mmio base/irq) is carried so restore
    /// can fail closed on a layout mismatch.
    pub fn capture_transport_states(&self) -> Vec<crate::snapshot::DeviceTransportState> {
        use crate::snapshot::{DeviceTransportState, QueueRegs};
        use devices::virtio::MmioTransport;
        use devices::BusDevice;
        let mut out = Vec::new();
        for ((dtype, id), dev_info) in self.id_to_dev_info.iter() {
            let DeviceType::Virtio(type_id) = *dtype else {
                continue;
            };
            let Some(dev) = self.get_device(*dtype, id) else {
                continue;
            };
            let guard = dev.lock().unwrap();
            let dev_ref: &dyn BusDevice = &*guard;
            let Some(mmio) = dev_ref.as_any().downcast_ref::<MmioTransport>() else {
                continue;
            };
            if mmio.device_status() == 0 {
                continue; // reset device — re-negotiated by the guest on resume, nothing to carry
            }
            let ts = mmio.transport_state();
            out.push(DeviceTransportState {
                type_id,
                mmio_base: dev_info.addr,
                irq: dev_info.irq,
                device_status: ts.device_status,
                acked_features: ts.acked_features,
                config_generation: ts.config_generation,
                isr: ts.isr,
                queues: ts
                    .queues
                    .into_iter()
                    .map(|q| QueueRegs {
                        size: q.size,
                        ready: q.ready,
                        desc: q.desc,
                        avail: q.avail,
                        used: q.used,
                    })
                    .collect(),
            });
        }
        out
    }

    /// limina M9.3: validate + log the captured device transports on restore. We do NOT replay the
    /// negotiation: the guest re-negotiates every virtio device ITSELF on s2idle thaw
    /// (`virtio_device_restore` resets + re-drives DRIVER_OK for each; the fresh worker's `activate()`
    /// fires from the guest's own writes). A device is captured only because it was DRIVER_OK at
    /// snapshot time — for the virtio-gpu that's just because it has no driver PM ops, so it's the sole
    /// device left non-INIT; it is still re-negotiated on thaw like the rest (RED-first + R4 verified,
    /// 2026-07-18). So this pass only (a) fails closed on layout drift — a captured device missing at
    /// its snapshot base/irq means the restore worker's device topology diverged, which would silently
    /// corrupt guest driver state — and (b) logs the captured transport for diagnostics.
    ///
    /// TODO(feature-drift, Fable): the one way the guest's self-revival can go wrong is a features_ok
    /// mismatch when a newer limina restores an older snapshot offering different device features. Add
    /// a fail-closed compare of `st.acked_features` against the fresh device's offered feature set here.
    pub fn validate_transport_states(
        &self,
        states: &[crate::snapshot::DeviceTransportState],
    ) -> std::result::Result<(), String> {
        use devices::virtio::MmioTransport;
        for st in states {
            // Locate the device by identity: same virtio type at the same mmio base + irq.
            let matched = self.id_to_dev_info.iter().find(|((dtype, _), info)| {
                matches!(*dtype, DeviceType::Virtio(t) if t == st.type_id)
                    && info.addr == st.mmio_base
                    && info.irq == st.irq
            });
            let Some(((dtype, id), _)) = matched else {
                return Err(format!(
                    "restore: no virtio device type={} @0x{:x} irq={} (layout drift)",
                    st.type_id, st.mmio_base, st.irq
                ));
            };
            info!(
                "restore: captured virtio type={} @0x{:x} status=0x{:x} acked_features=0x{:x} {} queue(s) \
                 (validated; queue regs seeded for the no-PM-ops thaw re-arm)",
                st.type_id,
                st.mmio_base,
                st.device_status,
                st.acked_features,
                st.queues.len()
            );
            // Seed the fresh transport's previous-activation queue register file from the capture.
            // The guest re-negotiates every device itself on s2idle thaw; a driver WITH PM ops
            // re-programs its queues and never looks at this. A driver WITHOUT PM ops (virtio-gpu,
            // verified absent through v7.1.2) resumes via the bus fallback — reset → features →
            // DRIVER_OK, no queue writes — and the transport re-arms from this seed at activation
            // (see MmioTransport::rearm_queues_from_stash). Without it, the guest drives DRIVER_OK
            // onto dead queues and every GPU command waits forever (the M9.3 restore wedge).
            if let Some(dev) = self.get_device(*dtype, id) {
                let mut guard = dev.lock().unwrap();
                let dev_ref: &mut dyn devices::BusDevice = &mut *guard;
                if let Some(mmio) = dev_ref.as_mut_any().downcast_mut::<MmioTransport>() {
                    mmio.seed_queue_regs(
                        st.queues
                            .iter()
                            .map(|q| devices::virtio::QueueRegs {
                                size: q.size,
                                ready: q.ready,
                                desc: q.desc,
                                avail: q.avail,
                                used: q.used,
                            })
                            .collect(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Private structure for storing information about the MMIO device registered at some address on the bus.
#[derive(Clone, Debug)]
pub struct MMIODeviceInfo {
    addr: u64,
    irq: u32,
    len: u64,
}

#[cfg(target_arch = "aarch64")]
impl DeviceInfoForFDT for MMIODeviceInfo {
    fn addr(&self) -> u64 {
        self.addr
    }
    fn irq(&self) -> u32 {
        self.irq
    }
    fn length(&self) -> u64 {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arch;
    use devices::legacy::DummyIrqChip;
    use devices::virtio::{
        ActivateResult, DeviceQueue, InterruptTransport, QueueConfig, VirtioDevice,
    };
    use std::sync::Arc;
    use vm_memory::{GuestAddress, GuestMemoryMmap};

    const QUEUE_SIZES: &[u16] = &[64];

    impl MMIODeviceManager {
        fn register_virtio_device(
            &mut self,
            guest_mem: GuestMemoryMmap,
            device: Arc<Mutex<dyn devices::virtio::VirtioDevice>>,
            _cmdline: &mut kernel_cmdline::Cmdline,
            type_id: u32,
            device_id: &str,
        ) -> Result<u64> {
            let mmio_device =
                devices::virtio::MmioTransport::new(guest_mem, DummyIrqChip::new().into(), device)
                    .unwrap();
            let (mmio_base, _irq) =
                self.register_mmio_device(mmio_device, type_id, device_id.to_string())?;
            Ok(mmio_base)
        }
    }

    #[allow(dead_code)]
    struct DummyDevice {
        dummy: u32,
        queue_config: Vec<QueueConfig>,
    }

    impl DummyDevice {
        pub fn new() -> Self {
            DummyDevice {
                dummy: 0,
                queue_config: QUEUE_SIZES.iter().map(|&s| QueueConfig::new(s)).collect(),
            }
        }
    }

    impl devices::virtio::VirtioDevice for DummyDevice {
        fn avail_features(&self) -> u64 {
            0
        }

        fn acked_features(&self) -> u64 {
            0
        }

        fn set_acked_features(&mut self, _: u64) {}

        fn device_type(&self) -> u32 {
            0
        }

        fn device_name(&self) -> &str {
            "dummy"
        }

        fn queue_config(&self) -> &[QueueConfig] {
            &self.queue_config
        }

        fn read_config(&self, offset: u64, data: &mut [u8]) {
            let _ = offset;
            let _ = data;
        }

        fn write_config(&mut self, offset: u64, data: &[u8]) {
            let _ = offset;
            let _ = data;
        }

        fn activate(
            &mut self,
            _mem: GuestMemoryMmap,
            _interrupt: InterruptTransport,
            _queues: Vec<DeviceQueue>,
        ) -> ActivateResult {
            Ok(())
        }

        fn is_activated(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_register_virtio_device() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let guest_mem =
            GuestMemoryMmap::from_ranges(&[(start_addr1, 0x1000), (start_addr2, 0x1000)]).unwrap();
        let mut device_manager =
            MMIODeviceManager::new(&mut 0xd000_0000, (arch::IRQ_BASE, arch::IRQ_MAX));

        let mut cmdline = kernel_cmdline::Cmdline::new(4096);
        let dummy = Arc::new(Mutex::new(DummyDevice::new()));

        assert!(
            device_manager
                .register_virtio_device(guest_mem, dummy, &mut cmdline, 0, "dummy")
                .is_ok()
        );
    }

    #[test]
    fn test_register_too_many_devices() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let guest_mem =
            GuestMemoryMmap::from_ranges(&[(start_addr1, 0x1000), (start_addr2, 0x1000)]).unwrap();
        let mut device_manager =
            MMIODeviceManager::new(&mut 0xd000_0000, (arch::IRQ_BASE, arch::IRQ_MAX));

        let mut cmdline = kernel_cmdline::Cmdline::new(4096);

        for _i in arch::IRQ_BASE..=arch::IRQ_MAX {
            device_manager
                .register_virtio_device(
                    guest_mem.clone(),
                    Arc::new(Mutex::new(DummyDevice::new())),
                    &mut cmdline,
                    0,
                    "dummy1",
                )
                .unwrap();
        }
        assert_eq!(
            format!(
                "{}",
                device_manager
                    .register_virtio_device(
                        guest_mem,
                        Arc::new(Mutex::new(DummyDevice::new())),
                        &mut cmdline,
                        0,
                        "dummy2"
                    )
                    .unwrap_err()
            ),
            "no more IRQs are available".to_string()
        );
    }

    #[test]
    fn test_dummy_device() {
        let dummy = DummyDevice::new();
        assert_eq!(dummy.device_type(), 0);
        assert_eq!(dummy.queue_config().len(), QUEUE_SIZES.len());
    }

    #[test]
    fn test_error_messages() {
        let device_manager =
            MMIODeviceManager::new(&mut 0xd000_0000, (arch::IRQ_BASE, arch::IRQ_MAX));
        let mut cmdline = kernel_cmdline::Cmdline::new(4096);
        let e = Error::Cmdline(
            cmdline
                .insert(
                    "virtio_mmio=device",
                    &format!(
                        "{}K@0x{:08x}:{}",
                        MMIO_LEN / 1024,
                        device_manager.mmio_base,
                        device_manager.irq
                    ),
                )
                .unwrap_err(),
        );
        assert_eq!(
            format!("{}", e),
            format!(
                "unable to add device to kernel command line: {}",
                kernel_cmdline::Error::HasEquals
            ),
        );
        assert_eq!(
            format!("{}", Error::UpdateFailed),
            "failed to update the mmio device"
        );
        assert_eq!(
            format!("{}", Error::BusError(devices::BusError::Overlap)),
            format!(
                "failed to perform bus operation: {}",
                devices::BusError::Overlap
            )
        );
        assert_eq!(
            format!("{}", Error::IrqsExhausted),
            "no more IRQs are available"
        );
        assert_eq!(
            format!("{}", Error::RegisterIoEvent),
            "failed to register IO event"
        );
        assert_eq!(
            format!("{}", Error::RegisterIrqFd),
            "failed to register irqfd"
        );
    }

    #[test]
    fn test_device_info() {
        let start_addr1 = GuestAddress(0x0);
        let start_addr2 = GuestAddress(0x1000);
        let guest_mem =
            GuestMemoryMmap::from_ranges(&[(start_addr1, 0x1000), (start_addr2, 0x1000)]).unwrap();
        let mut device_manager =
            MMIODeviceManager::new(&mut 0xd000_0000, (arch::IRQ_BASE, arch::IRQ_MAX));
        let mut cmdline = kernel_cmdline::Cmdline::new(4096);
        let dummy = Arc::new(Mutex::new(DummyDevice::new()));

        let type_id = 0;
        let id = String::from("foo");
        if let Ok(addr) =
            device_manager.register_virtio_device(guest_mem, dummy, &mut cmdline, type_id, &id)
        {
            assert!(
                device_manager
                    .get_device(DeviceType::Virtio(type_id), &id)
                    .is_some()
            );
            assert_eq!(
                addr,
                device_manager.id_to_dev_info[&(DeviceType::Virtio(type_id), id.clone())].addr
            );
            assert_eq!(
                arch::IRQ_BASE,
                device_manager.id_to_dev_info[&(DeviceType::Virtio(type_id), id.clone())].irq
            );
        }
        let id = "bar";
        assert!(
            device_manager
                .get_device(DeviceType::Virtio(type_id), &id)
                .is_none()
        );
    }
}
