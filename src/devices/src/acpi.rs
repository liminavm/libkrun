// SPDX-License-Identifier: Apache-2.0

//! limina: ACPI tables describing an aarch64 machine, for a guest booted through a firmware.
//!
//! The tables describe the same machine as the device tree from `fdt::create_fdt`, which the
//! firmware keeps reading for its own drivers. arm64 Linux uses one description or the other,
//! never both, so a firmware that installs these tables withholds the device tree from the OS.
//! The firmware finds the tables through the `libkrun,acpi-tables` node, RSDP first.

use std::collections::HashMap;
use std::fmt::Debug;

use acpi_tables::aml::{
    Device, Interrupt, Memory32Fixed, Method, Name, Notify, Path, ResourceTemplate, Scope,
};
use acpi_tables::fadt::{FADTBuilder, Flags};
use acpi_tables::madt::{
    EnabledStatus, GicVersion, Gicc, Gicd, Gicr, LocalInterruptController, MADT, Trigger,
};
use acpi_tables::pptt::{PPTT, ProcessorNode};
use acpi_tables::rsdp::Rsdp;
use acpi_tables::sdt::Sdt;
use acpi_tables::xsdt::XSDT;
use acpi_tables::{Aml, AmlSink};
use arch::aarch64::layout::{
    ACPI_SIZE, ACPI_START, GTIMER_HYP, GTIMER_PHYS, GTIMER_SEC, GTIMER_VIRT,
};
use vm_memory::{Bytes, GuestAddress, GuestMemoryBackend, GuestMemoryMmap};

use crate::DeviceType;
use crate::fdt::DeviceInfoForFDT;
use crate::legacy::IrqChip;
use crate::legacy::gic::GICDevice;

const OEM_ID: [u8; 6] = *b"LIMINA";
const OEM_REVISION: u32 = 1;

/// Private peripheral interrupts are numbered from 16 in the GIC's interrupt space, which is
/// what ACPI GSIVs use.
const PPI_BASE: u32 = 16;

/// FADT `ARM_BOOT_ARCH`: PSCI is implemented, and its conduit is HVC rather than SMC.
const ARM_BOOT_ARCH_PSCI_COMPLIANT: u16 = 1 << 0;
#[cfg(target_os = "linux")]
const ARM_BOOT_ARCH_PSCI_USE_HVC: u16 = 1 << 1;

/// The PL061 lines the device tree's `gpio-keys` node names, which the VMM pulses from its
/// power, sleep and wake eventfds. Under ACPI they are GPIO-signalled events instead: the
/// controller's `_AEI` lists them and its `_Exx` methods notify the power and sleep buttons.
/// The restart line (5) has no ACPI button to map to.
const GPIO_POWER_LINE: u16 = 3;
const GPIO_SLEEP_LINE: u16 = 4;
const GPIO_WAKE_LINE: u16 = 6;
/// Button notification values: the button was pressed, or it woke the system.
const NOTIFY_BUTTON_PRESSED: u8 = 0x80;
const NOTIFY_BUTTON_WAKE: u8 = 0x02;

/// GTDT timer flags.
const GTDT_TIMER_ALWAYS_ON: u32 = 1 << 2;
const GTDT_REVISION: u8 = 3;
/// A GTDT field that names no memory-mapped counter.
const GTDT_NO_COUNTER_BLOCK: u64 = u64::MAX;

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    /// The machine's interrupt controller is not a GICv3, the only one these tables describe.
    UnsupportedGic(String),
    /// The tables do not fit in the ACPI window.
    TooLarge(usize),
    /// A device lies above 4 GiB, which its 32-bit memory descriptor cannot name.
    DeviceAbove4G(u64),
    /// Writing the tables to guest memory failed.
    Write,
}

pub type Result<T> = std::result::Result<T, Error>;

/// Writes the tables into the ACPI window when guest memory has one. Returns whether it did.
pub fn create_acpi_tables<T: DeviceInfoForFDT + Clone + Debug>(
    guest_mem: &GuestMemoryMmap,
    vcpu_mpidr: &[u64],
    device_info: &HashMap<(DeviceType, String), T>,
    gic_device: &IrqChip,
) -> Result<bool> {
    if guest_mem.find_region(GuestAddress(ACPI_START)).is_none() {
        return Ok(false);
    }
    let gic = {
        let gic = gic_device.lock().unwrap();
        Gic {
            compatible: gic.fdt_compatibility(),
            properties: gic.device_properties(),
            maint_ppi: gic.fdt_maint_irq(),
        }
    };
    let blob = build_tables(ACPI_START, vcpu_mpidr, device_info, &gic)?;
    if blob.len() > ACPI_SIZE {
        return Err(Error::TooLarge(blob.len()));
    }
    guest_mem
        .write_slice(&blob, GuestAddress(ACPI_START))
        .map_err(|_| Error::Write)?;
    Ok(true)
}

struct Gic {
    compatible: String,
    /// `[distributor base, size, redistributors base, size]`.
    properties: Vec<u64>,
    maint_ppi: u32,
}

/// Lays the tables out from `base`: RSDP, XSDT, FADT, MADT, GTDT, PPTT, DSDT, each 8-byte
/// aligned.
fn build_tables<T: DeviceInfoForFDT + Clone + Debug>(
    base: u64,
    vcpu_mpidr: &[u64],
    device_info: &HashMap<(DeviceType, String), T>,
    gic: &Gic,
) -> Result<Vec<u8>> {
    if gic.compatible != "arm,gic-v3" {
        return Err(Error::UnsupportedGic(gic.compatible.clone()));
    }
    let madt = build_madt(vcpu_mpidr, &gic.properties, gic.maint_ppi);
    let gtdt = build_gtdt();
    let pptt = build_pptt(vcpu_mpidr.len());
    let dsdt = build_dsdt(vcpu_mpidr.len(), device_info)?;

    let rsdp_len = aml_bytes(&Rsdp::new(OEM_ID, 0)).len();
    let xsdt_len = build_xsdt(&[0, 0, 0, 0]).len();
    let fadt_len = build_fadt(0).len();

    let xsdt_addr = base + align8(rsdp_len);
    let fadt_addr = xsdt_addr + align8(xsdt_len);
    let madt_addr = fadt_addr + align8(fadt_len);
    let gtdt_addr = madt_addr + align8(madt.len());
    let pptt_addr = gtdt_addr + align8(gtdt.len());
    let dsdt_addr = pptt_addr + align8(pptt.len());

    let mut blob = Vec::new();
    for (addr, table) in [
        (base, aml_bytes(&Rsdp::new(OEM_ID, xsdt_addr))),
        (
            xsdt_addr,
            build_xsdt(&[fadt_addr, madt_addr, gtdt_addr, pptt_addr]),
        ),
        (fadt_addr, build_fadt(dsdt_addr)),
        (madt_addr, madt),
        (gtdt_addr, gtdt),
        (pptt_addr, pptt),
        (dsdt_addr, dsdt),
    ] {
        blob.resize((addr - base) as usize, 0);
        blob.extend_from_slice(&table);
    }
    Ok(blob)
}

fn aml_bytes(aml: &dyn Aml) -> Vec<u8> {
    let mut bytes = Vec::new();
    aml.to_aml_bytes(&mut bytes);
    bytes
}

fn align8(len: usize) -> u64 {
    len.next_multiple_of(8) as u64
}

fn build_xsdt(entries: &[u64]) -> Vec<u8> {
    let mut xsdt = XSDT::new(OEM_ID, *b"LIMNXSDT", OEM_REVISION);
    for &entry in entries {
        xsdt.add_entry(entry);
    }
    let mut bytes = Vec::new();
    xsdt.to_aml_bytes(&mut bytes);
    bytes
}

fn build_fadt(dsdt_addr: u64) -> Vec<u8> {
    let mut builder = FADTBuilder::new(OEM_ID, *b"LIMNFACP", OEM_REVISION)
        .dsdt_64(dsdt_addr)
        .flag(Flags::HwReducedAcpi);
    #[cfg(target_os = "linux")]
    let arm_boot_arch = ARM_BOOT_ARCH_PSCI_COMPLIANT | ARM_BOOT_ARCH_PSCI_USE_HVC;
    #[cfg(not(target_os = "linux"))]
    let arm_boot_arch = ARM_BOOT_ARCH_PSCI_COMPLIANT;
    builder.arm_boot_arch = arm_boot_arch.into();
    let mut bytes = Vec::new();
    builder.finalize().to_aml_bytes(&mut bytes);
    bytes
}

fn build_madt(vcpu_mpidr: &[u64], gic_properties: &[u64], maint_ppi: u32) -> Vec<u8> {
    let mut madt = MADT::new(
        OEM_ID,
        *b"LIMNAPIC",
        OEM_REVISION,
        LocalInterruptController::Address(0),
    );
    for (index, &mpidr) in vcpu_mpidr.iter().enumerate() {
        madt.add_structure(
            Gicc::new(EnabledStatus::Enabled)
                .cpu_interface_number(index as u32)
                .acpi_processor_uid(index as u32)
                .mpidr(mpidr & 0xFF_00FF_FFFF)
                .maintenance_interrupt(PPI_BASE + maint_ppi, Trigger::Level),
        );
    }
    madt.add_structure(Gicd::new(0, gic_properties[0], GicVersion::GICv3));
    madt.add_structure(Gicr::new(gic_properties[2], gic_properties[3] as u32));
    let mut bytes = Vec::new();
    madt.to_aml_bytes(&mut bytes);
    bytes
}

/// The architected timers, with the interrupts the device tree's `arm,armv8-timer` node names,
/// in the same roles: secure EL1, non-secure EL1, virtual EL1, EL2.
fn build_gtdt() -> Vec<u8> {
    let timer = |ppi: u32| [PPI_BASE + ppi, GTDT_TIMER_ALWAYS_ON];
    let mut gtdt = Sdt::new(
        *b"GTDT",
        36,
        GTDT_REVISION,
        OEM_ID,
        *b"LIMNGTDT",
        OEM_REVISION,
    );
    gtdt.append(GTDT_NO_COUNTER_BLOCK);
    gtdt.append(0u32);
    for ppi in [GTIMER_SEC, GTIMER_HYP, GTIMER_VIRT, GTIMER_PHYS] {
        gtdt.append(timer(ppi));
    }
    gtdt.append(GTDT_NO_COUNTER_BLOCK);
    // No platform timers; the offset is unused when the count is zero.
    gtdt.append([0u32, 0u32]);
    // The virtual EL2 timer (revision 3) is not provided.
    gtdt.append([0u32, 0u32]);
    gtdt.update_checksum();
    gtdt.as_slice().to_vec()
}

/// One package holding every vCPU. Without a PPTT, arm64 Linux cannot work out the CPU
/// topology under ACPI and its cacheinfo and topology sysfs setup fail.
fn build_pptt(vcpus: usize) -> Vec<u8> {
    let mut pptt = PPTT::new(OEM_ID, *b"LIMNPPTT", OEM_REVISION);
    let package = pptt.add_processor(ProcessorNode::new(None, 0).physical());
    for index in 0..vcpus {
        pptt.add_processor(
            ProcessorNode::new(Some(&package), index as u32)
                .valid()
                .leaf(),
        );
    }
    aml_bytes(&pptt)
}

/// The vCPUs as processor devices, then the platform devices. arm64 Linux registers a CPU's
/// device (`/sys/devices/system/cpu/cpuN`) only for a processor device whose `_UID` matches a
/// MADT GICC's processor UID; without them it boots with no CPU devices, so the topology and
/// cacheinfo sysfs setup fails and anything reading it (lscpu, qemu-ga's vCPU calls) breaks.
fn build_dsdt<T: DeviceInfoForFDT + Clone + Debug>(
    vcpus: usize,
    device_info: &HashMap<(DeviceType, String), T>,
) -> Result<Vec<u8>> {
    let mut devices: Vec<_> = device_info
        .iter()
        .map(|((device_type, _), info)| (*device_type, info))
        .collect();
    devices.sort_by_key(|(_, info)| info.addr());

    let mut body = Vec::new();
    for index in 0..vcpus {
        let hid = Name::new(Path::new("_HID"), &"ACPI0007");
        let uid = Name::new(Path::new("_UID"), &(index as u32));
        Device::new(Path::new(&format!("C{index:03X}")), vec![&hid, &uid]).to_aml_bytes(&mut body);
    }
    let pwrb_hid = Name::new(Path::new("_HID"), &"PNP0C0C");
    let slpb_hid = Name::new(Path::new("_HID"), &"PNP0C0E");
    let button_uid = Name::new(Path::new("_UID"), &0u32);

    let gpio_events = GpioEvents::new();
    let mut virtio_uid = 0u32;
    for (device_type, info) in devices {
        let mut extra: Vec<&dyn Aml> = Vec::new();
        let (name, hid, uid, coherent) = match device_type {
            DeviceType::Virtio(_) => {
                virtio_uid += 1;
                (
                    format!("VR{:02X}", virtio_uid - 1),
                    "LNRO0005",
                    virtio_uid - 1,
                    false,
                )
            }
            DeviceType::Serial => ("COM0".to_string(), "ARMH0011", 0, false),
            DeviceType::Xhci => ("USB0".to_string(), "PNP0D10", 0, true),
            DeviceType::Gpio => {
                Device::new(Path::new("PWRB"), vec![&pwrb_hid, &button_uid])
                    .to_aml_bytes(&mut body);
                Device::new(Path::new("SLPB"), vec![&slpb_hid, &button_uid])
                    .to_aml_bytes(&mut body);
                extra = gpio_events.children();
                ("GPO0".to_string(), "ARMH0061", 0, false)
            }
            // Not described yet: the virtual cpufreq controller and the TPM. The PL031 has no
            // ACPI binding; the OS reaches the clock through the firmware's GetTime.
            _ => continue,
        };
        body.extend(mmio_device(&name, hid, uid, info, coherent, &extra)?);
    }

    let mut dsdt = Sdt::new(*b"DSDT", 36, 2, OEM_ID, *b"LIMNDSDT", OEM_REVISION);
    dsdt.append_slice(&Scope::raw(Path::new("\\_SB_"), body));
    Ok(dsdt.as_slice().to_vec())
}

/// One MMIO window and one edge-triggered SPI: HVF's in-kernel GIC only pulses SPIs, which is
/// why the device tree declares them edge-triggered too.
fn mmio_device<T: DeviceInfoForFDT>(
    name: &str,
    hid: &'static str,
    uid: u32,
    info: &T,
    coherent: bool,
    extra: &[&dyn Aml],
) -> Result<Vec<u8>> {
    let base = u32::try_from(info.addr()).map_err(|_| Error::DeviceAbove4G(info.addr()))?;
    let mem = Memory32Fixed::new(true, base, info.length() as u32);
    let irq = Interrupt::new(true, true, false, false, info.irq());
    let resources = ResourceTemplate::new(vec![&mem, &irq]);
    let hid = Name::new(Path::new("_HID"), &hid);
    let uid = Name::new(Path::new("_UID"), &uid);
    let crs = Name::new(Path::new("_CRS"), &resources);
    let cca = Name::new(Path::new("_CCA"), &1u8);
    let mut children: Vec<&dyn Aml> = vec![&hid, &uid, &crs];
    if coherent {
        children.push(&cca);
    }
    children.extend_from_slice(extra);
    let mut bytes = Vec::new();
    Device::new(Path::new(name), children).to_aml_bytes(&mut bytes);
    Ok(bytes)
}

/// The PL061's `_AEI` and its event methods: the power and sleep lines press the buttons, and
/// the wake line, marked wake-capable so Linux arms its interrupt for wake during s2idle, tells
/// the sleep button it woke the system (a wake, not a press).
struct GpioEvents {
    aei: Name,
    methods: [EventMethod; 3],
}

impl GpioEvents {
    fn new() -> Self {
        let pins = [
            GpioInt::edge(GPIO_POWER_LINE, false),
            GpioInt::edge(GPIO_SLEEP_LINE, false),
            GpioInt::edge(GPIO_WAKE_LINE, true),
        ];
        let resources: Vec<&dyn Aml> = pins.iter().map(|pin| pin as &dyn Aml).collect();
        GpioEvents {
            aei: Name::new(Path::new("_AEI"), &ResourceTemplate::new(resources)),
            methods: [
                EventMethod::new(GPIO_POWER_LINE, "\\_SB_.PWRB", NOTIFY_BUTTON_PRESSED),
                EventMethod::new(GPIO_SLEEP_LINE, "\\_SB_.SLPB", NOTIFY_BUTTON_PRESSED),
                EventMethod::new(GPIO_WAKE_LINE, "\\_SB_.SLPB", NOTIFY_BUTTON_WAKE),
            ],
        }
    }

    fn children(&self) -> Vec<&dyn Aml> {
        let mut children: Vec<&dyn Aml> = vec![&self.aei];
        children.extend(self.methods.iter().map(|m| m as &dyn Aml));
        children
    }
}

/// `Method (_Exx) { Notify (<button>, <value>) }`, the handler ACPI runs for GPIO event line xx.
struct EventMethod {
    line: u16,
    button: Path,
    value: u8,
}

impl EventMethod {
    fn new(line: u16, button: &str, value: u8) -> Self {
        EventMethod {
            line,
            button: Path::new(button),
            value,
        }
    }
}

impl Aml for EventMethod {
    fn to_aml_bytes(&self, sink: &mut dyn AmlSink) {
        let notify = Notify::new(&self.button, &self.value);
        Method::new(
            Path::new(&format!("_E{:02X}", self.line)),
            0,
            false,
            vec![&notify],
        )
        .to_aml_bytes(sink);
    }
}

/// A GpioInt connection descriptor (ACPI 6.5, 6.4.3.8.1) for one edge-triggered, active-high
/// pin on `\_SB_.GPO0`.
struct GpioInt {
    pin: u16,
    wake: bool,
}

impl GpioInt {
    const SOURCE: &'static [u8] = b"\\_SB_.GPO0\0";
    const HEADER_LEN: u16 = 23;

    fn edge(pin: u16, wake: bool) -> Self {
        GpioInt { pin, wake }
    }
}

impl Aml for GpioInt {
    fn to_aml_bytes(&self, sink: &mut dyn AmlSink) {
        let pin_table = Self::HEADER_LEN;
        let source = pin_table + 2;
        let vendor = source + Self::SOURCE.len() as u16;
        // Edge-triggered, active high; exclusive, or exclusive and wake.
        let flags: u16 = 1 | if self.wake { 2 << 3 } else { 0 };
        sink.byte(0x8C);
        sink.word(vendor - 3);
        sink.byte(1); // revision
        sink.byte(0); // an interrupt connection
        sink.word(1); // consumer
        sink.word(flags);
        sink.byte(0); // default pull
        sink.word(0); // drive strength
        sink.word(0); // debounce
        sink.word(pin_table);
        sink.byte(0); // resource source index
        sink.word(source);
        sink.word(vendor);
        sink.word(0); // no vendor data
        sink.word(self.pin);
        sink.vec(Self::SOURCE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct Info {
        addr: u64,
        irq: u32,
    }

    impl DeviceInfoForFDT for Info {
        fn addr(&self) -> u64 {
            self.addr
        }
        fn irq(&self) -> u32 {
            self.irq
        }
        fn length(&self) -> u64 {
            0x1000
        }
    }

    fn checksum_ok(table: &[u8]) -> bool {
        table.iter().fold(0u8, |sum, b| sum.wrapping_add(*b)) == 0
    }

    fn table_at(blob: &[u8], offset: usize) -> &[u8] {
        let len = u32::from_le_bytes(blob[offset + 4..offset + 8].try_into().unwrap()) as usize;
        &blob[offset..offset + len]
    }

    fn u64_at(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
    }

    #[test]
    fn madt_has_a_gicc_per_vcpu_then_gicd_and_gicr() {
        let madt = build_madt(
            &[0x8000_0000, 0x8000_0001],
            &[0x1000, 0x1_0000, 0x2000, 0x4_0000],
            9,
        );
        assert_eq!(&madt[..4], b"APIC");
        assert!(checksum_ok(&madt));
        let mut types = Vec::new();
        let mut offset = 44;
        while offset < madt.len() {
            types.push(madt[offset]);
            offset += madt[offset + 1] as usize;
        }
        assert_eq!(types, [0xb, 0xb, 0xc, 0xe]);
        // The second GICC's MPIDR, with the non-affinity bits cleared.
        assert_eq!(u64_at(&madt, 44 + 82 + 68), 1);
    }

    #[test]
    fn gtdt_names_the_device_tree_timer_interrupts() {
        let gtdt = build_gtdt();
        assert_eq!(&gtdt[..4], b"GTDT");
        assert_eq!(gtdt.len(), 104);
        assert!(checksum_ok(&gtdt));
        let gsiv = |offset: usize| u32::from_le_bytes(gtdt[offset..offset + 4].try_into().unwrap());
        assert_eq!(
            [gsiv(48), gsiv(56), gsiv(64), gsiv(72)],
            [
                16 + GTIMER_SEC,
                16 + GTIMER_HYP,
                16 + GTIMER_VIRT,
                16 + GTIMER_PHYS
            ]
        );
    }

    #[test]
    fn dsdt_describes_virtio_serial_and_xhci() {
        let mut devices = HashMap::new();
        devices.insert(
            (DeviceType::Virtio(2), "blk".into()),
            Info {
                addr: 0x0a00_2000,
                irq: 34,
            },
        );
        devices.insert(
            (DeviceType::Serial, "uart".into()),
            Info {
                addr: 0x0a00_1000,
                irq: 33,
            },
        );
        devices.insert(
            (DeviceType::Xhci, "usb".into()),
            Info {
                addr: 0x0a00_3000,
                irq: 35,
            },
        );
        devices.insert(
            (DeviceType::RTC, "rtc".into()),
            Info {
                addr: 0x0a00_4000,
                irq: 36,
            },
        );
        let dsdt = build_dsdt(2, &devices).unwrap();
        assert!(checksum_ok(&dsdt));
        for needle in [
            &b"ACPI0007"[..],
            b"C001",
            b"LNRO0005",
            b"ARMH0011",
            b"PNP0D10",
            b"_CCA",
        ] {
            assert!(dsdt.windows(needle.len()).any(|w| w == needle));
        }
        // The RTC has no ACPI binding and is left out.
        let irq36 = [0x89, 6, 0, 0x03, 1, 36, 0, 0, 0];
        assert!(!dsdt.windows(irq36.len()).any(|w| w == irq36));
    }

    #[test]
    fn gpio_events_press_the_buttons_and_wake() {
        let mut devices = HashMap::new();
        devices.insert(
            (DeviceType::Gpio, "gpio".into()),
            Info {
                addr: 0x0a00_5000,
                irq: 37,
            },
        );
        let dsdt = build_dsdt(1, &devices).unwrap();
        assert!(checksum_ok(&dsdt));
        for needle in [
            &b"ARMH0061"[..],
            b"PNP0C0C",
            b"PNP0C0E",
            b"_AEI",
            b"_E03",
            b"_E04",
            b"_E06",
        ] {
            assert!(
                dsdt.windows(needle.len()).any(|w| w == needle),
                "{needle:?}"
            );
        }
    }

    #[test]
    fn gpio_int_descriptor_layout() {
        let mut bytes = Vec::new();
        GpioInt::edge(6, true).to_aml_bytes(&mut bytes);
        assert_eq!(bytes.len(), 36);
        assert_eq!(bytes[0], 0x8C);
        assert_eq!(u16::from_le_bytes([bytes[1], bytes[2]]), 33);
        // Edge-triggered, exclusive and wake.
        assert_eq!(u16::from_le_bytes([bytes[7], bytes[8]]), 0x11);
        // The pin, then the controller's path.
        assert_eq!(u16::from_le_bytes([bytes[23], bytes[24]]), 6);
        assert_eq!(&bytes[25..], b"\\_SB_.GPO0\0");
        let mut plain = Vec::new();
        GpioInt::edge(3, false).to_aml_bytes(&mut plain);
        assert_eq!(u16::from_le_bytes([plain[7], plain[8]]), 0x01);
    }

    #[test]
    fn rsdp_leads_to_every_table() {
        let gic = Gic {
            compatible: "arm,gic-v3".into(),
            properties: vec![0x1000, 0x1_0000, 0x2000, 0x4_0000],
            maint_ppi: 9,
        };
        let devices: HashMap<(DeviceType, String), Info> = HashMap::new();
        let base = 0x0408_0000;
        let blob = build_tables(base, &[0], &devices, &gic).unwrap();
        assert_eq!(&blob[..8], b"RSD PTR ");
        let xsdt = table_at(&blob, (u64_at(&blob, 24) - base) as usize);
        assert_eq!(&xsdt[..4], b"XSDT");
        let mut signatures = Vec::new();
        for entry in xsdt[36..].chunks(8) {
            let table = table_at(
                &blob,
                (u64::from_le_bytes(entry.try_into().unwrap()) - base) as usize,
            );
            assert!(checksum_ok(table));
            signatures.push(table[..4].to_vec());
            if &table[..4] == b"FACP" {
                let dsdt = table_at(&blob, (u64_at(table, 140) - base) as usize);
                assert_eq!(&dsdt[..4], b"DSDT");
                assert!(checksum_ok(dsdt));
            }
        }
        assert_eq!(signatures, [b"FACP", b"APIC", b"GTDT", b"PPTT"]);
    }
}
