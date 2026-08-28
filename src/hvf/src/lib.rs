// Copyright 2021 Red Hat, Inc.
// SPDX-License-Identifier: Apache-2.0

#[allow(non_camel_case_types)]
#[allow(improper_ctypes)]
#[allow(dead_code)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(deref_nullptr)]
pub mod bindings;
pub mod released_ram;

#[macro_use]
extern crate log;

use bindings::*;
pub use released_ram::{FaultOutcome, ReleasedRam, ReleasedRamStats};

#[cfg(target_arch = "aarch64")]
use std::arch::asm;
use std::cell::Cell;

use std::convert::TryInto;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
use arch::aarch64::sysreg::{SYSREG_MASK, sys_reg_name};
use log::{debug, error, warn};

unsafe extern "C" {
    pub fn mach_absolute_time() -> u64;
    fn getpagesize() -> i32;
}

/// The HOST page size (16 KiB on Apple Silicon). hv_vm_map/hv_vm_unmap operate at this
/// granularity and reject any size that is not a multiple of it with HV_BAD_ARGUMENT.
fn host_page_size() -> u64 {
    static PAGE: LazyLock<u64> = LazyLock::new(|| unsafe { getpagesize() } as u64);
    *PAGE
}

/// The granule hv_vm_map/hv_vm_unmap actually enforce for this VM: the one
/// [`HvfVm::new`] successfully set, or the host page size when it set none.
///
/// 0 means "not set", which is also the state of a process that never created a VM.
static MAP_GRANULE: AtomicU64 = AtomicU64::new(0);

/// What every address and size handed to hv_vm_map must be a multiple of.
fn map_granule() -> u64 {
    match MAP_GRANULE.load(Ordering::Relaxed) {
        0 => host_page_size(),
        g => g,
    }
}

/// Round `size` up to the mapping granule.
///
/// A guest whose page size is smaller than the granule produces mappings sized at *its*
/// granularity — e.g. a 0x21000-byte virtio-gpu blob (33 × 4 KiB, size%16k=4096) on the
/// 16 KiB granule — which hv_vm_map/hv_vm_unmap reject outright. Rounding the size up is
/// safe *because the granule is the unit HVF itself refuses below*: the tail can only run
/// into a neighbour whose own start is not granule-aligned, and mapping such a neighbour is
/// itself refused before it can reach these bytes. On the host side an mmap'ed region always
/// occupies whole host pages, and the granule never exceeds one, so the tail belongs to the
/// same host mapping.
///
/// That argument is why this rounds to the GRANULE and not to the host page: under a 4 KiB
/// granule a neighbouring blob at the next 4 KiB guest address is legal, and rounding to
/// 16 KiB would map over it — HVF answers an overlapping map with HV_ERROR, and the blob the
/// guest was promised never arrives.
///
/// Map and unmap must round identically or a successful rounded map would leak its tail at
/// unmap time.
fn round_up_to_map_granule(size: u64) -> u64 {
    round_up(size, map_granule())
}

fn round_up(size: u64, granule: u64) -> u64 {
    size.div_ceil(granule).saturating_mul(granule)
}

const HV_EXIT_REASON_CANCELED: hv_exit_reason_t = 0;
const HV_EXIT_REASON_EXCEPTION: hv_exit_reason_t = 1;
const HV_EXIT_REASON_VTIMER_ACTIVATED: hv_exit_reason_t = 2;

const TMR_CTL_ENABLE: u64 = 1 << 0;
const TMR_CTL_IMASK: u64 = 1 << 1;
const TMR_CTL_ISTATUS: u64 = 1 << 2;

const PSR_MODE_EL1H: u64 = 0x0000_0005;
const PSR_MODE_EL2H: u64 = 0x0000_0009;
const PSR_F_BIT: u64 = 0x0000_0040;
const PSR_I_BIT: u64 = 0x0000_0080;
const PSR_A_BIT: u64 = 0x0000_0100;
const PSR_D_BIT: u64 = 0x0000_0200;
const PSTATE_EL1_FAULT_BITS_64: u64 = PSR_MODE_EL1H | PSR_A_BIT | PSR_F_BIT | PSR_I_BIT | PSR_D_BIT;
const PSTATE_EL2_FAULT_BITS_64: u64 = PSR_MODE_EL2H | PSR_A_BIT | PSR_F_BIT | PSR_I_BIT | PSR_D_BIT;

const HCR_TLOR: u64 = 1 << 35;
const HCR_RW: u64 = 1 << 31;
const HCR_TSW: u64 = 1 << 22;
const HCR_TACR: u64 = 1 << 21;
const HCR_TIDCP: u64 = 1 << 20;
const HCR_TSC: u64 = 1 << 19;
const HCR_TID3: u64 = 1 << 18;
const HCR_TWE: u64 = 1 << 14;
const HCR_TWI: u64 = 1 << 13;
const HCR_BSU_IS: u64 = 1 << 10;
const HCR_FB: u64 = 1 << 9;
const HCR_AMO: u64 = 1 << 5;
const HCR_IMO: u64 = 1 << 4;
const HCR_FMO: u64 = 1 << 3;
const HCR_PTW: u64 = 1 << 2;
const HCR_SWIO: u64 = 1 << 1;
const HCR_VM: u64 = 1 << 0;
// Use the same bits as KVM uses in vcpu reset.
const HCR_EL2_BITS: u64 = HCR_TSC
    | HCR_TSW
    | HCR_TWE
    | HCR_TWI
    | HCR_VM
    | HCR_BSU_IS
    | HCR_FB
    | HCR_TACR
    | HCR_AMO
    | HCR_SWIO
    | HCR_TIDCP
    | HCR_RW
    | HCR_TLOR
    | HCR_FMO
    | HCR_IMO
    | HCR_PTW
    | HCR_TID3;

const CNTHCTL_EL0VCTEN: u64 = 1 << 1;
const CNTHCTL_EL0PCTEN: u64 = 1 << 0;
// Trap accesses to both virtual and physical counter registers.
const CNTHCTL_EL2_BITS: u64 = CNTHCTL_EL0VCTEN | CNTHCTL_EL0PCTEN;

const AA64PFR0_EL1_EL2EN: u64 = 1 << 8;
const AA64PFR0_EL1_GIC3EN: u64 = 1 << 24;
const AA64PFR1_EL1_SMEMASK: u64 = 0xf << 24;

const EC_WFX_TRAP: u64 = 0x1;
const EC_AA64_HVC: u64 = 0x16;
const EC_AA64_SMC: u64 = 0x17;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
const EC_SYSTEMREGISTERTRAP: u64 = 0x18;
const EC_INSTRABORT: u64 = 0x20;
const EC_DATAABORT: u64 = 0x24;
const EC_AA64_BKPT: u64 = 0x3c;

/// PSCI standard return: the requested function is not implemented (Arm DEN 0022, value -1).
/// A 64-bit -1 also reads as -1 in the 32-bit (w0) calling convention, so it's correct for both
/// SMC32 and SMC64 callers. Returned for any PSCI/SMC function we don't model, so a guest that
/// probes an optional function (e.g. PSCI_FEATURES) keeps running instead of crashing the VMM.
const PSCI_RET_NOT_SUPPORTED: u64 = -1i64 as u64;

/// PSCI standard return: the request is understood but refused in the current state (Arm DEN 0022,
/// value -3). Returned for a `SYSTEM_SUSPEND` issued while another vCPU is still online.
const PSCI_RET_DENIED: u64 = -3i64 as u64;

/// Whether to offer the guest PSCI 1.0 and `SYSTEM_SUSPEND`. On unless
/// `LIMINA_PSCI_SYSTEM_SUSPEND` is `0`/`off`/empty.
///
/// The kill switch for the whole feature, in the shape `LIMINA_VCPU_SCHED` established: read
/// here, once, so a worker inherits it with no plumbing and a bad dogfood run is bisectable
/// without a rebuild. Off, we report 0.2 exactly as before and refuse SYSTEM_SUSPEND, so the
/// guest never registers `psci_suspend_ops` and every suspend stays on the s2idle path —
/// guest-visible behaviour identical to a build without this code.
///
/// It needs a kill switch because advertising the call IS the rollout: Linux's
/// `mem_sleep_default` is `PM_SUSPEND_MAX`, so the moment PM_SUSPEND_MEM becomes valid a plain
/// `systemctl suspend` switches from s2idle to deep with no guest configuration at all.
fn system_suspend_offered() -> bool {
    static OFFERED: LazyLock<bool> = LazyLock::new(|| {
        !matches!(
            std::env::var("LIMINA_PSCI_SYSTEM_SUSPEND").as_deref(),
            Ok("0") | Ok("off") | Ok("")
        )
    });
    *OFFERED
}

/// The PSCI version we report: 1.0, encoded major<<16 | minor (Arm DEN 0022).
///
/// Reporting >= 1.0 is what makes Linux probe the 1.0 function set at all — `psci_probe()` only
/// calls `psci_init_system_suspend()` when the major version is >= 1 (`drivers/firmware/psci/psci.c`).
/// It also costs a handful of extra probe SMCs at boot (SMCCC_VERSION, CPU_SUSPEND features,
/// the KVM vendor-hypervisor UID), all of which we answer NOT_SUPPORTED.
const PSCI_VERSION_1_0: u64 = 0x0001_0000;

/// The PSCI version we reported before `SYSTEM_SUSPEND` existed, and still do when it is off.
const PSCI_VERSION_0_2: u64 = 0x0000_0002;

#[derive(Debug)]
pub enum Error {
    EnableEL2,
    SetIpaGranule,
    FindSymbol(libloading::Error),
    MemoryMap,
    MemoryUnmap,
    NestedCheck,
    /// The guest reached a vCPU state we can't emulate (unknown exit reason, exception class, or
    /// system register). Not fatal to the VMM: the run loop logs it and stops the VM cleanly
    /// instead of `panic!`ing the worker process. The specifics are logged at the trap site.
    Unhandled,
    /// A stage-2 fault on balloon-released guest RAM could not be healed (re-map failed);
    /// resuming would fault forever on the same access, so the VM must stop.
    StageTwoHeal,
    VcpuCreate,
    VcpuInitialRegisters,
    VcpuReadRegister,
    VcpuReadSystemRegister,
    VcpuRequestExit,
    VcpuRun,
    VcpuSetPendingIrq,
    VcpuSetRegister,
    VcpuSetSystemRegister(u16, u64),
    VcpuSetVtimerMask,
    /// A vCPU snapshot save/restore FFI call (SIMD, vtimer offset, or pending interrupt) failed.
    VcpuSnapshot,
    VmCreate,
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use self::Error::*;

        match self {
            EnableEL2 => write!(f, "Error enabling EL2 mode in HVF"),
            SetIpaGranule => write!(f, "Error setting the HVF stage-2 IPA granule"),
            FindSymbol(err) => write!(f, "Couldn't find symbol in HVF library: {err}"),
            MemoryMap => write!(f, "Error registering memory region in HVF"),
            MemoryUnmap => write!(f, "Error unregistering memory region in HVF"),
            Unhandled => write!(f, "Unhandled guest vCPU state (see log); stopping the VM"),
            StageTwoHeal => write!(
                f,
                "Failed to re-map balloon-released guest RAM on a stage-2 fault (see log); \
                 stopping the VM"
            ),
            NestedCheck => write!(
                f,
                "Nested virtualization was requested but it's not support in this system"
            ),
            VcpuCreate => write!(f, "Error creating HVF vCPU instance"),
            VcpuInitialRegisters => write!(f, "Error setting up initial HVF vCPU registers"),
            VcpuReadRegister => write!(f, "Error reading HVF vCPU register"),
            VcpuReadSystemRegister => write!(f, "Error reading HVF vCPU system register"),
            VcpuRequestExit => write!(f, "Error requesting HVF vCPU exit"),
            VcpuRun => write!(f, "Error running HVF vCPU"),
            VcpuSetPendingIrq => write!(f, "Error setting HVF vCPU pending irq"),
            VcpuSetRegister => write!(f, "Error setting HVF vCPU register"),
            VcpuSetSystemRegister(reg, val) => write!(
                f,
                "Error setting HVF vCPU system register 0x{reg:#x} to 0x{val:#x}"
            ),
            VcpuSetVtimerMask => write!(f, "Error setting HVF vCPU vtimer mask"),
            VcpuSnapshot => write!(f, "Error saving/restoring HVF vCPU snapshot state"),
            VmCreate => write!(f, "Error creating HVF VM instance"),
        }
    }
}

pub enum InterruptType {
    Irq,
    Fiq,
}

pub trait Vcpus {
    fn set_vtimer_irq(&self, vcpuid: u64);
    fn should_wait(&self, vcpuid: u64) -> bool;
    fn has_pending_irq(&self, vcpuid: u64) -> bool;
    fn get_pending_irq(&self, vcpuid: u64) -> u32;
    fn handle_sysreg_read(&self, vcpuid: u64, reg: u32) -> Option<u64>;
    fn handle_sysreg_write(&self, vcpuid: u64, reg: u32, val: u64) -> bool;
    /// Record a vCPU's PSCI power state (CPU hotplug): `false` on CPU_OFF, `true` on CPU_ON.
    /// Consulted by AFFINITY_INFO so the guest's offline reaper sees the CPU report OFF. A
    /// vcpuid outside `[0, cpu_count)` (a bad guest-supplied MPIDR) is ignored, never panics.
    fn set_online(&self, vcpuid: u64, online: bool);
    /// This vCPU's PSCI power state. Out-of-range vcpuids report offline (safe default).
    fn is_online(&self, vcpuid: u64) -> bool;
    /// Are all vCPUs other than `vcpuid` powered off (PSCI `CPU_OFF`)? Gates `SYSTEM_SUSPEND`,
    /// which the spec allows only when the caller is the last core standing; anything else must
    /// be refused with `DENIED`.
    fn others_all_offline(&self, vcpuid: u64) -> bool;
    /// Record whether this vCPU is blocked in a WFx wait. Set around the wait, so a
    /// caller can tell a guest that has genuinely stopped executing from one that is
    /// merely not taking vmexits.
    fn set_parked(&self, vcpuid: u64, parked: bool);
}

/// Save the in-kernel GICv3 **distributor + redistributor** state (VM-wide) as an opaque,
/// versioned blob (Apple's binary-plist format). The per-vCPU CPU-interface (ICC) registers are
/// deliberately NOT included — Apple's `hv_gic_state` omits them — so they're captured per-vCPU
/// in [`VcpuState`]. Call while all vCPUs are quiesced (M9 snapshot).
pub fn save_gic_state() -> Result<Vec<u8>, Error> {
    let state = unsafe { hv_gic_state_create() };
    if state.is_null() {
        return Err(Error::VcpuSnapshot);
    }
    let mut size: usize = 0;
    if unsafe { hv_gic_state_get_size(state, &mut size) } != HV_SUCCESS {
        return Err(Error::VcpuSnapshot);
    }
    let mut buf = vec![0u8; size];
    if unsafe { hv_gic_state_get_data(state, buf.as_mut_ptr() as *mut _) } != HV_SUCCESS {
        return Err(Error::VcpuSnapshot);
    }
    Ok(buf)
}

/// Restore the in-kernel GICv3 distributor + redistributor state from a [`save_gic_state`] blob.
/// Must run after the GIC is (re-)created and before the vCPUs run (the per-vCPU ICC state is
/// then restored on each vCPU thread by [`HvfVcpu::restore_state`]).
pub fn restore_gic_state(data: &[u8]) -> Result<(), Error> {
    if unsafe { hv_gic_set_state(data.as_ptr() as *const _, data.len()) } != HV_SUCCESS {
        return Err(Error::VcpuSnapshot);
    }
    Ok(())
}

pub fn vcpu_request_exit(vcpuid: u64) -> Result<(), Error> {
    let mut vcpu: u64 = vcpuid;
    let ret = unsafe { hv_vcpus_exit(&mut vcpu, 1) };

    if ret != HV_SUCCESS {
        Err(Error::VcpuRequestExit)
    } else {
        Ok(())
    }
}

pub fn vcpu_set_pending_irq(
    vcpuid: u64,
    irq_type: InterruptType,
    pending: bool,
) -> Result<(), Error> {
    let _type = match irq_type {
        InterruptType::Irq => hv_interrupt_type_t_HV_INTERRUPT_TYPE_IRQ,
        InterruptType::Fiq => hv_interrupt_type_t_HV_INTERRUPT_TYPE_FIQ,
    };

    let ret = unsafe { hv_vcpu_set_pending_interrupt(vcpuid, _type, pending) };

    if ret != HV_SUCCESS {
        Err(Error::VcpuSetPendingIrq)
    } else {
        Ok(())
    }
}

pub fn vcpu_set_vtimer_mask(vcpuid: u64, masked: bool) -> Result<(), Error> {
    let ret = unsafe { hv_vcpu_set_vtimer_mask(vcpuid, masked) };

    if ret != HV_SUCCESS {
        Err(Error::VcpuSetVtimerMask)
    } else {
        Ok(())
    }
}

/// Checks if Nested Virtualization is supported on the current system. Only
/// M3 or newer chips on macOS 15+ will satisfy the requirements.
pub fn check_nested_virt() -> Result<bool, Error> {
    type GetEL2Supported =
        libloading::Symbol<'static, unsafe extern "C" fn(*mut bool) -> hv_return_t>;

    let get_el2_supported: Result<GetEL2Supported, libloading::Error> =
        unsafe { HVF.get(b"hv_vm_config_get_el2_supported") };
    if get_el2_supported.is_err() {
        info!("cannot find hv_vm_config_get_el2_supported symbol");
        return Ok(false);
    }

    let mut el2_supported: bool = false;
    let ret = unsafe { (get_el2_supported.unwrap())(&mut el2_supported) };
    if ret != HV_SUCCESS {
        error!("hv_vm_config_get_el2_supported failed: {ret:?}");
        return Err(Error::NestedCheck);
    }

    Ok(el2_supported)
}

/// Granule of the stage-2 (guest-physical -> host) translation, chosen at VM creation.
///
/// macOS pins this to the host page size unless asked otherwise, and on Apple silicon that is
/// 16 KiB — so a 4 KiB-page guest cannot express its own memory layout to us. Every guest
/// physical address, size and offset we are handed has to be a multiple of the granule, and a
/// guest that packs things at 4 KiB (virtio-gpu host-visible blobs, most visibly) hands us
/// addresses `hv_vm_map` refuses outright with `HV_BAD_ARGUMENT`.
///
/// `HV_IPA_GRANULE_4KB` lifts that. Measured both ways in `spikes/hv-ipa-granule/`: with the
/// coarse granule a 4 KiB-aligned guest address is refused and two separate host allocations
/// cannot be presented contiguously across a 4 KiB boundary; with the fine one both work.
///
/// Creation-time only — there is no setter for a live VM — so this is a per-boot decision, and
/// a restored snapshot must use the granule its guest's layout was built under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpaGranule {
    FourK,
    SixteenK,
}

impl IpaGranule {
    /// The values of Apple's `hv_ipa_granule_t`, verified against the macOS 26.5 SDK rather
    /// than assumed from declaration order.
    fn raw(self) -> u32 {
        match self {
            IpaGranule::FourK => 0,
            IpaGranule::SixteenK => 1,
        }
    }

    /// The alignment this granule imposes on every hv_vm_map operand.
    fn bytes(self) -> u64 {
        match self {
            IpaGranule::FourK => 4096,
            IpaGranule::SixteenK => 16384,
        }
    }
}

pub struct HvfVm {}

static HVF: LazyLock<libloading::Library> = LazyLock::new(|| unsafe {
    libloading::Library::new(
        "/System/Library/Frameworks/Hypervisor.framework/Versions/A/Hypervisor",
    )
    .unwrap()
});

impl HvfVm {
    pub fn new(nested_enabled: bool, ipa_granule: Option<IpaGranule>) -> Result<Self, Error> {
        let config = unsafe { hv_vm_config_create() };

        // macOS 26+. Looked up rather than linked so an older host simply keeps its default
        // granule instead of failing to start: a VM on the coarse granule still boots, it just
        // cannot map a 4 KiB-packed guest's blobs.
        if let Some(granule) = ipa_granule {
            type SetIpaGranule =
                libloading::Symbol<'static, unsafe extern "C" fn(hv_vm_config_t, u32) -> hv_return_t>;
            let set_ipa_granule: Result<SetIpaGranule, libloading::Error> =
                unsafe { HVF.get(b"hv_vm_config_set_ipa_granule") };
            match set_ipa_granule {
                Ok(set) => {
                    let ret = unsafe { (set)(config, granule.raw()) };
                    if ret != HV_SUCCESS {
                        error!("hv_vm_config_set_ipa_granule({granule:?}) failed: {ret:?}");
                        return Err(Error::SetIpaGranule);
                    }
                    // Only now: the rounding in map_memory/unmap_memory must follow what HVF
                    // is really enforcing, not what we asked for.
                    MAP_GRANULE.store(granule.bytes(), Ordering::Relaxed);
                    info!("stage-2 IPA granule set to {granule:?}");
                }
                Err(_) => {
                    // Not an error: the caller asked for something this macOS cannot express.
                    // Say so once, loudly enough to explain a later blob-map refusal.
                    warn!(
                        "hv_vm_config_set_ipa_granule is unavailable (needs macOS 26);                          keeping the host default granule"
                    );
                }
            }
        }
        if nested_enabled {
            let set_el2_enabled: libloading::Symbol<
                'static,
                unsafe extern "C" fn(hv_vm_config_t, bool) -> hv_return_t,
            > = unsafe {
                HVF.get(b"hv_vm_config_set_el2_enabled")
                    .map_err(Error::FindSymbol)?
            };

            let ret = unsafe { (set_el2_enabled)(config, true) };
            if ret != HV_SUCCESS {
                return Err(Error::EnableEL2);
            }
        }

        let ret = unsafe { hv_vm_create(config) };

        if ret != HV_SUCCESS {
            Err(Error::VmCreate)
        } else {
            Ok(Self {})
        }
    }

    pub fn map_memory(
        &self,
        host_start_addr: u64,
        guest_start_addr: u64,
        size: u64,
    ) -> Result<(), Error> {
        // A guest may request sizes below the granule (see round_up_to_map_granule); the
        // addresses are NOT rounded — a misaligned address is a real caller bug.
        let size = round_up_to_map_granule(size);
        let ret = unsafe {
            hv_vm_map(
                host_start_addr as *mut core::ffi::c_void,
                guest_start_addr,
                size.try_into().unwrap(),
                (HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC).into(),
            )
        };
        if ret != HV_SUCCESS {
            // hv_vm_map rejects any host addr, guest addr, or size that is not a multiple of
            // the granule with HV_BAD_ARGUMENT (0xfae94003). Surface which operand is
            // misaligned to make blob-mapping bugs diagnosable — and note that an OVERLAPPING
            // map answers HV_ERROR (0xfae94001) instead, one digit away and easily misread as
            // an alignment refusal, so a line here with every remainder 0 means overlap.
            let granule = map_granule();
            error!(
                "hv_vm_map failed: ret={ret:#x} host={host_start_addr:#x} guest={guest_start_addr:#x} size={size:#x} \
                 (granule={granule:#x} host%g={} guest%g={} size%g={})",
                host_start_addr % granule,
                guest_start_addr % granule,
                size % granule,
            );
            Err(Error::MemoryMap)
        } else {
            Ok(())
        }
    }

    pub fn unmap_memory(&self, guest_start_addr: u64, size: u64) -> Result<(), Error> {
        // Mirror map_memory's rounding exactly, or a rounded map's tail would survive its
        // unmap.
        let size = round_up_to_map_granule(size);
        let ret = unsafe { hv_vm_unmap(guest_start_addr, size.try_into().unwrap()) };
        if ret != HV_SUCCESS {
            Err(Error::MemoryUnmap)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub enum VcpuExit<'a> {
    Breakpoint,
    Canceled,
    CpuOn(u64, u64, u64),
    /// PSCI `CPU_OFF` — the guest offlined THIS vCPU (CPU hotplug). The vmm must park the vCPU
    /// thread cleanly until a matching `CpuOn` re-onlines it; CPU_OFF does not return on success,
    /// so no result register is written. (limina addition — without it CPU_OFF returns
    /// NOT_SUPPORTED, the guest busy-spins in `cpu_die`, and the VM wedges.)
    CpuOff,
    HypervisorCall,
    MmioRead(u64, &'a mut [u8]),
    MmioWrite(u64, &'a [u8]),
    PsciHandled,
    SecureMonitorCall,
    Shutdown,
    /// PSCI `SYSTEM_RESET` — the guest asked to reboot (vs `Shutdown` = power off). (limina addition.)
    Reset,
    /// A stage-2 fault on balloon-released guest RAM was healed (the range was re-validated and
    /// re-mapped). The PC was deliberately NOT advanced: re-running the vCPU retries the faulting
    /// access against the restored mapping. (limina addition — the FRQ/balloon unmap fix.)
    StageTwoHealed,
    /// PSCI `SYSTEM_SUSPEND` — the guest's LAST running vCPU asked firmware to suspend the whole
    /// system (suspend-to-RAM). Carries the resume `entry_point_address` and `context_id` the
    /// guest passed; the vmm must park this vCPU and later re-arm it at `entry` (the same reset
    /// shape as [`Self::CpuOn`]) to resume the guest. (limina addition.)
    ///
    /// This is the exact, race-free quiesce signal the GPIO-pulse-and-poll bracket could only
    /// approximate: Linux issues it from `syscore_suspend()`, i.e. AFTER every secondary vCPU has
    /// gone through `CPU_OFF` and AFTER `timekeeping_suspend()`, so on arrival the guest is
    /// provably stopped and its clock baseline is already recorded.
    SystemSuspend {
        entry: u64,
        context_id: u64,
    },
    SystemRegister,
    VtimerActivated,
    WaitForEvent,
    WaitForEventExpired,
    WaitForEventTimeout(Duration),
}

struct MmioRead {
    addr: u64,
    len: usize,
    srt: u32,
}

/// A vCPU's PSCI power state at snapshot time.
///
/// Without this the restore is architecturally incoherent, not merely imprecise: every vCPU
/// resumes at its saved PC, and for a parked one that PC is the instruction after the PSCI call
/// it never returned from. A secondary then returns from `CPU_OFF` — which Linux requires never
/// to return — and the boot CPU returns from a `SYSTEM_SUSPEND` it did not take. Deep suspend
/// puts every secondary in `Offline` and the caller in `SystemSuspended`, so this is the normal
/// state of any snapshot taken through the suspend bracket, not a corner case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VcpuPower {
    /// Executing guest code (or in a WFx wait, which resumes on its own).
    Running,
    /// Powered off via PSCI `CPU_OFF`; resumes only on a `CPU_ON` naming it.
    Offline,
    /// Parked in PSCI `SYSTEM_SUSPEND`, to be re-armed at `entry` on wake.
    SystemSuspended { entry: u64, context_id: u64 },
}

/// A full snapshot of one vCPU's architectural state (M9 suspend/resume). Everything HVF
/// exposes that a resumed guest needs: the general-purpose registers + PC + PSTATE, the SIMD/FP
/// bank, the EL1/EL0 system-register set ([`SNAPSHOT_SYS_REGS`], values in the same order), the
/// virtual-timer offset, and the pending interrupt lines. Transient HVF-internal software state
/// (a pending WFx PC advance, a half-serviced MMIO read) is folded into the real register file
/// by [`HvfVcpu::save_state`] before capture, so nothing here is renderer-/emulator-internal.
#[derive(Clone)]
pub struct VcpuState {
    /// X0..X30.
    pub x: [u64; 31],
    pub pc: u64,
    pub cpsr: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    /// V0..V31 (128-bit SIMD/FP registers).
    pub q: [u128; 32],
    /// Values for [`SNAPSHOT_SYS_REGS`], in that order.
    pub sysregs: Vec<u64>,
    /// Values for the per-vCPU GIC CPU-interface registers [`SNAPSHOT_ICC_REGS`], in that order.
    /// Not covered by the VM-wide GIC blob, but needed or interrupts won't deliver after restore.
    pub icc: Vec<u64>,
    pub vtimer_offset: u64,
    /// Whether HVF's virtual-timer IRQ was masked at snapshot (`hv_vcpu_get_vtimer_mask`). libkrun
    /// masks it after a `VTIMER_ACTIVATED` exit until the guest EOIs, so a fresh vCPU's default
    /// (unmasked) can desync the mask dance and wedge the guest waiting on an IRQ that never
    /// re-fires — the mask must be restored alongside the offset (the round-trip spike does this).
    pub vtimer_masked: bool,
    pub pending_irq: bool,
    pub pending_fiq: bool,
    /// The PSCI power state this vCPU was in. See [`VcpuPower`].
    pub power: VcpuPower,
}

/// The per-vCPU GIC CPU-interface (`ICC_*_EL1`) registers captured in a [`VcpuState`], in
/// save/restore order. These are the settable interface registers (priority mask, binary points,
/// active-priority, control, SRE, group enables); `ICC_RPR_EL1` is read-only (running priority)
/// and `ICC_SRE_EL2` is EL2, so both are excluded. Restoring these is what keeps interrupts
/// deliverable after resume — the VM-wide `hv_gic_state` blob does not carry them.
#[rustfmt::skip]
static SNAPSHOT_ICC_REGS: &[hv_gic_icc_reg_t] = &[
    hv_gic_icc_reg_t_HV_GIC_ICC_REG_PMR_EL1, hv_gic_icc_reg_t_HV_GIC_ICC_REG_BPR0_EL1,
    hv_gic_icc_reg_t_HV_GIC_ICC_REG_AP0R0_EL1, hv_gic_icc_reg_t_HV_GIC_ICC_REG_AP1R0_EL1,
    hv_gic_icc_reg_t_HV_GIC_ICC_REG_BPR1_EL1, hv_gic_icc_reg_t_HV_GIC_ICC_REG_CTLR_EL1,
    hv_gic_icc_reg_t_HV_GIC_ICC_REG_SRE_EL1, hv_gic_icc_reg_t_HV_GIC_ICC_REG_IGRPEN0_EL1,
    hv_gic_icc_reg_t_HV_GIC_ICC_REG_IGRPEN1_EL1,
];

/// The EL1/EL0 system registers captured in a [`VcpuState`], in save/restore order. Excludes the
/// EL2 (nested-only) registers and `CNTHP_*` (they read `HV_UNSUPPORTED` without nested virt),
/// the derived `CNTP_TVAL_EL0` (a view of `CNTP_CVAL_EL0` — restoring both would double-apply),
/// and `MPIDR_EL1`/`MIDR_EL1` (set at vCPU creation). Every entry is proven to round-trip on a
/// fresh HVF vCPU by `spikes/m9-hvf-state-roundtrip` (get + set, no rejections).
#[rustfmt::skip]
static SNAPSHOT_SYS_REGS: &[hv_sys_reg_t] = &[
    hv_sys_reg_t_HV_SYS_REG_ACTLR_EL1, hv_sys_reg_t_HV_SYS_REG_AFSR0_EL1,
    hv_sys_reg_t_HV_SYS_REG_AFSR1_EL1, hv_sys_reg_t_HV_SYS_REG_AMAIR_EL1,
    hv_sys_reg_t_HV_SYS_REG_APDAKEYHI_EL1, hv_sys_reg_t_HV_SYS_REG_APDAKEYLO_EL1,
    hv_sys_reg_t_HV_SYS_REG_APDBKEYHI_EL1, hv_sys_reg_t_HV_SYS_REG_APDBKEYLO_EL1,
    hv_sys_reg_t_HV_SYS_REG_APGAKEYHI_EL1, hv_sys_reg_t_HV_SYS_REG_APGAKEYLO_EL1,
    hv_sys_reg_t_HV_SYS_REG_APIAKEYHI_EL1, hv_sys_reg_t_HV_SYS_REG_APIAKEYLO_EL1,
    hv_sys_reg_t_HV_SYS_REG_APIBKEYHI_EL1, hv_sys_reg_t_HV_SYS_REG_APIBKEYLO_EL1,
    hv_sys_reg_t_HV_SYS_REG_CNTKCTL_EL1, hv_sys_reg_t_HV_SYS_REG_CNTP_CTL_EL0,
    hv_sys_reg_t_HV_SYS_REG_CNTP_CVAL_EL0, hv_sys_reg_t_HV_SYS_REG_CNTV_CTL_EL0,
    hv_sys_reg_t_HV_SYS_REG_CNTV_CVAL_EL0, hv_sys_reg_t_HV_SYS_REG_CONTEXTIDR_EL1,
    hv_sys_reg_t_HV_SYS_REG_CPACR_EL1, hv_sys_reg_t_HV_SYS_REG_CSSELR_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR0_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR10_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR11_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR12_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR13_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR14_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR15_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR2_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR3_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR4_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR5_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR6_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR7_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBCR8_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBCR9_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR0_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR10_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR11_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR12_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR13_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR14_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR15_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR2_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR3_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR4_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR5_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR6_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR7_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGBVR8_EL1, hv_sys_reg_t_HV_SYS_REG_DBGBVR9_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR0_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR10_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR11_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR12_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR13_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR14_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR15_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR2_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR3_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR4_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR5_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR6_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR7_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWCR8_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWCR9_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR0_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR10_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR11_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR12_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR13_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR14_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR15_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR2_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR3_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR4_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR5_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR6_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR7_EL1,
    hv_sys_reg_t_HV_SYS_REG_DBGWVR8_EL1, hv_sys_reg_t_HV_SYS_REG_DBGWVR9_EL1,
    hv_sys_reg_t_HV_SYS_REG_ELR_EL1, hv_sys_reg_t_HV_SYS_REG_ESR_EL1,
    hv_sys_reg_t_HV_SYS_REG_FAR_EL1, hv_sys_reg_t_HV_SYS_REG_ID_AA64DFR0_EL1,
    hv_sys_reg_t_HV_SYS_REG_ID_AA64DFR1_EL1, hv_sys_reg_t_HV_SYS_REG_ID_AA64ISAR0_EL1,
    hv_sys_reg_t_HV_SYS_REG_ID_AA64ISAR1_EL1, hv_sys_reg_t_HV_SYS_REG_ID_AA64MMFR0_EL1,
    hv_sys_reg_t_HV_SYS_REG_ID_AA64MMFR1_EL1, hv_sys_reg_t_HV_SYS_REG_ID_AA64MMFR2_EL1,
    hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR0_EL1, hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_MAIR_EL1, hv_sys_reg_t_HV_SYS_REG_MDCCINT_EL1,
    hv_sys_reg_t_HV_SYS_REG_MDSCR_EL1, hv_sys_reg_t_HV_SYS_REG_PAR_EL1,
    hv_sys_reg_t_HV_SYS_REG_SCTLR_EL1, hv_sys_reg_t_HV_SYS_REG_SPSR_EL1,
    hv_sys_reg_t_HV_SYS_REG_SP_EL0, hv_sys_reg_t_HV_SYS_REG_SP_EL1,
    hv_sys_reg_t_HV_SYS_REG_TCR_EL1, hv_sys_reg_t_HV_SYS_REG_TPIDRRO_EL0,
    hv_sys_reg_t_HV_SYS_REG_TPIDR_EL0, hv_sys_reg_t_HV_SYS_REG_TPIDR_EL1,
    hv_sys_reg_t_HV_SYS_REG_TTBR0_EL1, hv_sys_reg_t_HV_SYS_REG_TTBR1_EL1,
    hv_sys_reg_t_HV_SYS_REG_VBAR_EL1,
];

pub struct HvfVcpu<'a> {
    vcpuid: hv_vcpu_t,
    vcpu_exit: &'a hv_vcpu_exit_t,
    cntfrq: u64,
    mmio_buf: [u8; 8],
    pending_mmio_read: Option<MmioRead>,
    pending_advance_pc: bool,
    vtimer_masked: bool,
    nested_enabled: bool,
    /// This vCPU's PSCI power state, maintained by the park sites so a snapshot can record it.
    power: VcpuPower,
    // Cached copy of the HVF vtimer offset (0 until a live pause advances it),
    // so the WFE deadline check doesn't issue a syscall on every wait.
    vtimer_offset: Cell<u64>,
    /// Balloon-released guest RAM (shared with the balloon's release path): stage-2 faults on
    /// ranges recorded here are healed in place instead of being treated as MMIO. `None` outside
    /// limina's macOS worker.
    released_ram: Option<Arc<ReleasedRam>>,
}

impl HvfVcpu<'_> {
    pub fn new(mpidr: u64, nested_enabled: bool) -> Result<Self, Error> {
        let mut vcpuid: hv_vcpu_t = 0;
        let mut vcpu_exit_ptr: *mut hv_vcpu_exit_t = std::ptr::null_mut();

        #[cfg(target_arch = "aarch64")]
        let cntfrq = {
            let cntfrq: u64;
            unsafe { asm!("mrs {}, cntfrq_el0", out(reg) cntfrq) };
            cntfrq
        };
        #[cfg(target_arch = "x86_64")]
        let cntfrq = 0u64;
        #[cfg(target_arch = "riscv64")]
        let cntfrq = 0u64;

        let ret = unsafe {
            hv_vcpu_create(
                &mut vcpuid,
                &mut vcpu_exit_ptr as *mut *mut _,
                std::ptr::null_mut(),
            )
        };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuCreate);
        }

        // We write vcpuid to Aff1 as otherwise it won't match the redistributor ID
        // when using HVF in-kernel GICv3.
        let ret = unsafe { hv_vcpu_set_sys_reg(vcpuid, hv_sys_reg_t_HV_SYS_REG_MPIDR_EL1, mpidr) };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuCreate);
        }

        let vcpu_exit: &hv_vcpu_exit_t = unsafe { vcpu_exit_ptr.as_mut().unwrap() };

        Ok(Self {
            vcpuid,
            vcpu_exit,
            cntfrq,
            mmio_buf: [0; 8],
            pending_mmio_read: None,
            pending_advance_pc: false,
            vtimer_masked: false,
            nested_enabled,
            power: VcpuPower::Running,
            vtimer_offset: Cell::new(0),
            released_ram: None,
        })
    }

    /// Wire up the balloon-released RAM tracker so this vCPU heals stage-2 faults on released
    /// ranges (see [`ReleasedRam`]). Call before the first `run`.
    pub fn set_released_ram(&mut self, released_ram: Arc<ReleasedRam>) {
        self.released_ram = Some(released_ram);
    }

    pub fn set_initial_state(&self, entry_addr: u64, fdt_addr: u64) -> Result<(), Error> {
        if self.nested_enabled {
            let ret = unsafe {
                hv_vcpu_set_reg(self.vcpuid, hv_reg_t_HV_REG_CPSR, PSTATE_EL2_FAULT_BITS_64)
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }

            let ret = unsafe {
                hv_vcpu_set_sys_reg(self.vcpuid, hv_sys_reg_t_HV_SYS_REG_HCR_EL2, HCR_EL2_BITS)
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }

            let ret = unsafe {
                hv_vcpu_set_sys_reg(
                    self.vcpuid,
                    hv_sys_reg_t_HV_SYS_REG_CNTHCTL_EL2,
                    CNTHCTL_EL2_BITS,
                )
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }

            // Enable EL2 and GICv3 in ID_AA64PFR0_EL1
            let mut val: u64 = 0;
            let ret = unsafe {
                hv_vcpu_get_sys_reg(
                    self.vcpuid,
                    hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR0_EL1,
                    &mut val as *mut _,
                )
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }
            let ret = unsafe {
                hv_vcpu_set_sys_reg(
                    self.vcpuid,
                    hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR0_EL1,
                    val | AA64PFR0_EL1_EL2EN | AA64PFR0_EL1_GIC3EN,
                )
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }
        } else {
            let ret = unsafe {
                hv_vcpu_set_reg(self.vcpuid, hv_reg_t_HV_REG_CPSR, PSTATE_EL1_FAULT_BITS_64)
            };
            if ret != HV_SUCCESS {
                return Err(Error::VcpuInitialRegisters);
            }
        }

        // Masked on every vCPU, not just the nested ones: a guest that sees SME
        // will use it, and what it gets is the host's SME without the SVE that
        // guest userspace assumes comes with it. On an M4 that combination makes
        // Chrome's SME path issue a non-streaming SVE instruction and take a
        // SIGILL. Under nested virt the same advertisement breaks the guest
        // outright once it enables the MMU, which is why this was already here —
        // it was just in the wrong branch.
        let mut val: u64 = 0;
        let ret = unsafe {
            hv_vcpu_get_sys_reg(
                self.vcpuid,
                hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR1_EL1,
                &mut val as *mut _,
            )
        };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuInitialRegisters);
        }
        let ret = unsafe {
            hv_vcpu_set_sys_reg(
                self.vcpuid,
                hv_sys_reg_t_HV_SYS_REG_ID_AA64PFR1_EL1,
                val & !AA64PFR1_EL1_SMEMASK,
            )
        };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuInitialRegisters);
        }

        let ret = unsafe { hv_vcpu_set_reg(self.vcpuid, hv_reg_t_HV_REG_PC, entry_addr) };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuInitialRegisters);
        }

        let ret = unsafe { hv_vcpu_set_reg(self.vcpuid, hv_reg_t_HV_REG_X0, fdt_addr) };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuInitialRegisters);
        }

        Ok(())
    }

    pub fn id(&self) -> u64 {
        self.vcpuid
    }

    fn read_reg(&self, reg: u32) -> Result<u64, Error> {
        let mut val: u64 = 0;
        let ret = unsafe { hv_vcpu_get_reg(self.vcpuid, reg, &mut val as *mut _) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuReadRegister)
        } else {
            Ok(val)
        }
    }

    pub fn write_reg(&self, rt: u32, val: u64) -> Result<(), Error> {
        let ret = unsafe { hv_vcpu_set_reg(self.vcpuid, rt, val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSetRegister)
        } else {
            Ok(())
        }
    }

    fn read_sys_reg(&self, reg: u16) -> Result<u64, Error> {
        let mut val: u64 = 0;
        let ret = unsafe { hv_vcpu_get_sys_reg(self.vcpuid, reg, &mut val as *mut _) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuReadSystemRegister)
        } else {
            Ok(val)
        }
    }

    /// Freeze the guest's CNTVCT across a live pause. The guest virtual counter
    /// is `mach_absolute_time() - vtimer_offset`; while paused the host counter
    /// keeps advancing, so on resume we add the ticks spent paused to the offset.
    /// The guest's counter then resumes where it stopped and armed deadlines stay
    /// in the near future instead of firing en masse to catch up the gap.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    pub fn advance_vtimer_offset(&self, paused_ticks: u64) -> Result<(), Error> {
        let offset = self.vtimer_offset.get().wrapping_add(paused_ticks);
        let ret = unsafe { hv_vcpu_set_vtimer_offset(self.vcpuid, offset) };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuSetRegister);
        }
        self.vtimer_offset.set(offset);
        Ok(())
    }

    /// Set an EL1/EL0 system register — the counterpart to the private [`Self::read_sys_reg`],
    /// used by the M9 snapshot restore path (there was no generic setter before).
    pub fn write_sys_reg(&self, reg: u16, val: u64) -> Result<(), Error> {
        let ret = unsafe { hv_vcpu_set_sys_reg(self.vcpuid, reg, val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSetSystemRegister(reg, val))
        } else {
            Ok(())
        }
    }

    fn read_simd(&self, reg: hv_simd_fp_reg_t) -> Result<u128, Error> {
        let mut val: hv_simd_fp_uchar16_t = 0;
        let ret = unsafe { hv_vcpu_get_simd_fp_reg(self.vcpuid, reg, &mut val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(val)
        }
    }

    fn write_simd(&self, reg: hv_simd_fp_reg_t, val: u128) -> Result<(), Error> {
        let ret = unsafe { hv_vcpu_set_simd_fp_reg(self.vcpuid, reg, val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(())
        }
    }

    fn read_vtimer_offset(&self) -> Result<u64, Error> {
        let mut val: u64 = 0;
        let ret = unsafe { hv_vcpu_get_vtimer_offset(self.vcpuid, &mut val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(val)
        }
    }

    fn write_vtimer_offset(&self, val: u64) -> Result<(), Error> {
        let ret = unsafe { hv_vcpu_set_vtimer_offset(self.vcpuid, val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(())
        }
    }

    fn read_vtimer_mask(&self) -> Result<bool, Error> {
        let mut masked = false;
        let ret = unsafe { hv_vcpu_get_vtimer_mask(self.vcpuid, &mut masked) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(masked)
        }
    }

    fn read_icc_reg(&self, reg: hv_gic_icc_reg_t) -> Result<u64, Error> {
        let mut val: u64 = 0;
        let ret = unsafe { hv_gic_get_icc_reg(self.vcpuid, reg, &mut val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(val)
        }
    }

    fn write_icc_reg(&self, reg: hv_gic_icc_reg_t, val: u64) -> Result<(), Error> {
        let ret = unsafe { hv_gic_set_icc_reg(self.vcpuid, reg, val) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(())
        }
    }

    fn read_pending_interrupt(&self, ty: hv_interrupt_type_t) -> Result<bool, Error> {
        let mut pending = false;
        let ret = unsafe { hv_vcpu_get_pending_interrupt(self.vcpuid, ty, &mut pending) };
        if ret != HV_SUCCESS {
            Err(Error::VcpuSnapshot)
        } else {
            Ok(pending)
        }
    }

    /// Fold this vCPU's transient software state into the real HVF register file before a
    /// snapshot: apply a half-serviced MMIO read to its destination register, then the pending
    /// WFx PC advance. Mirrors the top of [`Self::run`] (which normally applies these on the
    /// next entry). After this the register file is fully consistent, so a [`VcpuState`]
    /// captures everything and a restore needs to carry no HVF-internal software flags.
    fn flush_pending_state(&mut self) -> Result<(), Error> {
        if let Some(mmio_read) = self.pending_mmio_read.take() {
            if mmio_read.srt < 31 {
                let val = match mmio_read.len {
                    1 => u8::from_le_bytes(self.mmio_buf[0..1].try_into().unwrap()) as u64,
                    2 => u16::from_le_bytes(self.mmio_buf[0..2].try_into().unwrap()) as u64,
                    4 => u32::from_le_bytes(self.mmio_buf[0..4].try_into().unwrap()) as u64,
                    8 => u64::from_le_bytes(self.mmio_buf[0..8].try_into().unwrap()),
                    _ => return Err(Error::Unhandled),
                };
                self.write_reg(mmio_read.srt, val)?;
            }
        }
        if self.pending_advance_pc {
            let pc = self.read_reg(hv_reg_t_HV_REG_PC)?;
            self.write_reg(hv_reg_t_HV_REG_PC, pc + 4)?;
            self.pending_advance_pc = false;
        }
        Ok(())
    }

    /// Capture this vCPU's full architectural state for an M9 snapshot. **Must run on the vCPU's
    /// own thread** — HVF binds `hv_vcpu_*` register access to the creating thread. Transient
    /// software state is folded into the register file first ([`Self::flush_pending_state`]).
    pub fn save_state(&mut self) -> Result<VcpuState, Error> {
        self.flush_pending_state()?;

        let mut x = [0u64; 31];
        for (i, slot) in x.iter_mut().enumerate() {
            *slot = self.read_reg(hv_reg_t_HV_REG_X0 + i as u32)?;
        }
        let mut q = [0u128; 32];
        for (i, slot) in q.iter_mut().enumerate() {
            *slot = self.read_simd(hv_simd_fp_reg_t_HV_SIMD_FP_REG_Q0 + i as u32)?;
        }
        let mut sysregs = Vec::with_capacity(SNAPSHOT_SYS_REGS.len());
        for &reg in SNAPSHOT_SYS_REGS {
            sysregs.push(self.read_sys_reg(reg)?);
        }
        let mut icc = Vec::with_capacity(SNAPSHOT_ICC_REGS.len());
        for &reg in SNAPSHOT_ICC_REGS {
            icc.push(self.read_icc_reg(reg)?);
        }

        Ok(VcpuState {
            x,
            pc: self.read_reg(hv_reg_t_HV_REG_PC)?,
            cpsr: self.read_reg(hv_reg_t_HV_REG_CPSR)?,
            fpcr: self.read_reg(hv_reg_t_HV_REG_FPCR)?,
            fpsr: self.read_reg(hv_reg_t_HV_REG_FPSR)?,
            q,
            sysregs,
            icc,
            vtimer_offset: self.read_vtimer_offset()?,
            vtimer_masked: self.read_vtimer_mask()?,
            pending_irq: self.read_pending_interrupt(hv_interrupt_type_t_HV_INTERRUPT_TYPE_IRQ)?,
            pending_fiq: self.read_pending_interrupt(hv_interrupt_type_t_HV_INTERRUPT_TYPE_FIQ)?,
            // The register file alone cannot say this: a parked vCPU looks exactly like one that
            // is about to return from a PSCI call. The park sites set it (see `set_power`).
            power: self.power,
        })
    }

    /// Restore a vCPU from a [`VcpuState`] — the resume-path counterpart to
    /// [`Self::set_initial_state`]. Runs on the vCPU's own thread, before its run loop. MPIDR is
    /// already set at creation and the VM-wide GIC blob is restored by the caller beforehand;
    /// order otherwise follows the round-trip spike (sysregs, then GP/PC/PSTATE/SIMD, then the
    /// vtimer offset and pending lines).
    pub fn restore_state(&mut self, state: &VcpuState) -> Result<(), Error> {
        if state.sysregs.len() != SNAPSHOT_SYS_REGS.len()
            || state.icc.len() != SNAPSHOT_ICC_REGS.len()
        {
            return Err(Error::VcpuSnapshot);
        }
        // Order follows the round-trip spike: GIC CPU-interface (ICC) regs first (the VM-wide GIC
        // blob is already restored by the caller), then the EL1/EL0 sysregs, then GP/PC/PSTATE/SIMD,
        // then the vtimer offset + mask and the pending lines. ICC before sysregs matters: a stale
        // sysreg view must not sit atop a half-restored CPU interface.
        for (&reg, &val) in SNAPSHOT_ICC_REGS.iter().zip(state.icc.iter()) {
            self.write_icc_reg(reg, val)?;
        }
        for (&reg, &val) in SNAPSHOT_SYS_REGS.iter().zip(state.sysregs.iter()) {
            self.write_sys_reg(reg, val)?;
        }
        for (i, &val) in state.x.iter().enumerate() {
            self.write_reg(hv_reg_t_HV_REG_X0 + i as u32, val)?;
        }
        self.write_reg(hv_reg_t_HV_REG_PC, state.pc)?;
        self.write_reg(hv_reg_t_HV_REG_CPSR, state.cpsr)?;
        self.write_reg(hv_reg_t_HV_REG_FPCR, state.fpcr)?;
        self.write_reg(hv_reg_t_HV_REG_FPSR, state.fpsr)?;
        for (i, &val) in state.q.iter().enumerate() {
            self.write_simd(hv_simd_fp_reg_t_HV_SIMD_FP_REG_Q0 + i as u32, val)?;
        }
        self.write_vtimer_offset(state.vtimer_offset)?;
        // Restore HVF's vtimer mask (and mirror it into our worker-side tracking) — without it a
        // guest snapshotted mid-timer waits forever on an IRQ that never re-fires.
        vcpu_set_vtimer_mask(self.vcpuid, state.vtimer_masked)?;
        self.vtimer_masked = state.vtimer_masked;
        vcpu_set_pending_irq(self.vcpuid, InterruptType::Irq, state.pending_irq)?;
        vcpu_set_pending_irq(self.vcpuid, InterruptType::Fiq, state.pending_fiq)?;
        Ok(())
    }

    fn hvf_sync_vtimer(&mut self, vcpu_list: Arc<dyn Vcpus>) {
        if !self.vtimer_masked {
            return;
        }

        let ctl = self
            .read_sys_reg(hv_sys_reg_t_HV_SYS_REG_CNTV_CTL_EL0)
            .unwrap();
        let irq_state = (ctl & (TMR_CTL_ENABLE | TMR_CTL_IMASK | TMR_CTL_ISTATUS))
            == (TMR_CTL_ENABLE | TMR_CTL_ISTATUS);
        vcpu_list.set_vtimer_irq(self.vcpuid);
        if !irq_state {
            vcpu_set_vtimer_mask(self.vcpuid, false).unwrap();
            self.vtimer_masked = false;
        }
    }

    /// Re-arm a parked (offlined) vCPU for a fresh secondary entry after PSCI CPU_ON. Clears the
    /// transient per-exit software state left over from before the offline, then resets to a clean
    /// secondary-boot register state at `entry_addr` — the guest re-enters at its secondary-startup
    /// path, which re-initializes stack/MMU. Must run on the vCPU's OWN thread (HVF register access
    /// is thread-bound), which is why the parked vCPU thread calls this itself on re-online.
    pub fn reonline(&mut self, entry_addr: u64, fdt_addr: u64) -> Result<(), Error> {
        self.pending_advance_pc = false;
        self.pending_mmio_read = None;
        // The vtimer state is per-CPU; a re-onlined CPU reprograms it from scratch. Drop any stale
        // masked-flag from before the offline so `hvf_sync_vtimer` reconciles cleanly.
        self.vtimer_masked = false;
        // A running vCPU has the MMU + caches enabled. PSCI CPU_ON enters at a PHYSICAL address
        // (the kernel's secondary_entry) with the MMU OFF, exactly as a reset CPU would — so clear
        // the MMU (M), data-cache (C) and instruction-cache (I) enable bits before re-arming, or
        // the guest faults executing the physical entry PC. The kernel's secondary bringup
        // re-enables them. (At initial boot the fresh vCPU already has these clear, which is why
        // set_initial_state alone sufficed on that path.) A nested guest runs at EL2, so its
        // translation control is SCTLR_EL2; a normal guest's is SCTLR_EL1.
        const SCTLR_M: u64 = 1 << 0;
        const SCTLR_C: u64 = 1 << 2;
        const SCTLR_I: u64 = 1 << 12;
        let sctlr_reg = if self.nested_enabled {
            hv_sys_reg_t_HV_SYS_REG_SCTLR_EL2
        } else {
            hv_sys_reg_t_HV_SYS_REG_SCTLR_EL1
        };
        let sctlr = self.read_sys_reg(sctlr_reg)?;
        self.write_sys_reg(sctlr_reg, sctlr & !(SCTLR_M | SCTLR_C | SCTLR_I))?;
        self.set_initial_state(entry_addr, fdt_addr)
    }

    /// Record this vCPU's PSCI power state, so a snapshot taken while it is parked restores it
    /// into the same park rather than into a return from a call that never returns.
    pub fn set_power(&mut self, power: VcpuPower) {
        self.power = power;
    }

    /// Re-arm this vCPU after a PSCI `SYSTEM_SUSPEND`, at the resume entry point the guest gave us.
    ///
    /// Same reset shape as [`Self::reonline`] (physical entry, MMU/caches off — what arm64's
    /// `cpu_resume` expects), then X0 = `context_id` as the PSCI spec requires. Linux's
    /// `cpu_resume` zeroes X0 before using it, so the value is ceremony for Linux, but a guest
    /// that does honour it gets the right one.
    pub fn resume_from_system_suspend(
        &mut self,
        entry_addr: u64,
        fdt_addr: u64,
        context_id: u64,
    ) -> Result<(), Error> {
        self.reonline(entry_addr, fdt_addr)?;
        self.write_reg(hv_reg_t_HV_REG_X0, context_id)
    }

    fn handle_psci_request(&self, vcpu_list: &Arc<dyn Vcpus>) -> Result<VcpuExit<'_>, Error> {
        match self.read_reg(hv_reg_t_HV_REG_X0)? {
            0x8400_0000 /* PSCI_0_2_FN_PSCI_VERSION */ => {
                // 0.2 when SYSTEM_SUSPEND is not offered: Linux only probes the 1.0 function set
                // when the major version is >= 1, so this alone takes the guest back to the
                // s2idle-only world.
                let version = if system_suspend_offered() {
                    PSCI_VERSION_1_0
                } else {
                    PSCI_VERSION_0_2
                };
                self.write_reg(hv_reg_t_HV_REG_X0, version)?;
                Ok(VcpuExit::PsciHandled)
            },
            0x8400_000a /* PSCI_1_0_FN_PSCI_FEATURES */ => {
                // Feature discovery (1.0+). SUCCESS (0) = implemented with no optional flags;
                // NOT_SUPPORTED for everything else. Only the functions we actually dispatch may
                // be advertised: Linux takes a non-NOT_SUPPORTED answer as a promise, and
                // `psci_init_system_suspend()` registers its suspend ops on the strength of it.
                let queried = self.read_reg(hv_reg_t_HV_REG_X1)?;
                let supported = matches!(
                    queried,
                    0x8400_0000 /* PSCI_VERSION */
                    | 0x8400_0002 /* CPU_OFF */
                    | 0x8400_0004 | 0xc400_0004 /* AFFINITY_INFO */
                    | 0x8400_0006 /* MIGRATE_INFO_TYPE */
                    | 0x8400_0008 /* SYSTEM_OFF */
                    | 0x8400_0009 /* SYSTEM_RESET */
                    | 0x8400_000a /* PSCI_FEATURES */
                    | 0xc400_0003 /* CPU_ON */
                    | 0xc400_000e /* SYSTEM_SUSPEND */
                ) && (queried != 0xc400_000e || system_suspend_offered());
                let ret = if supported { 0 } else { PSCI_RET_NOT_SUPPORTED };
                self.write_reg(hv_reg_t_HV_REG_X0, ret)?;
                Ok(VcpuExit::PsciHandled)
            },
            0xc400_000e /* PSCI_1_0_FN64_SYSTEM_SUSPEND */ => {
                // Suspend-to-RAM for the whole VM. The spec requires the caller to be the only
                // core still on; Linux guarantees it (`pm_sleep_disable_secondary_cpus()` runs
                // before `syscore_suspend()`), but a guest that gets it wrong must be refused
                // rather than silently parked — a DENIED return unwinds cleanly in the guest's
                // `cpu_suspend()`, which treats any non-zero result as a failed suspend.
                if !system_suspend_offered() {
                    self.write_reg(hv_reg_t_HV_REG_X0, PSCI_RET_NOT_SUPPORTED)?;
                    return Ok(VcpuExit::PsciHandled);
                }
                if !vcpu_list.others_all_offline(self.vcpuid) {
                    warn!(
                        "vCPU {} asked for PSCI SYSTEM_SUSPEND while another vCPU is still online; \
                         refusing with DENIED",
                        self.vcpuid
                    );
                    self.write_reg(hv_reg_t_HV_REG_X0, PSCI_RET_DENIED)?;
                    return Ok(VcpuExit::PsciHandled);
                }
                let entry = self.read_reg(hv_reg_t_HV_REG_X1)?;
                let context_id = self.read_reg(hv_reg_t_HV_REG_X2)?;
                Ok(VcpuExit::SystemSuspend { entry, context_id })
            },
            0x8400_0002 /* QEMU_PSCI_0_2_FN_CPU_OFF */ => {
                // The guest is offlining THIS vCPU (CPU hotplug down). Record it OFF so the
                // reaper's AFFINITY_INFO poll (below) completes, then hand the vmm a CpuOff exit
                // to park this thread until re-online. CPU_OFF never returns on success — write
                // no result register.
                vcpu_list.set_online(self.vcpuid, false);
                Ok(VcpuExit::CpuOff)
            },
            0x8400_0004 /* QEMU_PSCI_0_2_FN_AFFINITY_INFO */
            | 0xc400_0004 /* QEMU_PSCI_0_2_FN64_AFFINITY_INFO */ => {
                // The offline reaper polls this for the dying CPU to report OFF. `target_affinity`
                // (X1) is the queried MPIDR = the vcpuid in our flat topology. 0 = ON, 1 = OFF.
                let target = self.read_reg(hv_reg_t_HV_REG_X1)?;
                let status = if vcpu_list.is_online(target) { 0 } else { 1 };
                self.write_reg(hv_reg_t_HV_REG_X0, status)?;
                Ok(VcpuExit::PsciHandled)
            },
            0x8400_0006 /* QEMU_PSCI_0_2_FN_MIGRATE_INFO_TYPE */ => {
                self.write_reg(hv_reg_t_HV_REG_X0, 2)?;
                Ok(VcpuExit::PsciHandled)
            },
            0x8400_0008 /* QEMU_PSCI_0_2_FN_SYSTEM_OFF */ => {
                Ok(VcpuExit::Shutdown)
            },
            0x8400_0009 /* QEMU_PSCI_0_2_FN_SYSTEM_RESET */ => {
                Ok(VcpuExit::Reset)
            },
            0xc400_0003 /* QEMU_PSCI_0_2_FN64_CPU_ON */ => {
                let mpidr = self.read_reg(hv_reg_t_HV_REG_X1)?;
                let entry = self.read_reg(hv_reg_t_HV_REG_X2)?;
                let context_id = self.read_reg(hv_reg_t_HV_REG_X3)?;
                // Mark the target ON (covers both initial secondary boot and re-online after a
                // CPU_OFF); the vmm delivers `entry` to the target vCPU's boot channel.
                vcpu_list.set_online(mpidr, true);
                self.write_reg(hv_reg_t_HV_REG_X0, 0)?;
                Ok(VcpuExit::CpuOn(mpidr, entry, context_id))
            }
            val => {
                // An unmodeled PSCI/SMC function. Standard PSCI behaviour is to return
                // NOT_SUPPORTED rather than fault — a stock guest probing an optional function
                // (SYSTEM_RESET2, STAT_COUNT, …) then degrades gracefully instead of taking
                // the whole VMM down.
                //
                // Reporting PSCI 1.0 makes Linux probe a fixed set of optional calls on every
                // boot (SMCCC_VERSION and ARCH_* in the 0x8000_xxxx range, the KVM vendor
                // hypervisor UID at 0x8600_xxxx, CPU_SUSPEND, and the debugfs feature table).
                // Those are routine discovery, not anomalies, so they log at debug; anything
                // outside the known probe ranges still warns.
                let routine_probe = matches!(val & 0xbfff_ffff,
                    0x8000_0000..=0x8000_ffff /* SMCCC_VERSION / ARCH_FEATURES / ARCH_WORKAROUND */
                    | 0x8400_0001 /* CPU_SUSPEND */
                    | 0x8400_0005 /* MIGRATE */
                    | 0x8400_0007 /* MIGRATE_INFO_UP_CPU */
                    | 0x8400_000b..=0x8400_0015 /* CPU_FREEZE … SYSTEM_OFF2 */
                    | 0x8600_0000..=0x8600_ffff /* vendor hypervisor service range */
                );
                if routine_probe {
                    debug!("PSCI/SMC probe 0x{val:x}; returning NOT_SUPPORTED");
                } else {
                    warn!("unhandled PSCI/SMC function 0x{val:x}; returning NOT_SUPPORTED");
                }
                self.write_reg(hv_reg_t_HV_REG_X0, PSCI_RET_NOT_SUPPORTED)?;
                Ok(VcpuExit::PsciHandled)
            }
        }
    }

    pub fn run(&mut self, vcpu_list: Arc<dyn Vcpus>) -> Result<VcpuExit<'_>, Error> {
        let pending_irq = vcpu_list.has_pending_irq(self.vcpuid);

        if let Some(mmio_read) = self.pending_mmio_read.take()
            && mmio_read.srt < 31
        {
            let val = match mmio_read.len {
                1 => u8::from_le_bytes(self.mmio_buf[0..1].try_into().unwrap()) as u64,
                2 => u16::from_le_bytes(self.mmio_buf[0..2].try_into().unwrap()) as u64,
                4 => u32::from_le_bytes(self.mmio_buf[0..4].try_into().unwrap()) as u64,
                8 => u64::from_le_bytes(self.mmio_buf[0..8].try_into().unwrap()),
                _ => {
                    error!(
                        "unsupported MMIO read size: pa=0x{:x} len={}; stopping the VM",
                        mmio_read.addr, mmio_read.len
                    );
                    return Err(Error::Unhandled);
                }
            };

            self.write_reg(mmio_read.srt, val)?;
        }

        if self.pending_advance_pc {
            let pc = self.read_reg(hv_reg_t_HV_REG_PC)?;
            self.write_reg(hv_reg_t_HV_REG_PC, pc + 4)?;
            self.pending_advance_pc = false;
        }

        if pending_irq {
            vcpu_set_pending_irq(self.vcpuid, InterruptType::Irq, true)?;
        }

        let ret = unsafe { hv_vcpu_run(self.vcpuid) };
        if ret != HV_SUCCESS {
            return Err(Error::VcpuRun);
        }

        match self.vcpu_exit.reason {
            HV_EXIT_REASON_EXCEPTION => { /* This is the main one, handle below. */ }
            HV_EXIT_REASON_VTIMER_ACTIVATED => {
                self.vtimer_masked = true;
                return Ok(VcpuExit::VtimerActivated);
            }
            HV_EXIT_REASON_CANCELED => return Ok(VcpuExit::Canceled),
            _ => {
                let pc = self.read_reg(hv_reg_t_HV_REG_PC)?;
                error!(
                    "unexpected HVF exit reason: vcpuid={} 0x{:x} at pc=0x{:x}; stopping the VM",
                    self.id(),
                    self.vcpu_exit.reason,
                    pc
                );
                return Err(Error::Unhandled);
            }
        }

        self.hvf_sync_vtimer(vcpu_list.clone());

        let syndrome = self.vcpu_exit.exception.syndrome;
        let ec = (syndrome >> 26) & 0x3f;
        match ec {
            EC_AA64_BKPT => {
                debug!("vcpu[{}]: BRK exit", self.vcpuid);
                Ok(VcpuExit::Breakpoint)
            }
            EC_DATAABORT => {
                let isv: bool = (syndrome & (1 << 24)) != 0;
                let iswrite: bool = ((syndrome >> 6) & 1) != 0;
                let s1ptw: bool = ((syndrome >> 7) & 1) != 0;
                let sas: u32 = ((syndrome >> 22) & 3) as u32;
                let len: usize = (1 << sas) as usize;
                let srt: u32 = ((syndrome >> 16) & 0x1f) as u32;
                let cm: u32 = ((syndrome >> 8) & 0x1) as u32;

                debug!(
                    "EC_DATAABORT {} {} {} {} {} {} {} {}",
                    syndrome, isv as u8, iswrite as u8, s1ptw as u8, sas, len, srt, cm
                );

                let pa = self.vcpu_exit.exception.physical_address;

                // A translation fault on balloon-released guest RAM is healed and retried, never
                // decoded as MMIO. Must run before the PC-advance is armed and before any ISS
                // interpretation: RAM re-touches are often ISV=0 (stp/ldp, SIMD), whose srt/sas
                // fields are meaningless.
                if let Some(released_ram) = &self.released_ram {
                    match released_ram.handle_fault(pa, syndrome & 0x3f) {
                        FaultOutcome::Healed | FaultOutcome::Retry => {
                            return Ok(VcpuExit::StageTwoHealed);
                        }
                        FaultOutcome::Fatal => return Err(Error::StageTwoHeal),
                        FaultOutcome::NotHandled => {}
                    }
                }

                self.pending_advance_pc = true;

                if iswrite {
                    let val = if srt < 31 {
                        self.read_reg(hv_reg_t_HV_REG_X0 + srt)?
                    } else {
                        0
                    };

                    match len {
                        1 => self.mmio_buf[0..1].copy_from_slice(&(val as u8).to_le_bytes()),
                        2 => self.mmio_buf[0..2].copy_from_slice(&(val as u16).to_le_bytes()),
                        4 => self.mmio_buf[0..4].copy_from_slice(&(val as u32).to_le_bytes()),
                        8 => self.mmio_buf[0..8].copy_from_slice(&val.to_le_bytes()),
                        _ => {
                            error!("unsupported MMIO write size: len={len}; stopping the VM");
                            return Err(Error::Unhandled);
                        }
                    };

                    Ok(VcpuExit::MmioWrite(pa, &self.mmio_buf[0..len]))
                } else {
                    self.pending_mmio_read = Some(MmioRead { addr: pa, srt, len });
                    Ok(VcpuExit::MmioRead(pa, &mut self.mmio_buf[0..len]))
                }
            }
            EC_INSTRABORT => {
                // The guest can legitimately EXECUTE from a balloon-released page: free pages
                // reallocated as page cache hold binaries. Same heal as the data-abort path; any
                // other instruction abort is a guest state we don't model.
                let pa = self.vcpu_exit.exception.physical_address;
                if let Some(released_ram) = &self.released_ram {
                    match released_ram.handle_fault(pa, syndrome & 0x3f) {
                        FaultOutcome::Healed | FaultOutcome::Retry => {
                            return Ok(VcpuExit::StageTwoHealed);
                        }
                        FaultOutcome::Fatal => return Err(Error::StageTwoHeal),
                        FaultOutcome::NotHandled => {}
                    }
                }
                let pc = self.read_reg(hv_reg_t_HV_REG_PC).unwrap_or(0);
                error!(
                    "unhandled instruction abort: pa=0x{pa:x} syndrome=0x{syndrome:x} \
                     pc=0x{pc:x}; stopping the VM"
                );
                Err(Error::Unhandled)
            }
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            EC_SYSTEMREGISTERTRAP => {
                let isread: bool = (syndrome & 1) != 0;
                let rt: u32 = ((syndrome >> 5) & 0x1f) as u32;
                let reg: u32 = syndrome as u32 & SYSREG_MASK;
                debug!(
                    "EC_SYSTEMREGISTERTRAP isread={}, syndrome={}, rt={}, reg={}, reg_name={}",
                    isread as u32,
                    syndrome,
                    rt,
                    reg,
                    sys_reg_name(reg).unwrap_or("unknown sysreg")
                );

                self.pending_advance_pc = true;

                if isread {
                    assert!(rt < 32);

                    // See https://developer.arm.com/documentation/dui0801/l/Overview-of-AArch64-state/Registers-in-AArch64-state
                    if rt == 31 {
                        return Ok(VcpuExit::SystemRegister);
                    }

                    match vcpu_list.handle_sysreg_read(self.vcpuid, reg) {
                        Some(val) => {
                            self.write_reg(rt, val)?;
                            Ok(VcpuExit::SystemRegister)
                        }
                        None => {
                            error!(
                                "unhandled system-register read: rt={} reg={} name={}; stopping the VM",
                                rt,
                                reg,
                                sys_reg_name(reg).unwrap_or("unknown sysreg")
                            );
                            Err(Error::Unhandled)
                        }
                    }
                } else {
                    assert!(rt < 32);

                    // See https://developer.arm.com/documentation/dui0801/l/Overview-of-AArch64-state/Registers-in-AArch64-state
                    let val = if rt == 31 { 0u64 } else { self.read_reg(rt)? };

                    if vcpu_list.handle_sysreg_write(self.vcpuid, reg, val) {
                        Ok(VcpuExit::SystemRegister)
                    } else {
                        error!(
                            "unhandled system-register write: reg={} name={}; stopping the VM",
                            reg,
                            sys_reg_name(reg).unwrap_or("unknown sysreg")
                        );
                        Err(Error::Unhandled)
                    }
                }
            }
            EC_WFX_TRAP => {
                let ctl = self.read_sys_reg(hv_sys_reg_t_HV_SYS_REG_CNTV_CTL_EL0)?;

                self.pending_advance_pc = true;
                if ((ctl & 1) == 0) || (ctl & 2) != 0 {
                    return Ok(VcpuExit::WaitForEvent);
                }

                // Also CNTV_CVAL & CNTV_CVAL_EL0
                let cval = self.read_sys_reg(hv_sys_reg_t_HV_SYS_REG_CNTV_CVAL_EL0)?;
                // The guest's deadline `cval` is in the CNTVCT domain, where
                // CNTVCT = mach_absolute_time() - vtimer_offset. The offset is 0
                // on a fresh boot but nonzero after a live pause (it keeps CNTVCT
                // continuous across the paused interval). Compare against the
                // corrected counter, not raw mach time — otherwise after a pause
                // `now` runs ahead of every armed deadline and the vCPU busy-loops
                // on WaitForEventExpired instead of parking, so the guest's virtual
                // timer never fires and timed sleeps hang.
                let now = unsafe { mach_absolute_time() }.saturating_sub(self.vtimer_offset.get());
                if now > cval {
                    return Ok(VcpuExit::WaitForEventExpired);
                }

                // Multiply before dividing (in u128 to avoid overflow on far-future deadlines):
                // dividing 1e9/cntfrq first truncates (24 MHz -> 41 ns/tick instead of 41.67), so
                // every WFI-timeout ran ~1.6% short, the vCPU woke early, the guest re-WFI'd, and a
                // single guest timer deadline cost two host wakeups.
                let timeout_ns = ((cval - now) as u128 * 1_000_000_000) / self.cntfrq as u128;
                let timeout = Duration::from_nanos(timeout_ns.min(u64::MAX as u128) as u64);
                Ok(VcpuExit::WaitForEventTimeout(timeout))
            }
            EC_AA64_HVC => self.handle_psci_request(&vcpu_list),
            EC_AA64_SMC => {
                self.pending_advance_pc = true;
                self.handle_psci_request(&vcpu_list)
            }
            _ => {
                let pc = self.read_reg(hv_reg_t_HV_REG_PC).unwrap_or(0);
                error!(
                    "unhandled exception class EC=0x{ec:x} syndrome=0x{syndrome:x} at pc=0x{pc:x}; stopping the VM"
                );
                Err(Error::Unhandled)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IpaGranule, map_granule, round_up};

    // The blob-map alignment guard: hv_vm_map/hv_vm_unmap take sizes at granule
    // granularity only, so the wrappers must round sub-granule sizes up (and by exactly
    // the same amount on both map and unmap). See round_up_to_map_granule.
    #[test]
    fn sizes_round_up_to_the_map_granule() {
        let page = map_granule();
        assert!(page.is_power_of_two() && page >= 4096);

        assert_eq!(round_up(0, page), 0);
        assert_eq!(round_up(1, page), page);
        assert_eq!(round_up(page, page), page);
        assert_eq!(round_up(page + 1, page), 2 * page);

        // The signature from the wild: a 4 KiB guest's 0x21000-byte blob.
        let odd = 0x21000u64;
        let rounded = round_up(odd, page);
        assert!(rounded >= odd);
        assert_eq!(rounded % page, 0);
        assert!(rounded - odd < page);
    }

    // Under a 4 KiB granule a neighbouring blob at the next 4 KiB guest address is legal, so
    // rounding a 4 KiB-granular size up to the 16 KiB HOST page would map over it and HVF
    // would refuse the neighbour. The rounding must follow the granule the VM was created
    // with, which is the whole reason it is not a function of getpagesize().
    #[test]
    fn a_four_k_granule_stops_the_rounding_from_reaching_a_neighbour() {
        let four_k = IpaGranule::FourK.bytes();
        assert_eq!(four_k, 4096);
        assert_eq!(round_up(0x21000, four_k), 0x21000);
        assert_eq!(round_up(0x21001, four_k), 0x22000);
        // The same size under the host page granule reaches 12 KiB past the blob, into
        // guest addresses a 4 KiB guest is entitled to hand its next blob.
        assert_eq!(round_up(0x21000, IpaGranule::SixteenK.bytes()), 0x24000);
    }
}
