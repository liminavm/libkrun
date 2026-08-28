use crossbeam_channel::Sender;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use arch::aarch64::layout::VTIMER_IRQ;
use arch::aarch64::sysreg::*;
use hvf::bindings::{
    HV_SUCCESS, hv_sys_reg_t_HV_SYS_REG_CNTHCTL_EL2, hv_sys_reg_t_HV_SYS_REG_MDCCINT_EL1,
    hv_vcpu_get_sys_reg, hv_vcpu_set_sys_reg,
};
use hvf::{Vcpus, vcpu_request_exit};

// See https://developer.arm.com/documentation/ddi0595/2020-12/AArch64-Registers/ICC-IAR0-EL1--Interrupt-Controller-Interrupt-Acknowledge-Register-0
const GIC_INTID_SPURIOUS: u32 = 1023;

enum VcpuStatus {
    Running,
    Waiting,
}

struct PerCPUInterruptControllerState {
    vcpuid: u64,
    status: VcpuStatus,
    pending_irqs: VecDeque<u32>,
    wfe_sender: Option<Sender<u32>>,
    /// PSCI power state for CPU hotplug: `true` = ON, `false` = OFF (parked after CPU_OFF).
    /// Read by AFFINITY_INFO so the guest's offline reaper sees the CPU report OFF. All vCPUs
    /// start ON (the boot model brings every vCPU up).
    online: bool,
    /// True while this vCPU is blocked in a WFx wait (or parked offline): it is executing
    /// no guest instructions and is not going to until something wakes it. Distinct from
    /// "no recent vmexits", which a vCPU spinning in guest code also satisfies.
    parked: bool,
}

impl PerCPUInterruptControllerState {
    fn set_irq_common(&mut self, irq: u32) {
        debug!(
            "[GICv3] SET_IRQ_COMMON vcpuid={}, irq_line={}",
            self.vcpuid, irq
        );
        self.pending_irqs.push_back(irq);

        // limina wake-probe: the return half of the venus wake chain starts here. `status` is
        // the interesting part — a Waiting vCPU is parked in WFI/WFE and has to come back
        // through its channel, a Running one is only kicked out of hv_vcpu_run, and those are
        // very different costs. No-op unless LIMINA_WAKE_PROBE is set and the gpu worker armed.
        crate::virtio::wake_probe::irq_raised(irq, matches!(self.status, VcpuStatus::Waiting));

        match self.status {
            VcpuStatus::Waiting => {
                self.wfe_sender
                    .as_mut()
                    .unwrap()
                    .send(self.vcpuid as u32)
                    .unwrap();
                self.status = VcpuStatus::Running;
            }
            VcpuStatus::Running => {
                vcpu_request_exit(self.vcpuid).unwrap();
            }
        }
    }

    fn should_wait(&mut self) -> bool {
        if self.pending_irqs.is_empty() {
            self.status = VcpuStatus::Waiting;
            return true;
        }
        false
    }

    /// Force this vCPU out to the top of its run loop so it notices a pending
    /// [`VcpuEvent`](crate) (e.g. a snapshot Pause). Mirrors [`Self::set_irq_common`]'s wake
    /// path but injects **no** interrupt: a WFE/WFI-parked vCPU (blocked in
    /// `wait_for_event`'s channel `recv`) is woken via its wfe channel, while a running one is
    /// kicked out of `hv_vcpu_run` with `hv_vcpus_exit`. Harmless if the vCPU is already at the
    /// loop top: the spurious wakeup just re-evaluates `should_wait`.
    fn kick(&mut self) {
        match self.status {
            VcpuStatus::Waiting => {
                self.wfe_sender
                    .as_mut()
                    .unwrap()
                    .send(self.vcpuid as u32)
                    .unwrap();
                self.status = VcpuStatus::Running;
            }
            VcpuStatus::Running => {
                vcpu_request_exit(self.vcpuid).unwrap();
            }
        }
    }

    fn has_pending_irq(&self) -> bool {
        !self.pending_irqs.is_empty()
    }

    fn get_pending_irq(&mut self) -> u32 {
        self.pending_irqs.pop_front().unwrap_or(GIC_INTID_SPURIOUS)
    }
}

pub struct VcpuList {
    cpu_count: u64,
    vcpus: Vec<Mutex<PerCPUInterruptControllerState>>,
    /// The vCPU currently parked in PSCI `SYSTEM_SUSPEND` (suspend-to-RAM), or -1 when the guest
    /// is not system-suspended. At most one vCPU can be here: the call is only accepted from the
    /// last core still online. Read by the VMM to decide whether the guest needs a host-driven
    /// wake (there is no vCPU left to take a wake interrupt) and which vCPU to send it to.
    system_suspended: AtomicI64,
}

impl VcpuList {
    pub fn new(cpu_count: u64) -> Self {
        let mut vcpus = Vec::with_capacity(cpu_count as usize);
        for vcpuid in 0..cpu_count {
            vcpus.push(Mutex::new(PerCPUInterruptControllerState {
                vcpuid,
                status: VcpuStatus::Running,
                pending_irqs: VecDeque::new(),
                wfe_sender: None,
                online: true,
                parked: false,
            }));
        }

        Self {
            cpu_count,
            vcpus,
            system_suspended: AtomicI64::new(-1),
        }
    }

    pub fn get_cpu_count(&self) -> u64 {
        self.cpu_count
    }

    pub fn set_irq_common(&self, vcpuid: u64, irq: u32) {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize]
            .lock()
            .unwrap()
            .set_irq_common(irq);
    }

    pub fn set_sgi_irq(&self, vcpuid: u64, irq: u32) {
        assert!(vcpuid < self.cpu_count);
        assert!(irq < 16);
        self.vcpus[vcpuid as usize]
            .lock()
            .unwrap()
            .set_irq_common(irq);
    }

    pub fn register(&self, vcpuid: u64, wfe_sender: Sender<u32>) {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize].lock().unwrap().wfe_sender = Some(wfe_sender);
    }

    /// Kick every vCPU out to the top of its run loop so each notices a pending
    /// [`VcpuEvent`](crate) — used by the snapshot path to quiesce all vCPUs (M9). See
    /// [`PerCPUInterruptControllerState::kick`].
    pub fn kick_all(&self) {
        for vcpu in &self.vcpus {
            vcpu.lock().unwrap().kick();
        }
    }

    /// True when every vCPU is blocked in a WFx wait (or parked offline) — the guest is
    /// executing no instructions on any CPU. This is the signal that an s2idle entry has
    /// actually completed: the `s2idle_enter` rendezvous needs every vCPU to be scheduled
    /// to reach `tick_freeze()`, and only once the last one has does the guest stop
    /// accounting elapsed time as running time.
    pub fn all_parked(&self) -> bool {
        self.park_holdouts().is_empty()
    }

    /// Record that `vcpuid` has parked in PSCI `SYSTEM_SUSPEND`, or clear it on resume.
    pub fn set_system_suspended(&self, vcpuid: Option<u64>) {
        self.system_suspended
            .store(vcpuid.map(|v| v as i64).unwrap_or(-1), Ordering::SeqCst);
    }

    /// The vCPU parked in PSCI `SYSTEM_SUSPEND`, if the guest is suspended to RAM.
    pub fn system_suspended(&self) -> Option<u64> {
        match self.system_suspended.load(Ordering::SeqCst) {
            -1 => None,
            v => Some(v as u64),
        }
    }

    /// The vcpuids still executing guest instructions. Empty iff [`Self::all_parked`].
    pub fn park_holdouts(&self) -> Vec<u64> {
        self.vcpus
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                let v = v.lock().unwrap();
                !(v.parked || !v.online)
            })
            .map(|(i, _)| i as u64)
            .collect()
    }
}

impl Vcpus for VcpuList {
    fn set_vtimer_irq(&self, vcpuid: u64) {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize]
            .lock()
            .unwrap()
            .set_irq_common(VTIMER_IRQ);
    }

    fn should_wait(&self, vcpuid: u64) -> bool {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize].lock().unwrap().should_wait()
    }

    fn has_pending_irq(&self, vcpuid: u64) -> bool {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize]
            .lock()
            .unwrap()
            .has_pending_irq()
    }

    fn set_online(&self, vcpuid: u64, online: bool) {
        // A guest-supplied MPIDR (CPU_ON/CPU_OFF operand) is untrusted — ignore out-of-range
        // rather than panic the VMM.
        if let Some(vcpu) = self.vcpus.get(vcpuid as usize) {
            vcpu.lock().unwrap().online = online;
        }
    }

    fn is_online(&self, vcpuid: u64) -> bool {
        // Out-of-range (a bad guest AFFINITY_INFO target) reports offline — a safe default that
        // never panics.
        self.vcpus
            .get(vcpuid as usize)
            .map(|v| v.lock().unwrap().online)
            .unwrap_or(false)
    }

    fn others_all_offline(&self, vcpuid: u64) -> bool {
        self.vcpus
            .iter()
            .enumerate()
            .filter(|(i, _)| *i as u64 != vcpuid)
            .all(|(_, v)| !v.lock().unwrap().online)
    }

    fn set_parked(&self, vcpuid: u64, parked: bool) {
        if let Some(vcpu) = self.vcpus.get(vcpuid as usize) {
            vcpu.lock().unwrap().parked = parked;
        }
    }

    fn get_pending_irq(&self, vcpuid: u64) -> u32 {
        assert!(vcpuid < self.cpu_count);
        self.vcpus[vcpuid as usize]
            .lock()
            .unwrap()
            .get_pending_irq()
    }

    fn handle_sysreg_read(&self, vcpuid: u64, reg: u32) -> Option<u64> {
        assert!(vcpuid < self.cpu_count);

        if is_id_sysreg(reg) {
            return Some(0);
        }

        match reg {
            SYSREG_ICC_IAR1_EL1 => Some(
                self.vcpus[vcpuid as usize]
                    .lock()
                    .unwrap()
                    .get_pending_irq() as u64,
            ),
            SYSREG_ICC_PMR_EL1 => Some(0),
            SYSREG_ICC_CTLR_EL1 => Some(
                (1 << ICC_CTLR_EL1_RSS_SHIFT)
                    | (1 << ICC_CTLR_EL1_A3V_SHIFT)
                    | (1 << ICC_CTLR_EL1_ID_BITS_SHIFT)
                    | (4 << ICC_CTLR_EL1_PRI_BITS_SHIFT),
            ),
            SYSREG_CNTHCTL_EL2 => {
                let val: u64 = 0;
                let ret = unsafe {
                    hv_vcpu_get_sys_reg(
                        vcpuid,
                        hv_sys_reg_t_HV_SYS_REG_CNTHCTL_EL2,
                        &val as *const _ as *mut _,
                    )
                };
                if ret == HV_SUCCESS { Some(val) } else { None }
            }
            SYSREG_MDCCINT_EL1 => {
                let val: u64 = 0;
                let ret = unsafe {
                    hv_vcpu_get_sys_reg(
                        vcpuid,
                        hv_sys_reg_t_HV_SYS_REG_MDCCINT_EL1,
                        &val as *const _ as *mut _,
                    )
                };
                if ret == HV_SUCCESS { Some(val) } else { None }
            }
            _ => None,
        }
    }

    fn handle_sysreg_write(&self, vcpuid: u64, reg: u32, val: u64) -> bool {
        assert!(vcpuid < self.cpu_count);

        if is_id_sysreg(reg) {
            return true;
        }

        match reg {
            SYSREG_ICC_SGI1R_EL1 => {
                let target_list = val & 0xffff;
                let intid = ((val >> 24) & 0xf) as u32;
                let irm = (val & (1 << 40)) >> 40;
                let is_broadcast = irm == 1;
                let aff3aff2aff1 = val & ((0xff << 48) | (0xff << 32) | (0xff << 16));
                let rs = (val & (0xf << 44)) >> 44;

                debug!("vCPU {vcpuid} GenerateSoftwareInterrupt={intid} (0x{val:x})");

                // A flat core hierarchy should be good enough, but if we ever start using
                // Aff[123] MPIDR fields (currently MPID is configured via DT), GICv3 support
                // will need to be added.
                assert_eq!(
                    aff3aff2aff1, 0,
                    "[GICv3] only flat core hierarchy supported for now"
                );

                assert!(
                    !is_broadcast,
                    "[GICv3] SGI broadcast is not implemented yet"
                );

                // for each core in target list
                for target_id in 0u64..=15u64 {
                    if (target_list >> target_id) & 1 == 1 {
                        self.set_sgi_irq(rs * 16 + target_id, intid)
                    }
                }

                true
            }
            SYSREG_CNTHCTL_EL2 => {
                let ret = unsafe {
                    hv_vcpu_set_sys_reg(vcpuid, hv_sys_reg_t_HV_SYS_REG_CNTHCTL_EL2, val)
                };
                ret == HV_SUCCESS
            }
            SYSREG_MDCCINT_EL1 => {
                let ret = unsafe {
                    hv_vcpu_set_sys_reg(vcpuid, hv_sys_reg_t_HV_SYS_REG_MDCCINT_EL1, val)
                };
                ret == HV_SUCCESS
            }
            SYSREG_ICC_EOIR1_EL1
            | SYSREG_ICC_IGRPEN1_EL1
            | SYSREG_ICC_PMR_EL1
            | SYSREG_ICC_BPR1_EL1
            | SYSREG_ICC_CTLR_EL1
            | SYSREG_ICC_AP1R0_EL1
            | SYSREG_LORC_EL1
            | SYSREG_OSLAR_EL1
            | SYSREG_OSDLR_EL1 => true,
            _ => false,
        }
    }
}
