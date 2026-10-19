// SPDX-License-Identifier: Apache-2.0

//! Virtual CPUFreq — the host end of Linux's `qemu,virtual-cpufreq` (limina).
//!
//! The guest driver (`drivers/cpufreq/virtual-cpufreq.c`, in-tree since 6.13 and shipped as a
//! module by stock Fedora) treats this as a performance *hint* channel: it writes the frequency
//! schedutil asks for and reads back what it is getting. Nothing here has to make a vCPU actually
//! run faster, and this device deliberately does not try — the host scheduler owns that.
//!
//! What it exists for is the two things the guest cannot synthesize on its own:
//!
//! 1. **Frequency invariance.** On binding, the driver installs itself as a
//!    `scale_freq_source`, which is one of the hard preconditions for Energy Aware Scheduling
//!    (`sched_is_eas_possible()` fails `arch_scale_freq_invariant()` without it).
//! 2. **A cpufreq policy per CPU, under schedutil**, which is the other precondition.
//!
//! EAS is the goal: with it the guest packs light load onto a subset of vCPUs and leaves the rest
//! fully idle, where NO_HZ stops their tick and they cost the host nothing — the same saving
//! vCPU offlining gets by deleting capacity, but *without* deleting it, so `nproc` keeps telling
//! the truth and `make -j$(nproc)` still sizes itself to the whole machine.
//!
//! Register map, per the driver (`PER_CPU_OFFSET` = 4 KiB, indexed by logical CPU):
//!
//! | offset | dir | meaning                                   |
//! |--------|-----|-------------------------------------------|
//! | `0x00` | R   | current performance state                 |
//! | `0x04` | W   | requested performance state               |
//! | `0x08` | R   | number of entries in this CPU's perf table|
//! | `0x0c` | W   | perf-table index to read next             |
//! | `0x10` | R   | the frequency at the selected index       |
//! | `0x14` | R   | this CPU's performance-domain id          |
//!
//! Two contract details worth stating, because both would be silent failures: the driver probes
//! `PERFTBL_LEN` for **every possible CPU** and refuses to load unless each answers `1..=64`, and
//! it groups CPUs into one cpufreq policy per distinct `PERF_DOMAIN` value.

use crate::BusDevice;

/// Register offsets within one CPU's window.
const REG_CUR_PERF_STATE: u64 = 0x00;
const REG_SET_PERF_STATE: u64 = 0x04;
const REG_PERFTBL_LEN: u64 = 0x08;
const REG_PERFTBL_SEL: u64 = 0x0c;
const REG_PERFTBL_RD: u64 = 0x10;
const REG_PERF_DOMAIN: u64 = 0x14;

/// Bytes of MMIO per CPU (the driver's `PER_CPU_OFFSET`).
pub const PER_CPU_OFFSET: u64 = 0x1000;

/// The driver refuses a table longer than this (`PERFTBL_MAX_ENTRIES`).
const PERFTBL_MAX_ENTRIES: usize = 64;

/// One CPU's performance domain: the frequencies it can be asked for, and which policy it
/// belongs to. CPUs sharing a `domain` end up in one cpufreq policy, exactly as CPUs sharing a
/// clock would on real hardware.
#[derive(Debug, Clone)]
pub struct VcpuPerfDomain {
    /// Ascending frequencies in kHz (cpufreq's unit). Never empty, never > 64 entries.
    pub freqs: Vec<u32>,
    /// Performance-domain id. The driver puts every CPU reporting the same value into one policy.
    pub domain: u32,
    /// `capacity-dmips-mhz` for this vCPU, relative to a 1024 maximum. **Not read by this
    /// device** — the FDT emits it on the CPU node — but it lives here because the kernel
    /// requires the two to agree: `em_dev_register_perf_domain()` refuses a perf domain whose
    /// CPUs do not all report the same `arch_scale_cpu_capacity`, so a capacity split that does
    /// not follow a domain split silently costs us the energy model, and with it EAS.
    pub capacity: u32,
}

/// The `capacity-dmips-mhz` a vCPU gets when every vCPU is the same. Any value works — the
/// kernel normalises against the largest — but 1024 is the normalised maximum, so using it
/// keeps the emitted numbers and the guest's `cpu_capacity` readings identical.
pub const CAPACITY_UNIFORM: u32 = 1024;

/// The `capacity-dmips-mhz` of a "little" vCPU, as a fraction of [`CAPACITY_UNIFORM`].
///
/// **Measured, not derived from the E-core/P-core ratio.** An identical CPU-bound loop pinned
/// inside the guest takes ~1385 ms on a big vCPU and ~5200 ms on a little one (M1 Max, 4 vCPUs,
/// steady state) — 3.75x, which is well past the ~2.3x that efficiency-core placement alone
/// would explain, because `QOS_CLASS_BACKGROUND` throttles as well as relocates.
///
/// The number has to be honest or the whole exercise backfires: EAS decides what fits on a CPU
/// from exactly this capacity, so overstating it makes the scheduler pack work onto a vCPU that
/// cannot carry it.
pub const CAPACITY_LITTLE: u32 = 1024 * 100 / 375;

/// The frequency ladder every vCPU advertises. The absolute numbers do not matter — nothing on
/// the host acts on a request — but the *ratios* do, because the guest derives its
/// frequency-invariance scale from `cur / max`.
pub const DEFAULT_FREQS_KHZ: [u32; 5] = [600_000, 1_200_000, 1_800_000, 2_400_000, 3_200_000];

impl VcpuPerfDomain {
    /// The per-vCPU topology for `num_cpus` vCPUs of which the **last** `little` are little.
    ///
    /// Little vCPUs come last so that CPU0 — which takes the boot path and the GIC's default
    /// interrupt affinity — stays big. `little == 0` gives a uniform machine: one domain, one
    /// capacity, and no asymmetry for the scheduler to find.
    ///
    /// The domain split always follows the capacity split, which is not a style choice: see
    /// [`VcpuPerfDomain::capacity`].
    pub fn topology(num_cpus: usize, little: usize) -> Vec<VcpuPerfDomain> {
        let little = little.min(num_cpus);
        (0..num_cpus)
            .map(|i| {
                let is_little = i >= num_cpus - little;
                VcpuPerfDomain {
                    freqs: DEFAULT_FREQS_KHZ.to_vec(),
                    domain: u32::from(is_little),
                    capacity: if is_little {
                        CAPACITY_LITTLE
                    } else {
                        CAPACITY_UNIFORM
                    },
                }
            })
            .collect()
    }
}

/// Per-CPU register state.
#[derive(Debug)]
struct CpuRegs {
    perf: VcpuPerfDomain,
    /// The frequency last requested by the guest; read back as the current state. Echoing the
    /// request is honest here: we do not model a ramp, and pretending to would only make the
    /// guest's frequency-invariance scaling lie in a different direction.
    cur: u32,
    /// The index `REG_PERFTBL_RD` will return next.
    sel: u32,
}

/// The device. One 4 KiB window per vCPU, in logical-CPU order.
#[derive(Debug)]
pub struct VirtCpuFreq {
    cpus: Vec<CpuRegs>,
}

impl VirtCpuFreq {
    /// Build the device from one performance domain description per vCPU, in logical order.
    ///
    /// Each CPU starts at the top of its own table, which is what a guest that has not yet run
    /// schedutil should see: no throttling it did not ask for.
    pub fn new(per_cpu: Vec<VcpuPerfDomain>) -> VirtCpuFreq {
        let cpus = per_cpu
            .into_iter()
            .map(|mut perf| {
                // Defend the driver's contract at construction rather than discovering a refusal
                // to bind at guest boot, which surfaces only as "no cpufreq" with no explanation.
                assert!(
                    !perf.freqs.is_empty() && perf.freqs.len() <= PERFTBL_MAX_ENTRIES,
                    "virtual-cpufreq needs 1..={PERFTBL_MAX_ENTRIES} frequencies per CPU, got {}",
                    perf.freqs.len()
                );
                perf.freqs.sort_unstable();
                perf.freqs.dedup();
                let cur = *perf.freqs.last().expect("non-empty");
                CpuRegs { perf, cur, sel: 0 }
            })
            .collect();
        VirtCpuFreq { cpus }
    }

    /// The MMIO window this device needs for `num_cpus` vCPUs.
    pub fn mmio_len(num_cpus: usize) -> u64 {
        (num_cpus as u64) * PER_CPU_OFFSET
    }

    /// The frequency the guest last asked for on `cpu`, if any — the whole point of the channel,
    /// and the hook a future host-side consumer (thread QoS, for instance) would read.
    pub fn requested_khz(&self, cpu: usize) -> Option<u32> {
        self.cpus.get(cpu).map(|c| c.cur)
    }
}

impl BusDevice for VirtCpuFreq {
    fn read(&mut self, _vcpuid: u64, offset: u64, data: &mut [u8]) {
        if data.len() != 4 {
            return;
        }
        let cpu = (offset / PER_CPU_OFFSET) as usize;
        let reg = offset % PER_CPU_OFFSET;
        let Some(state) = self.cpus.get(cpu) else {
            // A read outside the window we advertised. Answer zero rather than panicking: for
            // PERFTBL_LEN that is exactly the "invalid" the driver checks for, so it declines to
            // bind instead of running on nonsense.
            data.copy_from_slice(&0u32.to_le_bytes());
            return;
        };
        let value = match reg {
            REG_CUR_PERF_STATE => state.cur,
            REG_PERFTBL_LEN => state.perf.freqs.len() as u32,
            REG_PERFTBL_RD => state
                .perf
                .freqs
                .get(state.sel as usize)
                .copied()
                // An out-of-range index is the guest's error; the last entry is a safer answer
                // than 0, which cpufreq would take as a valid frequency of zero.
                .unwrap_or_else(|| *state.perf.freqs.last().expect("non-empty")),
            REG_PERF_DOMAIN => state.perf.domain,
            // SET/SEL are write-only; reads of them and of anything else read as zero.
            _ => 0,
        };
        data.copy_from_slice(&value.to_le_bytes());
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, data: &[u8]) {
        if data.len() != 4 {
            return;
        }
        let cpu = (offset / PER_CPU_OFFSET) as usize;
        let reg = offset % PER_CPU_OFFSET;
        let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let Some(state) = self.cpus.get_mut(cpu) else {
            return;
        };
        match reg {
            // The guest's requested frequency. We record it and report it straight back as the
            // current state; the host scheduler decides what the vCPU thread actually gets.
            REG_SET_PERF_STATE => state.cur = value,
            REG_PERFTBL_SEL => state.sel = value,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(n: usize) -> VirtCpuFreq {
        VirtCpuFreq::new(
            (0..n)
                .map(|i| VcpuPerfDomain {
                    freqs: vec![600_000, 1_200_000, 2_400_000],
                    domain: (i / 2) as u32,
                    capacity: CAPACITY_UNIFORM,
                })
                .collect(),
        )
    }

    fn read32(d: &mut VirtCpuFreq, cpu: usize, reg: u64) -> u32 {
        let mut buf = [0u8; 4];
        d.read(0, cpu as u64 * PER_CPU_OFFSET + reg, &mut buf);
        u32::from_le_bytes(buf)
    }

    fn write32(d: &mut VirtCpuFreq, cpu: usize, reg: u64, v: u32) {
        d.write(0, cpu as u64 * PER_CPU_OFFSET + reg, &v.to_le_bytes());
    }

    /// The driver reads PERFTBL_LEN for every possible CPU at probe and refuses to bind unless
    /// each answers 1..=64. A zero anywhere means no cpufreq at all, and therefore no EAS — with
    /// nothing in the guest log to say why.
    #[test]
    fn every_cpu_reports_a_table_length_the_driver_will_accept() {
        let mut d = dev(4);
        for cpu in 0..4 {
            let len = read32(&mut d, cpu, REG_PERFTBL_LEN);
            assert!(
                (1..=PERFTBL_MAX_ENTRIES as u32).contains(&len),
                "cpu{cpu} reported an unusable table length {len}"
            );
        }
    }

    /// The table is read by selecting an index and then reading; the pair has to be per-CPU, or
    /// two CPUs probing concurrently would read each other's entries.
    #[test]
    fn the_perf_table_reads_back_by_index_per_cpu() {
        let mut d = dev(2);
        write32(&mut d, 0, REG_PERFTBL_SEL, 1);
        write32(&mut d, 1, REG_PERFTBL_SEL, 2);
        assert_eq!(read32(&mut d, 0, REG_PERFTBL_RD), 1_200_000);
        assert_eq!(read32(&mut d, 1, REG_PERFTBL_RD), 2_400_000);
    }

    /// Frequencies must come back ascending whatever order they were given in — cpufreq's table
    /// helpers assume it, and a descending table silently mis-selects.
    #[test]
    fn frequencies_are_sorted_and_deduplicated() {
        let mut d = VirtCpuFreq::new(vec![VcpuPerfDomain {
            freqs: vec![2_400_000, 600_000, 1_200_000, 600_000],
            domain: 0,
            capacity: CAPACITY_UNIFORM,
        }]);
        assert_eq!(read32(&mut d, 0, REG_PERFTBL_LEN), 3);
        let got: Vec<u32> = (0..3)
            .map(|i| {
                write32(&mut d, 0, REG_PERFTBL_SEL, i);
                read32(&mut d, 0, REG_PERFTBL_RD)
            })
            .collect();
        assert_eq!(got, vec![600_000, 1_200_000, 2_400_000]);
    }

    /// A requested frequency reads straight back as the current one.
    #[test]
    fn a_requested_frequency_is_reported_back_as_current() {
        let mut d = dev(2);
        assert_eq!(read32(&mut d, 0, REG_CUR_PERF_STATE), 2_400_000);
        write32(&mut d, 0, REG_SET_PERF_STATE, 600_000);
        assert_eq!(read32(&mut d, 0, REG_CUR_PERF_STATE), 600_000);
        assert_eq!(d.requested_khz(0), Some(600_000));
        // ...and only on the CPU that asked.
        assert_eq!(read32(&mut d, 1, REG_CUR_PERF_STATE), 2_400_000);
    }

    /// Domains are what the driver groups cpufreq policies by.
    #[test]
    fn cpus_report_their_performance_domain() {
        let mut d = dev(4);
        assert_eq!(read32(&mut d, 0, REG_PERF_DOMAIN), 0);
        assert_eq!(read32(&mut d, 1, REG_PERF_DOMAIN), 0);
        assert_eq!(read32(&mut d, 2, REG_PERF_DOMAIN), 1);
        assert_eq!(read32(&mut d, 3, REG_PERF_DOMAIN), 1);
    }

    /// Accesses past the advertised window must not panic the VMM. A stray PERFTBL_LEN of 0 is
    /// also precisely the value that makes the driver decline rather than misbehave.
    #[test]
    fn out_of_range_access_is_inert() {
        let mut d = dev(1);
        assert_eq!(read32(&mut d, 9, REG_PERFTBL_LEN), 0);
        write32(&mut d, 9, REG_SET_PERF_STATE, 1_000_000);
        // A non-word access is ignored rather than mis-parsed.
        let mut byte = [0u8; 1];
        d.read(0, REG_CUR_PERF_STATE, &mut byte);
        assert_eq!(byte, [0]);
    }
    /// Every CPU in a perf domain must report the same capacity, or the guest's energy model
    /// refuses to register and EAS never turns on. The two splits are one split.
    #[test]
    fn the_domain_split_follows_the_capacity_split() {
        for (num, little) in [(4, 0), (4, 2), (8, 2), (2, 1), (3, 5), (1, 0)] {
            let topo = VcpuPerfDomain::topology(num, little);
            assert_eq!(topo.len(), num);
            let mut by_domain: std::collections::HashMap<u32, Vec<u32>> = Default::default();
            for t in &topo {
                by_domain.entry(t.domain).or_default().push(t.capacity);
            }
            for (domain, caps) in by_domain {
                assert!(
                    caps.windows(2).all(|w| w[0] == w[1]),
                    "domain {domain} of {num}/{little} mixes capacities: {caps:?}"
                );
            }
        }
    }

    /// CPU0 stays big: it takes the boot path and the GIC's default interrupt affinity.
    #[test]
    fn cpu0_is_never_little() {
        for little in 0..=8 {
            let topo = VcpuPerfDomain::topology(8, little);
            if little < 8 {
                assert_eq!(topo[0].capacity, CAPACITY_UNIFORM, "little={little}");
            }
        }
    }

    /// A uniform machine has exactly one domain and one capacity — nothing for EAS to find.
    #[test]
    fn no_littles_means_no_asymmetry() {
        let topo = VcpuPerfDomain::topology(4, 0);
        assert!(topo.iter().all(|t| t.domain == 0));
        assert!(topo.iter().all(|t| t.capacity == CAPACITY_UNIFORM));
    }
}
