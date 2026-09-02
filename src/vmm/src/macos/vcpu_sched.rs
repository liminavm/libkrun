// Copyright 2026 The limina Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Put a vCPU thread in the real-time scheduling band.
//!
//! An idle guest misses frame deadlines because its timer wakeups arrive late, and on macOS an
//! ordinary thread asking for a 16.667 ms deadline is served ~1.5 ms late at the median and tens
//! of milliseconds late in the tail. `THREAD_TIME_CONSTRAINT_POLICY` — the band CoreAudio's render
//! thread runs in — takes that to ~18 µs median, 52 µs worst (`spikes/macos-timer-wakeup/`).
//!
//! HVF parks an idle vCPU inside `hv_vcpu_run` rather than handing us the WFI trap, so the wait is
//! not ours. The *thread* still is: a scheduling band belongs to the thread, not to whoever called
//! into the kernel on it. Whether that reaches the wakeup that is actually late is the open
//! question — HVF also runs its own `VirtualClock` thread, which we do not own.
//!
//! A vCPU is not a 2 ms audio callback: it runs guest code for as long as the guest wants, and
//! xnu answers a real-time thread that computes for a whole second without blocking by demoting it
//! to `TH_MODE_TIMESHARE` for two (`osfmk/kern/priority.c::thread_quantum_expire`). A guest that
//! saturates its vCPUs while presenting therefore lands in the worst of both worlds. The
//! accumulator xnu tests is cleared in `thread_unblock()`, so a vCPU thread that genuinely parks —
//! even for 100 µs, even a few times a second — can never reach the limit. That is the heartbeat.
//!
//! Off unless `LIMINA_VCPU_SCHED` is set:
//!
//! * `rt` — the real-time band with the defaults; `rt:period,computation,constraint` in
//!   microseconds overrides them.
//! * `qos` — `QOS_CLASS_USER_INTERACTIVE` instead. No fail-safe to dodge, and no guarantee.
//! * `+hb` / `+hb<ms>` appended to `rt` arms the heartbeat (default 250 ms).
//! * `#<n>` appended limits the policy to the first `n` vCPUs, so the cost of banding *every*
//!   vCPU thread can be separated from the benefit of banding the one that carries the frame.
//!
//! `LIMINA_VCPU_RT` is still read as the old spelling of `rt`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;

/// Defaults: a 60 Hz arrival, a 1 ms slice, 2 ms to deliver it. `constraint - computation` is the
/// latency the scheduler promises, so the gap is deliberately small.
const DEFAULT_PERIOD: Duration = Duration::from_micros(16_667);
const DEFAULT_COMPUTATION: Duration = Duration::from_micros(1_000);
const DEFAULT_CONSTRAINT: Duration = Duration::from_micros(2_000);

#[repr(C)]
#[derive(Default)]
struct ThreadTimeConstraintPolicy {
    period: u32,
    computation: u32,
    constraint: u32,
    preemptible: u32,
}

#[repr(C)]
#[derive(Default)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_thread_self() -> u32;
    fn thread_info(thread: u32, flavor: u32, info: *mut u32, count: *mut u32) -> i32;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    fn thread_policy_set(thread: u32, flavor: u32, policy: *mut u32, count: u32) -> i32;
    fn mach_absolute_time() -> u64;
    fn mach_wait_until(deadline: u64) -> i32;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// `QOS_CLASS_USER_INTERACTIVE` from `sys/qos.h`.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
/// `QOS_CLASS_UTILITY` and `QOS_CLASS_BACKGROUND`, likewise.
///
/// Background is what a little vCPU gets, and utility is kept only as a comparison point:
/// **utility produces no asymmetry whatsoever.** An identical pinned loop in the guest ran in
/// ~1385 ms on both a big and a "little" vCPU under utility, and ~5200 ms on the little one
/// under background. Utility *prefers* an efficiency core; only background confines to one —
/// and throttles besides, which is where the extra slowdown comes from.
///
/// The cost of background is that a little vCPU holding a guest spinlock releases it ~4x
/// slower. That is a real hazard under host contention, and the reason the little count
/// defaults to zero rather than to something clever.
const QOS_CLASS_UTILITY: u32 = 0x11;
const QOS_CLASS_BACKGROUND: u32 = 0x09;

/// The vCPU topology, set once by the builder before any vCPU thread starts: `(num_cpus,
/// little)`. The last `little` vCPUs are the little ones — the same split the guest is told
/// about through `capacity-dmips-mhz` and the perf domains.
static TOPOLOGY: OnceLock<(u64, u64)> = OnceLock::new();

/// Declare which vCPUs are little. Call before starting the vCPU threads; later calls are
/// ignored, since a thread that has already picked its band will not revisit it.
pub fn set_topology(num_cpus: u64, little: u64) {
    let _ = TOPOLOGY.set((num_cpus, little.min(num_cpus)));
}

/// Whether this vCPU is one of the little ones.
pub fn is_little(vcpuid: u64) -> bool {
    match TOPOLOGY.get() {
        Some(&(num_cpus, little)) if little > 0 => vcpuid >= num_cpus - little,
        _ => false,
    }
}

/// The QoS class a little vCPU's thread runs at. `LIMINA_VCPU_LITTLE_QOS=utility` picks the
/// shallower class, which is useful for showing that it changes nothing.
fn little_qos() -> (u32, &'static str) {
    match std::env::var("LIMINA_VCPU_LITTLE_QOS") {
        Ok(v) if v.eq_ignore_ascii_case("utility") => (QOS_CLASS_UTILITY, "UTILITY"),
        _ => (QOS_CLASS_BACKGROUND, "BACKGROUND"),
    }
}

const THREAD_EXTENDED_POLICY: u32 = 1;
const THREAD_BASIC_INFO: u32 = 3;
/// `thread_basic_info_data_t` is ten `natural_t`s.
const THREAD_BASIC_INFO_COUNT: u32 = 10;

const DEFAULT_HEARTBEAT: Duration = Duration::from_millis(250);

/// How often the dynamic sampler looks at each vCPU thread's CPU share.
const SAMPLE_INTERVAL: Duration = Duration::from_millis(200);
/// Arm below this share of a core, disarm above the upper one. The gap is the hysteresis that
/// keeps a thread hovering at the boundary from flapping between policies every sample.
const ARM_BELOW: f64 = 0.35;
const DISARM_ABOVE: f64 = 0.60;
/// Long enough that the thread really parks — a deadline already behind us returns without ever
/// entering `TH_WAIT`, and then nothing is cleared.
const HEARTBEAT_PARK: Duration = Duration::from_micros(100);

fn ns_to_abs(ns: u64) -> u32 {
    let mut tb = MachTimebaseInfo::default();
    if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.numer == 0 {
        return ns as u32;
    }
    let abs = (ns as u128 * tb.denom as u128) / tb.numer as u128;
    abs.min(u32::MAX as u128) as u32
}

/// What a vCPU thread should ask the scheduler for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Band {
    /// `THREAD_TIME_CONSTRAINT_POLICY`, with the period/computation/constraint it was given.
    RealTime(Duration, Duration, Duration),
    /// `QOS_CLASS_USER_INTERACTIVE`.
    Qos,
}

/// How many vCPUs the policy applies to. `None` means all of them.
fn vcpu_limit() -> Option<u64> {
    let raw = std::env::var("LIMINA_VCPU_SCHED").unwrap_or_default();
    raw.split_once('#').and_then(|(_, n)| n.trim().parse().ok())
}

/// Whether the policy is applied dynamically, per vCPU, from that thread's own CPU share.
fn dynamic() -> bool {
    std::env::var("LIMINA_VCPU_SCHED")
        .map(|v| v.contains("+dyn"))
        .unwrap_or(false)
}

/// The policy and the heartbeat interval, from the environment. `None` for either means off.
fn requested() -> (Option<Band>, Option<Duration>) {
    let raw = std::env::var("LIMINA_VCPU_SCHED")
        .or_else(|_| std::env::var("LIMINA_VCPU_RT"))
        .unwrap_or_default();
    let raw = raw
        .split('#')
        .next()
        .unwrap_or_default()
        .replace("+dyn", "")
        .trim()
        .to_string();
    let raw = raw.as_str();
    if raw.is_empty() || raw == "0" || raw.eq_ignore_ascii_case("off") {
        return (None, None);
    }

    // `<policy>[+hb[ms]]`, and the policy half may carry its three durations after a colon.
    let (policy, hb) = match raw.split_once("+hb") {
        Some((policy, rest)) => {
            let ms: u64 = rest
                .trim()
                .parse()
                .unwrap_or(DEFAULT_HEARTBEAT.as_millis() as u64);
            (policy.trim(), Some(Duration::from_millis(ms.max(1))))
        }
        None => (raw, None),
    };

    if policy.eq_ignore_ascii_case("qos") {
        return (Some(Band::Qos), hb);
    }

    let durations = policy.split_once(':').map(|(_, d)| d).unwrap_or(policy);
    let parsed: Vec<u64> = durations
        .split(',')
        .filter_map(|p| p.trim().parse().ok())
        .collect();
    let band = if parsed.len() == 3 {
        Band::RealTime(
            Duration::from_micros(parsed[0]),
            Duration::from_micros(parsed[1]),
            Duration::from_micros(parsed[2]),
        )
    } else {
        Band::RealTime(DEFAULT_PERIOD, DEFAULT_COMPUTATION, DEFAULT_CONSTRAINT)
    };
    (Some(band), hb)
}

/// How long a vCPU thread may run without parking before the heartbeat makes it park. `None`
/// when the heartbeat is off. Read once.
pub fn heartbeat_interval() -> Option<Duration> {
    static CACHED: OnceLock<Option<Duration>> = OnceLock::new();
    *CACHED.get_or_init(|| requested().1)
}

/// Total forced parks across every vCPU thread, for the log line below.
static BEATS: AtomicU64 = AtomicU64::new(0);

/// The last time this thread was known to have blocked, in mach absolute units.
///
/// xnu clears `computation_metered` in `thread_unblock()`, so this is a conservative shadow of
/// the accumulator the fail-safe tests: any real park refreshes it, and we only force one when it
/// has gone stale.
#[derive(Default)]
pub struct Heartbeat {
    last_block: AtomicU64,
}

impl Heartbeat {
    pub fn new() -> Self {
        Self {
            last_block: AtomicU64::new(unsafe { mach_absolute_time() }),
        }
    }

    /// The thread is about to block for real (an idle vCPU parking on its WFI). Nothing to force.
    pub fn observed_block(&self) {
        self.last_block
            .store(unsafe { mach_absolute_time() }, Ordering::Relaxed);
    }

    /// Whether this thread has been computing for longer than the interval without parking.
    pub fn is_stale(&self, interval: Duration) -> bool {
        let now = unsafe { mach_absolute_time() };
        now.saturating_sub(self.last_block.load(Ordering::Relaxed))
            >= u64::from(ns_to_abs(interval.as_nanos() as u64))
    }

    /// Called at every exit from the guest: park briefly if this thread has been computing for
    /// longer than the interval, so the fail-safe accumulator never reaches its limit.
    pub fn beat(&self, interval: Duration) {
        if !self.is_stale(interval) {
            return;
        }
        unsafe {
            let deadline =
                mach_absolute_time() + u64::from(ns_to_abs(HEARTBEAT_PARK.as_nanos() as u64));
            mach_wait_until(deadline);
        }
        self.observed_block();
        // Whether the beat reaches a saturated vCPU at all is the thing to check first when the
        // band still misbehaves under load, so make it observable without a debugger.
        let n = BEATS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 200 == 0 {
            log::info!("[VCPU-RT] heartbeat parks so far: {n}");
        }
    }
}

/// One banded thread's mach port and the CPU time it had at the last sample.
struct Sampled {
    vcpuid: u64,
    port: u32,
    cpu_us: u64,
    armed: bool,
}

static REGISTRY: Mutex<Vec<Sampled>> = Mutex::new(Vec::new());

/// Total CPU time this thread has used, from `THREAD_BASIC_INFO`.
fn thread_cpu_us(port: u32) -> Option<u64> {
    let mut info = [0u32; THREAD_BASIC_INFO_COUNT as usize];
    let mut count = THREAD_BASIC_INFO_COUNT;
    if unsafe { thread_info(port, THREAD_BASIC_INFO, info.as_mut_ptr(), &mut count) } != 0 {
        return None;
    }
    // user_time then system_time, each a `time_value_t` of {seconds, microseconds}.
    let secs = u64::from(info[0]) + u64::from(info[2]);
    let usecs = u64::from(info[1]) + u64::from(info[3]);
    Some(secs * 1_000_000 + usecs)
}

/// Take `port` back out of the real-time band and return it to ordinary timeshare scheduling.
fn set_timeshare(port: u32) -> bool {
    let mut timeshare: u32 = 1;
    unsafe { thread_policy_set(port, THREAD_EXTENDED_POLICY, &mut timeshare, 1) == 0 }
}

/// Apply the band to another thread by port. Same policy as [`set_realtime_band`], which applies
/// it to the calling thread at startup.
fn set_band_on(port: u32, band: Band) -> bool {
    let Band::RealTime(period, computation, constraint) = band else {
        return false;
    };
    let mut policy = ThreadTimeConstraintPolicy {
        period: ns_to_abs(period.as_nanos() as u64),
        computation: ns_to_abs(computation.as_nanos() as u64),
        constraint: ns_to_abs(constraint.as_nanos() as u64),
        preemptible: 1,
    };
    let count = (size_of::<ThreadTimeConstraintPolicy>() / size_of::<u32>()) as u32;
    unsafe {
        thread_policy_set(
            port,
            THREAD_TIME_CONSTRAINT_POLICY,
            &mut policy as *mut _ as *mut u32,
            count,
        ) == 0
    }
}

/// Whether an armed thread should give its core back, or a disarmed one may take the band.
///
/// The band is a *reservation*: a banded thread preempts the host's own work, and every vCPU
/// thread banded at once leaves the present path nothing to run on. A vCPU that is mostly idle is
/// exactly the one that needs a punctual timer wake and the one that costs nothing to promise, so
/// arming follows each thread's own share of a core rather than a global switch.
fn next_state(armed: bool, share: f64) -> bool {
    if armed {
        share <= DISARM_ABOVE
    } else {
        share < ARM_BELOW
    }
}

/// Start the sampler that arms and disarms each registered vCPU thread. Idempotent.
fn start_sampler(band: Band) {
    static STARTED: OnceLock<()> = OnceLock::new();
    if STARTED.set(()).is_err() {
        return;
    }
    std::thread::Builder::new()
        .name("vcpu-band-sampler".into())
        .spawn(move || {
            loop {
                std::thread::sleep(SAMPLE_INTERVAL);
                let mut threads = REGISTRY.lock().unwrap();
                for t in threads.iter_mut() {
                    let Some(now_us) = thread_cpu_us(t.port) else {
                        continue;
                    };
                    let share = (now_us.saturating_sub(t.cpu_us)) as f64
                        / SAMPLE_INTERVAL.as_micros() as f64;
                    t.cpu_us = now_us;
                    let want = next_state(t.armed, share);
                    if want == t.armed {
                        continue;
                    }
                    let ok = if want {
                        set_band_on(t.port, band)
                    } else {
                        set_timeshare(t.port)
                    };
                    if ok {
                        t.armed = want;
                        log::debug!(
                            "[VCPU-RT] vCPU {} {} the band (share {:.0}%)",
                            t.vcpuid,
                            if want { "took" } else { "gave back" },
                            share * 100.0
                        );
                    }
                }
            }
        })
        .ok();
}

/// Move the *calling* thread into whichever band was asked for. Must run on the vCPU thread
/// itself, since both policies apply to the current thread.
pub fn set_realtime_band(vcpuid: u64) {
    // A little vCPU takes the low QoS class instead, and never the real-time band. The two are
    // not compatible in either direction: xnu does not serve a time-constraint thread on an
    // efficiency core, so banding a little vCPU would quietly undo the asymmetry the guest was
    // told about — and the guest, believing the CPU is slow, would keep packing work onto it.
    if is_little(vcpuid) {
        let (class, name) = little_qos();
        let ret = unsafe { pthread_set_qos_class_self_np(class, 0) };
        if ret == 0 {
            log::info!("[VCPU-RT] vCPU {vcpuid} is little: QOS_CLASS_{name}");
        } else {
            log::warn!("[VCPU-RT] vCPU {vcpuid}: little qos class refused (errno={ret})");
        }
        return;
    }
    if vcpu_limit().is_some_and(|limit| vcpuid >= limit) {
        return;
    }
    let (band, heartbeat) = requested();
    let Some(band) = band else {
        return;
    };
    let (period, computation, constraint) = match band {
        Band::Qos => {
            let ret = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
            if ret == 0 {
                log::info!("[VCPU-RT] vCPU {vcpuid} at QOS_CLASS_USER_INTERACTIVE");
            } else {
                log::warn!("[VCPU-RT] vCPU {vcpuid}: qos class refused (errno={ret})");
            }
            return;
        }
        Band::RealTime(p, c, k) => (p, c, k),
    };

    if dynamic() {
        // Register and let the sampler decide: a vCPU that is running guest code flat out must not
        // hold a real-time reservation, whatever it is doing right now.
        let port = unsafe { mach_thread_self() };
        REGISTRY.lock().unwrap().push(Sampled {
            vcpuid,
            port,
            cpu_us: thread_cpu_us(port).unwrap_or(0),
            armed: false,
        });
        start_sampler(band);
        log::info!(
            "[VCPU-RT] vCPU {vcpuid} joins the dynamic band (period={period:?} \
             computation={computation:?} constraint={constraint:?})"
        );
        return;
    }
    let mut policy = ThreadTimeConstraintPolicy {
        period: ns_to_abs(period.as_nanos() as u64),
        computation: ns_to_abs(computation.as_nanos() as u64),
        constraint: ns_to_abs(constraint.as_nanos() as u64),
        preemptible: 1,
    };
    let count = (size_of::<ThreadTimeConstraintPolicy>() / size_of::<u32>()) as u32;
    let ret = unsafe {
        thread_policy_set(
            mach_thread_self(),
            THREAD_TIME_CONSTRAINT_POLICY,
            &mut policy as *mut _ as *mut u32,
            count,
        )
    };
    if ret == 0 {
        log::info!(
            "[VCPU-RT] vCPU {vcpuid} in the real-time band (period={period:?} \
             computation={computation:?} constraint={constraint:?} heartbeat={heartbeat:?})"
        );
    } else {
        log::warn!("[VCPU-RT] vCPU {vcpuid}: thread_policy_set refused the band (kr={ret})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_duration_survives_the_trip_through_mach_units() {
        // The policy fields are mach absolute units, not nanoseconds. Converting and back must
        // land within a tick, or the band is asked for a period nothing like the one intended.
        let mut tb = MachTimebaseInfo::default();
        assert_eq!(unsafe { mach_timebase_info(&mut tb) }, 0);
        let ns = 16_667_000u64;
        let abs = ns_to_abs(ns) as u128;
        let back = (abs * tb.numer as u128) / tb.denom as u128;
        assert!(
            back.abs_diff(ns as u128) < 100,
            "round-tripped to {back} ns"
        );
    }

    #[test]
    fn arming_has_a_gap_a_thread_can_sit_in() {
        // Without hysteresis a vCPU hovering at the threshold changes policy every sample, and a
        // policy change is exactly the moment the present path can lose its core.
        assert!(ARM_BELOW < DISARM_ABOVE);
        // Idle: takes the band and keeps it.
        assert!(next_state(false, 0.02));
        assert!(next_state(true, 0.02));
        // In the gap: whatever it was, it stays.
        assert!(!next_state(false, 0.5));
        assert!(next_state(true, 0.5));
        // Running flat out: gives the core back and does not take it again.
        assert!(!next_state(true, 0.99));
        assert!(!next_state(false, 0.99));
    }

    #[test]
    fn a_heartbeat_park_is_far_enough_out_to_really_park() {
        // A deadline already behind us returns without entering TH_WAIT, which would leave the
        // fail-safe accumulator untouched — the whole point of the beat.
        assert!(HEARTBEAT_PARK >= Duration::from_micros(50));
    }

    #[test]
    fn the_band_is_off_without_the_variable() {
        // Safety net for the opt-in: a vCPU thread is not an audio callback, and nothing should
        // land in the real-time band by accident.
        unsafe { std::env::remove_var("LIMINA_VCPU_SCHED") };
        unsafe { std::env::remove_var("LIMINA_VCPU_RT") };
        assert_eq!(requested(), (None, None));
        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "0") };
        assert_eq!(requested(), (None, None));

        let rt_default = Band::RealTime(DEFAULT_PERIOD, DEFAULT_COMPUTATION, DEFAULT_CONSTRAINT);
        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "rt") };
        assert_eq!(requested(), (Some(rt_default), None));
        // The old spelling still works, so a recorded repro keeps reproducing.
        unsafe { std::env::remove_var("LIMINA_VCPU_SCHED") };
        unsafe { std::env::set_var("LIMINA_VCPU_RT", "1") };
        assert_eq!(requested(), (Some(rt_default), None));
        unsafe { std::env::remove_var("LIMINA_VCPU_RT") };

        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "rt:8000,500,900") };
        assert_eq!(
            requested().0,
            Some(Band::RealTime(
                Duration::from_micros(8_000),
                Duration::from_micros(500),
                Duration::from_micros(900)
            ))
        );

        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "qos") };
        assert_eq!(requested(), (Some(Band::Qos), None));

        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "rt+hb") };
        assert_eq!(requested(), (Some(rt_default), Some(DEFAULT_HEARTBEAT)));
        unsafe { std::env::set_var("LIMINA_VCPU_SCHED", "rt+hb50") };
        assert_eq!(
            requested(),
            (Some(rt_default), Some(Duration::from_millis(50)))
        );
        unsafe { std::env::remove_var("LIMINA_VCPU_SCHED") };
    }
}
