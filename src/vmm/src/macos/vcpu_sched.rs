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

use std::sync::{Mutex, OnceLock};
use std::time::Duration;

// Each vCPU's arm state is loom's in this crate's own tests under `--cfg loom`, so `loom_model`
// can race the sampler against the vCPU's own guard. The process-wide statics stay std's: loom's
// are not `const`.
#[cfg(all(test, loom))]
use loom::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
#[cfg(not(all(test, loom)))]
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

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
    fn sysctlbyname(
        name: *const u8,
        oldp: *mut core::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut core::ffi::c_void,
        newlen: usize,
    ) -> i32;
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

/// How many declared periods a banded thread may compute, without ever parking, before it takes
/// itself out of the band.
///
/// Two, because one is not yet evidence: a thread that has run for a whole period without parking
/// has already spent more than the `computation` it promised for that period, and a second says
/// it was not a one-off. The threshold is derived from the reservation rather than chosen, so it
/// moves with a caller's `rt:period,...` instead of silently not applying to it.
const SELF_DISARM_PERIODS: u32 = 2;

fn ns_to_abs(ns: u64) -> u32 {
    let mut tb = MachTimebaseInfo::default();
    if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.numer == 0 {
        return ns as u32;
    }
    let abs = (ns as u128 * tb.denom as u128) / tb.numer as u128;
    abs.min(u32::MAX as u128) as u32
}

/// The inverse of [`ns_to_abs`], for reading back a `mach_absolute_time` interval.
fn abs_to_ns(abs: u64) -> u64 {
    let mut tb = MachTimebaseInfo::default();
    if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.denom == 0 {
        return abs;
    }
    ((abs as u128 * tb.numer as u128) / tb.denom as u128).min(u64::MAX as u128) as u64
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
static BEATS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The last time this thread was known to have blocked, in mach absolute units.
///
/// xnu clears `computation_metered` in `thread_unblock()`, so this is a conservative shadow of
/// the accumulator the fail-safe tests: any real park refreshes it, and we only force one when it
/// has gone stale.
#[derive(Default)]
pub struct Heartbeat {
    last_block: std::sync::atomic::AtomicU64,
}

impl Heartbeat {
    pub fn new() -> Self {
        Self {
            last_block: std::sync::atomic::AtomicU64::new(unsafe { mach_absolute_time() }),
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
        if n.is_multiple_of(200) {
            log::info!("[VCPU-RT] heartbeat parks so far: {n}");
        }
    }
}

/// Read a `u32` sysctl by name, or `None` if it does not answer.
fn sysctl_u32(name: &[u8]) -> Option<u32> {
    debug_assert_eq!(
        name.last(),
        Some(&0),
        "a sysctl name reaches C as a C string"
    );
    let mut out: u32 = 0;
    let mut len = size_of::<u32>();
    // SAFETY: `name` is NUL-terminated by the assert above, and the buffer and its length
    // describe the same `u32` — the pair the kernel writes through.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut out as *mut u32 as *mut core::ffi::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == size_of::<u32>()).then_some(out)
}

/// How many vCPU threads may hold the band at once.
///
/// The band is a reservation, so this is the host's floor: the number of cores that can never be
/// promised away. **It is deliberately not derived from `hw.activecpu`**, which reports every
/// configured core whether or not any of them is parked — measured 10 of 10 on an idle M1 Max,
/// while a panic report from an M4 Pro at the same moment showed 10 of its 14 cores offline with
/// only the efficiency cluster running. A cap against a number that cannot fall is not a cap.
///
/// The efficiency cluster is what survives an idle machine, so half of it is the bound, floored
/// at one. That is 1 on a 2-E-core M1 Max and 2 on a 4-E-core M4 Pro. The only configuration
/// measured clean under a saturated guest is a single banded vCPU (`spikes/macos-timer-wakeup/`,
/// `rt#1`), so this is a bound and not a tuned value: it exists to keep an explicit `rt+dyn` from
/// owning the machine, not to say that everything under it performs well.
fn arm_cap() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let e_cores = sysctl_u32(b"hw.perflevel1.logicalcpu\0")
            .or_else(|| sysctl_u32(b"hw.perflevel0.logicalcpu\0"))
            .unwrap_or(2) as usize;
        (e_cores / 2).max(1)
    })
}

/// One banded thread's mach port and the CPU time it had at the last sample.
struct Sampled {
    vcpuid: u64,
    port: u32,
    cpu_us: u64,
    /// Shared with the vCPU thread itself, which takes itself out of the band without waiting to
    /// be told — see [`BandGuard`]. The sampler must therefore read it rather than remember it.
    hold: Arc<Hold>,
}

/// Whether a thread has the band, shared by the sampler and the thread's own [`BandGuard`].
///
/// Either party moves the thread in or out, and each move is two steps: the kernel call, and the
/// record of it that everyone else reads. Taken as two plain steps by each party, they interleaved
/// into a thread the kernel had banded while the record said it was not: a guard that had judged
/// its hold, preempted across the sampler taking the band back and handing it over again, cleared
/// the record after the new hold's kernel call (found by `loom_model`). Nothing ever takes such a
/// thread back out, because the guard and the sampler both believe it already out, and that is the
/// 2026-09-21 panic's precondition.
///
/// So a move is claimed before it is made, by a compare-and-swap on one word, and finished by
/// whoever claimed it; the other party sees the claim and leaves the thread alone. The word also
/// counts the holds, so a decision about one hold is never applied to the next: a guard that judged
/// the first may find, by the time it acts, a second on which nothing has been burned.
pub struct Hold {
    /// The phase in the low two bits ([`OUT`], [`ARMING`], [`IN`], [`DISARMING`]) and, above them,
    /// how many holds have been handed out.
    word: AtomicU64,
    /// `mach_absolute_time` when this hold began, and the thread's total CPU microseconds at that
    /// moment. The guard spends its budget from these, not from the last park it happens to have
    /// inherited — see [`should_disarm`]. Written before the word publishes the hold, and read
    /// after loading the word, whose release/acquire pair carries them.
    armed_at: AtomicU64,
    armed_cpu_us: AtomicU64,
}

const OUT: u64 = 0;
const ARMING: u64 = 1;
const IN: u64 = 2;
const DISARMING: u64 = 3;
const PHASE: u64 = 3;

impl Hold {
    fn new() -> Hold {
        Hold {
            word: AtomicU64::new(OUT),
            armed_at: AtomicU64::new(0),
            armed_cpu_us: AtomicU64::new(0),
        }
    }

    /// A thread already in its first hold, which began at `armed_at` with `cpu_us` burned.
    #[cfg(test)]
    fn banded_since(armed_at: u64, cpu_us: u64) -> Hold {
        Hold {
            word: AtomicU64::new((PHASE + 1) | IN),
            armed_at: AtomicU64::new(armed_at),
            armed_cpu_us: AtomicU64::new(cpu_us),
        }
    }

    fn load(&self) -> u64 {
        self.word.load(Ordering::Acquire)
    }

    /// Whether `word` counts against the cap: anything but out does, a move in flight included.
    fn counts(word: u64) -> bool {
        word & PHASE != OUT
    }

    /// Hand the band over, if the thread is out and nobody is already moving it.
    fn arm<O: BandOs>(&self, os: &O, port: u32, band: Band, cpu_now: u64) -> bool {
        let seen = self.load();
        if seen & PHASE != OUT {
            return false;
        }
        let hold = (seen & !PHASE) + (PHASE + 1);
        if self
            .word
            .compare_exchange(seen, hold | ARMING, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        if !os.set_band(port, band) {
            self.word.store(seen, Ordering::Release);
            return false;
        }
        self.armed_at.store(os.now(), Ordering::Relaxed);
        self.armed_cpu_us.store(cpu_now, Ordering::Relaxed);
        self.word.store(hold | IN, Ordering::Release);
        true
    }

    /// Take the band back, if the thread is still in the hold `seen` was loaded in and nobody is
    /// already moving it.
    fn disarm<O: BandOs>(&self, os: &O, port: u32, seen: u64) -> bool {
        if seen & PHASE != IN {
            return false;
        }
        let hold = seen & !PHASE;
        if self
            .word
            .compare_exchange(seen, hold | DISARMING, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        if os.set_timeshare(port) {
            self.word.store(hold | OUT, Ordering::Release);
            true
        } else {
            self.word.store(seen, Ordering::Release);
            false
        }
    }
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

/// What moving a thread in and out of the band asks of the kernel, and the clocks it is judged by.
/// A seam so `loom_model` can race the sampler against a vCPU's own guard over a stand-in kernel;
/// everything else is [`Mach`].
pub trait BandOs {
    /// `mach_absolute_time`.
    fn now(&self) -> u64;
    /// The thread's total CPU microseconds ([`thread_cpu_us`]).
    fn cpu_us(&self, port: u32) -> Option<u64>;
    fn set_band(&self, port: u32, band: Band) -> bool;
    fn set_timeshare(&self, port: u32) -> bool;
}

/// The real kernel.
pub struct Mach;

impl BandOs for Mach {
    fn now(&self) -> u64 {
        unsafe { mach_absolute_time() }
    }
    fn cpu_us(&self, port: u32) -> Option<u64> {
        thread_cpu_us(port)
    }
    fn set_band(&self, port: u32, band: Band) -> bool {
        set_band_on(port, band)
    }
    fn set_timeshare(&self, port: u32) -> bool {
        set_timeshare(port)
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

/// What a thread's armed state should become, or `None` to leave it alone.
///
/// `next_state` asks only about the thread; this adds the one question that is about the host.
/// The cap wins over the share in both directions: a thread idle enough to deserve the band does
/// not get it when there is no room, and a thread already holding one gives it back when the room
/// has gone — which is how a cap that is already exceeded walks itself back down instead of
/// waiting for every holder to get busy.
fn decide(armed: bool, share: f64, armed_now: usize, cap: usize) -> Option<bool> {
    if armed && armed_now > cap {
        return Some(false);
    }
    let want = next_state(armed, share);
    if want == armed || (want && armed_now >= cap) {
        return None;
    }
    Some(want)
}

/// What share of a hold must be CPU time before the thread counts as computing flat out.
///
/// Chosen from the separation actually measured, not from what a spin "should" look like. The
/// idle false positives this rate exists to reject ran at 0.0%, 2.4% and 8.5% of a core. A vCPU
/// under four saturating spinners, over eight disarms, ran at 73.7, 86.3, 87.8, 89.9, 99.3, 99.8,
/// 100.0 and 100.0 — so the gap to clear is 8.5% to 73.7%, and half a core sits in the middle of
/// it with room on both sides.
///
/// **90% would have been wrong**, and was the first guess: four of those eight real saturations
/// fall under it, so it would have missed them and quietly given back the protection this whole
/// mechanism exists for. The asymmetry is the reason to keep the margin generous — missing a real
/// spin is a host panic, rejecting an idle thread costs one re-arm.
const SATURATED_PERCENT: u32 = 50;

/// Whether a banded thread owes its band back.
///
/// Two conditions, and both are needed. **The budget** is CPU time burned since the arm, which is
/// what makes an idle vCPU cheap to promise: wall time alone disarms one because time passes
/// whether or not it runs, and time-since-park does too, because HVF parks an idle vCPU inside
/// `hv_vcpu_run` rather than handing us the WFI trap — `observed_block` never fires on the idle
/// path, so the timestamp ages while the thread sleeps. The 2026-09-21 log holds that
/// contradiction in plain sight: the sampler armed threads it measured at 0%, 1%, 2% and 9% of a
/// core, and the guard declared those same threads to have computed for 33 ms without parking.
///
/// **The rate** is what makes the budget mean "flat out" rather than "eventually". This runs only
/// at exits from the guest, so the window between two of them is unbounded — and an absolute
/// budget over an unbounded window is an arbitrarily low bar. Measured with the budget alone:
/// holds of 413 ms, 1.75 s and 97.5 s all ended on 35-45 ms of CPU, which is 8.5%, 2.4% and 0.0%
/// of a core. Those threads were idle and gave the band back anyway. Comparing the CPU against
/// the hold rather than against nothing is scale-free, so it says the same thing at any window.
///
/// The rate leaves the panic protection untouched: a saturated vCPU burns CPU 1:1 with the wall
/// clock and clears 90% comfortably — measured at 42.4 ms of CPU against a 33.3 ms budget, with
/// four spinners saturating a four-vCPU guest.
fn should_disarm(armed: bool, cpu_used: Duration, held: Duration, budget: Duration) -> bool {
    armed && cpu_used >= budget && cpu_used * 100 >= held * SATURATED_PERCENT
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
                // Read rather than remembered: a vCPU thread takes itself out of the band from
                // its own run loop, so this sampler's idea of who is armed is always downstream
                // of theirs.
                let mut armed_now = threads
                    .iter()
                    .filter(|t| Hold::counts(t.hold.load()))
                    .count();
                let cap = arm_cap();
                for t in threads.iter_mut() {
                    sample_one(&Mach, t, band, &mut armed_now, cap);
                }
            }
        })
        .ok();
}

/// One sample of one registered thread: measure its share of a core since the last sample, and
/// hand it the band or take the band back as [`decide`] says. `armed_now` is how many threads
/// hold the band, kept current as this changes it.
fn sample_one<O: BandOs>(os: &O, t: &mut Sampled, band: Band, armed_now: &mut usize, cap: usize) {
    let Some(now_us) = os.cpu_us(t.port) else {
        return;
    };
    let share = (now_us.saturating_sub(t.cpu_us)) as f64 / SAMPLE_INTERVAL.as_micros() as f64;
    t.cpu_us = now_us;
    let seen = t.hold.load();
    let Some(want) = decide(Hold::counts(seen), share, *armed_now, cap) else {
        return;
    };
    let ok = if want {
        t.hold.arm(os, t.port, band, now_us)
    } else {
        t.hold.disarm(os, t.port, seen)
    };
    if ok {
        *armed_now = if want {
            *armed_now + 1
        } else {
            armed_now.saturating_sub(1)
        };
        // Info, not debug: these are a few lines a minute, and their absence is
        // why the 2026-09-21 host panic cannot say how many vCPUs were banded.
        log::info!(
            "[VCPU-RT] vCPU {} {} the band (share {:.0}%, {}/{} armed)",
            t.vcpuid,
            if want { "took" } else { "gave back" },
            share * 100.0,
            *armed_now,
            cap,
        );
    }
}

/// A banded vCPU thread's own handle on the band, so it can give it back without being told.
///
/// The sampler cannot be the only thing that disarms. It is an ordinary-priority thread, and a
/// banded thread cannot be preempted by one: measured directly during a saturated collapse, every
/// other thread in the worker sat at 0.0% CPU while the vCPU threads held priority 97
/// (`spikes/macos-timer-wakeup/`, `starvation-probe.sh`). The sampler is one of those threads, so
/// the guard against over-committing the machine was itself the first thing the machine stopped
/// running -- it has to be scheduled to fix the condition that stops it being scheduled. On
/// 2026-09-21 that ended in a kernel panic: `watchdog timeout: no checkins from watchdogd in 94
/// seconds`, with four vCPU threads at priority 97 and only the efficiency cluster online.
///
/// A thread that is running is, by definition, scheduled. So the thread rescues itself, from its
/// own loop, with no lock and no syscall on the path that does not act -- which is why this is a
/// plain check at every exit from the guest rather than anything cleverer.
pub struct BandGuard<O: BandOs = Mach> {
    os: O,
    port: u32,
    /// The band's record, shared with the sampler. Its baseline is when the band was handed over
    /// and this thread's CPU microseconds then, so the budget below measures this hold and not
    /// the thread's past.
    hold: Arc<Hold>,
    /// How much CPU time this thread may burn, in one hold, before it gives the band back.
    disarm_after: Duration,
}

impl<O: BandOs> BandGuard<O> {
    /// How long this thread may compute without parking. The kicker that forces a saturated
    /// guest back out to us uses it too, since a check the thread never reaches is not a check.
    pub fn kick_interval(&self) -> Duration {
        self.disarm_after
    }

    /// How long this thread has held the band, in wall time.
    fn held_for(&self) -> Duration {
        let armed_at = self.hold.armed_at.load(Ordering::Relaxed);
        if armed_at == 0 {
            return Duration::ZERO;
        }
        let now = self.os.now();
        Duration::from_nanos(abs_to_ns(now.saturating_sub(armed_at)))
    }

    /// Give the band back if this thread has burned its whole budget of CPU time holding it.
    ///
    /// Call at every exit from the guest. Disarming only ever *lowers* this thread's claim on the
    /// machine, so it needs no agreement with the sampler beyond [`Hold`]'s claim on the move.
    ///
    /// The wall clock is only a **pre-filter**, so that the common case costs no syscall: until
    /// the budget could even have been spent, nothing needs asking. Once it could have been, one
    /// `thread_info` call settles whether it actually was. That is at most one call per budget
    /// per armed vCPU — and only while armed.
    ///
    /// A thread that passes the wall gate while genuinely idle re-baselines both clocks and keeps
    /// the band. That is the case the band exists for: it must not have to re-earn, every 33 ms,
    /// a reservation it is not spending.
    pub fn check(&self) {
        // The word first: it is what makes the baseline read below this hold's.
        let seen = self.hold.load();
        if seen & PHASE != IN {
            return;
        }
        let held = self.held_for();
        if held < self.disarm_after {
            return;
        }
        // A read that fails falls THROUGH to the disarm rather than returning: a guard which
        // cannot see its own input must not be the reason a real-time reservation is kept.
        let sampled = self.os.cpu_us(self.port).map(|cpu_now| {
            let used = cpu_now.saturating_sub(self.hold.armed_cpu_us.load(Ordering::Relaxed));
            (cpu_now, Duration::from_micros(used))
        });
        if let Some((cpu_now, cpu_used)) = sampled
            && !should_disarm(true, cpu_used, held, self.disarm_after)
        {
            // Idle after all: start a fresh window rather than asking again on the next exit.
            self.hold.armed_at.store(self.os.now(), Ordering::Relaxed);
            self.hold.armed_cpu_us.store(cpu_now, Ordering::Relaxed);
            return;
        }
        if self.hold.disarm(&self.os, self.port, seen) {
            match sampled {
                Some((_, cpu_used)) => log::info!(
                    "[VCPU-RT] a vCPU gave the band back from its own loop: it burned {cpu_used:?} of CPU over a {held:?} hold"
                ),
                None => log::warn!(
                    "[VCPU-RT] a vCPU gave the band back from its own loop: its CPU time could not be read over a {held:?} hold"
                ),
            }
        }
    }
}

/// Move the *calling* thread into whichever band was asked for. Must run on the vCPU thread
/// itself, since both policies apply to the current thread.
pub fn set_realtime_band(vcpuid: u64) -> Option<BandGuard> {
    // A little vCPU takes the low QoS class instead, and never the real-time band: the band would
    // stop it being little. xnu places a time-constraint thread on an efficiency core readily (a
    // mostly idle one runs ~96% of its time there, and still wakes ~20 µs late), but only while it
    // is idle: saturated, it runs on a performance core and at priority 97. The QoS class cannot
    // hold it back. QOS_CLASS_BACKGROUND set after the policy is refused (EPERM), and set before
    // it is overridden (0.01% of saturated samples on E). A banded little vCPU would therefore
    // quietly undo the asymmetry the guest was told about, and the guest, believing the CPU is
    // slow, would keep packing work onto it (`spikes/rt-ecore-placement/`).
    if is_little(vcpuid) {
        let (class, name) = little_qos();
        let ret = unsafe { pthread_set_qos_class_self_np(class, 0) };
        if ret == 0 {
            log::info!("[VCPU-RT] vCPU {vcpuid} is little: QOS_CLASS_{name}");
        } else {
            log::warn!("[VCPU-RT] vCPU {vcpuid}: little qos class refused (errno={ret})");
        }
        return None;
    }
    if vcpu_limit().is_some_and(|limit| vcpuid >= limit) {
        return None;
    }
    let (band, heartbeat) = requested();
    let band = band?;
    let (period, computation, constraint) = match band {
        Band::Qos => {
            let ret = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
            if ret == 0 {
                log::info!("[VCPU-RT] vCPU {vcpuid} at QOS_CLASS_USER_INTERACTIVE");
            } else {
                log::warn!("[VCPU-RT] vCPU {vcpuid}: qos class refused (errno={ret})");
            }
            return None;
        }
        Band::RealTime(p, c, k) => (p, c, k),
    };

    if dynamic() {
        // Register and let the sampler decide: a vCPU that is running guest code flat out must not
        // hold a real-time reservation, whatever it is doing right now.
        let port = unsafe { mach_thread_self() };
        let hold = Arc::new(Hold::new());
        REGISTRY.lock().unwrap().push(Sampled {
            vcpuid,
            port,
            cpu_us: thread_cpu_us(port).unwrap_or(0),
            hold: Arc::clone(&hold),
        });
        start_sampler(band);
        log::info!(
            "[VCPU-RT] vCPU {vcpuid} joins the dynamic band (period={period:?} \
             computation={computation:?} constraint={constraint:?}, at most {} armed at once)",
            arm_cap(),
        );
        // The thread keeps its own way out, which is the only one that still works once every
        // core is promised away.
        return Some(BandGuard {
            os: Mach,
            port,
            hold,
            disarm_after: period * SELF_DISARM_PERIODS,
        });
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
    // A statically banded thread holds the band for its whole life by construction, so there is
    // nothing here for a guard to give back.
    None
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    /// The cap bounds how much of the machine can be promised away, in both directions.
    ///
    /// This is the guarantee that does not depend on any thread being scheduled to enforce it,
    /// which is what the 2026-09-21 host panic cost: the sampler that should have disarmed four
    /// saturated banded vCPUs was an ordinary-priority thread those same vCPUs had starved.
    #[test]
    fn the_cap_bounds_the_band_whatever_the_threads_are_doing() {
        // An idle thread takes the band while there is room.
        assert_eq!(decide(false, 0.10, 0, 2), Some(true));
        // And is refused once there is not, however idle it is.
        assert_eq!(
            decide(false, 0.00, 2, 2),
            None,
            "a full cap refuses an idle thread"
        );
        // A thread already holding one gives it back when the room has gone -- the case that
        // walks an exceeded cap back down rather than waiting for its holders to get busy.
        assert_eq!(
            decide(true, 0.00, 3, 2),
            Some(false),
            "over the cap, the band goes back"
        );
        // Saturation still disarms on its own, which is the ordinary path.
        assert_eq!(decide(true, 0.95, 1, 2), Some(false));
        // And the hysteresis is not lost: between the thresholds, nothing changes.
        assert_eq!(decide(true, 0.50, 1, 2), None);
        assert_eq!(decide(false, 0.50, 1, 2), None);
    }

    /// The cap always leaves the host something, on any machine.
    ///
    /// A cap of zero would disable the band; a cap that counted every core would be the panic
    /// again. Note what it is *not* derived from: `hw.activecpu` reports every configured core
    /// whether or not it is parked, so a cap against it could not fall when the machine idles
    /// down -- which is precisely when every vCPU is armed.
    #[test]
    fn the_cap_leaves_the_host_at_least_one_core_and_never_zero() {
        let cap = arm_cap();
        assert!(
            cap >= 1,
            "a cap of zero would turn the band off rather than bound it"
        );
        let e_cores = sysctl_u32(b"hw.perflevel1.logicalcpu\0").unwrap_or(2) as usize;
        assert!(
            cap <= e_cores.max(1),
            "the cap ({cap}) may never exceed the cluster that survives an idle machine \
             ({e_cores} efficiency cores)"
        );
    }

    /// A banded thread that stops parking takes itself out of the band.
    ///
    /// The whole point is that this needs nobody else to run: the thread is holding a core by
    /// definition, so it is the one actor guaranteed to be schedulable.
    #[test]
    fn a_thread_that_stops_parking_gives_the_band_back_itself() {
        let port = unsafe { mach_thread_self() };
        let hold = Arc::new(Hold::banded_since(
            unsafe { mach_absolute_time() },
            thread_cpu_us(port).unwrap_or(0),
        ));
        let armed = || hold.load() & PHASE == IN;
        let guard = BandGuard {
            os: Mach,
            port,
            hold: Arc::clone(&hold),
            disarm_after: Duration::from_secs(3600),
        };

        // Nothing like the budget burned yet: nothing is owed.
        guard.check();
        assert!(armed(), "a thread that has burned nothing keeps the band");

        // Burning CPU past what it promised: it gives the band back without being asked. Spin
        // rather than sleep — sleeping is exactly what must NOT count against the budget.
        let hot = BandGuard {
            os: Mach,
            port,
            hold: Arc::clone(&hold),
            disarm_after: Duration::from_millis(2),
        };
        let spin_until = std::time::Instant::now() + Duration::from_millis(20);
        while std::time::Instant::now() < spin_until {
            std::hint::spin_loop();
        }
        hot.check();
        assert!(
            !armed(),
            "a thread burning CPU flat out must not still hold a reservation"
        );

        // Idempotent: a thread that has already stood down does not keep calling the kernel.
        hot.check();
        assert!(!armed());
    }

    /// A band that was just taken is not immediately owed back.
    ///
    /// The regression this pins: judging only on the last park gave the band back within
    /// microseconds of the sampler handing it over, because the thread inherited a park from
    /// before it was armed. Measured over one 103 s boot: 464 arm/disarm pairs, median hold
    /// 6.1 ms, maximum 27.8 ms — all of them under the 33.3 ms budget, so the budget was
    /// plainly not being spent from the arm. The band was held 3.3% of the time it was meant
    /// to be held.
    #[test]
    fn the_disarm_budget_is_spent_in_cpu_time_not_wall_time() {
        let budget = Duration::from_millis(33);
        let ms = Duration::from_millis;

        // An idle vCPU: hours may pass, but it burned nothing, so it owes nothing. This is the
        // first regression — judged on the wall clock, or on a park an idle vCPU never records
        // because HVF absorbs its WFI, it gave the band straight back and the sampler re-armed it
        // a moment later. Measured across two boots: 464 pairs at a 6.1 ms median (3.3% duty),
        // then 265 arms a minute at a 45 ms median (20%). Held for real, both are one arm.
        assert!(
            !should_disarm(true, Duration::ZERO, Duration::from_secs(3600), budget),
            "a parked vCPU owes nothing, however long it has been parked"
        );
        assert!(!should_disarm(true, ms(6), ms(40), budget));
        assert!(
            !should_disarm(true, ms(32), ms(33), budget),
            "just under the budget is still under it"
        );

        // A saturated vCPU burns CPU nearly 1:1 with the wall clock: the panic case, caught at
        // the same budget it always was.
        assert!(should_disarm(true, budget, budget, budget));
        // The WORST real saturation measured under four spinners — 33.55 ms over 38.87 ms, 86%.
        // A threshold set by intuition at 90% would have let this one through, and three of its
        // seven siblings with it.
        assert!(
            should_disarm(true, ms(34), ms(39), budget),
            "a vCPU at 86% of a core is spinning, whatever a round number suggests"
        );
        // And the worst of all eight, at 73.7%.
        assert!(should_disarm(true, ms(35), ms(47), budget));

        // The second regression: an absolute budget over an UNBOUNDED window is an arbitrarily
        // low bar, because this check only runs at exits from the guest. Every one of these was
        // observed disarming a real vCPU that was doing essentially nothing.
        assert!(
            !should_disarm(true, ms(35), ms(413), budget),
            "8.5% of a core is not computing flat out"
        );
        assert!(!should_disarm(
            true,
            ms(43),
            Duration::from_millis(1751),
            budget
        ));
        assert!(
            !should_disarm(true, ms(35), Duration::from_secs(97), budget),
            "0.0% of a core least of all"
        );

        // Not armed: nothing to give back, and no syscall on the path that does not act.
        assert!(!should_disarm(
            false,
            Duration::from_secs(5),
            Duration::from_secs(5),
            budget
        ));
    }

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
        const { assert!(ARM_BELOW < DISARM_ABOVE) };
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

/// loom model of the sampler and a vCPU's own [`BandGuard`] deciding about the same thread at once.
///
/// Run with `RUSTFLAGS="--cfg loom" cargo test --release --lib loom_model` in this crate (or
/// `cargo xtask check loom third_party/libkrun/src/vmm`). The thread's arm state is loom's, and so
/// is the stand-in kernel's lock, so every kernel call is a point loom can interleave against the
/// other party.
///
/// The thread starts banded and has computed flat out for its whole hold, so both parties want
/// the band back. The sampler takes it back and then, the thread having gone quiet, hands it over
/// again: a second hold, on which nothing has been burned. The guard checks once, concurrently.
/// After each run the kernel must agree with the flag everyone else reads — a thread banded while
/// flagged disarmed is one nothing will ever take back out, the 2026-09-21 panic's precondition —
/// and the guard must not have taken back the second hold, which it never judged.
#[cfg(all(test, loom))]
mod loom_model {
    use super::{Arc, Band, BandGuard, BandOs, Hold, IN, PHASE, Sampled, ns_to_abs, sample_one};
    use loom::sync::Mutex;
    use loom::thread;
    use std::time::Duration;

    const PORT: u32 = 7;
    /// What the thread's CPU clock reads throughout: a second, against a 100 ms hold.
    const CPU_US: u64 = 1_000_000;
    const BAND: Band = Band::RealTime(
        Duration::from_micros(16_667),
        Duration::from_micros(1_000),
        Duration::from_micros(2_000),
    );

    #[derive(Default)]
    struct State {
        banded: bool,
        /// How many times the band has been handed over; the thread starts in the first.
        hold: u64,
        /// Which holds the guard took back.
        guard_took: Vec<u64>,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Who {
        Sampler,
        Guard,
    }

    /// One party's view of the stand-in kernel: a clock that stands still, a CPU clock that
    /// reads [`CPU_US`], and the band.
    #[derive(Clone)]
    struct Kernel(Arc<Mutex<State>>, Who);

    fn now() -> u64 {
        u64::from(ns_to_abs(10_000_000_000))
    }

    impl BandOs for Kernel {
        fn now(&self) -> u64 {
            now()
        }
        fn cpu_us(&self, _port: u32) -> Option<u64> {
            Some(CPU_US)
        }
        fn set_band(&self, _port: u32, _band: Band) -> bool {
            let mut s = self.0.lock().unwrap();
            s.banded = true;
            s.hold += 1;
            true
        }
        fn set_timeshare(&self, _port: u32) -> bool {
            let mut s = self.0.lock().unwrap();
            if self.1 == Who::Guard && s.banded {
                let hold = s.hold;
                s.guard_took.push(hold);
            }
            s.banded = false;
            true
        }
    }

    #[test]
    fn the_sampler_and_the_guard_decide_at_once() {
        loom::model(|| {
            let state = Arc::new(Mutex::new(State {
                banded: true,
                hold: 1,
                guard_took: Vec::new(),
            }));
            // Banded 100 ms ago, having burned nothing then.
            let hold = Arc::new(Hold::banded_since(
                now() - u64::from(ns_to_abs(100_000_000)),
                0,
            ));

            let sampler = {
                let kernel = Kernel(state.clone(), Who::Sampler);
                let mut t = Sampled {
                    vcpuid: 0,
                    port: PORT,
                    cpu_us: 0,
                    hold: hold.clone(),
                };
                thread::spawn(move || {
                    let mut armed_now = 1;
                    // Flat out since the last sample: the band goes back.
                    sample_one(&kernel, &mut t, BAND, &mut armed_now, 1);
                    // Quiet since: it is handed over again.
                    sample_one(&kernel, &mut t, BAND, &mut armed_now, 1);
                })
            };
            let guard = BandGuard {
                os: Kernel(state.clone(), Who::Guard),
                port: PORT,
                hold: hold.clone(),
                disarm_after: Duration::from_millis(33),
            };
            guard.check();
            sampler.join().unwrap();

            let s = state.lock().unwrap();
            assert_eq!(
                s.banded,
                hold.load() & PHASE == IN,
                "the kernel and the flag disagree on whether the thread holds the band"
            );
            assert!(
                !s.guard_took.contains(&2),
                "the guard took back a hold it never judged (took {:?})",
                s.guard_took
            );
        });
    }
}
