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
//!
//! `LIMINA_VCPU_RT` is still read as the old spelling of `rt`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
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
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    fn thread_policy_set(thread: u32, flavor: u32, policy: *mut u32, count: u32) -> i32;
    fn mach_absolute_time() -> u64;
    fn mach_wait_until(deadline: u64) -> i32;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
}

/// `QOS_CLASS_USER_INTERACTIVE` from `sys/qos.h`.
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

const DEFAULT_HEARTBEAT: Duration = Duration::from_millis(250);
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

/// The policy and the heartbeat interval, from the environment. `None` for either means off.
fn requested() -> (Option<Band>, Option<Duration>) {
    let raw = std::env::var("LIMINA_VCPU_SCHED")
        .or_else(|_| std::env::var("LIMINA_VCPU_RT"))
        .unwrap_or_default();
    let raw = raw.trim();
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
    }
}

/// Move the *calling* thread into whichever band was asked for. Must run on the vCPU thread
/// itself, since both policies apply to the current thread.
pub fn set_realtime_band(vcpuid: u64) {
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
