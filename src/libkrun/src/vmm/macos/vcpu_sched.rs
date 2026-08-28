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
//! Off unless `LIMINA_VCPU_RT` is set. `1` takes the defaults; `period,computation,constraint` in
//! microseconds overrides them. A vCPU is not a 2 ms audio callback — it runs guest code for as
//! long as the guest wants — so this stays opt-in until the overrun behaviour is understood.

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
}

fn ns_to_abs(ns: u64) -> u32 {
    let mut tb = MachTimebaseInfo::default();
    if unsafe { mach_timebase_info(&mut tb) } != 0 || tb.numer == 0 {
        return ns as u32;
    }
    let abs = (ns as u128 * tb.denom as u128) / tb.numer as u128;
    abs.min(u32::MAX as u128) as u32
}

/// The three durations, from the environment or the defaults. `None` when the band is off.
fn requested() -> Option<(Duration, Duration, Duration)> {
    let v = std::env::var("LIMINA_VCPU_RT").ok()?;
    let v = v.trim();
    if v.is_empty() || v == "0" || v.eq_ignore_ascii_case("off") {
        return None;
    }
    let parsed: Vec<u64> = v.split(',').filter_map(|p| p.trim().parse().ok()).collect();
    if parsed.len() == 3 {
        return Some((
            Duration::from_micros(parsed[0]),
            Duration::from_micros(parsed[1]),
            Duration::from_micros(parsed[2]),
        ));
    }
    Some((DEFAULT_PERIOD, DEFAULT_COMPUTATION, DEFAULT_CONSTRAINT))
}

/// Move the *calling* thread into the real-time band. Must run on the vCPU thread itself, since
/// the policy applies to `mach_thread_self()`.
pub fn set_realtime_band(vcpuid: u64) {
    let Some((period, computation, constraint)) = requested() else {
        return;
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
             computation={computation:?} constraint={constraint:?})"
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
    fn the_band_is_off_without_the_variable() {
        // Safety net for the opt-in: a vCPU thread is not an audio callback, and nothing should
        // land in the real-time band by accident.
        unsafe { std::env::remove_var("LIMINA_VCPU_RT") };
        assert!(requested().is_none());
        unsafe { std::env::set_var("LIMINA_VCPU_RT", "0") };
        assert!(requested().is_none());
        unsafe { std::env::set_var("LIMINA_VCPU_RT", "1") };
        assert_eq!(
            requested(),
            Some((DEFAULT_PERIOD, DEFAULT_COMPUTATION, DEFAULT_CONSTRAINT))
        );
        unsafe { std::env::set_var("LIMINA_VCPU_RT", "8000,500,900") };
        assert_eq!(
            requested(),
            Some((
                Duration::from_micros(8_000),
                Duration::from_micros(500),
                Duration::from_micros(900)
            ))
        );
        unsafe { std::env::remove_var("LIMINA_VCPU_RT") };
    }
}
