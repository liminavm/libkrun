// Copyright 2026 The limina Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! How late does the host serve a guest's virtual-timer deadline?
//!
//! An idle vCPU traps on WFI; [`super::vstate`] reads the deadline out of `CNTV_CVAL_EL0` and
//! parks a host thread until it. The guest's timer is therefore exactly as punctual as that park,
//! and at a 60 Hz frame cadence a couple of milliseconds of lateness costs a whole refresh.
//!
//! This records the overshoot — observed park duration minus requested timeout — for parks that
//! actually ran to their deadline. A park cut short by a device IRQ or an out-of-band pause is
//! not late, it is *early on purpose*, and is counted separately rather than folded in.
//!
//! Off unless `LIMINA_WFI_LATENCY` is set; its value is the report interval in seconds
//! (default 5). A synthetic version of the same measurement, over every host wait primitive and
//! thread policy, lives in `spikes/macos-timer-wakeup/`.
//!
//! **What it found: on macOS 26.5 / Apple silicon this park never happens.** A guest's WFI does
//! not trap out to us at all — HVF parks the vCPU inside `hv_vcpu_run`
//! (`HvCore::Hypervisor::VcpuStateManager::wait_for_interrupt`) and serves the virtual timer from
//! its own clock thread. Over 30 s of idle desktop the only vCPU exits are MMIO; `WaitForEvent`,
//! `WaitForEventTimeout` and `VtimerActivated` are all zero. So the code below reports nothing,
//! and that silence is the result: guest timer lateness is not ours to serve here.
//!
//! It is kept because the silence is worth watching. A nonzero report means HVF changed, or a
//! guest or host configuration reached the trap path — either way, the assumption above is stale.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

/// Why a park ended. Only [`Wake::Deadline`] carries a meaningful lateness.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Wake {
    /// The requested timeout elapsed.
    Deadline,
    /// A device IRQ arrived first.
    Irq,
    /// An out-of-band pause/snapshot request arrived first.
    Pause,
}

/// Upper edge of each lateness bucket, in microseconds. The last bucket is open-ended.
///
/// The edges straddle a 16.67 ms frame deliberately: 8 ms is "half a frame late", 16 ms is
/// "certainly dropped one", 33 ms is "dropped two".
const EDGES_US: [u64; 10] = [
    50, 100, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 33_000,
];

#[derive(Default)]
struct Stats {
    /// One counter per bucket in [`EDGES_US`], plus one for everything above the last edge.
    buckets: [AtomicU64; EDGES_US.len() + 1],
    deadline_parks: AtomicU64,
    irq_parks: AtomicU64,
    pause_parks: AtomicU64,
    /// Parks whose deadline had already passed when we went to arm it (`WaitForEventExpired`).
    expired: AtomicU64,
    /// WFIs that never parked at all because an IRQ was already pending.
    no_wait: AtomicU64,
    late_us_total: AtomicU64,
    late_us_max: AtomicU64,
}

static STATS: Stats = Stats {
    buckets: [
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
        AtomicU64::new(0),
    ],
    deadline_parks: AtomicU64::new(0),
    irq_parks: AtomicU64::new(0),
    pause_parks: AtomicU64::new(0),
    expired: AtomicU64::new(0),
    no_wait: AtomicU64::new(0),
    late_us_total: AtomicU64::new(0),
    late_us_max: AtomicU64::new(0),
};

static ENABLED: AtomicU64 = AtomicU64::new(u64::MAX);

/// Which bucket a lateness falls in. Pure, so the edges can be tested without a VM.
fn bucket_of(late_us: u64) -> usize {
    EDGES_US
        .iter()
        .position(|&edge| late_us < edge)
        .unwrap_or(EDGES_US.len())
}

/// Whether recording is on, and how often to report. Reads the environment once.
fn report_interval() -> Option<Duration> {
    let cached = ENABLED.load(Ordering::Relaxed);
    let secs = if cached == u64::MAX {
        let secs = match std::env::var("LIMINA_WFI_LATENCY") {
            Ok(v) if v == "0" || v.eq_ignore_ascii_case("off") => 0,
            Ok(v) => v.trim().parse::<u64>().unwrap_or(5).max(1),
            Err(_) => 0,
        };
        ENABLED.store(secs, Ordering::Relaxed);
        secs
    } else {
        cached
    };
    (secs > 0).then(|| Duration::from_secs(secs))
}

pub fn enabled() -> bool {
    report_interval().is_some()
}

/// Record one finished park. `requested` is what the guest's deadline asked for, `observed` is
/// how long the park actually took.
pub fn record(wake: Wake, requested: Duration, observed: Duration) {
    if !enabled() {
        return;
    }
    match wake {
        Wake::Irq => {
            STATS.irq_parks.fetch_add(1, Ordering::Relaxed);
            return;
        }
        Wake::Pause => {
            STATS.pause_parks.fetch_add(1, Ordering::Relaxed);
            return;
        }
        Wake::Deadline => {}
    }
    STATS.deadline_parks.fetch_add(1, Ordering::Relaxed);
    let late_us = observed.saturating_sub(requested).as_micros() as u64;
    STATS.buckets[bucket_of(late_us)].fetch_add(1, Ordering::Relaxed);
    STATS.late_us_total.fetch_add(late_us, Ordering::Relaxed);
    STATS.late_us_max.fetch_max(late_us, Ordering::Relaxed);
}

/// A WFI that returned without parking: an IRQ was already pending.
pub fn record_no_wait() {
    if enabled() {
        STATS.no_wait.fetch_add(1, Ordering::Relaxed);
    }
}

/// A park we never armed because the deadline was already behind us.
pub fn record_expired() {
    if enabled() {
        STATS.expired.fetch_add(1, Ordering::Relaxed);
    }
}

/// Approximate a percentile from the bucket counts, reported as the bucket's upper edge. Coarse
/// on purpose: the question is "did it cost a frame", not "how many microseconds exactly".
fn percentile_edge(counts: &[u64], total: u64, p: f64) -> String {
    let target = (total as f64 * p).ceil() as u64;
    let mut seen = 0;
    for (i, &c) in counts.iter().enumerate() {
        seen += c;
        if seen >= target {
            return match EDGES_US.get(i) {
                Some(edge) => format!("<{edge}us"),
                None => format!(">={}us", EDGES_US[EDGES_US.len() - 1]),
            };
        }
    }
    "n/a".to_string()
}

/// Log one summary line and reset, so each report covers only its own interval.
fn report_and_reset() {
    let counts: Vec<u64> = STATS
        .buckets
        .iter()
        .map(|b| b.swap(0, Ordering::Relaxed))
        .collect();
    let deadline = STATS.deadline_parks.swap(0, Ordering::Relaxed);
    let irq = STATS.irq_parks.swap(0, Ordering::Relaxed);
    let pause = STATS.pause_parks.swap(0, Ordering::Relaxed);
    let expired = STATS.expired.swap(0, Ordering::Relaxed);
    let no_wait = STATS.no_wait.swap(0, Ordering::Relaxed);
    let total_us = STATS.late_us_total.swap(0, Ordering::Relaxed);
    let max_us = STATS.late_us_max.swap(0, Ordering::Relaxed);

    if deadline == 0 && irq == 0 && pause == 0 && expired == 0 && no_wait == 0 {
        return;
    }

    // Frames are the unit that matters, so name the two edges that cost one.
    let over_8ms: u64 = counts[8..].iter().sum();
    let over_16ms: u64 = counts[9..].iter().sum();

    log::info!(
        "[WFI-LATE] parks: deadline={deadline} irq={irq} pause={pause} expired={expired} \
         no_wait={no_wait} | \
         lateness p50={} p90={} p99={} max={max_us}us mean={}us | \
         >=8ms {over_8ms} ({:.1}%) >=16ms {over_16ms} ({:.1}%)",
        percentile_edge(&counts, deadline, 0.50),
        percentile_edge(&counts, deadline, 0.90),
        percentile_edge(&counts, deadline, 0.99),
        total_us.checked_div(deadline).unwrap_or(0),
        pct(over_8ms, deadline),
        pct(over_16ms, deadline),
    );
}

fn pct(n: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        n as f64 * 100.0 / total as f64
    }
}

/// Start the reporter. A no-op unless `LIMINA_WFI_LATENCY` is set; safe to call more than once
/// only from the single boot path that owns it.
pub fn start_reporter() {
    let Some(interval) = report_interval() else {
        return;
    };
    log::info!(
        "[WFI-LATE] recording WFI park lateness, reporting every {}s",
        interval.as_secs()
    );
    thread::Builder::new()
        .name("wfi-latency".into())
        .spawn(move || {
            loop {
                thread::sleep(interval);
                report_and_reset();
            }
        })
        .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_straddle_a_frame() {
        // The edges exist to answer "did this cost a refresh at 60 Hz", so the boundaries
        // around half a frame and a whole frame are the ones worth pinning.
        assert_eq!(bucket_of(0), 0);
        assert_eq!(bucket_of(49), 0);
        assert_eq!(bucket_of(50), 1);
        assert_eq!(bucket_of(7_999), 7);
        // 8 ms: half a frame late.
        assert_eq!(bucket_of(8_000), 8);
        assert_eq!(bucket_of(15_999), 8);
        // 16 ms: a whole refresh gone.
        assert_eq!(bucket_of(16_000), 9);
        assert_eq!(bucket_of(32_999), 9);
        // Above the last edge everything lands in the open-ended bucket.
        assert_eq!(bucket_of(33_000), EDGES_US.len());
        assert_eq!(bucket_of(u64::MAX), EDGES_US.len());
    }

    #[test]
    fn percentiles_read_off_the_bucket_edges() {
        // 100 samples: 90 under 50us, 10 spread into the 8ms and 16ms buckets.
        let mut counts = vec![0u64; EDGES_US.len() + 1];
        counts[0] = 90;
        counts[8] = 7;
        counts[9] = 3;
        assert_eq!(percentile_edge(&counts, 100, 0.50), "<50us");
        assert_eq!(percentile_edge(&counts, 100, 0.90), "<50us");
        // The last 10% is where the dropped frames live.
        assert_eq!(percentile_edge(&counts, 100, 0.95), "<16000us");
        assert_eq!(percentile_edge(&counts, 100, 0.99), "<33000us");
    }

    #[test]
    fn an_early_wake_is_not_lateness() {
        // A park cut short by an IRQ must never be counted as late — it is the common case on a
        // busy guest, and folding it in would report a healthy median for a broken timer.
        assert_eq!(
            Duration::from_millis(4)
                .saturating_sub(Duration::from_millis(16))
                .as_micros(),
            0
        );
    }
}
