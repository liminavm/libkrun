// limina wake-chain probe (`LIMINA_WAKE_PROBE=1`): stamp the guest doorbell so the device
// worker can measure how long it took to get scheduled.
//
// gnome-shell-rs (docs/fork/venus-cost.md §9.4) measures the first venus submit after an idle
// gap at ~1 ms against ~0.05 ms back-to-back. virglrenderer's LIMINA_RING_WAKE_PROFILE showed
// the hops it can see are cheap — cnd_signal -> ring thread running averages 6-74 us and
// ring-thread-running -> decode is ~0 — so the bulk of that millisecond is spent BEFORE
// vkr_ring_notify is ever called, somewhere in guest ioctl -> VM exit -> this worker.
//
// The doorbell write happens on a vCPU thread inside the MMIO exit handler (`mmio.rs`, queue
// notify at reg 0x50); the work happens on the device's worker thread, woken via the queue
// eventfd. Nothing connects those two, so this does: the vCPU stamps CLOCK_MONOTONIC into a
// cell, the worker reads it when it wakes and gets `kick -> wake`, the one hop that spans the
// exit boundary.
//
// Only one fd is watched (whoever registers), because only the venus control queue is in
// question and a per-fd table would put a lookup on the vCPU path. Registration is a plain
// store: with the probe off the fd stays -1 and the notify handler's only cost is a relaxed
// load that never matches.

use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::OnceLock;

static WATCHED_FD: AtomicI32 = AtomicI32::new(-1);
static KICK_NS: AtomicU64 = AtomicU64::new(0);
/// Kicks that arrived while a previous one was still unconsumed. The worker coalesces (one
/// eventfd wake can drain many descriptors), so a nonzero count here means `kick -> wake`
/// samples are attributed to the OLDEST unconsumed kick — which is the honest reading, but
/// the count has to be visible or the averages look worse than they are.
static COALESCED: AtomicU64 = AtomicU64::new(0);

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("LIMINA_WAKE_PROBE").as_deref(),
            Ok("1") | Ok("calib")
        )
    })
}

/// `LIMINA_WAKE_PROBE=calib` additionally runs the fixed-work control (see `calibrate`). It is
/// opt-in on top of the probe because it burns ~0.1 ms of the worker's critical section per
/// doorbell, which is small but not nothing when the thing under study is a ~1 ms latency.
fn calib_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LIMINA_WAKE_PROBE").as_deref() == Ok("calib"))
}

pub fn now_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, properly aligned timespec we own for the duration of the call.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Watch this queue eventfd's doorbell. Called by the worker once, at activation.
pub fn watch(fd: RawFd) {
    WATCHED_FD.store(fd, Ordering::Relaxed);
}

/// Called from the MMIO queue-notify handler, on a vCPU thread, before the eventfd write.
/// Keeps the OLDEST unconsumed stamp: if the worker hasn't run yet, the wait it is about to
/// report started at the first kick, not the last.
pub fn kick(fd: RawFd) {
    if fd != WATCHED_FD.load(Ordering::Relaxed) {
        return;
    }
    if KICK_NS
        .compare_exchange(0, now_ns(), Ordering::Release, Ordering::Relaxed)
        .is_err()
    {
        COALESCED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Called from the worker when it wakes for the watched fd. Returns the doorbell stamp and
/// clears the cell, or `None` if this wake had no doorbell behind it (a spurious wake, or a
/// kick the worker already consumed in a previous drain).
pub fn consume_kick() -> Option<u64> {
    match KICK_NS.swap(0, Ordering::Acquire) {
        0 => None,
        ns => Some(ns),
    }
}

pub fn take_coalesced() -> u64 {
    COALESCED.swap(0, Ordering::Relaxed)
}

/// Idle-gap buckets, identical to virglrenderer's LIMINA_RING_WAKE_PROFILE so the two tables
/// can be read side by side: the effect being chased grows with how long the path was idle,
/// and both ends have to be cut on that same axis for the comparison to mean anything.
const BUCKET_BOUNDS_NS: [u64; 3] = [1_000_000, 4_000_000, 16_000_000];
const BUCKET_NAMES: [&str; 4] = ["<1ms  ", "1-4ms ", "4-16ms", ">=16ms"];

#[derive(Default, Clone, Copy)]
struct Bucket {
    n: u64,
    kick_wake_sum: u64,
    kick_wake_max: u64,
    wake_drain_sum: u64,
    wake_drain_max: u64,
    drain_sig_sum: u64,
    drain_sig_max: u64,
    calib_sum: u64,
    calib_max: u64,
}

impl Bucket {
    fn add(&mut self, kick_wake: u64, wake_drain: u64, drain_sig: u64) {
        self.n += 1;
        self.kick_wake_sum += kick_wake;
        self.kick_wake_max = self.kick_wake_max.max(kick_wake);
        self.wake_drain_sum += wake_drain;
        self.wake_drain_max = self.wake_drain_max.max(wake_drain);
        self.drain_sig_sum += drain_sig;
        self.drain_sig_max = self.drain_sig_max.max(drain_sig);
    }
}

/// Fixed-work control. Runs the identical arithmetic every time, timed on the worker thread
/// right after the wake it is attributed to, and touches no lock, syscall or shared line — so
/// its only input is how fast this core happens to be running at that instant.
///
/// This is the discriminator the rest of the table needs. Every measured hop doing MORE work
/// after a longer idle gap has two possible explanations: something in the path is
/// idle-dependent, or the machine itself is slower coming out of idle (DVFS ramp, E-core
/// placement, core wake). A hop can't tell those apart; this can, because it has no path.
/// If `calib` is flat across the idle buckets, the growth is real work and worth chasing in
/// the code. If `calib` scales with the buckets the same way the hops do, the hops are just
/// measuring the CPU and no amount of restructuring them will help.
fn calibrate() -> u64 {
    let t0 = now_ns();
    let mut x: u64 = 0x9e3779b97f4a7c15;
    for i in 0..20_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    std::hint::black_box(x);
    now_ns().saturating_sub(t0)
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

/// Worker-thread accumulator for the VMM half of the wake chain. One instance per activation,
/// owned by the worker loop, so no synchronization beyond the doorbell cell above.
pub struct Profile {
    buckets: [Bucket; 4],
    /// Wakes on the watched fd with no doorbell stamp behind them: either spurious, or a kick
    /// the previous drain already covered. High counts mean the per-wake numbers describe
    /// fewer doorbells than there were.
    no_kick: u64,
    last_kick_ns: u64,
    report_at_ns: u64,
}

impl Profile {
    pub fn new() -> Option<Self> {
        if !enabled() {
            return None;
        }
        Some(Self {
            buckets: [Bucket::default(); 4],
            no_kick: 0,
            last_kick_ns: 0,
            report_at_ns: now_ns() + 5_000_000_000,
        })
    }

    /// Record one doorbell's worth of the VMM path. `kick_ns` is the stamp taken on the vCPU
    /// thread; the rest are taken by the worker around its own work.
    pub fn record(&mut self, kick_ns: u64, wake_ns: u64, drained_ns: u64, signaled_ns: u64) {
        let gap = kick_ns.saturating_sub(self.last_kick_ns);
        self.last_kick_ns = kick_ns;

        let idx = BUCKET_BOUNDS_NS.iter().position(|&b| gap < b).unwrap_or(3);
        let b = &mut self.buckets[idx];
        b.add(
            wake_ns.saturating_sub(kick_ns),
            drained_ns.saturating_sub(wake_ns),
            signaled_ns.saturating_sub(drained_ns),
        );
        // After the hops are recorded, so the control never inflates what it is controlling for.
        if calib_enabled() {
            let c = calibrate();
            b.calib_sum += c;
            b.calib_max = b.calib_max.max(c);
        }
    }

    pub fn record_no_kick(&mut self) {
        self.no_kick += 1;
    }

    /// Emit one line per non-empty bucket every ~5s, then reset. Called from the worker loop.
    pub fn maybe_report(&mut self) {
        let now = now_ns();
        if now < self.report_at_ns {
            return;
        }
        self.report_at_ns = now + 5_000_000_000;

        let coalesced = take_coalesced();
        for (i, b) in self.buckets.iter().enumerate() {
            if b.n == 0 {
                continue;
            }
            let n = b.n as f64;
            eprintln!(
                "[GPUWAKE idle {}] n={} kick->wake avg {:.3} ms max {:.3} ms | \
                 wake->drained avg {:.3} ms max {:.3} ms | drained->signal avg {:.3} ms max {:.3} ms \
                 | calib avg {:.3} ms max {:.3} ms",
                BUCKET_NAMES[i],
                b.n,
                ms(b.kick_wake_sum) / n,
                ms(b.kick_wake_max),
                ms(b.wake_drain_sum) / n,
                ms(b.wake_drain_max),
                ms(b.drain_sig_sum) / n,
                ms(b.drain_sig_max),
                ms(b.calib_sum) / n,
                ms(b.calib_max),
            );
        }
        if self.no_kick > 0 || coalesced > 0 {
            eprintln!("[GPUWAKE] no_kick={} coalesced={coalesced}", self.no_kick);
        }
        self.buckets = [Bucket::default(); 4];
        self.no_kick = 0;
    }
}
