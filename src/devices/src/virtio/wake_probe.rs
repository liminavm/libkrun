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

/// A log2-spaced latency histogram, ~12% wide per bin. Reports p50/p95 rather than a mean,
/// because the numbers this is compared against — gnome-shell-rs's idle sweep — are medians,
/// and because on a real desktop the traffic is heavy-tailed enough that a mean describes
/// nothing anyone experiences. `max` is kept alongside: one bad sample is worth seeing, it
/// just must not be mistaken for the distribution.
#[derive(Clone, Copy)]
struct Hist {
    /// Bin i covers [2^(i/8), 2^((i+1)/8)) ns, so 8 bins per octave up to ~4 s.
    bins: [u32; 256],
    n: u64,
    max: u64,
}

impl Default for Hist {
    fn default() -> Self {
        Self {
            bins: [0; 256],
            n: 0,
            max: 0,
        }
    }
}

impl Hist {
    fn add(&mut self, ns: u64) {
        self.n += 1;
        self.max = self.max.max(ns);
        // 8 bins per octave: bin = 8*log2(ns), read off the exponent plus 3 mantissa bits.
        let bin = if ns < 2 {
            0
        } else {
            let oct = 63 - ns.leading_zeros() as usize;
            let mant = ((ns >> (oct.saturating_sub(3))) & 0x7) as usize;
            (oct * 8 + mant).min(255)
        };
        self.bins[bin] += 1;
    }

    /// Upper edge of the bin holding the `p`th percentile. Coarse by construction (~12%), which
    /// is far finer than the effect being measured.
    fn pct(&self, p: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let target = (self.n as f64 * p).ceil() as u64;
        let mut seen = 0u64;
        for (i, &c) in self.bins.iter().enumerate() {
            seen += c as u64;
            if seen >= target {
                return 2f64.powf((i + 1) as f64 / 8.0);
            }
        }
        self.max as f64
    }
}

#[derive(Default, Clone, Copy)]
struct Bucket {
    n: u64,
    kick_wake: Hist,
    wake_drain: Hist,
    drain_sig: Hist,
    calib: Hist,
}

impl Bucket {
    fn add(&mut self, kick_wake: u64, wake_drain: u64, drain_sig: u64) {
        self.n += 1;
        self.kick_wake.add(kick_wake);
        self.wake_drain.add(wake_drain);
        self.drain_sig.add(drain_sig);
    }
}

// ---------------------------------------------------------------------------------------
// The return half: host raises the used-queue IRQ -> the guest's handler acknowledges it.
//
// This looked at first like it needed a guest-side stamp, because `drained->signal` only times
// the CALL that raises the interrupt, not when the guest sees it. It does not — but the obvious
// hook is the wrong one, which is worth recording.
//
// The obvious hook is the guest's GIC acknowledge (ICC_IAR1_EL1), which traps to
// `legacy/vcpu.rs::handle_sysreg_read`. It produced ZERO samples, because macOS 26 gives us
// Apple's IN-KERNEL GIC (`HvfGicV3`, `hv_gic_set_spi`): injection and acknowledge both happen
// inside HVF and never come out to us. That hook only fires on the userspace `GicV3` fallback.
//
// The hook that always works is one layer up, in the transport. A virtio-mmio driver's ISR
// handler acknowledges by WRITING InterruptACK (reg 0x64), and MMIO always traps. So the
// measurement is: raise the used-queue interrupt -> the guest's virtio-gpu interrupt handler
// runs and acks. That spans the vCPU coming back (out of WFI or hv_vcpu_run), the injection,
// guest IRQ entry, and the driver's handler reaching its ack — every host-adjacent part of the
// return path. Attribution is exact because the ack is per-device: it is keyed on the same
// queue eventfd the doorbell is.
//
// The GIC hook is kept anyway, for the one thing the transport ack cannot tell us: whether the
// vCPU was parked. `set_irq_common` knows (`VcpuStatus::Waiting`, woken through its channel,
// vs Running, kicked out with hv_vcpus_exit), and those are very different costs — if the
// idle-gap effect lives anywhere on the host, a parked vCPU is where to expect it. That split
// is only available on the userspace GIC, and is reported only there.

static IRQ_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static IRQ_BUCKET: AtomicI32 = AtomicI32::new(-1);
static IRQ_RAISE_NS: AtomicU64 = AtomicU64::new(0);
static IRQ_PARKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// [bucket][parked] -> (count, ns sum, ns max). Written from the vCPU thread on ack, read by
/// the worker thread when it reports; atomics rather than the `Profile`'s plain fields because
/// the two ends are different threads.
static IRQ_STATS: [[(AtomicU64, AtomicU64, AtomicU64); 2]; 4] =
    [const { [const { (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)) }; 2] }; 4];
/// Raises that landed on a line the guest had not yet serviced. Expected to dwarf the sample
/// count — it is the coalescing ratio, not an error.
static IRQ_COALESCED: AtomicU64 = AtomicU64::new(0);
/// Whether the return half is measurable at all. macOS 26 gives us Apple's IN-KERNEL GIC
/// (`HvfGicV3`, `hv_gic_set_spi`), which injects and acknowledges entirely inside HVF: neither
/// `set_irq_common` nor the ICC_IAR1_EL1 trap is on that path, so there is nothing to hook.
/// Only the userspace `GicV3` fallback can be timed. Defaults to false so the common case
/// reports the hop as UNAVAILABLE rather than silently printing no column — an absent number
/// that looks like a quiet path is how you talk yourself into a wrong conclusion.
static SOFTWARE_GIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn note_software_gic() {
    SOFTWARE_GIC.store(true, Ordering::Relaxed);
}

/// Arm the return-path measurement, immediately before the used-queue interrupt is raised.
///
/// Keeps the OLDEST unacked raise, like the doorbell cell does. The guest coalesces heavily —
/// the used-queue interrupt is one level-triggered SPI, so one ISR entry and one ack can cover
/// a few hundred raises, and in a busy window raises outnumber acks ~250:1. Only the raise that
/// found the line idle actually causes an ISR entry; the rest are absorbed into an interrupt
/// already pending. Overwriting on each raise (the first version of this) measured "last raise
/// before the ack -> ack", which is a fraction of a delivery and reads about 5x too fast.
pub fn irq_arm(bucket: usize) {
    if IRQ_ARMED.swap(true, Ordering::AcqRel) {
        // Already armed: this raise landed on a line the guest has not serviced yet.
        IRQ_COALESCED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    IRQ_BUCKET.store(bucket as i32, Ordering::Relaxed);
    IRQ_PARKED.store(false, Ordering::Relaxed);
    IRQ_RAISE_NS.store(now_ns(), Ordering::Release);
}

/// Called from `set_irq_common` (userspace GIC only) with whether the vCPU was parked. Records
/// nothing by itself; it just labels the sample the transport ack will close.
pub fn irq_raised(_irq: u32, parked: bool) {
    if parked && IRQ_ARMED.load(Ordering::Relaxed) {
        IRQ_PARKED.store(true, Ordering::Relaxed);
    }
}

/// Called from the virtio-mmio InterruptACK write (reg 0x64) with the acking device's first
/// queue eventfd. Closes the sample when it is the device being watched.
pub fn irq_acked_transport(fd: RawFd) {
    if fd != WATCHED_FD.load(Ordering::Relaxed) {
        return;
    }
    // Close first, so a raise racing this one re-arms cleanly rather than being counted twice.
    if !IRQ_ARMED.swap(false, Ordering::AcqRel) {
        return;
    }
    let bucket = IRQ_BUCKET.load(Ordering::Relaxed);
    if bucket < 0 {
        return;
    }
    let dt = now_ns().saturating_sub(IRQ_RAISE_NS.load(Ordering::Acquire));
    let slot = &IRQ_STATS[bucket as usize][IRQ_PARKED.load(Ordering::Relaxed) as usize];
    slot.0.fetch_add(1, Ordering::Relaxed);
    slot.1.fetch_add(dt, Ordering::Relaxed);
    slot.2.fetch_max(dt, Ordering::Relaxed);
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

// ---------------------------------------------------------------------------------------
// Outlier attribution.
//
// The 5 s buckets report a max per hop, which is enough to know a stall happened and useless
// for finding out why. The first dogfood run showed exactly that: two windows where a single
// control-queue drain took 25.0 ms and 18.7 ms while the median in those same windows was
// 0.012 ms — long enough to be a visible hitch, with nothing to say about the cause.
//
// So: time every command inside the drain, keep the worst one, and when a drain overruns the
// threshold emit ONE line immediately naming it. The wall-clock stamp is CLOCK_REALTIME rather
// than MONOTONIC on purpose — the guest's clock is anchored to the host's (libkrun 0088 +
// control-plane TimeSync), so a realtime stamp can be lined up against the compositor's own
// frame log, which a host monotonic stamp cannot.

/// Milliseconds a single drain may take before it is reported individually. Chosen below one
/// 60 Hz frame so anything capable of dropping a frame is caught, and far above the ~0.04 ms
/// median so normal traffic is silent.
const OUTLIER_MS: f64 = 5.0;

/// CLOCK_REALTIME seconds, for stamping every line this module prints.
///
/// On EVERY line, not just the outliers. The periodic bucket lines shipped without one, and the
/// first thing that cost was a wrong conclusion: a burst of 163 slow RESOURCE_FLUSHes read as
/// "the desktop is stalling on every frame" when converting the stamps against VM start showed
/// they were GRUB and the GNOME session coming up. The outlier lines could be checked because
/// they carried a stamp; the bucket lines could not, and an earlier run's 25 ms stall is now
/// permanently unattributable for exactly that reason. A measurement you cannot place in time
/// is a measurement you cannot rule out boot from.
fn realtime_s() -> f64 {
    let mut rt = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `rt` is a valid, properly aligned timespec we own for the duration of the call.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut rt) };
    rt.tv_sec as f64 + rt.tv_nsec as f64 / 1e9
}

static CMD_WORST_NS: AtomicU64 = AtomicU64::new(0);
static CMD_WORST_TYPE: AtomicU64 = AtomicU64::new(0);
static CMD_COUNT: AtomicU64 = AtomicU64::new(0);

/// Start timing one virtio-gpu command. Returns 0 (and costs one relaxed load) when off.
pub fn cmd_start() -> u64 {
    if enabled() {
        now_ns()
    } else {
        0
    }
}

/// Finish timing one virtio-gpu command, keeping the worst of the current drain.
pub fn cmd_end(start_ns: u64, cmd_type: u32) {
    if start_ns == 0 {
        return;
    }
    let dt = now_ns().saturating_sub(start_ns);
    CMD_COUNT.fetch_add(1, Ordering::Relaxed);
    if dt > CMD_WORST_NS.load(Ordering::Relaxed) {
        CMD_WORST_NS.store(dt, Ordering::Relaxed);
        CMD_WORST_TYPE.store(cmd_type as u64, Ordering::Relaxed);
    }
}

/// virtio-gpu control command names, for the outlier line. Unknown codes print as hex rather
/// than being dropped — an unnamed stall is still worth seeing.
fn cmd_name(t: u32) -> String {
    match t {
        0x0100 => "GET_DISPLAY_INFO".into(),
        0x0101 => "RESOURCE_CREATE_2D".into(),
        0x0102 => "RESOURCE_UNREF".into(),
        0x0103 => "SET_SCANOUT".into(),
        0x0104 => "RESOURCE_FLUSH".into(),
        0x0105 => "TRANSFER_TO_HOST_2D".into(),
        0x0106 => "RESOURCE_ATTACH_BACKING".into(),
        0x0107 => "RESOURCE_DETACH_BACKING".into(),
        0x0108 => "GET_CAPSET_INFO".into(),
        0x0109 => "GET_CAPSET".into(),
        0x010a => "GET_EDID".into(),
        0x010b => "RESOURCE_ASSIGN_UUID".into(),
        0x010c => "RESOURCE_CREATE_BLOB".into(),
        0x010d => "SET_SCANOUT_BLOB".into(),
        0x0200 => "CTX_CREATE".into(),
        0x0201 => "CTX_DESTROY".into(),
        0x0202 => "CTX_ATTACH_RESOURCE".into(),
        0x0203 => "CTX_DETACH_RESOURCE".into(),
        0x0204 => "RESOURCE_CREATE_3D".into(),
        0x0205 => "TRANSFER_TO_HOST_3D".into(),
        0x0206 => "TRANSFER_FROM_HOST_3D".into(),
        0x0207 => "SUBMIT_3D".into(),
        0x0208 => "RESOURCE_MAP_BLOB".into(),
        0x0209 => "RESOURCE_UNMAP_BLOB".into(),
        other => format!("0x{other:04x}"),
    }
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
    /// Bucket chosen by `arm`, consumed by `record`. The two are separate calls because the
    /// return-path measurement has to be armed BEFORE the interrupt is raised, while the
    /// forward-path sample can only be completed after.
    pending_idx: Option<usize>,
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
            pending_idx: None,
        })
    }

    /// Classify this doorbell by its idle gap. Called by the worker after the drain.
    pub fn arm(&mut self, kick_ns: u64) {
        let gap = kick_ns.saturating_sub(self.last_kick_ns);
        self.last_kick_ns = kick_ns;
        self.pending_idx = Some(BUCKET_BOUNDS_NS.iter().position(|&b| gap < b).unwrap_or(3));
    }

    /// Arm the return-path measurement, immediately before the used-queue interrupt is raised.
    ///
    /// Separate from `arm` because most drains raise NO interrupt: with EVENT_IDX the guest
    /// asks for one only when it wants it (`needs_notification`), and a drain that used no
    /// descriptors never signals at all. Arming on every doorbell instead of only on the ones
    /// that actually interrupt left `irq_unacked` in the thousands against a handful of
    /// samples — an unacked count that size means the arm point is wrong, not that the guest
    /// is dropping interrupts.
    pub fn arm_irq(&self) {
        if let Some(idx) = self.pending_idx {
            irq_arm(idx);
        }
    }

    /// Record one doorbell's worth of the VMM path. `kick_ns` is the stamp taken on the vCPU
    /// thread; the rest are taken by the worker around its own work.
    pub fn record(&mut self, kick_ns: u64, wake_ns: u64, drained_ns: u64, signaled_ns: u64) {
        let Some(idx) = self.pending_idx.take() else {
            return;
        };
        let b = &mut self.buckets[idx];
        b.add(
            wake_ns.saturating_sub(kick_ns),
            drained_ns.saturating_sub(wake_ns),
            signaled_ns.saturating_sub(drained_ns),
        );
        // After the hops are recorded, so the control never inflates what it is controlling for.
        if calib_enabled() {
            b.calib.add(calibrate());
        }

        // Per-drain command tally, consumed whether or not this drain overran, so the next one
        // starts clean.
        let cmds = CMD_COUNT.swap(0, Ordering::Relaxed);
        let worst_ns = CMD_WORST_NS.swap(0, Ordering::Relaxed);
        let worst_type = CMD_WORST_TYPE.swap(0, Ordering::Relaxed) as u32;

        let drain_ns = drained_ns.saturating_sub(wake_ns);
        let total_ns = signaled_ns.saturating_sub(kick_ns);
        if ms(drain_ns) < OUTLIER_MS {
            return;
        }
        // CLOCK_REALTIME so this can be lined up against the guest's own frame log; the guest
        // clock is anchored to ours.
        let mut rt = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `rt` is a valid, properly aligned timespec we own for the duration of the call.
        unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut rt) };
        eprintln!(
            "[GPUWAKE OUTLIER] realtime={}.{:09} idle={} total {:.3} ms | kick->wake {:.3} \
             wake->drained {:.3} drained->signal {:.3} | {cmds} cmds, worst {} {:.3} ms",
            rt.tv_sec,
            rt.tv_nsec,
            BUCKET_NAMES[idx].trim(),
            ms(total_ns),
            ms(wake_ns.saturating_sub(kick_ns)),
            ms(drain_ns),
            ms(signaled_ns.saturating_sub(drained_ns)),
            cmd_name(worst_type),
            ms(worst_ns),
        );
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
            // p50/p95/max per hop, in ms. Add the three p50s for the VMM-side floor.
            let h = |x: &Hist| {
                format!(
                    "{:.3}/{:.3}/{:.3}",
                    ms(x.pct(0.50) as u64),
                    ms(x.pct(0.95) as u64),
                    ms(x.max)
                )
            };
            let mut line = format!(
                "[GPUWAKE rt={:.3} idle {}] n={} p50/p95/max ms | kick->wake {} \
                 | wake->drained {} | drained->signal {}",
                realtime_s(),
                BUCKET_NAMES[i],
                b.n,
                h(&b.kick_wake),
                h(&b.wake_drain),
                h(&b.drain_sig),
            );
            // Return path. The parked/running split needs the userspace GIC; the hop itself
            // does not, so only the labels change.
            let sw_gic = SOFTWARE_GIC.load(Ordering::Relaxed);
            for (parked, label) in [(0usize, "irq->ack"), (1usize, "irq->ack PARKED")] {
                let slot = &IRQ_STATS[i][parked];
                let n = slot.0.swap(0, Ordering::Relaxed);
                let sum = slot.1.swap(0, Ordering::Relaxed);
                let mx = slot.2.swap(0, Ordering::Relaxed);
                if n > 0 {
                    let label = if parked == 0 && sw_gic {
                        "irq->ack running"
                    } else {
                        label
                    };
                    line.push_str(&format!(
                        " | {label} n={n} avg {:.3} max {:.3}",
                        ms(sum) / n as f64,
                        ms(mx)
                    ));
                }
            }
            if b.calib.n > 0 {
                line.push_str(&format!(" | calib {}", h(&b.calib)));
            }
            eprintln!("{line}");
        }
        let irq_coal = IRQ_COALESCED.swap(0, Ordering::Relaxed);
        if self.no_kick > 0 || coalesced > 0 || irq_coal > 0 {
            eprintln!(
                "[GPUWAKE rt={:.3}] no_kick={} coalesced={coalesced} irq_coalesced={irq_coal}",
                realtime_s(),
                self.no_kick
            );
        }
        self.buckets = [Bucket::default(); 4];
        self.no_kick = 0;
    }
}
