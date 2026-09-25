// Copyright 2026 The libkrun Authors
// SPDX-License-Identifier: Apache-2.0

//! Stage-2 tracking for balloon-released guest RAM.
//!
//! When the balloon device returns guest pages to the host (`MADV_FREE_REUSABLE`), the
//! backing pages become disposable to the OS — but a stage-2 mapping that stays live keeps
//! *offering* those same physical pages to the guest, and the virtio free-page-reporting
//! contract lets the guest reuse reported pages at any time without notice. On Linux/KVM,
//! mmu notifiers tear down the stage-2 mapping together with the host PTEs and the reuse
//! faults back in through GUP; Hypervisor.framework has no notifier, so this module is the
//! hand-rolled equivalent: `release` unmaps the range from the guest (then marks the host
//! pages reusable under the same lock), and a later guest touch takes a stage-2 fault that
//! [`ReleasedRam::handle_fault`] heals by re-validating (`MADV_FREE_REUSE`) and re-mapping
//! a chunk around the fault. Without the REUSE the re-touched pages would sit dirty but
//! reusable-marked — invisible to the task's footprint until the pageout scan happens to
//! reprocess them.
//!
//! The released set must be exact: `hv_vm_map` fails on any overlap with a live mapping
//! (even partial), so the fault handler can only map back precisely what was unmapped.

use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

// The locks are loom's under `--cfg loom`, so its model can interleave release, heal and
// reclaim. The atomics stay std: they are statistics and the sweep's signal-handler state,
// neither of which orders anything the model checks.
#[cfg(loom)]
use loom::sync::Mutex;
#[cfg(not(loom))]
use std::sync::Mutex;
use std::time::Instant;

use crate::bindings::*;
use crate::host_page_size;

/// ESR xFSC translation-fault range (levels 0-3): the only fault kind a missing stage-2
/// mapping produces. Anything else in guest RAM is not ours to heal.
const XFSC_TRANSLATION_L0: u64 = 0b000100;
const XFSC_TRANSLATION_L3: u64 = 0b000111;

/// How often to log a heal after the first one.
const HEAL_LOG_EVERY: u64 = 256;

/// Cumulative counters since boot. `released_bytes - remapped_bytes` is the RAM currently
/// handed back to the OS; `heals` counts stage-2 faults taken (each one a guest reuse the
/// balloon did not anticipate), `stray_faults` counts translation faults in guest RAM with
/// no released range covering them (should stay 0).
#[derive(Copy, Clone, Debug, Default)]
pub struct ReleasedRamStats {
    pub heals: u64,
    pub released_bytes: u64,
    pub remapped_bytes: u64,
    pub stray_faults: u64,
    pub sweeps: u64,
    pub sweep_debited_bytes: u64,
    pub sweep_ms: u64,
    pub sweep_faults: u64,
}

pub enum FaultOutcome {
    /// The fault hit a released range; it has been re-validated and re-mapped. Re-run the
    /// vCPU without advancing the PC so the faulting access retries against the new mapping.
    Healed,
    /// A translation fault on guest RAM with no released range covering it. Almost always a
    /// heal race: another vCPU healed the range between this vCPU's fault and our lookup, so
    /// the mapping exists again — re-run without advancing the PC and the access succeeds.
    /// Falling through instead would MMIO-decode a RAM access and silently swallow the
    /// guest's load/store. A per-PA cap turns a genuine bookkeeping hole into [`Fatal`].
    Retry,
    /// Not a released-RAM fault (outside guest RAM, or not a translation fault) — fall
    /// through to the caller's existing handling.
    NotHandled,
    /// The range was released but could not be re-mapped (or the same PA keeps stray-faulting
    /// past the cap); resuming the guest would livelock or corrupt. The VM must stop.
    Fatal,
}

struct RamRegion {
    gpa: u64,
    host: u64,
    len: u64,
}

/// What [`ReleasedRam`] does to the guest's stage-2 mappings and to the host pages behind them.
/// [`Hvf`] is Hypervisor.framework and `madvise`; the tests put a model of both here.
///
/// Measured HVF semantics the bookkeeping rests on: `hv_vm_unmap` is page-wise and idempotent (it
/// succeeds over any mix of mapped and unmapped pages), and `hv_vm_map` fails on any overlap with
/// a live mapping.
pub trait Stage2: Send + Sync {
    fn unmap(&self, gpa: u64, len: u64) -> hv_return_t;
    fn map(&self, host: u64, gpa: u64, len: u64) -> hv_return_t;
    /// Zero the host range if `zero`, then mark it `MADV_FREE_REUSABLE`.
    ///
    /// # Safety
    ///
    /// `[host, host + len)` lies inside a guest RAM region's host mapping and is unmapped from the
    /// guest, so nothing else writes it.
    unsafe fn discard(&self, host: u64, len: u64, zero: bool) -> std::io::Result<()>;
    /// Mark the host range `MADV_FREE_REUSE`.
    fn reuse(&self, host: u64, len: u64) -> std::io::Result<()>;
}

pub struct Hvf;

impl Stage2 for Hvf {
    fn unmap(&self, gpa: u64, len: u64) -> hv_return_t {
        unsafe { hv_vm_unmap(gpa, len as usize) }
    }

    fn map(&self, host: u64, gpa: u64, len: u64) -> hv_return_t {
        unsafe {
            hv_vm_map(
                host as *mut core::ffi::c_void,
                gpa,
                len as usize,
                (HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC).into(),
            )
        }
    }

    unsafe fn discard(&self, host: u64, len: u64, zero: bool) -> std::io::Result<()> {
        if zero {
            // SAFETY: the caller's contract: guest RAM, unmapped from the guest.
            unsafe { std::ptr::write_bytes(host as *mut u8, 0, len as usize) };
        }
        madvise(host, len, libc::MADV_FREE_REUSABLE)
    }

    fn reuse(&self, host: u64, len: u64) -> std::io::Result<()> {
        madvise(host, len, libc::MADV_FREE_REUSE)
    }
}

fn madvise(host: u64, len: u64, advice: libc::c_int) -> std::io::Result<()> {
    if unsafe { libc::madvise(host as *mut libc::c_void, len as usize, advice) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

pub struct ReleasedRam<S: Stage2 = Hvf> {
    stage2: S,
    regions: Vec<RamRegion>,
    /// Released GPA ranges: start -> len, disjoint, coalesced. Exact by construction: every
    /// byte in here is stage-2 unmapped and only bytes in here are (balloon-released) ones.
    released: Mutex<BTreeMap<u64, u64>>,
    /// Heal window: on a fault, everything released within the chunk-aligned window around
    /// the fault is re-mapped in one go, bounding the fault *count* for a linear refill.
    chunk: u64,
    heals: AtomicU64,
    released_bytes: AtomicU64,
    remapped_bytes: AtomicU64,
    stray_faults: AtomicU64,
    /// Consecutive-stray livelock guard: (page, consecutive count) of the last stray PA.
    /// A heal-race stray resolves on retry, so consecutive repeats of the SAME page mean a
    /// genuine hole in the bookkeeping — cap and stop instead of spinning.
    last_stray: Mutex<(u64, u32)>,
    /// Which release paths zero the range before marking it REUSABLE (see [`ZeroOnRelease`]).
    zero_on_release: ZeroOnRelease,
    /// The regions as host-VA ranges, leaked so the sweep fault handler (a signal handler,
    /// which cannot take locks or allocate) can classify fault addresses.
    handler_regions: &'static [(u64, u64)],
    sweeps: AtomicU64,
    sweep_debited_bytes: AtomicU64,
    sweep_ms: AtomicU64,
}

/// `LIMINA_BALLOON_RELEASE_MEMSET` — which release paths zero the range before the
/// `MADV_FREE_REUSABLE`: unset/other = `queue` (the default: inflate-queue releases only),
/// `0`/`none` = no zeroing, `1` = every path. Zeroing settles the compressed-slot residue a
/// plain REUSABLE leaves behind at scale (retention-testbed A/B: post-scrub pool residue
/// 2.67G → 0.69G, timings unchanged), and inflate-queue releases only happen while the
/// balloon inflates, so the default costs nothing at steady state. Zeroing the
/// free-page-reporting path re-dirties pages at churn rate (+3.5G steady-state resident
/// under FRQ churn) — hence the per-path gate rather than all-paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ZeroOnRelease {
    None,
    InflateQueue,
    All,
}

/// Consecutive stray faults on one page before we declare a real bookkeeping hole.
const STRAY_RETRY_CAP: u32 = 64;

impl ReleasedRam {
    /// `regions` are the guest RAM regions as `(gpa, host_va, len)`. Regions not aligned to
    /// the host page granule are dropped (loudly): release/heal must never round.
    pub fn new(regions: Vec<(u64, u64, u64)>) -> Self {
        let chunk_mib = std::env::var("LIMINA_BALLOON_REMAP_CHUNK_MIB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(2);
        let zero_on_release = match std::env::var("LIMINA_BALLOON_RELEASE_MEMSET").as_deref() {
            Ok("1") => ZeroOnRelease::All,
            Ok("0") | Ok("none") => ZeroOnRelease::None,
            _ => ZeroOnRelease::InflateQueue,
        };
        Self::with_stage2(Hvf, regions, chunk_mib << 20, zero_on_release)
    }
}

impl<S: Stage2> ReleasedRam<S> {
    fn with_stage2(
        stage2: S,
        regions: Vec<(u64, u64, u64)>,
        chunk: u64,
        zero_on_release: ZeroOnRelease,
    ) -> Self {
        let page = host_page_size();
        let regions: Vec<RamRegion> = regions
            .into_iter()
            .filter(|&(gpa, host, len)| {
                let ok = gpa % page == 0 && host % page == 0 && len % page == 0;
                if !ok {
                    error!(
                        "released-ram: dropping misaligned RAM region gpa={gpa:#x} host={host:#x} \
                         len={len:#x} (granule {page:#x}); balloon release disabled for it"
                    );
                }
                ok
            })
            .map(|(gpa, host, len)| RamRegion { gpa, host, len })
            .collect();

        let chunk = chunk.next_power_of_two().max(page);

        let handler_regions: &'static [(u64, u64)] = Box::leak(
            regions
                .iter()
                .map(|r: &RamRegion| (r.host, r.host + r.len))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );

        Self {
            stage2,
            regions,
            released: Mutex::new(BTreeMap::new()),
            chunk,
            heals: AtomicU64::new(0),
            released_bytes: AtomicU64::new(0),
            remapped_bytes: AtomicU64::new(0),
            stray_faults: AtomicU64::new(0),
            last_stray: Mutex::new((u64::MAX, 0)),
            zero_on_release,
            handler_regions,
            sweeps: AtomicU64::new(0),
            sweep_debited_bytes: AtomicU64::new(0),
            sweep_ms: AtomicU64::new(0),
        }
    }

    fn region_of(&self, gpa: u64) -> Option<&RamRegion> {
        self.regions
            .iter()
            .find(|r| gpa >= r.gpa && gpa < r.gpa + r.len)
    }

    fn host_of(region: &RamRegion, gpa: u64) -> u64 {
        region.host + (gpa - region.gpa)
    }

    /// Release `[gpa, gpa + len)`: record it, unmap it from the guest, and hand the host
    /// pages back to the OS with `MADV_FREE_REUSABLE`. Returns false (with the set rolled
    /// back and no madvise issued) if the range is invalid or the unmap failed — the caller
    /// must then leave the pages alone. `from_inflate_queue` says which balloon path is
    /// releasing (the inflate queue vs free-page reporting) — it selects whether the range
    /// is zeroed first under the [`ZeroOnRelease`] gate.
    ///
    /// The madvise happens under the released-set lock so a concurrent guest touch can
    /// never interleave a heal (REUSE + remap) between our unmap and our REUSABLE, which
    /// would re-mark live-again pages as disposable.
    pub fn release(&self, gpa: u64, len: u64, from_inflate_queue: bool) -> bool {
        let page = host_page_size();
        if len == 0 || gpa % page != 0 || len % page != 0 {
            error!("released-ram: misaligned release gpa={gpa:#x} len={len:#x}; ignoring");
            return false;
        }
        let Some(region) = self.region_of(gpa) else {
            error!("released-ram: release outside guest RAM: gpa={gpa:#x} len={len:#x}");
            return false;
        };
        if gpa + len > region.gpa + region.len {
            error!("released-ram: release crosses a region boundary: gpa={gpa:#x} len={len:#x}");
            return false;
        }

        let mut released = self.released.lock().unwrap();
        let ret = self.stage2.unmap(gpa, len);
        if ret != HV_SUCCESS {
            // The set is left alone: part of the range may be released already, and rolling
            // an insert back would forget those pages while they stay unmapped.
            error!("released-ram: hv_vm_unmap(gpa={gpa:#x}, len={len:#x}) failed: {ret:#x}");
            return false;
        }
        insert_range(&mut released, gpa, len);
        let host = Self::host_of(region, gpa);
        let zero = match self.zero_on_release {
            ZeroOnRelease::All => true,
            ZeroOnRelease::InflateQueue => from_inflate_queue,
            ZeroOnRelease::None => false,
        };
        // SAFETY: the range was just unmapped from the guest (above, under the lock), is
        // balloon-owned, and lies inside this region's host mapping — no guest access can race
        // the write; a touch faults and heals afterward.
        if let Err(e) = unsafe { self.stage2.discard(host, len, zero) } {
            // The unmap stands (a guest touch will fault and heal); only the host-side
            // reclaim didn't happen, so the pages simply stay resident.
            warn!(
                "released-ram: madvise(MADV_FREE_REUSABLE) at {host:#x} len={len:#x} failed: {e}"
            );
        }
        self.released_bytes.fetch_add(len, Ordering::Relaxed);
        true
    }

    /// Take `[gpa, gpa + len)` back for the guest ahead of a known reuse (deflate): every
    /// released byte in the range is re-validated (`MADV_FREE_REUSE`) and re-mapped, exactly
    /// like a fault heal but without paying for the fault. Best-effort: correctness never
    /// depends on this being called — an untaken range heals through the fault path.
    pub fn reclaim(&self, gpa: u64, len: u64) -> bool {
        let mut released = self.released.lock().unwrap();
        let ranges = remove_overlaps(&mut released, gpa, gpa + len);
        self.reuse_and_map_all(&mut released, &ranges)
    }

    /// Heal a stage-2 fault at `pa` (data or instruction abort with translation-fault
    /// `xfsc`). See [`FaultOutcome`].
    pub fn handle_fault(&self, pa: u64, xfsc: u64) -> FaultOutcome {
        if !(XFSC_TRANSLATION_L0..=XFSC_TRANSLATION_L3).contains(&xfsc) {
            return FaultOutcome::NotHandled;
        }
        let Some(region) = self.region_of(pa) else {
            return FaultOutcome::NotHandled;
        };

        let mut released = self.released.lock().unwrap();
        if !contains_point(&released, pa) {
            // A translation fault inside guest RAM with no released range: almost always the
            // heal race — another vCPU healed this range between our fault and the lookup
            // (we block on the released lock while its REUSE+remap completes), so the
            // mapping exists again and a retry succeeds. Retrying is the ONLY safe answer:
            // falling through would MMIO-decode a RAM access and swallow the guest's
            // load/store. Consecutive repeats of the same page mean the mapping is really
            // gone with no bookkeeping — cap and stop before the guest spins forever.
            let strays = self.stray_faults.fetch_add(1, Ordering::Relaxed) + 1;
            let page = pa & !(host_page_size() - 1);
            let mut last = self.last_stray.lock().unwrap();
            *last = if last.0 == page {
                (page, last.1 + 1)
            } else {
                (page, 1)
            };
            if last.1 > STRAY_RETRY_CAP {
                error!(
                    "released-ram: page {page:#x} stray-faulted {} times consecutively — a \
                     stage-2 hole outside the released set; stopping the VM",
                    last.1
                );
                return FaultOutcome::Fatal;
            }
            if strays <= 8 || strays.is_multiple_of(1024) {
                warn!(
                    "released-ram: stray stage-2 fault at pa={pa:#x} (guest RAM, not in the \
                     released set — lost heal race); retrying (stray #{strays})"
                );
            }
            return FaultOutcome::Retry;
        }
        // A covered fault resets the consecutive-stray guard: the vCPU is making progress.
        *self.last_stray.lock().unwrap() = (u64::MAX, 0);

        let aligned = pa & !(self.chunk - 1);
        let window_start = aligned.max(region.gpa);
        let window_end = (aligned + self.chunk).min(region.gpa + region.len);
        let ranges = remove_overlaps(&mut released, window_start, window_end);
        if !self.reuse_and_map_all(&mut released, &ranges) {
            return FaultOutcome::Fatal;
        }

        let heals = self.heals.fetch_add(1, Ordering::Relaxed) + 1;
        if heals == 1 || heals.is_multiple_of(HEAL_LOG_EVERY) {
            info!(
                "balloon: stage-2 heal #{heals} pa={pa:#x} (released {} MiB, remapped {} MiB \
                 cumulative)",
                self.released_bytes.load(Ordering::Relaxed) >> 20,
                self.remapped_bytes.load(Ordering::Relaxed) >> 20,
            );
        }
        FaultOutcome::Healed
    }

    pub fn stats(&self) -> ReleasedRamStats {
        ReleasedRamStats {
            heals: self.heals.load(Ordering::Relaxed),
            released_bytes: self.released_bytes.load(Ordering::Relaxed),
            remapped_bytes: self.remapped_bytes.load(Ordering::Relaxed),
            stray_faults: self.stray_faults.load(Ordering::Relaxed),
            sweeps: self.sweeps.load(Ordering::Relaxed),
            sweep_debited_bytes: self.sweep_debited_bytes.load(Ordering::Relaxed),
            sweep_ms: self.sweep_ms.load(Ordering::Relaxed),
            sweep_faults: SWEEP_FAULTS.load(Ordering::Relaxed),
        }
    }

    /// [`Self::reuse_and_map`] each extracted range in turn. On the first failure the ranges
    /// not yet mapped go back into the set with the failed one, since they are still unmapped.
    ///
    /// The set coalesces ranges that touch, so one range can span two regions adjacent in GPA
    /// but not on the host. Each is mapped a region at a time, from that region's own host base.
    fn reuse_and_map_all(&self, released: &mut BTreeMap<u64, u64>, ranges: &[(u64, u64)]) -> bool {
        let ranges: Vec<(u64, u64)> = ranges
            .iter()
            .flat_map(|&(start, len)| {
                self.regions.iter().filter_map(move |r| {
                    let s = start.max(r.gpa);
                    let e = (start + len).min(r.gpa + r.len);
                    (s < e).then(|| (s, e - s))
                })
            })
            .collect();
        for (i, &(start, len)) in ranges.iter().enumerate() {
            if !self.reuse_and_map(released, start, len) {
                for &(s, l) in &ranges[i + 1..] {
                    insert_range(released, s, l);
                }
                return false;
            }
        }
        true
    }

    /// `MADV_FREE_REUSE` + `hv_vm_map` one extracted range. On map failure the range is
    /// reinserted (bookkeeping stays exact) and false is returned. REUSE on pages the OS
    /// never actually reclaimed is a no-op, so no per-page state is needed.
    fn reuse_and_map(&self, released: &mut BTreeMap<u64, u64>, gpa: u64, len: u64) -> bool {
        let region = self
            .region_of(gpa)
            .expect("released range outside every RAM region");
        let host = Self::host_of(region, gpa);
        if let Err(e) = self.stage2.reuse(host, len) {
            warn!("released-ram: madvise(MADV_FREE_REUSE) at {host:#x} len={len:#x} failed: {e}");
        }
        let ret = self.stage2.map(host, gpa, len);
        if ret != HV_SUCCESS {
            error!(
                "released-ram: hv_vm_map(host={host:#x}, gpa={gpa:#x}, len={len:#x}) failed: \
                 {ret:#x}"
            );
            insert_range(released, gpa, len);
            return false;
        }
        self.remapped_bytes.fetch_add(len, Ordering::Relaxed);
        true
    }

    /// Settle the task-pmap ledger share of live guest RAM.
    ///
    /// xnu bills `phys_footprint`/`resident_size` once per pmap, so every page the VMM
    /// writes through its task mapping AND the guest touches through stage-2 (all disk-fed
    /// guest memory, by construction) is billed twice — Activity Monitor shows up to 2× the
    /// VM's real memory. An `mprotect(PROT_NONE)` disconnects the task-pmap PTEs, debiting
    /// exactly that share; the immediate restore to the mapping's original RW leaves lazy
    /// re-population to the next host touch. The guest never notices: stage-2 PTEs are
    /// untouched, and HVF populates missing stage-2 entries in-kernel without consulting
    /// the task mapping's protection (measured: a 1 GiB guest first-touch pass through an
    /// open window — zero vCPU exits, same physical pages).
    ///
    /// The two actors that CAN trip on an open window are worker threads touching guest
    /// RAM from userspace (virtqueue rings, GPU transfers — fielded by the sweep fault
    /// handler, which retries the access once the window closes) and kernel copyio at
    /// syscalls reading/writing guest buffers (those sites retry transient `EFAULT`).
    ///
    /// Sweeps only what is live: released ranges are stage-2 unmapped and REUSABLE — a
    /// sweep there would fight the heal path — so each chunk's live sub-ranges are computed
    /// and flipped under the released lock, which also serializes against release/heal.
    /// Single-flight; concurrent calls are dropped.
    pub fn settle_sweep(&self) -> Option<SweepReport> {
        if self.regions.is_empty() {
            return None;
        }
        if SWEEP_ACTIVE.swap(true, Ordering::AcqRel) {
            warn!("released-ram: settle sweep already running; dropping this request");
            return None;
        }
        install_sweep_fault_handler();
        SWEEP_REGIONS.store(
            self.handler_regions as *const [(u64, u64)] as *mut (u64, u64),
            Ordering::Release,
        );
        SWEEP_REGIONS_LEN.store(self.handler_regions.len() as u64, Ordering::Release);

        let started = Instant::now();
        let before = phys_footprint();
        let chunk = sweep_chunk_bytes();
        for region in &self.regions {
            let region_end = region.gpa + region.len;
            let mut pos = region.gpa;
            while pos < region_end {
                let chunk_end = region_end.min(pos.saturating_add(chunk));
                let released = self.released.lock().unwrap();
                for &(start, len) in &live_complement(&released, pos, chunk_end) {
                    self.flip_window(Self::host_of(region, start), len);
                }
                drop(released);
                pos = chunk_end;
            }
        }
        let debited = before.saturating_sub(phys_footprint());
        let ms = started.elapsed().as_millis() as u64;
        SWEEP_ACTIVE.store(false, Ordering::Release);

        let sweeps = self.sweeps.fetch_add(1, Ordering::Relaxed) + 1;
        self.sweep_debited_bytes.store(debited, Ordering::Relaxed);
        self.sweep_ms.store(ms, Ordering::Relaxed);
        info!(
            "released-ram: settle sweep #{sweeps} debited {} MiB off the task ledger in {ms} ms",
            debited >> 20
        );
        Some(SweepReport {
            debited_bytes: debited,
            ms,
        })
    }

    /// One NONE→RW protection flip. The window bounds are published for the fault handler
    /// before the PTEs disconnect and cleared after the restore. A failed disconnect is
    /// skipped (that range just stays double-billed); a failed restore would leave a
    /// `PROT_NONE` hole in guest RAM — unsurvivable, so it retries and ultimately panics.
    fn flip_window(&self, host: u64, len: u64) {
        let p = host as *mut libc::c_void;
        SWEEP_WINDOW_START.store(host, Ordering::Release);
        SWEEP_WINDOW_END.store(host + len, Ordering::Release);
        if unsafe { libc::mprotect(p, len as usize, libc::PROT_NONE) } != 0 {
            warn!(
                "released-ram: sweep mprotect(PROT_NONE) at {host:#x} len={len:#x} failed: {}",
                std::io::Error::last_os_error()
            );
        } else {
            let mut tries = 0;
            while unsafe { libc::mprotect(p, len as usize, libc::PROT_READ | libc::PROT_WRITE) }
                != 0
            {
                tries += 1;
                if tries > 100 {
                    panic!(
                        "released-ram: cannot restore guest RAM protection at {host:#x} \
                         len={len:#x}: {}",
                        std::io::Error::last_os_error()
                    );
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        SWEEP_WINDOW_END.store(0, Ordering::Release);
        SWEEP_WINDOW_START.store(0, Ordering::Release);
    }
}

pub struct SweepReport {
    pub debited_bytes: u64,
    pub ms: u64,
}

static SWEEP_ACTIVE: AtomicBool = AtomicBool::new(false);
static SWEEP_WINDOW_START: AtomicU64 = AtomicU64::new(0);
static SWEEP_WINDOW_END: AtomicU64 = AtomicU64::new(0);
/// The sweeping instance's guest-RAM host ranges, for the fault handler. Published as a
/// raw pointer + length because a signal handler can only do atomic loads.
static SWEEP_REGIONS: AtomicPtr<(u64, u64)> = AtomicPtr::new(std::ptr::null_mut());
static SWEEP_REGIONS_LEN: AtomicU64 = AtomicU64::new(0);
/// Worker-thread touches fielded by the sweep fault handler, cumulative. Global (not per
/// instance) because a signal handler can only reach statics; there is one guest per
/// process. This is the field oracle for "something touches guest RAM during windows".
static SWEEP_FAULTS: AtomicU64 = AtomicU64::new(0);

fn sweep_chunk_bytes() -> u64 {
    std::env::var("LIMINA_LEDGER_SWEEP_CHUNK_MIB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .map(|v| v << 20)
        .unwrap_or(256 << 20)
        .max(host_page_size())
}

fn phys_footprint() -> u64 {
    let mut info: libc::rusage_info_v2 = unsafe { std::mem::zeroed() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t,
        )
    };
    if rc == 0 { info.ri_phys_footprint } else { 0 }
}

/// The live complement: sub-ranges of `[start, end)` NOT covered by the released set.
fn live_complement(map: &BTreeMap<u64, u64>, start: u64, end: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    let mut pos = start;
    if let Some((&s, &l)) = map.range(..=start).next_back()
        && s + l > start
    {
        pos = (s + l).min(end);
    }
    for (&s, &l) in map.range(start..end) {
        if s > pos {
            out.push((pos, s - pos));
        }
        pos = (s + l).min(end);
        if pos >= end {
            break;
        }
    }
    if pos < end {
        out.push((pos, end - pos));
    }
    out
}

/// A saved pre-sweep signal action. Written once under [`Once`], read-only afterwards
/// (including from the signal handler), hence the manual `Sync`.
struct SavedAction(UnsafeCell<libc::sigaction>);
unsafe impl Sync for SavedAction {}
unsafe impl Send for SavedAction {}

static OLD_SIGBUS: OnceLock<SavedAction> = OnceLock::new();
static OLD_SIGSEGV: OnceLock<SavedAction> = OnceLock::new();

fn install_sweep_fault_handler() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sweep_fault_handler as usize;
        sa.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&mut sa.sa_mask);
        for (sig, slot) in [(libc::SIGBUS, &OLD_SIGBUS), (libc::SIGSEGV, &OLD_SIGSEGV)] {
            let mut old: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(sig, &sa, &mut old) != 0 {
                error!(
                    "released-ram: sigaction({sig}) for the sweep fault handler failed: {}",
                    std::io::Error::last_os_error()
                );
                continue;
            }
            let _ = slot.set(SavedAction(UnsafeCell::new(old)));
        }
    });
}

/// SIGBUS/SIGSEGV handler covering worker-thread touches of guest RAM during a sweep
/// window. Guest RAM is always mapped read-write outside a window, so ANY fault at a
/// guest-region address is the sweep's doing: wait out the current window (it closes in
/// microseconds) and return, retrying the faulting access. Everything else chains to the
/// previously installed action (e.g. Rust's stack-overflow reporter).
///
/// The guest-region check deliberately does NOT require `SWEEP_ACTIVE`: a fault can land
/// in the last window of a sweep and reach the handler after the sweep finished, and
/// chaining it would restore `SIG_DFL` permanently (installation is `Once`) — the next
/// sweep's first fielded fault would then kill the process. If the fault's window is
/// already closed, the mapping is back to read-write and the plain return retries fine.
///
/// Async-signal-safety: atomic loads and `sched_yield` only.
unsafe extern "C" fn sweep_fault_handler(
    sig: libc::c_int,
    info: *mut libc::siginfo_t,
    ctx: *mut libc::c_void,
) {
    let addr = unsafe { (*info).si_addr } as u64;
    let ptr = SWEEP_REGIONS.load(Ordering::Acquire);
    let len = SWEEP_REGIONS_LEN.load(Ordering::Acquire) as usize;
    if !ptr.is_null() {
        let regions = unsafe { std::slice::from_raw_parts(ptr, len) };
        if regions.iter().any(|&(s, e)| addr >= s && addr < e) {
            SWEEP_FAULTS.fetch_add(1, Ordering::Relaxed);
            while SWEEP_ACTIVE.load(Ordering::Acquire)
                && addr >= SWEEP_WINDOW_START.load(Ordering::Acquire)
                && addr < SWEEP_WINDOW_END.load(Ordering::Acquire)
            {
                unsafe { libc::sched_yield() };
            }
            return;
        }
    }

    let old = match sig {
        libc::SIGBUS => OLD_SIGBUS.get(),
        libc::SIGSEGV => OLD_SIGSEGV.get(),
        _ => None,
    };
    let Some(old) = old else {
        // No saved action (installation failed): fall back to the default disposition so
        // the crash still surfaces instead of refaulting forever.
        unsafe {
            libc::signal(sig, libc::SIG_DFL);
        }
        return;
    };
    let old = unsafe { &*old.0.get() };
    match old.sa_sigaction {
        libc::SIG_DFL => {
            // Restore the default action and return; the refault then produces the real
            // crash report (right signal, right address) instead of a nested one here.
            unsafe {
                libc::sigaction(sig, old, std::ptr::null_mut());
            }
        }
        libc::SIG_IGN => {}
        handler => unsafe {
            if old.sa_flags & libc::SA_SIGINFO != 0 {
                let f: unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
                    std::mem::transmute(handler);
                f(sig, info, ctx);
            } else {
                let f: unsafe extern "C" fn(libc::c_int) = std::mem::transmute(handler);
                f(sig);
            }
        },
    }
}

/// Insert `[start, start + len)`, coalescing with any adjacent or overlapping ranges.
fn insert_range(map: &mut BTreeMap<u64, u64>, start: u64, len: u64) {
    let mut new_start = start;
    let mut new_end = start + len;

    if let Some((&s, &l)) = map.range(..=start).next_back()
        && s + l >= new_start
    {
        new_start = s;
        new_end = new_end.max(s + l);
        map.remove(&s);
    }
    let overlapping: Vec<u64> = map.range(new_start..=new_end).map(|(&s, _)| s).collect();
    for s in overlapping {
        let l = map.remove(&s).unwrap();
        new_end = new_end.max(s + l);
    }
    map.insert(new_start, new_end - new_start);
}

/// Remove and return every sub-range of the set intersecting `[start, end)`. Parts of
/// intersected ranges outside the window are reinserted, so the set stays exact.
fn remove_overlaps(map: &mut BTreeMap<u64, u64>, start: u64, end: u64) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    // Ranges are disjoint and sorted, so walking down from the window end can stop at the
    // first range that ends at/before the window start — O(overlapping), not O(set).
    let mut candidates: Vec<u64> = Vec::new();
    for (&s, &l) in map.range(..end).rev() {
        if s + l <= start {
            break;
        }
        candidates.push(s);
    }
    candidates.reverse();
    for s in candidates {
        let l = map.remove(&s).unwrap();
        let e = s + l;
        if s < start {
            map.insert(s, start - s);
        }
        if e > end {
            map.insert(end, e - end);
        }
        let is = s.max(start);
        let ie = e.min(end);
        out.push((is, ie - is));
    }
    out
}

fn contains_point(map: &BTreeMap<u64, u64>, p: u64) -> bool {
    map.range(..=p)
        .next_back()
        .is_some_and(|(&s, &l)| p < s + l)
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::{
        FaultOutcome, ReleasedRam, STRAY_RETRY_CAP, contains_point, insert_range, live_complement,
        remove_overlaps,
    };
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Sweeps are single-flight through the global `SWEEP_ACTIVE`, so tests that sweep
    /// must not overlap or one of them gets its sweep dropped.
    static SWEEP_LOCK: Mutex<()> = Mutex::new(());

    /// The heal-race stray: a guest-RAM translation fault outside the released set must
    /// RETRY (the racing heal already restored the mapping) — never fall through to MMIO
    /// decode, which would swallow the access. Consecutive strays on one page cap to Fatal
    /// (a genuine stage-2 hole); any other page resets the guard.
    #[test]
    fn stray_faults_retry_then_cap() {
        let rr = ReleasedRam::new(vec![(0x8000_0000, 0x1_0000_0000, 1 << 20)]);
        let xfsc = 0b000101; // translation fault, level 1

        for _ in 0..STRAY_RETRY_CAP {
            assert!(matches!(
                rr.handle_fault(0x8000_4000, xfsc),
                FaultOutcome::Retry
            ));
        }
        assert!(matches!(
            rr.handle_fault(0x8000_4000, xfsc),
            FaultOutcome::Fatal
        ));

        // A different page resets the consecutive guard, and the original page starts over.
        assert!(matches!(
            rr.handle_fault(0x8000_8000, xfsc),
            FaultOutcome::Retry
        ));
        assert!(matches!(
            rr.handle_fault(0x8000_4000, xfsc),
            FaultOutcome::Retry
        ));

        // Outside guest RAM and non-translation faults keep falling through.
        assert!(matches!(
            rr.handle_fault(0x1000_0000, xfsc),
            FaultOutcome::NotHandled
        ));
        assert!(matches!(
            rr.handle_fault(0x8000_4000, 0b001001),
            FaultOutcome::NotHandled
        ));
    }

    fn set(ranges: &[(u64, u64)]) -> BTreeMap<u64, u64> {
        let mut m = BTreeMap::new();
        for &(s, l) in ranges {
            insert_range(&mut m, s, l);
        }
        m
    }

    #[test]
    fn insert_coalesces_adjacent_and_overlapping() {
        let m = set(&[(0x4000, 0x4000), (0x8000, 0x4000)]);
        assert_eq!(m, set(&[(0x4000, 0x8000)]));

        let m = set(&[(0x4000, 0x4000), (0x10000, 0x4000), (0x0, 0x20000)]);
        assert_eq!(m, set(&[(0x0, 0x20000)]));

        let m = set(&[(0x8000, 0x4000), (0x4000, 0x8000)]);
        assert_eq!(m, set(&[(0x4000, 0x8000)]));
    }

    #[test]
    fn insert_keeps_disjoint_ranges_apart() {
        let m = set(&[(0x0, 0x4000), (0x8000, 0x4000)]);
        assert_eq!(m.len(), 2);
        assert!(contains_point(&m, 0x0));
        assert!(contains_point(&m, 0x3fff));
        assert!(!contains_point(&m, 0x4000));
        assert!(!contains_point(&m, 0x7fff));
        assert!(contains_point(&m, 0x8000));
        assert!(!contains_point(&m, 0xc000));
    }

    #[test]
    fn remove_overlaps_extracts_exact_intersections() {
        // A range straddling the window start, one inside, one straddling the end.
        let mut m = set(&[(0x0, 0x8000), (0xc000, 0x4000), (0x14000, 0x8000)]);
        let got = remove_overlaps(&mut m, 0x4000, 0x18000);
        assert_eq!(
            got,
            vec![(0x4000, 0x4000), (0xc000, 0x4000), (0x14000, 0x4000)]
        );
        // The parts outside the window survive, exactly.
        assert_eq!(m, set(&[(0x0, 0x4000), (0x18000, 0x4000)]));
    }

    #[test]
    fn remove_overlaps_on_disjoint_window_is_empty() {
        let mut m = set(&[(0x0, 0x4000)]);
        assert!(remove_overlaps(&mut m, 0x8000, 0x10000).is_empty());
        assert_eq!(m, set(&[(0x0, 0x4000)]));
    }

    #[test]
    fn remove_overlaps_window_inside_one_range_splits_it() {
        let mut m = set(&[(0x0, 0x20000)]);
        let got = remove_overlaps(&mut m, 0x8000, 0xc000);
        assert_eq!(got, vec![(0x8000, 0x4000)]);
        assert_eq!(m, set(&[(0x0, 0x8000), (0xc000, 0x14000)]));
    }

    #[test]
    fn live_complement_inverts_the_released_set() {
        // Empty set: the whole window is live.
        let m = BTreeMap::new();
        assert_eq!(live_complement(&m, 0x4000, 0x10000), vec![(0x4000, 0xc000)]);

        // A released range straddling the window start, one inside, one straddling the end.
        let m = set(&[(0x0, 0x8000), (0xc000, 0x4000), (0x14000, 0x8000)]);
        assert_eq!(
            live_complement(&m, 0x4000, 0x18000),
            vec![(0x8000, 0x4000), (0x10000, 0x4000)]
        );

        // Window fully inside one released range: nothing live.
        let m = set(&[(0x0, 0x20000)]);
        assert!(live_complement(&m, 0x8000, 0xc000).is_empty());

        // Released range fully inside the window: live head and tail.
        let m = set(&[(0x8000, 0x4000)]);
        assert_eq!(
            live_complement(&m, 0x0, 0x10000),
            vec![(0x0, 0x8000), (0xc000, 0x4000)]
        );
    }

    /// The sweep must flip only live ranges (skipping released ones — flipping those would
    /// fight the heal path) and leave the memory readable, writable, and intact. Exercised
    /// against a real anonymous mapping; no VM is needed because the flip is pure task-side
    /// mprotect. hv_vm_unmap inside release() fails without a VM, so the released set is
    /// seeded directly.
    #[test]
    fn settle_sweep_flips_live_ranges_and_preserves_content() {
        let _serialize = SWEEP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let page = crate::host_page_size();
        let len = 64 * page;
        let host = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(host, libc::MAP_FAILED);
        let host = host as u64;
        unsafe { std::ptr::write_bytes(host as *mut u8, 0x5a, len as usize) };

        let gpa = 0x8000_0000u64;
        let rr = ReleasedRam::new(vec![(gpa, host, len)]);
        super::insert_range(&mut rr.released.lock().unwrap(), gpa + 8 * page, 4 * page);

        let report = rr.settle_sweep().expect("sweep should run");
        assert_eq!(rr.stats().sweeps, 1);
        assert_eq!(rr.stats().sweep_ms, report.ms);

        // Every byte survived and the mapping is writable again.
        let slice = unsafe { std::slice::from_raw_parts(host as *const u8, len as usize) };
        assert!(slice.iter().all(|&b| b == 0x5a));
        unsafe { std::ptr::write_bytes(host as *mut u8, 0xa5, len as usize) };

        // A second sweep is fine; a concurrent one would be dropped (single-flight is
        // covered by the SWEEP_ACTIVE swap, not testable without threads racing).
        assert!(rr.settle_sweep().is_some());
        assert_eq!(rr.stats().sweeps, 2);

        unsafe { libc::munmap(host as *mut libc::c_void, len as usize) };
    }

    /// The sweep fault handler must actually FIELD concurrent touches, not merely exist:
    /// a toucher thread writes every page of the region in a tight loop while sweeps flip
    /// windows over it. Any write landing in an open window faults; a broken handler kills
    /// the process on the default disposition, and `sweep_faults` proves collisions really
    /// happened rather than the timing never producing one. The toucher finishes each full
    /// pass before checking its stop flag, so afterwards every page must hold the final
    /// pass's value — a write torn or lost in a window would leave a mismatch.
    #[test]
    fn sweep_fault_handler_fields_concurrent_touches() {
        let _serialize = SWEEP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let page = crate::host_page_size();
        let len = 4096 * page;
        let host = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(host, libc::MAP_FAILED);
        let host = host as u64;
        let rr = ReleasedRam::new(vec![(0x8000_0000, host, len)]);

        static STOP: AtomicBool = AtomicBool::new(false);
        STOP.store(false, Ordering::Relaxed);
        let base = host as usize;
        let pages = (len / page) as usize;
        let step = page as usize;
        let toucher = std::thread::spawn(move || {
            let mut pass = 1u64;
            while !STOP.load(Ordering::Relaxed) {
                for i in 0..pages {
                    unsafe { std::ptr::write_volatile((base + i * step) as *mut u64, pass) };
                }
                pass += 1;
            }
            pass
        });

        let faults0 = rr.stats().sweep_faults;
        // Bounded by time, not a sweep count: a toucher starved by a busy host (the tests run
        // in parallel) can manage a single pass in 50 sweeps.
        let started = std::time::Instant::now();
        let mut sweeps = 0;
        while rr.stats().sweep_faults == faults0
            && started.elapsed() < std::time::Duration::from_secs(10)
        {
            rr.settle_sweep()
                .expect("nothing else sweeps under the test lock");
            sweeps += 1;
        }
        STOP.store(true, Ordering::Relaxed);
        let final_pass = toucher.join().unwrap() - 1;

        assert!(
            rr.stats().sweep_faults > faults0,
            "no toucher write collided with a sweep window in {sweeps} sweeps \
             ({final_pass} toucher passes) — the windows never opened under load"
        );
        for i in 0..pages {
            let v = unsafe { std::ptr::read_volatile((base + i * step) as *const u64) };
            assert_eq!(
                v, final_pass,
                "page {i} lost the final pass's write across the sweep windows"
            );
        }
        unsafe { libc::munmap(host as *mut libc::c_void, len as usize) };
    }
}

/// Every sequence of releases, reclaims, guest touches, raw stage-2 faults and injected
/// unmap/map failures, to depth four, over seven pages: region A (four pages) and region B (two),
/// adjacent in GPA but not on the host, then a page that is not RAM. A starts one page past a
/// heal-window boundary, so windows are clipped at its start and at the A|B boundary.
///
/// The stage-2 stand-in follows `spikes/hv-unmap-semantics` in limina: an unmap succeeds on any
/// mix of mapped and unmapped pages, a map over a live page fails. It also reports what HVF
/// would not catch: a page mapped to the wrong host address, a page mapped while still reusable,
/// and a page made reusable while the guest can still reach it.
///
/// After every step the released set and the stage-2 state must both equal a model kept outside
/// the code: a bool per page, set by a release the code must accept and cleared by a reclaim or
/// by a heal of the window the model computes.
#[cfg(all(test, not(loom)))]
mod every_sequence {
    use super::{FaultOutcome, ReleasedRam, Stage2, ZeroOnRelease, contains_point, hv_return_t};
    use crate::bindings::{HV_ERROR, HV_SUCCESS};
    use crate::host_page_size;
    use std::ops::Range;
    use std::sync::Mutex;

    const PAGES: usize = 7;
    const REGION_A: Range<usize> = 0..4;
    const REGION_B: Range<usize> = 4..6;
    const HOST_A: u64 = 0x10_0000_0000;
    const HOST_B: u64 = 0x20_0000_0000;
    const WINDOW_PAGES: u64 = 4;
    const TRANSLATION_FAULT: u64 = 0b000101;
    const DEPTH: usize = 4;

    fn page() -> u64 {
        host_page_size()
    }
    fn base() -> u64 {
        0x4000_0000 + page()
    }
    fn gpa(p: usize) -> u64 {
        base() + p as u64 * page()
    }
    fn region(p: usize) -> Option<Range<usize>> {
        [REGION_A, REGION_B].into_iter().find(|r| r.contains(&p))
    }
    fn host_of(p: usize) -> u64 {
        if REGION_A.contains(&p) {
            HOST_A + (p - REGION_A.start) as u64 * page()
        } else {
            HOST_B + (p - REGION_B.start) as u64 * page()
        }
    }

    #[derive(Default)]
    struct State {
        unmapped: [bool; PAGES],
        reusable: [bool; PAGES],
        fail_unmap: bool,
        fail_map: bool,
        wrong: Vec<String>,
    }

    #[derive(Default)]
    struct Model(Mutex<State>);

    impl Model {
        fn pages_of_gpa(s: &mut State, gpa: u64, len: u64) -> Vec<usize> {
            let pages: Vec<usize> = (0..len / page())
                .map(|i| ((gpa - base()) / page() + i) as usize)
                .collect();
            if let Some(p) = pages.iter().find(|&&p| region(p).is_none()) {
                s.wrong
                    .push(format!("a stage-2 call reaches page {p}, which is not RAM"));
            }
            pages.into_iter().filter(|&p| region(p).is_some()).collect()
        }

        fn pages_of_host(s: &mut State, host: u64, len: u64) -> Vec<usize> {
            (0..len / page())
                .filter_map(|i| {
                    let h = host + i * page();
                    let p = (0..PAGES).find(|&p| region(p).is_some() && host_of(p) == h);
                    if p.is_none() {
                        s.wrong
                            .push(format!("host address {h:#x} backs no guest page"));
                    }
                    p
                })
                .collect()
        }
    }

    impl Stage2 for Model {
        fn unmap(&self, gpa: u64, len: u64) -> hv_return_t {
            let mut s = self.0.lock().unwrap();
            if std::mem::take(&mut s.fail_unmap) {
                return HV_ERROR as hv_return_t;
            }
            for p in Self::pages_of_gpa(&mut s, gpa, len) {
                s.unmapped[p] = true;
            }
            HV_SUCCESS as hv_return_t
        }

        fn map(&self, host: u64, gpa: u64, len: u64) -> hv_return_t {
            let mut s = self.0.lock().unwrap();
            let pages = Self::pages_of_gpa(&mut s, gpa, len);
            for (i, &p) in pages.iter().enumerate() {
                let h = host + i as u64 * page();
                if h != host_of(p) {
                    s.wrong.push(format!(
                        "page {p} mapped to host {h:#x}, not its own {:#x}",
                        host_of(p)
                    ));
                }
                if s.reusable[p] {
                    s.wrong
                        .push(format!("page {p} mapped while still reusable"));
                }
            }
            if pages.iter().any(|&p| !s.unmapped[p]) {
                s.wrong.push(format!("a map over live pages in {pages:?}"));
                return HV_ERROR as hv_return_t;
            }
            if std::mem::take(&mut s.fail_map) {
                return HV_ERROR as hv_return_t;
            }
            for p in pages {
                s.unmapped[p] = false;
            }
            HV_SUCCESS as hv_return_t
        }

        unsafe fn discard(&self, host: u64, len: u64, _zero: bool) -> std::io::Result<()> {
            let mut s = self.0.lock().unwrap();
            for p in Self::pages_of_host(&mut s, host, len) {
                if !s.unmapped[p] {
                    s.wrong
                        .push(format!("page {p} made reusable while the guest reaches it"));
                }
                s.reusable[p] = true;
            }
            Ok(())
        }

        fn reuse(&self, host: u64, len: u64) -> std::io::Result<()> {
            let mut s = self.0.lock().unwrap();
            for p in Self::pages_of_host(&mut s, host, len) {
                s.reusable[p] = false;
            }
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Release(usize, usize),
        Reclaim(usize, usize),
        /// The guest touches a page, which faults only if the page is unmapped.
        Touch(usize),
        /// A stage-2 fault reported whatever the page's state, as when another vCPU's heal
        /// raced it.
        Fault(usize),
        FailUnmap,
        FailMap,
    }

    fn alphabet() -> Vec<Op> {
        let mut ops = Vec::new();
        for f in 0..PAGES {
            for n in 1..=2 {
                ops.push(Op::Release(f, n));
            }
        }
        ops.extend([
            Op::Reclaim(0, PAGES),
            Op::Reclaim(3, 2),
            Op::Reclaim(0, 1),
            Op::Reclaim(4, 1),
        ]);
        ops.extend((0..PAGES - 1).map(Op::Touch));
        ops.extend((0..PAGES).map(Op::Fault));
        ops.extend([Op::FailUnmap, Op::FailMap]);
        ops
    }

    /// The pages a heal of page `p` maps back: the window-aligned chunk around it, clipped to
    /// its region.
    fn window(p: usize) -> Range<usize> {
        let r = region(p).unwrap();
        let start = gpa(p) & !(WINDOW_PAGES * page() - 1);
        let first = start.saturating_sub(base()) / page();
        let end = (start + WINDOW_PAGES * page() - base()) / page();
        (first as usize).max(r.start)..(end as usize).min(r.end)
    }

    #[derive(Default)]
    struct Expect {
        released: [bool; PAGES],
        fail_unmap: bool,
        fail_map: bool,
    }

    impl Expect {
        fn heal(&mut self, p: usize) -> &'static str {
            if std::mem::take(&mut self.fail_map) {
                return "Fatal";
            }
            for q in window(p) {
                self.released[q] = false;
            }
            "Healed"
        }
    }

    fn outcome(o: FaultOutcome) -> &'static str {
        match o {
            FaultOutcome::Healed => "Healed",
            FaultOutcome::Retry => "Retry",
            FaultOutcome::NotHandled => "NotHandled",
            FaultOutcome::Fatal => "Fatal",
        }
    }

    fn step(rr: &ReleasedRam<Model>, e: &mut Expect, op: Op) -> Result<(), String> {
        let (got, want) = match op {
            Op::Release(f, n) => {
                let valid = region(f).is_some_and(|r| r.contains(&(f + n - 1)));
                let want = valid && !std::mem::take(&mut e.fail_unmap);
                if want {
                    (f..f + n).for_each(|p| e.released[p] = true);
                }
                let got = rr.release(gpa(f), n as u64 * page(), f % 2 == 0);
                (got.to_string(), want.to_string())
            }
            Op::Reclaim(f, n) => {
                let any = (f..f + n).any(|p| e.released[p]);
                let want = !any || !std::mem::take(&mut e.fail_map);
                if want {
                    (f..f + n).for_each(|p| e.released[p] = false);
                }
                let got = rr.reclaim(gpa(f), n as u64 * page());
                (got.to_string(), want.to_string())
            }
            Op::Touch(p) => {
                if !rr.stage2.0.lock().unwrap().unmapped[p] {
                    return Ok(());
                }
                let want = if e.released[p] { e.heal(p) } else { "Retry" };
                let got = outcome(rr.handle_fault(gpa(p), TRANSLATION_FAULT));
                (got.to_string(), want.to_string())
            }
            Op::Fault(p) => {
                let want = match region(p) {
                    None => "NotHandled",
                    Some(_) if e.released[p] => e.heal(p),
                    Some(_) => "Retry",
                };
                let got = outcome(rr.handle_fault(gpa(p), TRANSLATION_FAULT));
                (got.to_string(), want.to_string())
            }
            Op::FailUnmap => {
                e.fail_unmap = true;
                rr.stage2.0.lock().unwrap().fail_unmap = true;
                return Ok(());
            }
            Op::FailMap => {
                e.fail_map = true;
                rr.stage2.0.lock().unwrap().fail_map = true;
                return Ok(());
            }
        };
        if got != want {
            return Err(format!("returned {got}, the model says {want}"));
        }
        Ok(())
    }

    fn check(rr: &ReleasedRam<Model>, e: &Expect) -> Result<(), String> {
        let s = rr.stage2.0.lock().unwrap();
        if let Some(w) = s.wrong.first() {
            return Err(w.clone());
        }
        let set = rr.released.lock().unwrap();
        let not = |b: bool| if b { "" } else { "not " };
        for p in (0..PAGES).filter(|&p| region(p).is_some()) {
            let in_set = contains_point(&set, gpa(p));
            if in_set != e.released[p] {
                return Err(format!(
                    "page {p} is {}in the released set; the model says it is {}released",
                    not(in_set),
                    not(e.released[p])
                ));
            }
            if s.unmapped[p] != e.released[p] {
                return Err(format!(
                    "page {p} is {}mapped in stage 2; the model says it is {}released",
                    not(!s.unmapped[p]),
                    not(e.released[p])
                ));
            }
        }
        Ok(())
    }

    fn run(seq: &[Op]) -> Result<(), String> {
        let regions = [REGION_A, REGION_B]
            .into_iter()
            .map(|r| (gpa(r.start), host_of(r.start), r.len() as u64 * page()))
            .collect();
        let rr = ReleasedRam::with_stage2(
            Model::default(),
            regions,
            WINDOW_PAGES * page(),
            ZeroOnRelease::InflateQueue,
        );
        let mut e = Expect::default();
        for (i, &op) in seq.iter().enumerate() {
            step(&rr, &mut e, op)
                .and_then(|()| check(&rr, &e))
                .map_err(|m| format!("{:?}, at step {}: {m}", &seq[..=i], i + 1))?;
        }
        Ok(())
    }

    fn walk(ops: &[Op], seq: &mut Vec<Op>, walked: &mut u64) {
        if seq.len() == DEPTH {
            *walked += 1;
            replay(seq);
            return;
        }
        for &op in ops {
            seq.push(op);
            walk(ops, seq, walked);
            seq.pop();
        }
    }

    fn replay(seq: &[Op]) {
        if let Err(m) = run(seq) {
            panic!("{m}");
        }
    }

    /// A release that repeats an earlier one, whose unmap then fails, must leave the earlier
    /// release recorded: its pages are still unmapped.
    #[test]
    fn a_failed_release_keeps_the_pages_released_before_it() {
        replay(&[Op::Release(0, 1), Op::FailUnmap, Op::Release(0, 2)]);
    }

    /// A reclaim whose first map fails must keep every range it had not mapped yet.
    #[test]
    fn a_failed_reclaim_keeps_the_ranges_it_did_not_reach() {
        replay(&[
            Op::Release(0, 1),
            Op::Release(2, 1),
            Op::FailMap,
            Op::Reclaim(0, PAGES),
        ]);
    }

    /// Releases on each side of the A|B boundary coalesce into one range; mapping it back must
    /// take each region's pages from that region's host mapping.
    #[test]
    fn a_heal_across_adjacent_regions_maps_each_from_its_own_host() {
        replay(&[Op::Release(3, 1), Op::Release(4, 1), Op::Reclaim(3, 2)]);
    }

    #[test]
    fn every_release_and_heal_sequence_matches_the_model() {
        let ops = alphabet();
        let mut walked = 0;
        walk(&ops, &mut Vec::with_capacity(DEPTH), &mut walked);
        assert_eq!(walked, (ops.len() as u64).pow(DEPTH as u32));
    }
}

/// loom models of release, heal and reclaim racing one another, over one two-page region whose
/// heal window is the whole region.
///
/// Run with `RUSTFLAGS="--cfg loom" cargo test --release --lib loom_model` in this crate. The
/// released-set lock is loom's, and so is the lock of the stage-2 stand-in, so a vCPU's check that
/// its page is mapped is a point loom can interleave against the other thread's calls. After each
/// run the set must equal what stage 2 has unmapped, no page may have been made reusable while the
/// guest could reach it or mapped while still reusable, and every vCPU access must have landed.
#[cfg(all(test, loom))]
mod loom_model {
    use super::{FaultOutcome, ReleasedRam, Stage2, ZeroOnRelease, contains_point, hv_return_t};
    use crate::bindings::{HV_ERROR, HV_SUCCESS};
    use crate::host_page_size;
    use loom::sync::{Arc, Mutex};
    use loom::thread;

    const GPA: u64 = 0x4000_0000;
    const HOST: u64 = 0x10_0000_0000;
    const PAGES: usize = 2;
    const TRANSLATION_FAULT: u64 = 0b000101;
    /// A vCPU's access that faults, heals or retries, then faults again, has not landed.
    const ATTEMPTS: usize = 3;

    fn page() -> u64 {
        host_page_size()
    }
    fn index(addr: u64, base: u64) -> usize {
        ((addr - base) / page()) as usize
    }

    #[derive(Default)]
    struct State {
        unmapped: [bool; PAGES],
        reusable: [bool; PAGES],
        wrong: Vec<String>,
    }

    #[derive(Default)]
    struct Model(Mutex<State>);

    impl Stage2 for Model {
        fn unmap(&self, gpa: u64, len: u64) -> hv_return_t {
            let mut s = self.0.lock().unwrap();
            for p in index(gpa, GPA)..index(gpa + len, GPA) {
                s.unmapped[p] = true;
            }
            HV_SUCCESS as hv_return_t
        }

        fn map(&self, _host: u64, gpa: u64, len: u64) -> hv_return_t {
            let mut s = self.0.lock().unwrap();
            let pages = index(gpa, GPA)..index(gpa + len, GPA);
            if pages.clone().any(|p| !s.unmapped[p]) {
                s.wrong.push(format!("a map over live pages in {pages:?}"));
                return HV_ERROR as hv_return_t;
            }
            for p in pages {
                if s.reusable[p] {
                    s.wrong
                        .push(format!("page {p} mapped while still reusable"));
                }
                s.unmapped[p] = false;
            }
            HV_SUCCESS as hv_return_t
        }

        unsafe fn discard(&self, host: u64, len: u64, _zero: bool) -> std::io::Result<()> {
            let mut s = self.0.lock().unwrap();
            for p in index(host, HOST)..index(host + len, HOST) {
                if !s.unmapped[p] {
                    s.wrong
                        .push(format!("page {p} made reusable while the guest reaches it"));
                }
                s.reusable[p] = true;
            }
            Ok(())
        }

        fn reuse(&self, host: u64, len: u64) -> std::io::Result<()> {
            let mut s = self.0.lock().unwrap();
            for p in index(host, HOST)..index(host + len, HOST) {
                s.reusable[p] = false;
            }
            Ok(())
        }
    }

    fn ram(released: &[usize]) -> Arc<ReleasedRam<Model>> {
        let rr = ReleasedRam::with_stage2(
            Model::default(),
            vec![(GPA, HOST, PAGES as u64 * page())],
            PAGES as u64 * page(),
            ZeroOnRelease::InflateQueue,
        );
        for &p in released {
            assert!(rr.release(GPA + p as u64 * page(), page(), false));
        }
        Arc::new(rr)
    }

    /// A vCPU touching page `p`: the access lands once the page is mapped, and a fault on the
    /// way there must heal or retry, never anything else.
    fn touch(rr: &ReleasedRam<Model>, p: usize) {
        for _ in 0..ATTEMPTS {
            if !rr.stage2.0.lock().unwrap().unmapped[p] {
                return;
            }
            match rr.handle_fault(GPA + p as u64 * page(), TRANSLATION_FAULT) {
                FaultOutcome::Healed | FaultOutcome::Retry => {}
                FaultOutcome::NotHandled => panic!("a fault on page {p} was not handled"),
                FaultOutcome::Fatal => panic!("a fault on page {p} stopped the VM"),
            }
        }
        panic!("page {p} was still unmapped after {ATTEMPTS} faults");
    }

    fn check(rr: &ReleasedRam<Model>) {
        let s = rr.stage2.0.lock().unwrap();
        if let Some(w) = s.wrong.first() {
            panic!("{w}");
        }
        let set = rr.released.lock().unwrap();
        for p in 0..PAGES {
            let in_set = contains_point(&set, GPA + p as u64 * page());
            assert_eq!(
                in_set, s.unmapped[p],
                "page {p}: in the released set {in_set}, unmapped {}",
                s.unmapped[p]
            );
        }
    }

    /// A release while a vCPU faults on the other page of the same heal window. The heal can
    /// run before, after or (if the lock did not cover it) between the release's unmap and its
    /// discard, which would leave a page the guest reaches marked reusable.
    #[test]
    fn a_release_races_a_heal_of_its_window() {
        loom::model(|| {
            let rr = ram(&[0]);
            let releaser = {
                let rr = rr.clone();
                thread::spawn(move || assert!(rr.release(GPA + page(), page(), true)))
            };
            touch(&rr, 0);
            releaser.join().unwrap();
            check(&rr);
        });
    }

    /// Two vCPUs fault on one released page. One heals it; the other finds it gone from the
    /// set and must retry into the mapping, not stop the VM.
    #[test]
    fn two_vcpus_fault_on_one_page() {
        loom::model(|| {
            let rr = ram(&[0]);
            let other = {
                let rr = rr.clone();
                thread::spawn(move || touch(&rr, 0))
            };
            touch(&rr, 0);
            other.join().unwrap();
            check(&rr);
        });
    }

    /// A deflate's reclaim takes back the pages a vCPU is faulting on.
    #[test]
    fn a_reclaim_races_a_heal() {
        loom::model(|| {
            let rr = ram(&[0, 1]);
            let reclaimer = {
                let rr = rr.clone();
                thread::spawn(move || assert!(rr.reclaim(GPA, PAGES as u64 * page())))
            };
            touch(&rr, 1);
            reclaimer.join().unwrap();
            check(&rr);
        });
    }
}
