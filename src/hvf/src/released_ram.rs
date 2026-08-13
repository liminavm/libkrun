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
use std::sync::{Mutex, Once, OnceLock};
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

pub struct ReleasedRam {
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

        let chunk_mib = std::env::var("LIMINA_BALLOON_REMAP_CHUNK_MIB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(2);
        let chunk = (chunk_mib << 20).next_power_of_two().max(page);

        let handler_regions: &'static [(u64, u64)] = Box::leak(
            regions
                .iter()
                .map(|r: &RamRegion| (r.host, r.host + r.len))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        );

        Self {
            regions,
            released: Mutex::new(BTreeMap::new()),
            chunk,
            heals: AtomicU64::new(0),
            released_bytes: AtomicU64::new(0),
            remapped_bytes: AtomicU64::new(0),
            stray_faults: AtomicU64::new(0),
            last_stray: Mutex::new((u64::MAX, 0)),
            zero_on_release: match std::env::var("LIMINA_BALLOON_RELEASE_MEMSET").as_deref() {
                Ok("1") => ZeroOnRelease::All,
                Ok("0") | Ok("none") => ZeroOnRelease::None,
                _ => ZeroOnRelease::InflateQueue,
            },
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
        insert_range(&mut released, gpa, len);
        let ret = unsafe { hv_vm_unmap(gpa, len as usize) };
        if ret != HV_SUCCESS {
            error!("released-ram: hv_vm_unmap(gpa={gpa:#x}, len={len:#x}) failed: {ret:#x}");
            remove_overlaps(&mut released, gpa, gpa + len);
            return false;
        }
        let host = Self::host_of(region, gpa);
        let zero = match self.zero_on_release {
            ZeroOnRelease::All => true,
            ZeroOnRelease::InflateQueue => from_inflate_queue,
            ZeroOnRelease::None => false,
        };
        if zero {
            // SAFETY: the range was just unmapped from the guest (above, under the lock),
            // is balloon-owned, and lies inside this region's host mapping — no guest
            // access can race the write; a touch faults and heals afterward.
            unsafe { std::ptr::write_bytes(host as *mut u8, 0, len as usize) };
        }
        let rc = unsafe {
            libc::madvise(
                host as *mut libc::c_void,
                len as usize,
                libc::MADV_FREE_REUSABLE,
            )
        };
        if rc != 0 {
            // The unmap stands (a guest touch will fault and heal); only the host-side
            // reclaim didn't happen, so the pages simply stay resident.
            warn!(
                "released-ram: madvise(MADV_FREE_REUSABLE) at {host:#x} len={len:#x} failed: {}",
                std::io::Error::last_os_error()
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
        for &(start, rlen) in &ranges {
            if !self.reuse_and_map(&mut released, start, rlen) {
                return false;
            }
        }
        true
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

        let window_start = (pa & !(self.chunk - 1)).max(region.gpa);
        let window_end = (window_start + self.chunk).min(region.gpa + region.len);
        let ranges = remove_overlaps(&mut released, window_start, window_end);
        for &(start, len) in &ranges {
            if !self.reuse_and_map(&mut released, start, len) {
                return FaultOutcome::Fatal;
            }
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

    /// `MADV_FREE_REUSE` + `hv_vm_map` one extracted range. On map failure the range is
    /// reinserted (bookkeeping stays exact) and false is returned. REUSE on pages the OS
    /// never actually reclaimed is a no-op, so no per-page state is needed.
    fn reuse_and_map(&self, released: &mut BTreeMap<u64, u64>, gpa: u64, len: u64) -> bool {
        let region = self
            .region_of(gpa)
            .expect("released range outside every RAM region");
        let host = Self::host_of(region, gpa);
        let rc = unsafe {
            libc::madvise(
                host as *mut libc::c_void,
                len as usize,
                libc::MADV_FREE_REUSE,
            )
        };
        if rc != 0 {
            warn!(
                "released-ram: madvise(MADV_FREE_REUSE) at {host:#x} len={len:#x} failed: {}",
                std::io::Error::last_os_error()
            );
        }
        let ret = unsafe {
            hv_vm_map(
                host as *mut core::ffi::c_void,
                gpa,
                len as usize,
                (HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC).into(),
            )
        };
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

#[cfg(test)]
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
        let mut sweeps = 0;
        while rr.stats().sweep_faults == faults0 && sweeps < 50 {
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
