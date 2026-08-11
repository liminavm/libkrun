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

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

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
}

pub enum FaultOutcome {
    /// The fault hit a released range; it has been re-validated and re-mapped. Re-run the
    /// vCPU without advancing the PC so the faulting access retries against the new mapping.
    Healed,
    /// Not a released-RAM fault (outside guest RAM, not a translation fault, or the range
    /// is not in the released set) — fall through to the caller's existing handling.
    NotHandled,
    /// The range was released but could not be re-mapped; resuming the guest would livelock
    /// on the same fault. The VM must stop.
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
}

impl ReleasedRam {
    /// `regions` are the guest RAM regions as `(gpa, host_va, len)`. Regions not aligned to
    /// the host page granule are dropped (loudly): release/heal must never round.
    pub fn new(regions: Vec<(u64, u64, u64)>) -> Self {
        let page = host_page_size();
        let regions = regions
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

        Self {
            regions,
            released: Mutex::new(BTreeMap::new()),
            chunk,
            heals: AtomicU64::new(0),
            released_bytes: AtomicU64::new(0),
            remapped_bytes: AtomicU64::new(0),
            stray_faults: AtomicU64::new(0),
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
    /// must then leave the pages alone.
    ///
    /// The madvise happens under the released-set lock so a concurrent guest touch can
    /// never interleave a heal (REUSE + remap) between our unmap and our REUSABLE, which
    /// would re-mark live-again pages as disposable.
    pub fn release(&self, gpa: u64, len: u64) -> bool {
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
            // A translation fault inside guest RAM that we did not create: either stale
            // bookkeeping or an unmapped-RAM access we don't know about. Loud (it should
            // never happen), then fall through to the caller's existing path.
            let strays = self.stray_faults.fetch_add(1, Ordering::Relaxed) + 1;
            if strays <= 8 || strays.is_multiple_of(1024) {
                error!(
                    "released-ram: stage-2 translation fault at pa={pa:#x} (in guest RAM) with \
                     no released range covering it (stray #{strays}); not healing"
                );
            }
            return FaultOutcome::NotHandled;
        }

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
    use super::{contains_point, insert_range, remove_overlaps};
    use std::collections::BTreeMap;

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
}
