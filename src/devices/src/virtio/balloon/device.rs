use std::cmp;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use utils::eventfd::{EventFd, EFD_NONBLOCK};
use vm_memory::{ByteValued, Bytes, GuestAddress, GuestMemory, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, BalloonError, DeviceQueue, DeviceState, QueueConfig,
    VirtioDevice,
};
use super::{defs, defs::uapi};
use crate::virtio::InterruptTransport;
#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Memory::DiscardVirtualMemory;

// Inflate queue.
pub(crate) const IFQ_INDEX: usize = 0;
// Deflate queue.
pub(crate) const DFQ_INDEX: usize = 1;
// Stats queue.
pub(crate) const STQ_INDEX: usize = 2;
// Page-hinting queue.
pub(crate) const PHQ_INDEX: usize = 3;
// Free page reporting queue.
pub(crate) const FRQ_INDEX: usize = 4;

// Supported features. DEFLATE_ON_OOM lets the guest return ballooned pages under its own memory
// pressure (the OOM safety net) — advertised together with the inflate/deflate handlers so we never
// drive a target the guest can't back out of.
pub(crate) const AVAIL_FEATURES: u64 = (1 << uapi::VIRTIO_F_VERSION_1 as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_STATS_VQ as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_DEFLATE_ON_OOM as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_FREE_PAGE_HINT as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_REPORTING as u64);

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct VirtioBalloonConfig {
    /* Number of pages host wants Guest to give up. */
    num_pages: u32,
    /* Number of pages we've actually got in balloon. */
    actual: u32,
    /* Free page report command id, readonly by guest */
    free_page_report_cmd_id: u32,
    /* Stores PAGE_POISON if page poisoning is in use */
    poison_val: u32,
}

// Safe because it only has data and has no implicit padding.
unsafe impl ByteValued for VirtioBalloonConfig {}

// virtio-balloon's page unit is always 4 KiB (VIRTIO_BALLOON_PFN_SHIFT == 12), independent of the
// guest's MMU page size. Free-page reporting (FRQ) hands us byte ranges of free guest memory; we
// track them at this finest granularity so the 4 KiB-guest / 16 KiB-host mismatch is handled
// correctly.
const GUEST_PAGE: usize = 4096;

/// The host page size (16 KiB on Apple Silicon). `madvise` operates on whole host pages, so reclaim
/// must never hand a host page back to macOS unless *every* guest page inside it is free.
fn host_page_size() -> usize {
    // SAFETY: `sysconf(_SC_PAGESIZE)` is always valid and returns a positive power of two.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Accumulates guest-reported *free* 4 KiB runs and yields only the host pages that are provably
/// safe to return to macOS with `MADV_FREE_REUSABLE`.
///
/// INVARIANT — the whole safety proof for the 4 KiB-guest / 16 KiB-host page-size mismatch: a host
/// page is emitted by [`Self::take_full_pages`] **only** when every one of its constituent guest
/// pages was reported free, so the `madvise` can never discard a still-live guest page. Sub-page
/// (unaligned) fringes of a reported run are rounded *inward* (start up, end down) so a partially
/// free guest page is never counted as free, regardless of how `MADV_FREE_REUSABLE` itself rounds.
struct ReclaimCoalescer {
    host_page: usize,
    /// `(1 << (host_page / GUEST_PAGE)) - 1` — the all-sub-pages-free mask (`0b1111` on 16K/4K).
    full_mask: u64,
    /// host-page base address -> bitmask of which of its 4 KiB sub-pages have been reported free.
    partial: std::collections::HashMap<usize, u64>,
}

impl ReclaimCoalescer {
    fn new(host_page: usize) -> Self {
        let sub = host_page / GUEST_PAGE;
        debug_assert!(
            (1..=64).contains(&sub),
            "host/guest page ratio {sub} out of bitmask range"
        );
        let full_mask = if sub >= 64 {
            u64::MAX
        } else {
            (1u64 << sub) - 1
        };
        Self {
            host_page,
            full_mask,
            partial: std::collections::HashMap::new(),
        }
    }

    /// Record a reported-free run `[addr, addr + len)` (host addresses). Never madvises. Rounds the
    /// run *inward* to whole guest pages so an unaligned fringe never marks a partially-live page
    /// free. In the real FRQ path runs are already page-aligned page-multiples; the rounding is a
    /// correctness-by-construction guard, not an expected case.
    fn add(&mut self, addr: usize, len: usize) {
        if len < GUEST_PAGE {
            return;
        }
        let start = (addr + GUEST_PAGE - 1) & !(GUEST_PAGE - 1); // round up
        let end = (addr + len) & !(GUEST_PAGE - 1); // round down
        let mut p = start;
        while p < end {
            let base = p & !(self.host_page - 1);
            let sub = (p - base) / GUEST_PAGE;
            *self.partial.entry(base).or_insert(0) |= 1u64 << sub;
            p += GUEST_PAGE;
        }
    }

    /// Drain the host pages whose every sub-page is now free, returning `(base, host_page_len)`
    /// ranges safe to `MADV_FREE_REUSABLE`. Partially covered host pages are retained (FRQ callers
    /// discard the coalescer after each head, so retained partials are simply not reclaimed).
    fn take_full_pages(&mut self) -> Vec<(usize, usize)> {
        let full = self.full_mask;
        let hp = self.host_page;
        let mut out = Vec::new();
        self.partial.retain(|&base, &mut mask| {
            if mask == full {
                out.push((base, hp));
                false
            } else {
                true
            }
        });
        out
    }
}

/// Balloon statistics surfaced to the host policy (limina drives the target; libkrun only reports).
#[derive(Copy, Clone, Debug, Default)]
pub struct BalloonStats {
    /// Pages (4 KiB units) the guest currently holds in the balloon (its self-reported `actual`).
    pub actual_pages: u32,
    /// Cumulative bytes handed back to macOS via `MADV_FREE_REUSABLE` (free-page reporting + fully
    /// inflated host pages). A rough "how much we've returned" counter, not a live residency.
    pub reclaimed_bytes: u64,
}

/// limina: a cloneable, thread-safe handle that lets the host push a balloon target into the live
/// device after boot, from any thread (limina-vmm's control-socket listener funnels here). Mechanism
/// only — *policy* (when, and to what target) lives in limina (the PSI autoballoon loop). Obtained
/// from [`Balloon::balloon_control_handle`] before boot and held by limina-vmm. Mirrors the shipped
/// [`DisplayResizeHandle`](crate::virtio::DisplayResizeHandle).
///
/// `set_target_pages` stores the target and kicks the balloon's event loop; the device sets
/// `num_pages` and raises a virtio-balloon config-change interrupt so the guest inflates/deflates
/// toward it. `actual`/`reclaimed` are published by the device for the policy to read back.
#[derive(Clone)]
pub struct BalloonControlHandle {
    /// Wakes the balloon's EventManager subscriber; shared (same eventfd) with the device.
    target_evt: Arc<EventFd>,
    /// The latest target in 4 KiB pages; the device takes it on wake (coalescing).
    pending_target: Arc<Mutex<Option<u32>>>,
    /// The guest's self-reported balloon size (4 KiB pages), published by the device.
    actual: Arc<AtomicU32>,
    /// Cumulative bytes reclaimed via `MADV_FREE_REUSABLE`, published by the device.
    reclaimed: Arc<AtomicU64>,
}

impl BalloonControlHandle {
    /// Request the guest balloon to `target` 4 KiB pages. Stores the target (coalescing with any
    /// not-yet-applied one) and kicks the device. Returns `false` (logged) if the device can't be
    /// woken.
    pub fn set_target_pages(&self, target: u32) -> bool {
        *self.pending_target.lock().unwrap() = Some(target);
        if let Err(e) = self.target_evt.write(1) {
            error!("balloon: failed to kick the device for a new target: {e}");
            return false;
        }
        true
    }

    /// The guest's current balloon size in 4 KiB pages (its self-reported `actual`).
    pub fn get_actual(&self) -> u32 {
        self.actual.load(Ordering::Relaxed)
    }

    /// A snapshot of balloon stats for the policy.
    pub fn get_stats(&self) -> BalloonStats {
        BalloonStats {
            actual_pages: self.actual.load(Ordering::Relaxed),
            reclaimed_bytes: self.reclaimed.load(Ordering::Relaxed),
        }
    }
}

pub struct Balloon {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    config: VirtioBalloonConfig,
    /// Host page size, cached once (16 KiB on Apple Silicon).
    host_page: usize,
    /// `(1 << (host_page / GUEST_PAGE)) - 1` — the all-sub-pages-inflated mask for inflate
    /// coalescing.
    full_mask: u64,
    /// Inflate coalescing state: host-page base -> bitmask of its 4 KiB sub-pages currently in the
    /// balloon. A host page is reclaimed only when fully inflated (every sub-page balloon-owned), so
    /// `MADV_FREE_REUSABLE` never discards a live guest page. Persisted across heads (unlike FRQ):
    /// inflated pages stay balloon-owned until deflated.
    inflate_mask: HashMap<usize, u64>,
    /// Host-page bases currently reclaimed via inflate, so deflate knows which to un-track.
    ballooned_pages: HashSet<usize>,
    /// Wakes this subscriber when the host sets a new target. Registered with the EventManager at
    /// activate, alongside the queue eventfds.
    pub(crate) target_evt: Arc<EventFd>,
    /// The pending host target (4 KiB pages); taken on a `target_evt` wake.
    pending_target: Arc<Mutex<Option<u32>>>,
    /// The guest's self-reported balloon size (4 KiB pages), published to the control handle.
    actual_shared: Arc<AtomicU32>,
    /// Cumulative bytes reclaimed via `MADV_FREE_REUSABLE`, published to the control handle.
    reclaimed_bytes: Arc<AtomicU64>,
}

impl Balloon {
    pub fn new() -> super::Result<Balloon> {
        let host_page = host_page_size();
        let sub = host_page / GUEST_PAGE;
        let full_mask = if sub >= 64 {
            u64::MAX
        } else {
            (1u64 << sub) - 1
        };
        Ok(Balloon {
            queues: None,
            avail_features: AVAIL_FEATURES,
            acked_features: 0,
            activate_evt: EventFd::new(EFD_NONBLOCK).map_err(BalloonError::EventFd)?,
            device_state: DeviceState::Inactive,
            config: VirtioBalloonConfig::default(),
            host_page,
            full_mask,
            inflate_mask: HashMap::new(),
            ballooned_pages: HashSet::new(),
            target_evt: Arc::new(EventFd::new(EFD_NONBLOCK).map_err(BalloonError::EventFd)?),
            pending_target: Arc::new(Mutex::new(None)),
            actual_shared: Arc::new(AtomicU32::new(0)),
            reclaimed_bytes: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn id(&self) -> &str {
        defs::BALLOON_DEV_ID
    }

    /// limina: a handle for pushing balloon targets into this live device. Grab it before boot and
    /// hold it host-side (limina-vmm); see [`BalloonControlHandle`].
    pub fn balloon_control_handle(&self) -> BalloonControlHandle {
        BalloonControlHandle {
            target_evt: self.target_evt.clone(),
            pending_target: self.pending_target.clone(),
            actual: self.actual_shared.clone(),
            reclaimed: self.reclaimed_bytes.clone(),
        }
    }

    pub fn process_frq(&mut self) -> bool {
        debug!("balloon: process_frq()");
        let host_page = self.host_page;
        let reclaimed = self.reclaimed_bytes.clone();
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem,
            // This should never happen, it's been already validated in the event handler.
            DeviceState::Inactive => unreachable!(),
        };

        let queues = self
            .queues
            .as_mut()
            .expect("queues should exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[FRQ_INDEX].queue.pop(mem) {
            let index = head.index;
            // Coalesce this head's reported runs into whole host pages. FRQ coalescing is strictly
            // per-head and never persisted: a reported page stays guest-owned and may be
            // reallocated once we add_used it, so carrying a partial across heads could later
            // "complete" a host page whose sub-page is live again -> corruption.
            let mut coalescer = ReclaimCoalescer::new(host_page);
            for desc in head.into_iter() {
                let host_addr = mem.get_host_address(desc.addr).unwrap() as usize;
                coalescer.add(host_addr, desc.len as usize);
            }
            // madvise BEFORE add_used: page_reporting keeps these pages isolated from the guest
            // allocator until the descriptor is marked used, which closes the reallocation window.
            // MADV_FREE_REUSABLE (not MADV_DONTNEED, which returns nothing on macOS — see
            // spikes/balloon-madvise/RESULTS.md) actually debits the worker's phys_footprint.
            for (base, len) in coalescer.take_full_pages() {
                // SAFETY: `base`/`len` are host-page-aligned and every guest page inside the range
                // was reported free in this head, so reclaiming the whole host page cannot drop
                // live guest data.
                let rc = unsafe {
                    libc::madvise(base as *mut libc::c_void, len, libc::MADV_FREE_REUSABLE)
                };
                if rc != 0 {
                    warn!(
                        "balloon: madvise(MADV_FREE_REUSABLE) at {base:#x} len={len} failed: {}",
                        std::io::Error::last_os_error()
                    );
                } else {
                    reclaimed.fetch_add(len as u64, Ordering::Relaxed);
                }
            }

            have_used = true;
            if let Err(e) = queues[FRQ_INDEX].queue.add_used(mem, index, 0) {
                error!("failed to add used elements to the queue: {e:?}");
            }
        }

        have_used
    }

    /// Inflate: the guest handed us page-frame numbers (arrays of `__le32` at
    /// `VIRTIO_BALLOON_PFN_SHIFT`) for pages it has placed in the balloon and promises not to touch
    /// until deflate. We reclaim each fully covered 16 KiB host page with `MADV_FREE_REUSABLE`. The
    /// inflate coalescer **persists** across heads (unlike FRQ): inflated pages stay balloon-owned,
    /// so accumulating sub-pages from different heads is safe and recovers cross-head host pages on a
    /// stock 4 KiB guest. A host page is reclaimed only once all four sub-pages are inflated, so we
    /// never discard a still-live (non-inflated) guest page.
    pub fn process_ifq(&mut self) -> bool {
        let host_page = self.host_page;
        let full_mask = self.full_mask;
        let reclaimed = self.reclaimed_bytes.clone();
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };
        let queues = self
            .queues
            .as_mut()
            .expect("queues should exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[IFQ_INDEX].queue.pop(&mem) {
            let index = head.index;
            for desc in head.into_iter() {
                let pfn_count = (desc.len as usize) / 4;
                for i in 0..pfn_count {
                    let pfn: u32 = match mem.read_obj(GuestAddress(desc.addr.0 + (i * 4) as u64)) {
                        Ok(p) => p,
                        Err(e) => {
                            error!("balloon: failed to read inflate PFN: {e:?}");
                            break;
                        }
                    };
                    let guest = GuestAddress((pfn as u64) << uapi::VIRTIO_BALLOON_PFN_SHIFT);
                    let host_addr = match mem.get_host_address(guest) {
                        Ok(p) => p as usize,
                        Err(e) => {
                            error!("balloon: inflate PFN {pfn:#x} not in guest memory: {e:?}");
                            continue;
                        }
                    };
                    let base = host_addr & !(host_page - 1);
                    let sub = (host_addr - base) / GUEST_PAGE;
                    let entry = self.inflate_mask.entry(base).or_insert(0);
                    *entry |= 1u64 << sub;
                    if *entry == full_mask && self.ballooned_pages.insert(base) {
                        // SAFETY: every 4 KiB sub-page of this host page is now balloon-owned (the
                        // guest promised not to touch them), so reclaiming the whole page is safe.
                        let rc = unsafe {
                            libc::madvise(
                                base as *mut libc::c_void,
                                host_page,
                                libc::MADV_FREE_REUSABLE,
                            )
                        };
                        if rc != 0 {
                            warn!(
                                "balloon: inflate madvise(REUSABLE) at {base:#x} failed: {}",
                                std::io::Error::last_os_error()
                            );
                            self.ballooned_pages.remove(&base);
                        } else {
                            reclaimed.fetch_add(host_page as u64, Ordering::Relaxed);
                        }
                    }
                }
            }
            have_used = true;
            if let Err(e) = queues[IFQ_INDEX].queue.add_used(&mem, index, 0) {
                error!("balloon: inflate add_used failed: {e:?}");
            }
        }

        have_used
    }

    /// Deflate: the guest is taking pages back out of the balloon (lowered target, or
    /// `DEFLATE_ON_OOM` under guest pressure). We only need to drop them from the inflate bookkeeping
    /// so a host page is no longer considered fully inflated — the deflated pages return to the guest
    /// free pool and re-fault zero-filled on next use, which is correct. We deliberately do **not**
    /// `MADV_FREE_REUSE` here: re-validating would needlessly re-commit the *other* still-inflated
    /// sub-pages of the same host page.
    pub fn process_dfq(&mut self) -> bool {
        let host_page = self.host_page;
        let mem = match self.device_state {
            DeviceState::Activated(ref mem, _) => mem.clone(),
            DeviceState::Inactive => unreachable!(),
        };
        let queues = self
            .queues
            .as_mut()
            .expect("queues should exist when activated");
        let mut have_used = false;

        while let Some(head) = queues[DFQ_INDEX].queue.pop(&mem) {
            let index = head.index;
            for desc in head.into_iter() {
                let pfn_count = (desc.len as usize) / 4;
                for i in 0..pfn_count {
                    let pfn: u32 = match mem.read_obj(GuestAddress(desc.addr.0 + (i * 4) as u64)) {
                        Ok(p) => p,
                        Err(e) => {
                            error!("balloon: failed to read deflate PFN: {e:?}");
                            break;
                        }
                    };
                    let guest = GuestAddress((pfn as u64) << uapi::VIRTIO_BALLOON_PFN_SHIFT);
                    let Ok(host_addr) = mem.get_host_address(guest) else {
                        continue;
                    };
                    let base = (host_addr as usize) & !(host_page - 1);
                    let sub = ((host_addr as usize) - base) / GUEST_PAGE;
                    if let Some(entry) = self.inflate_mask.get_mut(&base) {
                        *entry &= !(1u64 << sub);
                        if *entry == 0 {
                            self.inflate_mask.remove(&base);
                        }
                    }
                    self.ballooned_pages.remove(&base);
                }
            }
            have_used = true;
            if let Err(e) = queues[DFQ_INDEX].queue.add_used(&mem, index, 0) {
                error!("balloon: deflate add_used failed: {e:?}");
            }
        }

        have_used
    }

    /// Apply a host-set target: stash it into `num_pages` and raise a config-change interrupt so the
    /// guest balloon driver inflates/deflates toward it. Called from the event loop on a
    /// `target_evt` wake. Returns true if a target was pending (the caller signals config-change).
    pub fn apply_target(&mut self) -> bool {
        let _ = self.target_evt.read();
        let target = self.pending_target.lock().unwrap().take();
        match target {
            Some(pages) => {
                self.config.num_pages = pages;
                debug!("balloon: target set to {pages} pages");
                true
            }
            None => false,
        }
    }
}

impl VirtioDevice for Balloon {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features
    }

    fn device_type(&self) -> u32 {
        uapi::VIRTIO_ID_BALLOON
    }

    fn device_name(&self) -> &str {
        "balloon"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_slice = self.config.as_slice();
        let config_len = config_slice.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(&config_slice[offset as usize..cmp::min(end, config_len) as usize])
                .unwrap();
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        // The guest writes the `actual` field (offset 4) to report its current balloon size as it
        // inflates/deflates toward `num_pages` (virtio-balloon spec). Apply the write, then publish
        // `actual` to the control handle so the host policy can read it back.
        let config_slice = self.config.as_mut_slice();
        let config_len = config_slice.len() as u64;
        let Some(end) = offset.checked_add(data.len() as u64) else {
            return;
        };
        if offset >= config_len || end > config_len {
            warn!(
                "balloon: out-of-bounds config write (offset={offset:x}, len={:x})",
                data.len()
            );
            return;
        }
        config_slice[offset as usize..end as usize].copy_from_slice(data);
        let actual = self.config.actual;
        self.actual_shared.store(actual, Ordering::Relaxed);
        debug!("balloon: guest reported actual={actual} pages");
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if queues.len() != defs::NUM_QUEUES {
            error!(
                "Cannot perform activate. Expected {} queue(s), got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }

        if self.activate_evt.write(1).is_err() {
            error!("Cannot write to activate_evt",);
            return Err(ActivateError::BadActivate);
        }

        self.queues = Some(queues);
        self.device_state = DeviceState::Activated(mem, interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }
}

#[cfg(test)]
mod tests {
    //! The 4 KiB-guest / 16 KiB-host coalescing safety proof, exercised deterministically without a
    //! boot. The invariant under test: [`ReclaimCoalescer`] emits a host page for reclaim **only**
    //! when every one of its constituent guest pages was reported free, and never counts a
    //! partially-free (unaligned-fringe) guest page.
    use super::{ReclaimCoalescer, GUEST_PAGE};

    const HOST_16K: usize = 16384;
    const BASE: usize = 0x4000_0000; // 16 KiB-aligned, mirrors the guest-RAM base.

    #[test]
    fn full_host_page_emitted_only_when_all_subpages_free() {
        // 3 of 4 sub-pages free -> nothing reclaimed.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        c.add(BASE, 3 * GUEST_PAGE);
        assert!(
            c.take_full_pages().is_empty(),
            "a 3-of-4 host page must never be reclaimed"
        );

        // The 4th completes the host page -> exactly that one host page is reclaimed. This is also
        // the enhanced-tier 16 KiB-guest case (a single 16 KiB-aligned run).
        let mut c = ReclaimCoalescer::new(HOST_16K);
        c.add(BASE, 4 * GUEST_PAGE);
        assert_eq!(c.take_full_pages(), vec![(BASE, HOST_16K)]);
    }

    #[test]
    fn unaligned_start_does_not_count_the_partial_leading_page() {
        // A run starting 2 KiB into the first guest page: that page is only partially free and must
        // not be counted. The remaining full pages alone can't complete the host page.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        c.add(BASE + GUEST_PAGE / 2, 4 * GUEST_PAGE);
        assert!(
            c.take_full_pages().is_empty(),
            "an unaligned fringe must never reclaim its partial page"
        );
    }

    #[test]
    fn boundary_spanning_run_completes_two_host_pages() {
        // 8 contiguous guest pages spanning two 16 KiB host pages -> both reclaimed.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        c.add(BASE, 8 * GUEST_PAGE);
        let mut got = c.take_full_pages();
        got.sort();
        assert_eq!(got, vec![(BASE, HOST_16K), (BASE + HOST_16K, HOST_16K)]);
    }

    #[test]
    fn split_descriptors_in_one_head_accumulate() {
        // The four sub-pages arriving as separate add() calls (separate descriptors within one
        // head) still complete the host page.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        for i in 0..4 {
            c.add(BASE + i * GUEST_PAGE, GUEST_PAGE);
        }
        assert_eq!(c.take_full_pages(), vec![(BASE, HOST_16K)]);
    }

    #[test]
    fn trailing_partial_page_is_not_reclaimed() {
        // A run that covers the first host page fully then 2 KiB into the next: only the full one
        // is reclaimed; the straddled second host page is left mapped.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        c.add(BASE, 4 * GUEST_PAGE + GUEST_PAGE / 2);
        assert_eq!(c.take_full_pages(), vec![(BASE, HOST_16K)]);
    }
}
