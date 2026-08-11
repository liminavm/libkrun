use std::cmp;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
use hvf::ReleasedRam;
use utils::eventfd::{EFD_NONBLOCK, EventFd};
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

// Supported features. DEFLATE_ON_OOM is OFF BY DEFAULT (limina M6 addendum 2026-07-20,
// "transparent balloon accounting"): Linux keeps ballooned pages in MemTotal (they read as
// *used*) exactly when this bit is negotiated, and subtracts them from the totals without it
// — so masking it makes a fresh dynamic-memory VM show its effective RAM instead of looking
// almost out of memory. The bit's guest-side OOM net is preempted by systemd-oomd on modern
// guests anyway (oomd reads the balloon's apparent usage as real pressure and kills first);
// the host-side release policy is the actual pressure response. Re-advertised per VM via the
// `deflate_on_oom` constructor knob (the vm.toml escape hatch) while confidence builds.
pub(crate) const AVAIL_FEATURES: u64 = (1 << uapi::VIRTIO_F_VERSION_1 as u64)
    | (1 << uapi::VIRTIO_BALLOON_F_STATS_VQ as u64)
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
/// safe to release (stage-2 unmap + `MADV_FREE_REUSABLE`).
///
/// INVARIANT — the whole safety proof for the 4 KiB-guest / 16 KiB-host page-size mismatch: a host
/// page is emitted by [`Self::take_full_pages`] **only** when every one of its constituent guest
/// pages was reported free, so releasing it can never drop a still-live guest page. Sub-page
/// (unaligned) fringes of a reported run are rounded *inward* (start up, end down) so a partially
/// free guest page is never counted as free, regardless of how `MADV_FREE_REUSABLE` itself rounds.
struct ReclaimCoalescer {
    host_page: usize,
    /// `(1 << (host_page / GUEST_PAGE)) - 1` — the all-sub-pages-free mask (`0b1111` on 16K/4K).
    full_mask: u64,
    /// host-page base address -> (bitmask of which of its 4 KiB sub-pages have been reported free,
    /// guest-physical address of the host page base — carried so the release can unmap the range
    /// from the guest).
    partial: std::collections::HashMap<usize, (u64, u64)>,
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

    /// Record a reported-free run `[addr, addr + len)` (host addresses; `gpa` is the guest-physical
    /// address of `addr`). Never releases anything. Rounds the run *inward* to whole guest pages so
    /// an unaligned fringe never marks a partially-live page free. In the real FRQ path runs are
    /// already page-aligned page-multiples; the rounding is a correctness-by-construction guard,
    /// not an expected case.
    fn add(&mut self, addr: usize, gpa: u64, len: usize) {
        if len < GUEST_PAGE {
            return;
        }
        let start = (addr + GUEST_PAGE - 1) & !(GUEST_PAGE - 1); // round up
        let end = (addr + len) & !(GUEST_PAGE - 1); // round down
        let mut p = start;
        while p < end {
            let base = p & !(self.host_page - 1);
            let sub = (p - base) / GUEST_PAGE;
            // GPA of the host-page base: shift the run's gpa by the same offset its host base
            // sits at. `base` can precede `addr` (the run starts mid-host-page), so keep the
            // arithmetic ordered to stay in range.
            let gpa_base = gpa + (p - addr) as u64 - (p - base) as u64;
            let entry = self.partial.entry(base).or_insert((0, gpa_base));
            debug_assert_eq!(
                entry.1, gpa_base,
                "one host page reported under two GPAs — regions overlap?"
            );
            entry.0 |= 1u64 << sub;
            p += GUEST_PAGE;
        }
    }

    /// Drain the host pages whose every sub-page is now free, returning `(base, gpa, host_page_len)`
    /// ranges safe to release. Partially covered host pages are retained (FRQ callers discard the
    /// coalescer after each head, so retained partials are simply not released).
    fn take_full_pages(&mut self) -> Vec<(usize, u64, usize)> {
        let full = self.full_mask;
        let hp = self.host_page;
        let mut out = Vec::new();
        self.partial.retain(|&base, &mut (mask, gpa)| {
            if mask == full {
                out.push((base, gpa, hp));
                false
            } else {
                true
            }
        });
        out
    }
}

/// Merge per-host-page `(host, gpa, len)` triples into maximal runs contiguous in BOTH address
/// spaces, so a large reported-free block costs one release call instead of one per host page.
fn merge_runs(mut pages: Vec<(usize, u64, usize)>) -> Vec<(usize, u64, usize)> {
    pages.sort_unstable();
    let mut out: Vec<(usize, u64, usize)> = Vec::new();
    for (host, gpa, len) in pages {
        match out.last_mut() {
            Some(last) if last.0 + last.2 == host && last.1 + last.2 as u64 == gpa => {
                last.2 += len;
            }
            _ => out.push((host, gpa, len)),
        }
    }
    out
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
    /// Balloon-released guest RAM (shared with the vCPUs' stage-2 fault healing): releasing a
    /// range here unmaps it from the guest *before* marking the host pages reusable, so the
    /// guest can never trample pages the OS considers disposable.
    #[cfg(target_os = "macos")]
    released_ram: Arc<ReleasedRam>,
}

impl Balloon {
    /// Create a balloon device. `free_page_reporting` gates `VIRTIO_BALLOON_F_REPORTING` (the FRQ
    /// fast-reclaim path): when false we do NOT advertise it, so the guest's `virtio_balloon` never
    /// starts a page-reporting worker. This is masked by default because a Linux guest with the
    /// feature enabled crashes on suspend-to-idle — upstream `virtballoon_freeze()` frees the balloon
    /// virtqueues without stopping the page-reporting work (which runs on the non-freezable system
    /// `events` workqueue), so the worker use-after-frees the dead reporting vq mid-s2idle and the
    /// guest wedges in `dpm_resume`. limina re-enables it per-VM only for enhanced-tier guests that
    /// carry the kernel fix (`virtballoon_freeze` unregisters page-reporting first). Stock guests keep
    /// the coarser `MADV_FREE_REUSABLE`-on-inflate reclaim and s2idle safely (two-tier: degraded but
    /// working).
    /// `deflate_on_oom` gates `VIRTIO_BALLOON_F_DEFLATE_ON_OOM` (see the AVAIL_FEATURES
    /// comment): off by default for transparent balloon accounting; the per-VM escape
    /// hatch re-advertises it.
    pub fn new(
        free_page_reporting: bool,
        deflate_on_oom: bool,
        #[cfg(target_os = "macos")] released_ram: Arc<ReleasedRam>,
    ) -> super::Result<Balloon> {
        let host_page = host_page_size();
        let sub = host_page / GUEST_PAGE;
        let full_mask = if sub >= 64 {
            u64::MAX
        } else {
            (1u64 << sub) - 1
        };
        let mut avail_features = if free_page_reporting {
            AVAIL_FEATURES
        } else {
            AVAIL_FEATURES & !(1 << uapi::VIRTIO_BALLOON_F_REPORTING as u64)
        };
        if deflate_on_oom {
            avail_features |= 1 << uapi::VIRTIO_BALLOON_F_DEFLATE_ON_OOM as u64;
        }
        Ok(Balloon {
            queues: None,
            avail_features,
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
            #[cfg(target_os = "macos")]
            released_ram,
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
        #[cfg(target_os = "macos")]
        let released_ram = self.released_ram.clone();
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
                // desc.addr is guest-reported; a bad/out-of-range FRQ entry must skip, not
                // panic the worker (a guest-triggerable DoS otherwise).
                let Ok(host_addr) = mem.get_host_address(desc.addr) else {
                    warn!(
                        "balloon: FRQ descriptor addr {:#x} outside guest memory; skipping",
                        desc.addr.0
                    );
                    continue;
                };
                coalescer.add(host_addr as usize, desc.addr.0, desc.len as usize);
            }
            // Release BEFORE add_used: page_reporting keeps these pages isolated from the guest
            // allocator until the descriptor is marked used, which closes the reallocation window.
            // The release unmaps the range from the guest FIRST and only then hands the host
            // pages back with MADV_FREE_REUSABLE (not MADV_DONTNEED, which returns nothing on
            // macOS — see spikes/balloon-madvise/RESULTS.md): a reusable-marked page must never
            // stay reachable through a live stage-2 mapping, or the guest's spec-sanctioned reuse
            // tramples pages the OS considers disposable with no event fired (see
            // spikes/hv-ledger-gap round 8c). The guest's next touch takes a stage-2 fault the
            // vCPU heals (ReleasedRam::handle_fault).
            for (base, gpa, len) in merge_runs(coalescer.take_full_pages()) {
                // SAFETY (both branches): the range is host-page-aligned and every guest page
                // inside it was reported free in this head, so releasing the whole host page
                // cannot drop live guest data.
                let ok;
                #[cfg(target_os = "macos")]
                {
                    let _ = base;
                    ok = released_ram.release(gpa, len as u64);
                }
                #[cfg(not(target_os = "macos"))]
                {
                    let _ = gpa;
                    let rc = unsafe {
                        libc::madvise(base as *mut libc::c_void, len, libc::MADV_FREE_REUSABLE)
                    };
                    if rc != 0 {
                        warn!(
                            "balloon: madvise(MADV_FREE_REUSABLE) at {base:#x} len={len} failed: {}",
                            std::io::Error::last_os_error()
                        );
                    }
                    ok = rc == 0;
                }
                if ok {
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
    /// until deflate. We release each fully covered 16 KiB host page (stage-2 unmap +
    /// `MADV_FREE_REUSABLE`; the unmap is defensive hardening here — the guest promised not to
    /// touch these, so any fault the vCPU heals on them is also a true guest-bug detector, except
    /// under `DEFLATE_ON_OOM` where the spec lets the guest take pages back before notifying). The
    /// inflate coalescer **persists** across heads (unlike FRQ): inflated pages stay balloon-owned,
    /// so accumulating sub-pages from different heads is safe and recovers cross-head host pages on a
    /// stock 4 KiB guest. A host page is released only once all four sub-pages are inflated, so we
    /// never discard a still-live (non-inflated) guest page.
    pub fn process_ifq(&mut self) -> bool {
        let host_page = self.host_page;
        let full_mask = self.full_mask;
        let reclaimed = self.reclaimed_bytes.clone();
        #[cfg(target_os = "macos")]
        let released_ram = self.released_ram.clone();
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
                        // SAFETY (both branches): every 4 KiB sub-page of this host page is now
                        // balloon-owned (the guest promised not to touch them), so releasing the
                        // whole page is safe.
                        let ok;
                        #[cfg(target_os = "macos")]
                        {
                            let gpa_base = guest.0 - (host_addr - base) as u64;
                            ok = released_ram.release(gpa_base, host_page as u64);
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
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
                            }
                            ok = rc == 0;
                        }
                        if ok {
                            reclaimed.fetch_add(host_page as u64, Ordering::Relaxed);
                        } else {
                            self.ballooned_pages.remove(&base);
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
    /// `DEFLATE_ON_OOM` under guest pressure). Drop them from the inflate bookkeeping so a host
    /// page is no longer considered fully inflated, and — for host pages that were actually
    /// released — take the range back for the guest right away (`ReleasedRam::reclaim`: REUSE +
    /// re-map). The proactive reclaim is an optimization, not a correctness requirement: anything
    /// missed here (including a `DEFLATE_ON_OOM` guest touching pages before notifying us) heals
    /// through the vCPU stage-2 fault path.
    pub fn process_dfq(&mut self) -> bool {
        let host_page = self.host_page;
        #[cfg(target_os = "macos")]
        let released_ram = self.released_ram.clone();
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
                    if self.ballooned_pages.remove(&base) {
                        #[cfg(target_os = "macos")]
                        {
                            let gpa_base = guest.0 - ((host_addr as usize) - base) as u64;
                            released_ram.reclaim(gpa_base, host_page as u64);
                        }
                    }
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

    /// Deactivate on the virtio reset the guest issues when re-initialising the device — notably on
    /// resume from suspend-to-idle. Returning `false` here (the trait default) leaves the transport
    /// marking the device FAILED (`device_status 0x8f`), so the guest's re-init writes get dropped
    /// and the balloon never comes back. Unlike net/block there's no dedicated worker to stop: the
    /// balloon runs under the shared EventManager and its queue eventfds are stable across a
    /// transport reset (the transport reuses `queue_evts`), so they stay registered and route to a
    /// fresh `activate`.
    ///
    /// Clear the inflate bookkeeping: the guest balloon driver deflates and resets its state at
    /// suspend, so any deflate that hadn't been processed when the reset landed would otherwise be
    /// lost while our `inflate_mask`/`ballooned_pages` persist — a later re-inflate could then
    /// complete a host page whose other sub-pages hold live guest data and get `MADV_FREE_REUSABLE`d
    /// (data loss). Clearing is conservative-safe: worst case an already-reclaimed page stays
    /// resident until the guest inflates it again; it can never cause a wrong madvise. Also drop any
    /// pending host target so a stale kick isn't applied to the freshly re-activated device.
    fn reset(&mut self) -> bool {
        self.inflate_mask.clear();
        self.ballooned_pages.clear();
        *self.pending_target.lock().unwrap() = None;
        self.device_state = DeviceState::Inactive;
        true
    }
}

#[cfg(test)]
mod tests {
    //! The 4 KiB-guest / 16 KiB-host coalescing safety proof, exercised deterministically without a
    //! boot. The invariant under test: [`ReclaimCoalescer`] emits a host page for reclaim **only**
    //! when every one of its constituent guest pages was reported free, and never counts a
    //! partially-free (unaligned-fringe) guest page.
    use super::{GUEST_PAGE, ReclaimCoalescer, merge_runs};

    const HOST_16K: usize = 16384;
    const BASE: usize = 0x4000_0000; // 16 KiB-aligned, mirrors the guest-RAM base.
    /// Host VA and GPA differ by a fixed per-region offset; mirror that in the tests.
    const GPA_DELTA: usize = 0x3000_0000;

    fn gpa(host: usize) -> u64 {
        (host - GPA_DELTA) as u64
    }

    /// add() with the GPA derived the same way the FRQ path derives it.
    fn add(c: &mut ReclaimCoalescer, host: usize, len: usize) {
        c.add(host, gpa(host), len);
    }

    #[test]
    fn full_host_page_emitted_only_when_all_subpages_free() {
        // 3 of 4 sub-pages free -> nothing reclaimed.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        add(&mut c, BASE, 3 * GUEST_PAGE);
        assert!(
            c.take_full_pages().is_empty(),
            "a 3-of-4 host page must never be reclaimed"
        );

        // The 4th completes the host page -> exactly that one host page is reclaimed, carrying
        // the GPA of its base. This is also the enhanced-tier 16 KiB-guest case (a single
        // 16 KiB-aligned run).
        let mut c = ReclaimCoalescer::new(HOST_16K);
        add(&mut c, BASE, 4 * GUEST_PAGE);
        assert_eq!(c.take_full_pages(), vec![(BASE, gpa(BASE), HOST_16K)]);
    }

    #[test]
    fn unaligned_start_does_not_count_the_partial_leading_page() {
        // A run starting 2 KiB into the first guest page: that page is only partially free and must
        // not be counted. The remaining full pages alone can't complete the host page.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        add(&mut c, BASE + GUEST_PAGE / 2, 4 * GUEST_PAGE);
        assert!(
            c.take_full_pages().is_empty(),
            "an unaligned fringe must never reclaim its partial page"
        );
    }

    #[test]
    fn boundary_spanning_run_completes_two_host_pages() {
        // 8 contiguous guest pages spanning two 16 KiB host pages -> both reclaimed, each with
        // its own base GPA.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        add(&mut c, BASE, 8 * GUEST_PAGE);
        let mut got = c.take_full_pages();
        got.sort();
        assert_eq!(
            got,
            vec![
                (BASE, gpa(BASE), HOST_16K),
                (BASE + HOST_16K, gpa(BASE + HOST_16K), HOST_16K)
            ]
        );
    }

    #[test]
    fn split_descriptors_in_one_head_accumulate() {
        // The four sub-pages arriving as separate add() calls (separate descriptors within one
        // head) still complete the host page — and a mid-host-page descriptor start must still
        // attribute the host page base's GPA, not its own.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        for i in 0..4 {
            add(&mut c, BASE + i * GUEST_PAGE, GUEST_PAGE);
        }
        assert_eq!(c.take_full_pages(), vec![(BASE, gpa(BASE), HOST_16K)]);
    }

    #[test]
    fn trailing_partial_page_is_not_reclaimed() {
        // A run that covers the first host page fully then 2 KiB into the next: only the full one
        // is reclaimed; the straddled second host page is left mapped.
        let mut c = ReclaimCoalescer::new(HOST_16K);
        add(&mut c, BASE, 4 * GUEST_PAGE + GUEST_PAGE / 2);
        assert_eq!(c.take_full_pages(), vec![(BASE, gpa(BASE), HOST_16K)]);
    }

    #[test]
    fn merge_runs_joins_only_pages_contiguous_in_both_spaces() {
        // Host-contiguous AND gpa-contiguous pages merge into one run.
        let contiguous = vec![
            (BASE + HOST_16K, gpa(BASE + HOST_16K), HOST_16K),
            (BASE, gpa(BASE), HOST_16K),
        ];
        assert_eq!(
            merge_runs(contiguous),
            vec![(BASE, gpa(BASE), 2 * HOST_16K)]
        );

        // A host-space gap keeps runs apart.
        let gapped = vec![
            (BASE, gpa(BASE), HOST_16K),
            (BASE + 3 * HOST_16K, gpa(BASE + 3 * HOST_16K), HOST_16K),
        ];
        assert_eq!(merge_runs(gapped.clone()), gapped);

        // Host-contiguous but a GPA discontinuity (a region boundary) must NOT merge.
        let split_gpa = vec![
            (BASE, gpa(BASE), HOST_16K),
            (BASE + HOST_16K, gpa(BASE) + 0x1000_0000, HOST_16K),
        ];
        assert_eq!(merge_runs(split_gpa.clone()), split_gpa);
    }
}
