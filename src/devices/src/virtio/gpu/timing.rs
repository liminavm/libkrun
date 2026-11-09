//! limina: the timing page — the host display's vsync grid, in the guest's own counter time.
//!
//! One page of host memory, mapped into the guest at the first slot of the extension's arena
//! window and offered as its own shm region ([`SHM_ID_TIMING`]). Per scanout it says where the
//! guest's vblank should fall: an **anchor** (one vblank, as a guest virtual-counter value) and a
//! **period** (counter ticks per frame, 32.32 fixed point). The guest's driver times its vblank from
//! it instead of from a free-running timer of its own, so its frame clock ticks on the host
//! display's grid. The embedder says what the grid is ([`TimingPage::set_grid`]): the host
//! vsync's own time and the lead to put the guest's vblank ahead of it by.
//!
//! The guest's counter is `mach_absolute_time() - vtimer_offset`, in the same 24 MHz ticks, so a
//! host time becomes a guest one by subtracting the offset; when the offset moves (a live pause, a
//! restore) every entry is rewritten ([`TimingPage::set_vtimer_offset`]).
//!
//! Layout, little endian: a 64-byte header, then [`TIMING_ENTRIES`] 64-byte entries. An entry is
//! read under its sequence number: odd while the host writes it, and a reader that sees it change
//! across its read reads again.

use std::hint;
use std::io;
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

/// Shm region of the timing page.
pub const SHM_ID_TIMING: u32 = 0x82;
/// The page's size: one host page, which is what `hv_vm_map` maps.
pub const TIMING_SIZE: u64 = 16 * 1024;
/// Header magic, "LVBL".
pub const TIMING_MAGIC: u32 = 0x4c42_564c;
pub const TIMING_VERSION: u32 = 1;
/// Entries, one per possible scanout.
pub const TIMING_ENTRIES: usize = 16;
pub const TIMING_HEADER_SIZE: usize = 64;
pub const TIMING_ENTRY_SIZE: usize = 64;
/// Entry flag: the entry holds a grid; without it the guest keeps its own timer.
pub const TIMING_VALID: u32 = 1;

// Entry field offsets.
const SEQ: usize = 0;
const FLAGS: usize = 4;
const ANCHOR: usize = 8;
const PERIOD: usize = 16;
const LEAD_NS: usize = 24;
const GENERATION: usize = 28;

/// Where one scanout's guest vblanks should fall, in host terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// One host vsync, in `mach_absolute_time` ticks.
    pub vsync: u64,
    /// Ticks per frame, 32.32 fixed point.
    pub period_fp: u64,
    /// How far ahead of the vsync the guest's vblank falls, in ticks.
    pub lead_ticks: u64,
    /// The same lead in nanoseconds, written for diagnostics only.
    pub lead_ns: u32,
}

/// The host's side of the page.
pub struct TimingPage {
    page: Page,
    state: Mutex<State>,
}

struct State {
    grids: [Option<Grid>; TIMING_ENTRIES],
    vtimer_offset: u64,
}

/// The page's memory: page-aligned, never moved or freed while the page lives.
struct Page {
    base: *mut u8,
}

// SAFETY: the memory is only written through `TimingPage`, under its mutex, with atomic stores;
// the pointer itself is never reassigned.
unsafe impl Send for Page {}
unsafe impl Sync for Page {}

impl Page {
    fn new() -> io::Result<Self> {
        // SAFETY: an anonymous private mapping of TIMING_SIZE bytes, checked below.
        let base = unsafe {
            libc::mmap(
                ptr::null_mut(),
                TIMING_SIZE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        if base == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            base: base as *mut u8,
        })
    }

    fn u32_at(&self, off: usize) -> &AtomicU32 {
        debug_assert!(off.is_multiple_of(4) && off + 4 <= TIMING_SIZE as usize);
        // SAFETY: in bounds and aligned (checked above); the mapping outlives `self`.
        unsafe { &*(self.base.add(off) as *const AtomicU32) }
    }

    fn u64_at(&self, off: usize) -> &AtomicU64 {
        debug_assert!(off.is_multiple_of(8) && off + 8 <= TIMING_SIZE as usize);
        // SAFETY: as above.
        unsafe { &*(self.base.add(off) as *const AtomicU64) }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        // SAFETY: the mapping made in `new`.
        unsafe { libc::munmap(self.base as *mut libc::c_void, TIMING_SIZE as usize) };
    }
}

fn entry_off(scanout: usize) -> usize {
    TIMING_HEADER_SIZE + scanout * TIMING_ENTRY_SIZE
}

impl TimingPage {
    pub fn new() -> io::Result<Self> {
        let page = Page::new()?;
        page.u32_at(0).store(TIMING_MAGIC, Ordering::Relaxed);
        page.u32_at(4).store(TIMING_VERSION, Ordering::Relaxed);
        page.u32_at(8)
            .store(TIMING_ENTRIES as u32, Ordering::Relaxed);
        page.u32_at(12)
            .store(TIMING_ENTRY_SIZE as u32, Ordering::Release);
        Ok(Self {
            page,
            state: Mutex::new(State {
                grids: [None; TIMING_ENTRIES],
                vtimer_offset: 0,
            }),
        })
    }

    /// The page's host address, for mapping it into the guest.
    pub fn host_addr(&self) -> u64 {
        self.page.base as u64
    }

    /// Set (or, with `None`, clear) one scanout's grid. Scanouts past the page are ignored.
    pub fn set_grid(&self, scanout: u32, grid: Option<Grid>) {
        let i = scanout as usize;
        if i >= TIMING_ENTRIES {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state.grids[i] = grid;
        self.write(i, grid, state.vtimer_offset);
    }

    /// The guest's counter offset changed; every entry is rewritten in the new counter time.
    pub fn set_vtimer_offset(&self, offset: u64) {
        let mut state = self.state.lock().unwrap();
        if state.vtimer_offset == offset {
            return;
        }
        state.vtimer_offset = offset;
        for i in 0..TIMING_ENTRIES {
            if state.grids[i].is_some() {
                self.write(i, state.grids[i], offset);
            }
        }
    }

    fn write(&self, i: usize, grid: Option<Grid>, vtimer_offset: u64) {
        let e = entry_off(i);
        let seq = self.page.u32_at(e + SEQ);
        let s = seq.load(Ordering::Relaxed);
        seq.store(s.wrapping_add(1) | 1, Ordering::Relaxed);
        fence(Ordering::Release);
        match grid {
            Some(g) => {
                let anchor = g
                    .vsync
                    .wrapping_sub(g.lead_ticks)
                    .wrapping_sub(vtimer_offset);
                self.page
                    .u64_at(e + ANCHOR)
                    .store(anchor, Ordering::Relaxed);
                self.page
                    .u64_at(e + PERIOD)
                    .store(g.period_fp, Ordering::Relaxed);
                self.page
                    .u32_at(e + LEAD_NS)
                    .store(g.lead_ns, Ordering::Relaxed);
                self.page
                    .u32_at(e + FLAGS)
                    .store(TIMING_VALID, Ordering::Relaxed);
            }
            None => self.page.u32_at(e + FLAGS).store(0, Ordering::Relaxed),
        }
        let generation = self.page.u32_at(e + GENERATION);
        generation.store(
            generation.load(Ordering::Relaxed).wrapping_add(1),
            Ordering::Relaxed,
        );
        seq.store((s.wrapping_add(1) | 1).wrapping_add(1), Ordering::Release);
    }

    /// Read one entry back as the guest does: `(flags, anchor, period_fp, lead_ns)`.
    pub fn read(&self, scanout: usize) -> Option<(u32, u64, u64, u32)> {
        if scanout >= TIMING_ENTRIES {
            return None;
        }
        let e = entry_off(scanout);
        loop {
            let s0 = self.page.u32_at(e + SEQ).load(Ordering::Acquire);
            if s0 & 1 == 1 {
                hint::spin_loop();
                continue;
            }
            let out = (
                self.page.u32_at(e + FLAGS).load(Ordering::Relaxed),
                self.page.u64_at(e + ANCHOR).load(Ordering::Relaxed),
                self.page.u64_at(e + PERIOD).load(Ordering::Relaxed),
                self.page.u32_at(e + LEAD_NS).load(Ordering::Relaxed),
            );
            fence(Ordering::Acquire);
            if self.page.u32_at(e + SEQ).load(Ordering::Relaxed) == s0 {
                return Some(out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: u64 = 400_000 << 32; // 60 Hz at 24 MHz

    fn grid(vsync: u64) -> Grid {
        Grid {
            vsync,
            period_fp: FRAME,
            lead_ticks: 96_000, // 4 ms
            lead_ns: 4_000_000,
        }
    }

    #[test]
    fn the_header_names_the_layout() {
        let page = TimingPage::new().unwrap();
        assert_eq!(page.page.u32_at(0).load(Ordering::Relaxed), TIMING_MAGIC);
        assert_eq!(page.page.u32_at(4).load(Ordering::Relaxed), TIMING_VERSION);
        assert_eq!(
            page.page.u32_at(8).load(Ordering::Relaxed),
            TIMING_ENTRIES as u32
        );
        assert_eq!(page.host_addr() % TIMING_SIZE, 0, "a host page");
        // No grid yet: every entry tells the guest to keep its own timer.
        assert!((0..TIMING_ENTRIES).all(|i| page.read(i).unwrap().0 == 0));
    }

    #[test]
    fn a_grid_is_written_in_guest_counter_time_ahead_by_the_lead() {
        let page = TimingPage::new().unwrap();
        page.set_vtimer_offset(1_000);
        page.set_grid(2, Some(grid(5_000_000)));
        let (flags, anchor, period, lead_ns) = page.read(2).unwrap();
        assert_eq!(flags, TIMING_VALID);
        assert_eq!(anchor, 5_000_000 - 96_000 - 1_000);
        assert_eq!(period, FRAME);
        assert_eq!(lead_ns, 4_000_000);
        assert_eq!(page.read(1).unwrap().0, 0, "other scanouts untouched");
    }

    #[test]
    fn a_new_offset_rewrites_every_grid() {
        let page = TimingPage::new().unwrap();
        page.set_grid(0, Some(grid(5_000_000)));
        page.set_grid(3, Some(grid(7_000_000)));
        page.set_vtimer_offset(24_000_000);
        assert_eq!(
            page.read(0).unwrap().1,
            5_000_000u64.wrapping_sub(96_000).wrapping_sub(24_000_000)
        );
        assert_eq!(
            page.read(3).unwrap().1,
            7_000_000u64.wrapping_sub(96_000).wrapping_sub(24_000_000)
        );
    }

    #[test]
    fn clearing_a_grid_hands_the_scanout_back() {
        let page = TimingPage::new().unwrap();
        page.set_grid(0, Some(grid(5_000_000)));
        let s0 = page.page.u32_at(entry_off(0)).load(Ordering::Relaxed);
        page.set_grid(0, None);
        assert_eq!(page.read(0).unwrap().0, 0);
        let s1 = page.page.u32_at(entry_off(0)).load(Ordering::Relaxed);
        assert!(
            s1 > s0 && s1.is_multiple_of(2),
            "each write leaves the sequence even and higher"
        );
        page.set_grid(TIMING_ENTRIES as u32, Some(grid(1))); // past the page: ignored
    }
}
