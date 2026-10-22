// Copyright 2026 The libkrun Authors
// SPDX-License-Identifier: Apache-2.0

//! Guest RAM built from memory objects this task owns.
//!
//! An anonymous mapping bills a page to `phys_footprint` once for every pmap that maps it, so a
//! page both the VMM and the guest touch (any page a device fills from disk) is billed twice. A
//! memory entry the task creates `MAP_MEM_NAMED_CREATE | MAP_MEM_LEDGER_TAGGED` is owned by it and
//! billed once, to its own footprint, however many pmaps map it; `hv_vm_allocate` builds guest
//! RAM the same way. The cost is that `MADV_FREE_REUSABLE` does nothing on such an object: its
//! pages, resident or compressed, go only when the object dies. So guest RAM is a reservation
//! filled with blocks, each its own object, and a block is released by mapping a fresh one over
//! it, which reads as zeros.

use std::io;

#[allow(non_camel_case_types)]
type kern_return_t = i32;
#[allow(non_camel_case_types)]
type mach_port_t = u32;
#[allow(non_camel_case_types)]
type vm_prot_t = i32;

const KERN_SUCCESS: kern_return_t = 0;
const VM_PROT_NONE: vm_prot_t = 0x0;
const VM_PROT_READ: vm_prot_t = 0x1;
const VM_PROT_WRITE: vm_prot_t = 0x2;
const VM_INHERIT_NONE: u32 = 2;
const VM_FLAGS_FIXED: i32 = 0x0;
const VM_FLAGS_ANYWHERE: i32 = 0x1;
const VM_FLAGS_OVERWRITE: i32 = 0x4000;
const MAP_MEM_LEDGER_TAGGED: vm_prot_t = 0x002000;
const MAP_MEM_NAMED_CREATE: vm_prot_t = 0x020000;

unsafe extern "C" {
    static mach_task_self_: mach_port_t;
    fn mach_make_memory_entry_64(
        target_task: mach_port_t,
        size: *mut u64,
        offset: u64,
        permission: vm_prot_t,
        object_handle: *mut mach_port_t,
        parent_entry: mach_port_t,
    ) -> kern_return_t;
    #[allow(clippy::too_many_arguments)]
    fn mach_vm_map(
        target_task: mach_port_t,
        address: *mut u64,
        size: u64,
        mask: u64,
        flags: i32,
        object: mach_port_t,
        offset: u64,
        copy: i32,
        cur_protection: vm_prot_t,
        max_protection: vm_prot_t,
        inheritance: u32,
    ) -> kern_return_t;
    fn mach_port_deallocate(task: mach_port_t, name: mach_port_t) -> kern_return_t;
}

fn task() -> mach_port_t {
    // SAFETY: set by the runtime before main and never written again.
    unsafe { mach_task_self_ }
}

/// The block sizes over a guest RAM region: each `(offset, block)` applies from `offset` bytes
/// into the region up to the next entry's offset, or to the region's end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockLayout(Vec<(u64, u64)>);

impl BlockLayout {
    /// Spans must start at 0 and ascend, and each block must be a power of two no smaller than
    /// the host page, dividing its span's offset.
    pub fn new(spans: Vec<(u64, u64)>) -> Result<Self, String> {
        let page = crate::host_page_size();
        if spans.first().map(|s| s.0) != Some(0) {
            return Err("the first span must start at offset 0".into());
        }
        for (i, &(offset, block)) in spans.iter().enumerate() {
            if !block.is_power_of_two() || block < page {
                return Err(format!(
                    "block {block:#x} is not a power of two of at least {page:#x}"
                ));
            }
            if !offset.is_multiple_of(block) {
                return Err(format!(
                    "span at {offset:#x} is not aligned to its block {block:#x}"
                ));
            }
            if i > 0 && offset <= spans[i - 1].0 {
                return Err(format!("span at {offset:#x} does not ascend"));
            }
        }
        Ok(Self(spans))
    }

    /// One block size over the whole region.
    pub fn uniform(block: u64) -> Result<Self, String> {
        Self::new(vec![(0, block)])
    }

    /// The block holding `offset`, as `(start, len)`, clipped to a region of `size` bytes.
    pub fn block_at(&self, offset: u64, size: u64) -> (u64, u64) {
        let i = self.0.partition_point(|&(o, _)| o <= offset) - 1;
        let (span, block) = self.0[i];
        let start = span + (offset - span) / block * block;
        let end = (start + block)
            .min(size)
            .min(self.0.get(i + 1).map_or(u64::MAX, |s| s.0));
        (start, end - start)
    }

    /// The blocks of a region of `size` bytes that overlap `[start, end)`, in order.
    pub fn blocks(&self, start: u64, end: u64, size: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let end = end.min(size);
        let mut at = start;
        std::iter::from_fn(move || {
            (at < end).then(|| {
                let block = self.block_at(at, size);
                at = block.0 + block.1;
                block
            })
        })
    }
}

/// Reserve `size` bytes of address space and fill it with owned blocks laid out by `layout`.
/// Returns the base address; the mapping is never released, it lives as long as the VM.
pub fn allocate(size: u64, layout: &BlockLayout) -> io::Result<u64> {
    let mut addr = 0u64;
    // SAFETY: a fresh, no-access reservation anywhere in this task.
    let kr = unsafe {
        mach_vm_map(
            task(),
            &mut addr,
            size,
            0,
            VM_FLAGS_ANYWHERE,
            0,
            0,
            0,
            VM_PROT_NONE,
            VM_PROT_READ | VM_PROT_WRITE,
            VM_INHERIT_NONE,
        )
    };
    check("reserving guest RAM", kr)?;
    for (start, len) in layout.blocks(0, size, size) {
        // SAFETY: inside the reservation just made, which nothing else uses yet.
        unsafe { place_block(addr + start, len)? };
    }
    Ok(addr)
}

/// Map a fresh owned object over `[addr, addr + len)`, dropping whatever object was there and,
/// with it, the pages it held.
///
/// # Safety
///
/// The range is one block of guest RAM made by [`allocate`], and nothing reads or writes it
/// while this runs.
pub unsafe fn place_block(addr: u64, len: u64) -> io::Result<()> {
    let mut size = len;
    let mut entry = 0;
    // SAFETY: plain out-parameters; a named entry of `len` bytes owned by this task.
    let kr = unsafe {
        mach_make_memory_entry_64(
            task(),
            &mut size,
            0,
            MAP_MEM_NAMED_CREATE | MAP_MEM_LEDGER_TAGGED | VM_PROT_READ | VM_PROT_WRITE,
            &mut entry,
            0,
        )
    };
    check("creating a guest RAM block", kr)?;
    let mut at = addr;
    // SAFETY: the caller's contract: `addr` is one of our blocks and nothing touches it.
    let kr = unsafe {
        mach_vm_map(
            task(),
            &mut at,
            len,
            0,
            VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
            entry,
            0,
            0,
            VM_PROT_READ | VM_PROT_WRITE,
            VM_PROT_READ | VM_PROT_WRITE,
            VM_INHERIT_NONE,
        )
    };
    // SAFETY: the mapping holds its own reference to the object; ours goes.
    unsafe { mach_port_deallocate(task(), entry) };
    check("mapping a guest RAM block", kr)
}

fn check(what: &str, kr: kern_return_t) -> io::Result<()> {
    if kr == KERN_SUCCESS {
        Ok(())
    } else {
        Err(io::Error::other(format!("{what}: kern_return {kr}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const PAGE: u64 = 16 << 10;

    #[test]
    fn a_layout_must_start_at_zero_ascend_and_align() {
        assert!(BlockLayout::new(vec![]).is_err());
        assert!(BlockLayout::new(vec![(MIB, 2 * MIB)]).is_err());
        assert!(BlockLayout::new(vec![(0, 3 * MIB)]).is_err());
        assert!(BlockLayout::new(vec![(0, 2 * MIB), (3 * MIB, 2 * MIB)]).is_err());
        assert!(BlockLayout::new(vec![(0, 2 * MIB), (2 * MIB, PAGE), (2 * MIB, PAGE)]).is_err());
        assert!(BlockLayout::new(vec![(0, 2 * MIB), (4 * MIB, PAGE)]).is_ok());
    }

    #[test]
    fn blocks_follow_their_span_and_clip_at_its_end() {
        // A span ending inside one of its blocks clips that block; so does the region's end.
        let layout = BlockLayout::new(vec![(0, 2 * MIB), (5 * MIB, MIB)]).unwrap();
        let size = 8 * MIB;
        let blocks: Vec<_> = layout.blocks(0, size, size).collect();
        assert_eq!(
            blocks,
            vec![
                (0, 2 * MIB),
                (2 * MIB, 2 * MIB),
                (4 * MIB, MIB),
                (5 * MIB, MIB),
                (6 * MIB, MIB),
                (7 * MIB, MIB),
            ]
        );
        assert_eq!(layout.block_at(3 * MIB, size), (2 * MIB, 2 * MIB));
        assert_eq!(layout.block_at(9 * MIB / 2, size), (4 * MIB, MIB));
        assert_eq!(
            BlockLayout::uniform(2 * MIB)
                .unwrap()
                .block_at(7 * MIB, 7 * MIB + PAGE),
            (6 * MIB, MIB + PAGE)
        );
    }

    #[test]
    fn a_replaced_block_reads_as_zeros_and_its_neighbours_keep_their_data() {
        let layout = BlockLayout::uniform(2 * MIB).unwrap();
        let size = 6 * MIB;
        let base = allocate(size, &layout).unwrap();
        let mem = unsafe { std::slice::from_raw_parts_mut(base as *mut u8, size as usize) };
        mem.fill(0xa5);
        unsafe { place_block(base + 2 * MIB, 2 * MIB).unwrap() };
        assert!(mem[..2 * MIB as usize].iter().all(|&b| b == 0xa5));
        assert!(
            mem[2 * MIB as usize..4 * MIB as usize]
                .iter()
                .all(|&b| b == 0)
        );
        assert!(mem[4 * MIB as usize..].iter().all(|&b| b == 0xa5));
    }
}
