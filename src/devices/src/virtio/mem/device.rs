//! limina: a virtio-mem device. The guest plugs and unplugs blocks of a region above its RAM;
//! the host decides how much it may plug (`requested_size`), and an unplugged block goes back to
//! the host whole.
//!
//! On macOS the region is built from owned memory blocks of the device's block size
//! (`hvf::guest_ram`) and is stage-2 mapped whole from boot: the guest never touches a block it
//! has not plugged (`VIRTIO_MEM_F_UNPLUGGED_INACCESSIBLE`), so a mapped, unplugged block costs
//! nothing. Unplugging unmaps the blocks, replaces each with a fresh one, which frees its pages,
//! and maps them back.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use utils::eventfd::{EFD_NONBLOCK, EventFd};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

use super::super::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, QueueConfig, VirtioDevice,
};
use super::plug::{Plugged, Request};
use super::{MemError, defs, defs::uapi::*};
use crate::virtio::InterruptTransport;

pub(crate) const REQ_INDEX: usize = 0;

pub(crate) const AVAIL_FEATURES: u64 = (1 << VIRTIO_F_VERSION_1 as u64)
    | (1 << VIRTIO_MEM_F_UNPLUGGED_INACCESSIBLE as u64)
    | (1 << VIRTIO_MEM_F_PERSISTENT_SUSPEND as u64);

/// State the device shares with its control handle.
struct Shared {
    requested: AtomicU64,
    plugged: AtomicU64,
    bound: AtomicBool,
    /// Wakes the device to tell the guest a new requested size.
    requested_evt: EventFd,
}

pub struct Mem {
    pub(crate) queues: Option<Vec<DeviceQueue>>,
    pub(crate) acked_features: u64,
    pub(crate) activate_evt: EventFd,
    pub(crate) device_state: DeviceState,
    addr: u64,
    region_size: u64,
    block_size: u64,
    plugged: Plugged,
    shared: Arc<Shared>,
}

/// What a snapshot carries of the device: where its region is, what the host asked for, and
/// which blocks the guest has plugged, as `(gpa, len)` runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemState {
    pub addr: u64,
    pub region_size: u64,
    pub requested_size: u64,
    pub plugged: Vec<(u64, u64)>,
}

/// How big the region is, how much of it the host asks the guest to plug, and how much is.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemStats {
    pub region_size: u64,
    pub requested_size: u64,
    pub plugged_size: u64,
    /// The guest's driver has activated the device at least once.
    pub bound: bool,
}

/// Drives the live device's requested size. Mechanism only: limina decides the size.
#[derive(Clone)]
pub struct MemControlHandle {
    shared: Arc<Shared>,
    region_size: u64,
    block_size: u64,
}

impl MemControlHandle {
    /// Ask the guest to plug `bytes`, rounded down to the block size and capped at the region.
    /// Returns the size asked for.
    pub fn set_requested_size(&self, bytes: u64) -> u64 {
        let size = bytes.min(self.region_size) / self.block_size * self.block_size;
        self.shared.requested.store(size, Ordering::Relaxed);
        if let Err(e) = self.shared.requested_evt.write(1) {
            error!("virtio-mem: failed to kick the device for a new requested size: {e}");
        }
        size
    }

    pub fn stats(&self) -> MemStats {
        MemStats {
            region_size: self.region_size,
            requested_size: self.shared.requested.load(Ordering::Relaxed),
            plugged_size: self.shared.plugged.load(Ordering::Relaxed),
            bound: self.shared.bound.load(Ordering::Relaxed),
        }
    }
}

impl Mem {
    /// A device over `[addr, addr + region_size)` in blocks of `block_size`, which must be a
    /// power of two dividing both.
    pub fn new(addr: u64, region_size: u64, block_size: u64) -> super::Result<Mem> {
        if !block_size.is_power_of_two()
            || !addr.is_multiple_of(block_size)
            || !region_size.is_multiple_of(block_size)
        {
            return Err(MemError::Geometry(format!(
                "region {addr:#x}+{region_size:#x} in blocks of {block_size:#x}"
            )));
        }
        Ok(Mem {
            queues: None,
            acked_features: 0,
            activate_evt: EventFd::new(EFD_NONBLOCK).map_err(MemError::EventFd)?,
            device_state: DeviceState::Inactive,
            addr,
            region_size,
            block_size,
            plugged: Plugged::new(addr, block_size, (region_size / block_size) as usize),
            shared: Arc::new(Shared {
                requested: AtomicU64::new(0),
                plugged: AtomicU64::new(0),
                bound: AtomicBool::new(false),
                requested_evt: EventFd::new(EFD_NONBLOCK).map_err(MemError::EventFd)?,
            }),
        })
    }

    pub fn id(&self) -> &str {
        defs::MEM_DEV_ID
    }

    pub fn control_handle(&self) -> MemControlHandle {
        MemControlHandle {
            shared: self.shared.clone(),
            region_size: self.region_size,
            block_size: self.block_size,
        }
    }

    pub fn save_state(&self) -> MemState {
        MemState {
            addr: self.addr,
            region_size: self.region_size,
            requested_size: self.shared.requested.load(Ordering::Relaxed),
            plugged: self.plugged.runs(),
        }
    }

    /// Take the state a snapshot carries, before the restored guest runs: its driver reads
    /// `plugged_size` back on thaw and expects the blocks it plugged.
    pub fn restore_state(&mut self, state: &MemState) -> Result<(), String> {
        if (state.addr, state.region_size) != (self.addr, self.region_size) {
            return Err(format!(
                "the snapshot's virtio-mem region {:#x}+{:#x} is not this one, {:#x}+{:#x}",
                state.addr, state.region_size, self.addr, self.region_size
            ));
        }
        self.plugged.set_runs(&state.plugged)?;
        self.shared
            .requested
            .store(state.requested_size, Ordering::Relaxed);
        self.shared
            .plugged
            .store(self.plugged.plugged_size(), Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn queue_event(&self, idx: usize) -> &Arc<EventFd> {
        &self.queues.as_ref().expect("queues should exist")[idx].event
    }

    pub(crate) fn requested_evt(&self) -> &EventFd {
        &self.shared.requested_evt
    }

    /// Answer every queued request. Returns whether any was answered.
    pub fn process_req(&mut self) -> bool {
        let DeviceState::Activated(ref mem, _) = self.device_state else {
            unreachable!("checked by the event handler");
        };
        let mem = mem.clone();
        let mut have_used = false;
        loop {
            let queue = &mut self.queues.as_mut().expect("activated")[REQ_INDEX].queue;
            let Some(head) = queue.pop(&mem) else { break };
            let index = head.index;
            let mut req = None;
            let mut resp_addr = None;
            for desc in head.into_iter() {
                if desc.is_write_only() {
                    if resp_addr.is_none() && desc.len as usize >= RESP_SIZE {
                        resp_addr = Some(desc.addr);
                    }
                } else if req.is_none() && desc.len as usize >= REQ_SIZE {
                    let mut raw = [0u8; REQ_SIZE];
                    if mem.read_slice(&mut raw, desc.addr).is_ok() {
                        req = Some(Request::decode(&raw));
                    }
                }
            }
            let mut written = 0;
            match (req, resp_addr) {
                (Some(req), Some(resp_addr)) => {
                    let requested = self.shared.requested.load(Ordering::Relaxed);
                    let outcome = self.plugged.handle(req, requested);
                    for &(first, count) in &outcome.unplugged {
                        let gpa = self.addr + first as u64 * self.block_size;
                        self.release(&mem, gpa, count as u64 * self.block_size);
                    }
                    self.shared
                        .plugged
                        .store(self.plugged.plugged_size(), Ordering::Relaxed);
                    debug!("virtio-mem: {req:?} -> {}", outcome.resp);
                    match mem.write_slice(&outcome.encode(), resp_addr) {
                        Ok(()) => written = RESP_SIZE as u32,
                        Err(e) => error!("virtio-mem: failed to write a response: {e:?}"),
                    }
                }
                _ => error!("virtio-mem: malformed request chain at index {index}"),
            }
            let queue = &mut self.queues.as_mut().expect("activated")[REQ_INDEX].queue;
            if let Err(e) = queue.add_used(&mem, index, written) {
                error!("virtio-mem: failed to add a used element: {e:?}");
            }
            have_used = true;
        }
        have_used
    }

    /// Give the host back the pages of `[gpa, gpa + len)`, which the guest just unplugged.
    #[cfg(target_os = "macos")]
    fn release(&self, mem: &GuestMemoryMmap, gpa: u64, len: u64) {
        use hvf::bindings::HV_SUCCESS;
        use hvf::released_ram::{Hvf, Stage2};
        use vm_memory::GuestMemoryBackend;

        let host = match mem.get_host_address(GuestAddress(gpa)) {
            Ok(host) => host as u64,
            Err(e) => {
                error!("virtio-mem: unplugged range {gpa:#x}+{len:#x} is not guest memory: {e:?}");
                return;
            }
        };
        let ret = Hvf.unmap(gpa, len);
        if ret != HV_SUCCESS as _ {
            error!("virtio-mem: hv_vm_unmap({gpa:#x}, {len:#x}) failed: {ret:#x}");
            return;
        }
        for offset in (0..len).step_by(self.block_size as usize) {
            // SAFETY: one block of the region's owned memory, unmapped from the guest just above,
            // and unplugged, so no device holds a buffer in it.
            if let Err(e) = unsafe { Hvf.renew(host + offset, self.block_size) } {
                warn!(
                    "virtio-mem: renewing the block at {:#x} failed: {e}",
                    gpa + offset
                );
            }
        }
        let ret = Hvf.map(host, gpa, len);
        if ret != HV_SUCCESS as _ {
            error!("virtio-mem: hv_vm_map({gpa:#x}, {len:#x}) failed: {ret:#x}");
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn release(&self, _mem: &GuestMemoryMmap, gpa: u64, len: u64) {
        debug!("virtio-mem: unplugged {gpa:#x}+{len:#x}; its pages stay with the guest's backing");
    }

    fn config(&self) -> [u8; 56] {
        let mut config = [0u8; 56];
        let mut put = |offset: usize, value: u64| {
            config[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        };
        put(0, self.block_size);
        put(16, self.addr);
        put(24, self.region_size);
        put(32, self.region_size);
        put(40, self.plugged.plugged_size());
        put(48, self.shared.requested.load(Ordering::Relaxed));
        config
    }
}

impl VirtioDevice for Mem {
    fn avail_features(&self) -> u64 {
        AVAIL_FEATURES
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features
    }

    fn device_type(&self) -> u32 {
        VIRTIO_ID_MEM
    }

    fn device_name(&self) -> &str {
        "mem"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &defs::QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let config = self.config();
        let Some(src) = usize::try_from(offset)
            .ok()
            .and_then(|o| config.get(o..o.checked_add(data.len())?))
        else {
            warn!(
                "virtio-mem: config read out of range at {offset:#x}+{}",
                data.len()
            );
            return;
        };
        data.copy_from_slice(src);
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "virtio-mem: guest driver attempted to write device config (offset={offset:#x}, len={:#x})",
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if queues.len() != defs::NUM_QUEUES {
            error!(
                "virtio-mem: expected {} queue(s), got {}",
                defs::NUM_QUEUES,
                queues.len()
            );
            return Err(ActivateError::BadActivate);
        }
        if self.activate_evt.write(1).is_err() {
            error!("virtio-mem: cannot write to activate_evt");
            return Err(ActivateError::BadActivate);
        }
        self.queues = Some(queues);
        self.device_state = DeviceState::Activated(mem, interrupt);
        self.shared.bound.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    /// The plugged blocks outlive a reset: a driver that resets the device to suspend finds its
    /// memory as it left it (`VIRTIO_MEM_F_PERSISTENT_SUSPEND`), and one that probes it afresh
    /// sees a nonzero `plugged_size` and unplugs everything first.
    fn reset(&mut self) -> bool {
        self.queues = None;
        self.device_state = DeviceState::Inactive;
        true
    }
}
