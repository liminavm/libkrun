//! limina: a native-context-style extension of the virtio-gpu device.
//!
//! Behind one feature bit, the device offers two more shared-memory regions and five more control
//! commands, and leaves everything else (display, venus, vrend) as it was:
//!
//! - **Arenas** (shm id [`SHM_ID_ARENAS`]): host GPU memory an embedder-provided renderer
//!   allocates per guest context, mapped into this window wherever the device places it. The guest
//!   sub-allocates inside an arena and the renderer reads commands and data from it directly.
//! - **Doorbells** (shm id [`SHM_ID_DOORBELLS`]): a window with no memory behind it. Each context
//!   owns one [`DOORBELL_STRIDE`]-sized slot; a store anywhere in it exits to the VMM on the vCPU
//!   thread, which tells the renderer without going through any queue.
//!
//! The device owns the mechanism (window placement, mappings, slots, lifetimes); what an arena or
//! a doorbell means is the embedder's, through [`GpuExtension`]. A guest that does not negotiate the
//! feature sees a plain virtio-gpu.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(target_os = "macos")]
use crossbeam_channel::{Sender, unbounded};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;
use vm_memory::ByteValued;

use crate::bus::BusDevice;

/// Feature bit: the device implements this extension.
pub const VIRTIO_GPU_F_LIMINA_EXT: u32 = 23;
/// Shm region of the arena window.
pub const SHM_ID_ARENAS: u32 = 0x80;
/// Shm region of the doorbell window.
pub const SHM_ID_DOORBELLS: u32 = 0x81;
/// Bytes per doorbell slot: a 16 KiB page, so a 4 KiB and a 16 KiB guest can both hand each
/// context a page of its own.
pub const DOORBELL_STRIDE: u64 = 16 * 1024;
/// Arena placement granule: the host page, which `hv_vm_map` requires.
pub const ARENA_ALIGN: u64 = 16 * 1024;

pub const VIRTIO_GPU_CMD_LIMINA_CTX_CREATE: u32 = 0x4c00;
pub const VIRTIO_GPU_CMD_LIMINA_CTX_DESTROY: u32 = 0x4c01;
pub const VIRTIO_GPU_CMD_LIMINA_ARENA_CREATE: u32 = 0x4c02;
pub const VIRTIO_GPU_CMD_LIMINA_ARENA_DESTROY: u32 = 0x4c03;
pub const VIRTIO_GPU_CMD_LIMINA_RING_ATTACH: u32 = 0x4c04;

pub const VIRTIO_GPU_RESP_OK_LIMINA_CTX: u32 = 0x1180;
pub const VIRTIO_GPU_RESP_OK_LIMINA_ARENA: u32 = 0x1181;

/// `CTX_CREATE`: the context is `hdr.ctx_id`, in the extension's own namespace.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_ctx_create {
    pub flags: u32,
    pub padding: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_ctx_create {}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_arena_create {
    pub arena_id: u32,
    pub padding: u32,
    pub size: u64,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_arena_create {}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_arena_destroy {
    pub arena_id: u32,
    pub padding: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_arena_destroy {}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_ring_attach {
    pub arena_id: u32,
    pub data_size: u32,
    pub offset: u64,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_ring_attach {}

/// What a successful `CTX_CREATE` answers, after the header.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_resp_ctx {
    /// The context's doorbell slot in the doorbell window.
    pub doorbell: u32,
    pub padding: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_resp_ctx {}

/// What a successful `ARENA_CREATE` answers, after the header.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_resp_arena {
    /// Where the arena lies in the arena window.
    pub shm_offset: u64,
    /// The arena's size, rounded up by the host.
    pub size: u64,
    /// GPU address of the arena's first byte.
    pub gpu_base: u64,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_resp_arena {}

/// An arena the embedder allocated: its pages are mapped at `host_addr` in this process for as
/// long as the arena lives.
#[derive(Copy, Clone, Debug)]
pub struct ExtArena {
    pub host_addr: u64,
    pub size: u64,
    pub gpu_base: u64,
}

/// The embedder's half of the extension: the renderer behind contexts, arenas and doorbells.
pub trait GpuExtension: Send + Sync {
    fn create_context(&self, ctx: u32, doorbell: u32) -> Result<(), String>;
    fn destroy_context(&self, ctx: u32);
    fn create_arena(&self, ctx: u32, arena: u32, size: u64) -> Result<ExtArena, String>;
    /// Called after the guest mapping is gone; the embedder may unmap `host_addr`.
    fn destroy_arena(&self, ctx: u32, arena: u32);
    fn attach_ring(&self, ctx: u32, arena: u32, offset: u64, data_size: u32) -> Result<(), String>;
    /// A store to `doorbell`'s slot. Runs on a vCPU thread: must not block.
    fn doorbell(&self, doorbell: u32);
}

/// Where the device put the extension's windows in guest-physical space.
#[derive(Clone)]
pub struct GpuExtConfig {
    pub ext: Arc<dyn GpuExtension>,
    pub arenas_guest_addr: u64,
    pub arenas_size: u64,
    pub doorbells_guest_addr: u64,
    pub doorbells_size: u64,
}

impl GpuExtConfig {
    pub fn doorbell_slots(&self) -> u32 {
        (self.doorbells_size / DOORBELL_STRIDE).min(u32::MAX as u64) as u32
    }
}

/// The doorbell window on the MMIO bus.
pub struct DoorbellDevice {
    ext: Arc<dyn GpuExtension>,
}

impl DoorbellDevice {
    pub fn new(ext: Arc<dyn GpuExtension>) -> Self {
        Self { ext }
    }
}

impl BusDevice for DoorbellDevice {
    fn read(&mut self, _vcpuid: u64, _offset: u64, data: &mut [u8]) {
        data.fill(0);
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, _data: &[u8]) {
        self.ext.doorbell((offset / DOORBELL_STRIDE) as u32);
    }
}

struct Context {
    doorbell: u32,
    /// arena id → (window offset, size)
    arenas: HashMap<u32, (u64, u64)>,
}

/// Why a command failed, as the guest sees it.
#[derive(Debug, PartialEq, Eq)]
pub enum ExtError {
    InvalidContext,
    InvalidParameter,
    OutOfMemory,
    Unspec,
}

/// The device side of the extension, owned by the GPU worker thread.
pub struct ExtState {
    cfg: GpuExtConfig,
    #[cfg(target_os = "macos")]
    map_sender: Sender<WorkerMessage>,
    contexts: HashMap<u32, Context>,
    /// Window offsets in use, sorted: (offset, size).
    used: Vec<(u64, u64)>,
}

impl ExtState {
    pub fn new(
        cfg: GpuExtConfig,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
    ) -> Self {
        Self {
            cfg,
            #[cfg(target_os = "macos")]
            map_sender,
            contexts: HashMap::new(),
            used: Vec::new(),
        }
    }

    pub fn create_context(&mut self, ctx: u32) -> Result<virtio_gpu_limina_resp_ctx, ExtError> {
        if ctx == 0 || self.contexts.contains_key(&ctx) {
            return Err(ExtError::InvalidContext);
        }
        let taken: Vec<u32> = self.contexts.values().map(|c| c.doorbell).collect();
        let doorbell = (0..self.cfg.doorbell_slots())
            .find(|s| !taken.contains(s))
            .ok_or(ExtError::OutOfMemory)?;
        self.cfg.ext.create_context(ctx, doorbell).map_err(|e| {
            warn!("limina ext: context {ctx}: {e}");
            ExtError::Unspec
        })?;
        self.contexts.insert(
            ctx,
            Context {
                doorbell,
                arenas: HashMap::new(),
            },
        );
        Ok(virtio_gpu_limina_resp_ctx {
            doorbell,
            padding: 0,
        })
    }

    pub fn destroy_context(&mut self, ctx: u32) -> Result<(), ExtError> {
        let c = self.contexts.remove(&ctx).ok_or(ExtError::InvalidContext)?;
        for (arena, (off, size)) in c.arenas {
            self.unmap(off, size);
            self.release(off);
            self.cfg.ext.destroy_arena(ctx, arena);
        }
        self.cfg.ext.destroy_context(ctx);
        Ok(())
    }

    /// Tears every context down: the guest driver went away (device reset).
    pub fn reset(&mut self) {
        let ids: Vec<u32> = self.contexts.keys().copied().collect();
        for ctx in ids {
            let _ = self.destroy_context(ctx);
        }
    }

    fn place(&self, size: u64) -> Option<u64> {
        let mut at = 0u64;
        for &(off, len) in &self.used {
            if off >= at && off - at >= size {
                return Some(at);
            }
            at = at.max((off + len).next_multiple_of(ARENA_ALIGN));
        }
        (at.checked_add(size)? <= self.cfg.arenas_size).then_some(at)
    }

    fn reserve(&mut self, off: u64, size: u64) {
        let i = self.used.partition_point(|&(o, _)| o < off);
        self.used.insert(i, (off, size));
    }

    fn release(&mut self, off: u64) {
        self.used.retain(|&(o, _)| o != off);
    }

    #[cfg(target_os = "macos")]
    fn map(&self, host_addr: u64, off: u64, size: u64) -> bool {
        let (tx, rx) = unbounded();
        self.map_sender
            .send(WorkerMessage::GpuAddMapping(
                tx,
                host_addr,
                self.cfg.arenas_guest_addr + off,
                size,
            ))
            .is_ok()
            && rx.recv().unwrap_or(false)
    }

    #[cfg(target_os = "macos")]
    fn unmap(&self, off: u64, size: u64) {
        let (tx, rx) = unbounded();
        if self
            .map_sender
            .send(WorkerMessage::GpuRemoveMapping(
                tx,
                self.cfg.arenas_guest_addr + off,
                size,
            ))
            .is_ok()
        {
            let _ = rx.recv();
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn map(&self, _host_addr: u64, _off: u64, _size: u64) -> bool {
        false
    }

    #[cfg(not(target_os = "macos"))]
    fn unmap(&self, _off: u64, _size: u64) {}

    pub fn create_arena(
        &mut self,
        ctx: u32,
        req: virtio_gpu_limina_arena_create,
    ) -> Result<virtio_gpu_limina_resp_arena, ExtError> {
        let c = self.contexts.get(&ctx).ok_or(ExtError::InvalidContext)?;
        if req.size == 0 || c.arenas.contains_key(&req.arena_id) {
            return Err(ExtError::InvalidParameter);
        }
        let a = self
            .cfg
            .ext
            .create_arena(ctx, req.arena_id, req.size)
            .map_err(|e| {
                warn!("limina ext: arena {}/{}: {e}", ctx, req.arena_id);
                ExtError::OutOfMemory
            })?;
        let Some(off) = self.place(a.size) else {
            self.cfg.ext.destroy_arena(ctx, req.arena_id);
            return Err(ExtError::OutOfMemory);
        };
        if a.host_addr % ARENA_ALIGN != 0 || !self.map(a.host_addr, off, a.size) {
            self.cfg.ext.destroy_arena(ctx, req.arena_id);
            return Err(ExtError::Unspec);
        }
        self.reserve(off, a.size);
        self.contexts
            .get_mut(&ctx)
            .unwrap()
            .arenas
            .insert(req.arena_id, (off, a.size));
        Ok(virtio_gpu_limina_resp_arena {
            shm_offset: off,
            size: a.size,
            gpu_base: a.gpu_base,
        })
    }

    pub fn destroy_arena(&mut self, ctx: u32, arena: u32) -> Result<(), ExtError> {
        let c = self
            .contexts
            .get_mut(&ctx)
            .ok_or(ExtError::InvalidContext)?;
        let (off, size) = c.arenas.remove(&arena).ok_or(ExtError::InvalidParameter)?;
        self.unmap(off, size);
        self.release(off);
        self.cfg.ext.destroy_arena(ctx, arena);
        Ok(())
    }

    pub fn attach_ring(
        &mut self,
        ctx: u32,
        req: virtio_gpu_limina_ring_attach,
    ) -> Result<(), ExtError> {
        let c = self.contexts.get(&ctx).ok_or(ExtError::InvalidContext)?;
        if !c.arenas.contains_key(&req.arena_id) {
            return Err(ExtError::InvalidParameter);
        }
        self.cfg
            .ext
            .attach_ring(ctx, req.arena_id, req.offset, req.data_size)
            .map_err(|e| {
                warn!("limina ext: ring {}/{}: {e}", ctx, req.arena_id);
                ExtError::InvalidParameter
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Fake {
        log: Mutex<Vec<String>>,
    }

    impl GpuExtension for Fake {
        fn create_context(&self, ctx: u32, doorbell: u32) -> Result<(), String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("ctx {ctx} bell {doorbell}"));
            Ok(())
        }
        fn destroy_context(&self, ctx: u32) {
            self.log.lock().unwrap().push(format!("-ctx {ctx}"));
        }
        fn create_arena(&self, _ctx: u32, _arena: u32, size: u64) -> Result<ExtArena, String> {
            Ok(ExtArena {
                host_addr: 0x1_0000_0000,
                size: size.next_multiple_of(ARENA_ALIGN),
                gpu_base: 0x5000,
            })
        }
        fn destroy_arena(&self, ctx: u32, arena: u32) {
            self.log
                .lock()
                .unwrap()
                .push(format!("-arena {ctx}/{arena}"));
        }
        fn attach_ring(&self, _: u32, _: u32, _: u64, _: u32) -> Result<(), String> {
            Ok(())
        }
        fn doorbell(&self, _: u32) {}
    }

    fn state() -> (Arc<Fake>, ExtState) {
        let fake = Arc::new(Fake::default());
        let cfg = GpuExtConfig {
            ext: fake.clone(),
            arenas_guest_addr: 0x40_0000_0000,
            arenas_size: 1 << 20,
            doorbells_guest_addr: 0x3f_0000_0000,
            doorbells_size: 4 * DOORBELL_STRIDE,
        };
        #[cfg(target_os = "macos")]
        let (tx, rx) = unbounded();
        #[cfg(target_os = "macos")]
        std::thread::spawn(move || {
            while let Ok(m) = rx.recv() {
                match m {
                    WorkerMessage::GpuAddMapping(r, ..)
                    | WorkerMessage::GpuRemoveMapping(r, ..) => {
                        let _ = r.send(true);
                    }
                    _ => {}
                }
            }
        });
        (
            fake,
            ExtState::new(
                cfg,
                #[cfg(target_os = "macos")]
                tx,
            ),
        )
    }

    #[test]
    fn doorbell_slots_are_unique_and_reused() {
        let (_f, mut s) = state();
        let a = s.create_context(1).unwrap().doorbell;
        let b = s.create_context(2).unwrap().doorbell;
        assert_ne!(a, b);
        s.destroy_context(1).unwrap();
        assert_eq!(s.create_context(3).unwrap().doorbell, a);
        assert_eq!(s.create_context(3).err(), Some(ExtError::InvalidContext));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn arenas_are_placed_without_overlap_and_holes_are_reused() {
        let (_f, mut s) = state();
        s.create_context(1).unwrap();
        let req = |arena_id, size| virtio_gpu_limina_arena_create {
            arena_id,
            padding: 0,
            size,
        };
        let a = s.create_arena(1, req(1, 100_000)).unwrap();
        let b = s.create_arena(1, req(2, 300_000)).unwrap();
        assert_eq!(a.shm_offset, 0);
        assert_eq!(a.size, 100_000u64.next_multiple_of(ARENA_ALIGN));
        assert_eq!(b.shm_offset, a.size);
        s.destroy_arena(1, 1).unwrap();
        let c = s.create_arena(1, req(3, 16_384)).unwrap();
        assert_eq!(c.shm_offset, 0, "the hole is reused");
        assert_eq!(
            s.create_arena(1, req(4, 1 << 20)).err(),
            Some(ExtError::OutOfMemory),
            "the window is full"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn destroying_a_context_releases_its_arenas() {
        let (f, mut s) = state();
        s.create_context(7).unwrap();
        s.create_arena(
            7,
            virtio_gpu_limina_arena_create {
                arena_id: 1,
                padding: 0,
                size: 1 << 20,
            },
        )
        .unwrap();
        s.destroy_context(7).unwrap();
        assert!(f.log.lock().unwrap().contains(&"-arena 7/1".to_string()));
        s.create_context(8).unwrap();
        assert!(
            s.create_arena(
                8,
                virtio_gpu_limina_arena_create {
                    arena_id: 1,
                    padding: 0,
                    size: 1 << 20,
                },
            )
            .is_ok(),
            "the whole window is free again"
        );
    }
}
