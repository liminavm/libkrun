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
//! - **Shared arenas**: an arena created with [`ARENA_FLAG_BLOB`] is not placed in the arena
//!   window. The guest wraps it in a host3d blob resource instead (`blob_id` = the arena id), which
//!   the device maps into virtio-gpu's host-visible region like any blob; the resource is what other
//!   guest processes receive (as a dma-buf) and import with `ARENA_IMPORT`. An exported arena lives
//!   until its last holder lets go: the arena id in its own context, the blob resource, and every
//!   import.
//!
//! - **Fences**: `FENCE` is sent with `VIRTIO_GPU_FLAG_FENCE`; its response is held until the
//!   context's ring has completed the seqno it names, so a guest `dma_fence` follows GPU work the
//!   control queue never carried. Every fence is answered: when its context goes away, it is
//!   answered then.
//!
//! The device owns the mechanism (window placement, mappings, slots, lifetimes); what an arena or
//! a doorbell means is the embedder's, through [`GpuExtension`]. A guest that does not negotiate the
//! feature sees a plain virtio-gpu.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

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
/// `ARENA_CREATE` flag: the arena is shared through a blob resource, not the arena window.
pub const ARENA_FLAG_BLOB: u32 = 1;
/// `shm_offset` of an arena that is not in the window.
pub const NOT_IN_WINDOW: u64 = u64::MAX;

pub const VIRTIO_GPU_CMD_LIMINA_CTX_CREATE: u32 = 0x4c00;
pub const VIRTIO_GPU_CMD_LIMINA_CTX_DESTROY: u32 = 0x4c01;
pub const VIRTIO_GPU_CMD_LIMINA_ARENA_CREATE: u32 = 0x4c02;
pub const VIRTIO_GPU_CMD_LIMINA_ARENA_DESTROY: u32 = 0x4c03;
pub const VIRTIO_GPU_CMD_LIMINA_RING_ATTACH: u32 = 0x4c04;
pub const VIRTIO_GPU_CMD_LIMINA_ARENA_IMPORT: u32 = 0x4c05;
pub const VIRTIO_GPU_CMD_LIMINA_FENCE: u32 = 0x4c06;
pub const VIRTIO_GPU_CMD_LIMINA_SURFACE_CREATE: u32 = 0x4c07;
pub const VIRTIO_GPU_CMD_LIMINA_SURFACE_IMPORT: u32 = 0x4c08;

pub const VIRTIO_GPU_RESP_OK_LIMINA_CTX: u32 = 0x1180;
pub const VIRTIO_GPU_RESP_OK_LIMINA_ARENA: u32 = 0x1181;
pub const VIRTIO_GPU_RESP_OK_LIMINA_SURFACE: u32 = 0x1182;

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
    /// `ARENA_FLAG_*`.
    pub flags: u32,
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

/// `ARENA_IMPORT`: arena `arena_id` of `hdr.ctx_id` becomes the memory of blob resource
/// `resource_id`, which another context exported. Answered like `ARENA_CREATE`.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_arena_import {
    pub arena_id: u32,
    pub resource_id: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_arena_import {}

/// `FENCE`: answered once `hdr.ctx_id`'s ring has completed `seqno`.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_fence {
    pub seqno: u64,
}
// SAFETY: a plain integer.
unsafe impl ByteValued for virtio_gpu_limina_fence {}

/// `SURFACE_CREATE`: resource `resource_id` becomes a surface the device mints, `width` x
/// `height` of DRM format `drm_fourcc` with rows `stride` bytes apart, that `hdr.ctx_id` renders
/// into and every other context reaches as an ordinary resource.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_surface_create {
    pub resource_id: u32,
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub stride: u32,
    pub padding: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_surface_create {}

/// `SURFACE_IMPORT`: `hdr.ctx_id` is given the surface resource `resource_id` is presented from,
/// whichever renderer made it. Answered with its shape.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_surface_import {
    pub resource_id: u32,
    pub padding: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_surface_import {}

/// What a successful `SURFACE_IMPORT` answers, after the header.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct virtio_gpu_limina_resp_surface {
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub stride: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_resp_surface {}

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
    /// Where the arena lies in the arena window, or [`NOT_IN_WINDOW`].
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
    /// Makes arena `arena` of `ctx` reachable as `export` from any context, and alive (pages
    /// mapped at the returned `host_addr` included) until [`Self::release_export`], whatever
    /// happens to `ctx`.
    fn export_arena(&self, ctx: u32, arena: u32, export: u32) -> Result<ExtArena, String>;
    /// Makes arena `arena` of `ctx` the memory of `export`.
    fn import_arena(&self, ctx: u32, arena: u32, export: u32) -> Result<ExtArena, String>;
    fn release_export(&self, export: u32);
    /// Calls `signal` once `ctx`'s ring has completed `seqno` or the context is lost, from any
    /// thread; at once if it already has.
    fn fence(&self, ctx: u32, seqno: u64, signal: Signal);
    /// Gives `ctx` the surface `port` (a send right, moved) names, as `surface`.
    fn bind_surface(&self, ctx: u32, surface: u32, port: u32) -> Result<(), String>;
    /// `ctx` gives up `surface`; what it already made over it keeps it.
    fn release_surface(&self, ctx: u32, surface: u32);
}

/// What answers a fence; see [`GpuExtension::fence`].
pub type Signal = Box<dyn FnOnce() + Send>;

/// A fence's answer, shared by the embedder and the device: whichever comes first sends it.
#[derive(Clone)]
struct PendingFence(Arc<Mutex<Option<Signal>>>);

impl PendingFence {
    fn fire(&self) {
        let signal = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(signal) = signal {
            signal();
        }
    }

    fn fired(&self) -> bool {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).is_none()
    }
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
    /// Fences not answered yet, answered when the context goes away.
    fences: Vec<PendingFence>,
    /// arena id → its (window offset, size), or `None` for one shared through a blob resource.
    arenas: HashMap<u32, Option<(u64, u64)>>,
}

struct Export {
    arena: ExtArena,
    mapped: Option<u64>,
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
    /// Blob resources that are exported arenas, with where the blob is mapped in the
    /// host-visible region.
    exports: HashMap<u32, Export>,
    /// Surfaces given to contexts, as (context, resource): one resource is held by the context
    /// that made it and by every one that imported it.
    surfaces: HashSet<(u32, u32)>,
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
            exports: HashMap::new(),
            surfaces: HashSet::new(),
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
                fences: Vec::new(),
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
        for (arena, window) in c.arenas {
            if let Some((off, size)) = window {
                self.unmap(off, size);
                self.release(off);
            }
            self.cfg.ext.destroy_arena(ctx, arena);
        }
        self.cfg.ext.destroy_context(ctx);
        // The embedder's context took its surfaces with it; the resources live on.
        self.surfaces.retain(|&(c, _)| c != ctx);
        for f in c.fences {
            f.fire();
        }
        Ok(())
    }

    /// Answers `signal` once `ctx` has completed `seqno`, or when the context goes away; at once
    /// if there is no such context, so no fence is left unanswered.
    pub fn fence(&mut self, ctx: u32, seqno: u64, signal: Signal) -> Result<(), ExtError> {
        let pending = PendingFence(Arc::new(Mutex::new(Some(signal))));
        let Some(c) = self.contexts.get_mut(&ctx) else {
            pending.fire();
            return Err(ExtError::InvalidContext);
        };
        c.fences.retain(|f| !f.fired());
        c.fences.push(pending.clone());
        self.cfg
            .ext
            .fence(ctx, seqno, Box::new(move || pending.fire()));
        Ok(())
    }

    /// Tears every context down: the guest driver went away (device reset).
    pub fn reset(&mut self) {
        let ids: Vec<u32> = self.contexts.keys().copied().collect();
        for ctx in ids {
            let _ = self.destroy_context(ctx);
        }
        for (export, _) in self.exports.drain() {
            self.cfg.ext.release_export(export);
        }
    }

    pub fn has_context(&self, ctx: u32) -> bool {
        self.contexts.contains_key(&ctx)
    }

    /// Blob resource `resource` wraps arena `arena` of `ctx`, which must have been created to be
    /// shared. Answers where its pages are, for the device to map them into the host-visible
    /// region.
    /// `size` must be the arena's: the guest reserves that much of the host-visible region.
    pub fn export_arena(
        &mut self,
        ctx: u32,
        arena: u32,
        resource: u32,
        size: u64,
    ) -> Result<ExtArena, ExtError> {
        let c = self.contexts.get(&ctx).ok_or(ExtError::InvalidContext)?;
        if c.arenas.get(&arena) != Some(&None) || self.exports.contains_key(&resource) {
            return Err(ExtError::InvalidParameter);
        }
        let a = self
            .cfg
            .ext
            .export_arena(ctx, arena, resource)
            .map_err(|e| {
                warn!("limina ext: export {ctx}/{arena} as {resource}: {e}");
                ExtError::InvalidParameter
            })?;
        if a.size != size {
            self.cfg.ext.release_export(resource);
            return Err(ExtError::InvalidParameter);
        }
        self.exports.insert(
            resource,
            Export {
                arena: a,
                mapped: None,
            },
        );
        Ok(a)
    }

    /// The exported arena behind blob resource `resource`, and where it is mapped.
    pub fn export(&self, resource: u32) -> Option<(ExtArena, Option<u64>)> {
        self.exports.get(&resource).map(|e| (e.arena, e.mapped))
    }

    pub fn set_mapped(&mut self, resource: u32, offset: Option<u64>) {
        if let Some(e) = self.exports.get_mut(&resource) {
            e.mapped = offset;
        }
    }

    /// The blob resource is gone; the arena lives on while its context or an import holds it.
    pub fn release_export(&mut self, resource: u32) {
        if self.exports.remove(&resource).is_some() {
            self.cfg.ext.release_export(resource);
        }
    }

    /// Resource `resource`, a surface, is given to `ctx` through `port`.
    pub fn bind_surface(&mut self, ctx: u32, resource: u32, port: u32) -> Result<(), ExtError> {
        if !self.contexts.contains_key(&ctx) {
            return Err(ExtError::InvalidContext);
        }
        if self.surfaces.contains(&(ctx, resource)) {
            return Err(ExtError::InvalidParameter);
        }
        self.cfg
            .ext
            .bind_surface(ctx, resource, port)
            .map_err(|e| {
                warn!("limina ext: ctx {ctx} surface {resource}: {e}");
                ExtError::Unspec
            })?;
        self.surfaces.insert((ctx, resource));
        Ok(())
    }

    pub fn holds_surface(&self, ctx: u32, resource: u32) -> bool {
        self.surfaces.contains(&(ctx, resource))
    }

    /// `ctx` let go of the resource; the others holding it keep it.
    pub fn detach_surface(&mut self, ctx: u32, resource: u32) {
        if self.surfaces.remove(&(ctx, resource)) {
            self.cfg.ext.release_surface(ctx, resource);
        }
    }

    /// The resource is gone; every context holding it gives it back.
    pub fn release_surface(&mut self, resource: u32) {
        let holders: Vec<u32> = self
            .surfaces
            .iter()
            .filter(|&&(_, r)| r == resource)
            .map(|&(c, _)| c)
            .collect();
        for ctx in holders {
            self.detach_surface(ctx, resource);
        }
    }

    pub fn is_export(&self, resource: u32) -> bool {
        self.exports.contains_key(&resource)
    }

    pub fn import_arena(
        &mut self,
        ctx: u32,
        req: virtio_gpu_limina_arena_import,
    ) -> Result<virtio_gpu_limina_resp_arena, ExtError> {
        let c = self.contexts.get(&ctx).ok_or(ExtError::InvalidContext)?;
        if c.arenas.contains_key(&req.arena_id) || !self.exports.contains_key(&req.resource_id) {
            return Err(ExtError::InvalidParameter);
        }
        let a = self
            .cfg
            .ext
            .import_arena(ctx, req.arena_id, req.resource_id)
            .map_err(|e| {
                warn!(
                    "limina ext: import {} as {ctx}/{}: {e}",
                    req.resource_id, req.arena_id
                );
                ExtError::InvalidParameter
            })?;
        self.contexts
            .get_mut(&ctx)
            .unwrap()
            .arenas
            .insert(req.arena_id, None);
        Ok(virtio_gpu_limina_resp_arena {
            shm_offset: NOT_IN_WINDOW,
            size: a.size,
            gpu_base: a.gpu_base,
        })
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
        if req.size == 0
            || c.arenas.contains_key(&req.arena_id)
            || req.flags & !ARENA_FLAG_BLOB != 0
        {
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
        if req.flags & ARENA_FLAG_BLOB != 0 {
            self.contexts
                .get_mut(&ctx)
                .unwrap()
                .arenas
                .insert(req.arena_id, None);
            return Ok(virtio_gpu_limina_resp_arena {
                shm_offset: NOT_IN_WINDOW,
                size: a.size,
                gpu_base: a.gpu_base,
            });
        }
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
            .insert(req.arena_id, Some((off, a.size)));
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
        let window = c.arenas.remove(&arena).ok_or(ExtError::InvalidParameter)?;
        if let Some((off, size)) = window {
            self.unmap(off, size);
            self.release(off);
        }
        self.cfg.ext.destroy_arena(ctx, arena);
        Ok(())
    }

    pub fn attach_ring(
        &mut self,
        ctx: u32,
        req: virtio_gpu_limina_ring_attach,
    ) -> Result<(), ExtError> {
        let c = self.contexts.get(&ctx).ok_or(ExtError::InvalidContext)?;
        if !matches!(c.arenas.get(&req.arena_id), Some(Some(_))) {
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

    #[derive(Default)]
    struct Fake {
        log: Mutex<Vec<String>>,
        /// Fences the "GPU" has not completed: (ctx, seqno, signal).
        fences: Mutex<Vec<(u32, u64, Signal)>>,
    }

    impl Fake {
        /// Completes every fence of `ctx` up to `seqno`.
        fn complete(&self, ctx: u32, seqno: u64) {
            let mut fences = self.fences.lock().unwrap();
            let (done, rest) = std::mem::take(&mut *fences)
                .into_iter()
                .partition(|f| f.0 == ctx && f.1 <= seqno);
            *fences = rest;
            drop(fences);
            for (_, _, signal) in done {
                signal();
            }
        }
    }

    fn logging_signal(f: &Arc<Fake>, what: &str) -> Signal {
        let f = f.clone();
        let what = what.to_string();
        Box::new(move || f.log.lock().unwrap().push(what))
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
        fn export_arena(&self, ctx: u32, arena: u32, export: u32) -> Result<ExtArena, String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("export {ctx}/{arena} as {export}"));
            Ok(ExtArena {
                host_addr: 0x1_0000_0000,
                size: 1 << 20,
                gpu_base: 0x5000,
            })
        }
        fn import_arena(&self, ctx: u32, arena: u32, export: u32) -> Result<ExtArena, String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("import {export} as {ctx}/{arena}"));
            Ok(ExtArena {
                host_addr: 0x1_0000_0000,
                size: 1 << 20,
                gpu_base: 0x5000,
            })
        }
        fn release_export(&self, export: u32) {
            self.log.lock().unwrap().push(format!("-export {export}"));
        }
        fn fence(&self, ctx: u32, seqno: u64, signal: Signal) {
            self.fences.lock().unwrap().push((ctx, seqno, signal));
        }
        fn bind_surface(&self, ctx: u32, surface: u32, port: u32) -> Result<(), String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("surface {surface} port {port} to {ctx}"));
            Ok(())
        }
        fn release_surface(&self, ctx: u32, surface: u32) {
            self.log
                .lock()
                .unwrap()
                .push(format!("-surface {surface} of {ctx}"));
        }
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
            flags: 0,
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
                flags: 0,
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
                    flags: 0,
                    size: 1 << 20,
                },
            )
            .is_ok(),
            "the whole window is free again"
        );
    }

    #[test]
    fn a_shared_arena_is_not_placed_in_the_window() {
        let (_f, mut s) = state();
        s.create_context(1).unwrap();
        let blob = virtio_gpu_limina_arena_create {
            arena_id: 1,
            flags: ARENA_FLAG_BLOB,
            size: 1 << 20,
        };
        assert_eq!(s.create_arena(1, blob).unwrap().shm_offset, NOT_IN_WINDOW);
        assert!(s.used.is_empty(), "the window is untouched");
        assert_eq!(
            s.create_arena(
                1,
                virtio_gpu_limina_arena_create {
                    arena_id: 2,
                    flags: 4,
                    size: 1
                }
            )
            .err(),
            Some(ExtError::InvalidParameter),
            "unknown flags are refused"
        );
    }

    #[test]
    fn an_export_outlives_its_context_until_released() {
        let (f, mut s) = state();
        s.create_context(1).unwrap();
        s.create_context(2).unwrap();
        let blob = virtio_gpu_limina_arena_create {
            arena_id: 1,
            flags: ARENA_FLAG_BLOB,
            size: 1 << 20,
        };
        s.create_arena(1, blob).unwrap();
        s.export_arena(1, 1, 40, 1 << 20).unwrap();
        assert_eq!(
            s.export_arena(1, 1, 40, 1 << 20).err(),
            Some(ExtError::InvalidParameter),
            "one resource per export"
        );
        s.destroy_context(1).unwrap();
        assert!(s.is_export(40), "the resource still holds the arena");
        assert!(!f.log.lock().unwrap().contains(&"-export 40".to_string()));

        let imported = s
            .import_arena(
                2,
                virtio_gpu_limina_arena_import {
                    arena_id: 9,
                    resource_id: 40,
                },
            )
            .unwrap();
        assert_eq!(imported.shm_offset, NOT_IN_WINDOW);
        assert!(
            f.log
                .lock()
                .unwrap()
                .contains(&"import 40 as 2/9".to_string())
        );

        s.release_export(40);
        assert!(f.log.lock().unwrap().contains(&"-export 40".to_string()));
        assert_eq!(
            s.import_arena(
                2,
                virtio_gpu_limina_arena_import {
                    arena_id: 10,
                    resource_id: 40,
                },
            )
            .err(),
            Some(ExtError::InvalidParameter),
            "a released resource imports nothing"
        );
    }

    #[test]
    fn a_surface_is_given_back_once_and_goes_with_its_context() {
        let (f, mut s) = state();
        let log = |f: &Arc<Fake>, line: &str| f.log.lock().unwrap().contains(&line.to_string());
        assert_eq!(
            s.bind_surface(1, 50, 7).err(),
            Some(ExtError::InvalidContext),
            "no context to give it to"
        );
        s.create_context(1).unwrap();
        s.bind_surface(1, 50, 7).unwrap();
        assert!(log(&f, "surface 50 port 7 to 1"));
        assert_eq!(
            s.bind_surface(1, 50, 8).err(),
            Some(ExtError::InvalidParameter),
            "a context holds a surface once"
        );
        s.release_surface(50);
        s.release_surface(50);
        assert_eq!(
            f.log
                .lock()
                .unwrap()
                .iter()
                .filter(|l| *l == "-surface 50 of 1")
                .count(),
            1,
            "given back once"
        );

        s.bind_surface(1, 51, 9).unwrap();
        s.destroy_context(1).unwrap();
        s.release_surface(51);
        assert!(
            !log(&f, "-surface 51 of 1"),
            "the context took its surfaces with it"
        );
    }

    #[test]
    fn an_imported_surface_has_a_holder_per_context() {
        let (f, mut s) = state();
        let gone =
            |f: &Arc<Fake>, line: &str| f.log.lock().unwrap().iter().filter(|l| *l == line).count();
        s.create_context(1).unwrap();
        s.create_context(2).unwrap();
        s.create_context(3).unwrap();
        s.bind_surface(1, 60, 7).unwrap();
        s.bind_surface(2, 60, 8).unwrap();
        s.bind_surface(3, 60, 9).unwrap();

        s.detach_surface(2, 60);
        s.detach_surface(2, 60);
        assert_eq!(gone(&f, "-surface 60 of 2"), 1, "the importer let go, once");
        assert_eq!(gone(&f, "-surface 60 of 1"), 0, "the others still hold it");

        s.release_surface(60);
        assert_eq!(gone(&f, "-surface 60 of 1"), 1);
        assert_eq!(
            gone(&f, "-surface 60 of 3"),
            1,
            "every holder, when the resource goes"
        );
        assert_eq!(gone(&f, "-surface 60 of 2"), 1);
    }

    #[test]
    fn only_shared_arenas_export() {
        let (_f, mut s) = state();
        s.create_context(1).unwrap();
        #[cfg(target_os = "macos")]
        {
            s.create_arena(
                1,
                virtio_gpu_limina_arena_create {
                    arena_id: 1,
                    flags: 0,
                    size: 1 << 16,
                },
            )
            .unwrap();
            assert_eq!(
                s.export_arena(1, 1, 3, 1 << 16).err(),
                Some(ExtError::InvalidParameter),
                "a window arena is not shareable"
            );
        }
        assert_eq!(
            s.export_arena(1, 2, 3, 1 << 16).err(),
            Some(ExtError::InvalidParameter),
            "no such arena"
        );
    }

    #[test]
    fn a_fence_is_answered_when_its_seqno_completes_and_only_once() {
        let (f, mut s) = state();
        s.create_context(1).unwrap();
        s.fence(1, 5, logging_signal(&f, "fence 5")).unwrap();
        f.complete(1, 4);
        assert!(!f.log.lock().unwrap().contains(&"fence 5".to_string()));
        f.complete(1, 5);
        s.destroy_context(1).unwrap();
        let log = f.log.lock().unwrap();
        assert_eq!(log.iter().filter(|l| *l == "fence 5").count(), 1);
    }

    #[test]
    fn destroying_a_context_answers_its_pending_fences() {
        let (f, mut s) = state();
        s.create_context(1).unwrap();
        s.fence(1, 9, logging_signal(&f, "fence 9")).unwrap();
        s.destroy_context(1).unwrap();
        let log = f.log.lock().unwrap().clone();
        let answered = log.iter().position(|l| l == "fence 9");
        assert!(answered.is_some(), "the fence was answered: {log:?}");
        drop(f.fences.lock().unwrap().drain(..));
    }

    #[test]
    fn a_fence_without_a_context_is_answered_at_once() {
        let (f, mut s) = state();
        assert_eq!(
            s.fence(7, 1, logging_signal(&f, "fence")).err(),
            Some(ExtError::InvalidContext)
        );
        assert!(f.log.lock().unwrap().contains(&"fence".to_string()));
    }

    #[test]
    fn a_reset_answers_every_fence() {
        let (f, mut s) = state();
        s.create_context(1).unwrap();
        s.create_context(2).unwrap();
        s.fence(1, 1, logging_signal(&f, "a")).unwrap();
        s.fence(2, 1, logging_signal(&f, "b")).unwrap();
        s.reset();
        let log = f.log.lock().unwrap();
        assert!(log.contains(&"a".to_string()) && log.contains(&"b".to_string()));
    }
}
