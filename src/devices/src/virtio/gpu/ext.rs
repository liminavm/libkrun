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

use std::collections::HashMap;
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
pub const VIRTIO_GPU_CMD_LIMINA_INFO: u32 = 0x4c09;

pub const VIRTIO_GPU_RESP_OK_LIMINA_CTX: u32 = 0x1180;
pub const VIRTIO_GPU_RESP_OK_LIMINA_ARENA: u32 = 0x1181;
pub const VIRTIO_GPU_RESP_OK_LIMINA_SURFACE: u32 = 0x1182;
pub const VIRTIO_GPU_RESP_OK_LIMINA_INFO: u32 = 0x1183;

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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct virtio_gpu_limina_resp_surface {
    pub width: u32,
    pub height: u32,
    pub drm_fourcc: u32,
    pub stride: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_resp_surface {}

/// What `INFO` answers, after the header: enough for a guest driver to decline a host it cannot
/// speak to without creating a context.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct virtio_gpu_limina_resp_info {
    /// The version of the protocol the embedder's renderer speaks on a context's ring.
    pub wire_version: u32,
    pub flags: u32,
}
// SAFETY: plain integers, no padding.
unsafe impl ByteValued for virtio_gpu_limina_resp_info {}

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
    /// The version of the protocol contexts speak on their rings, which the device only relays.
    fn wire_version(&self) -> u32;
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
    /// `ctx` as it travels across a snapshot, opaque to the device. Answered once its committed
    /// work has finished or been given up on; given up on, the context is lost and its fences are
    /// answered.
    fn export_context(&self, ctx: u32) -> Result<Vec<u8>, String>;
    /// Arena `arena` of `ctx` again, `size` bytes at `gpu_base`, in a restore. Arenas come in
    /// address order, before any ring restarts. `ctx` 0 is an arena only its export still held.
    fn restore_arena(
        &self,
        ctx: u32,
        arena: u32,
        size: u64,
        gpu_base: u64,
    ) -> Result<ExtArena, String>;
    /// `ctx`'s ring restarted where [`Self::export_context`] left it, with what that exported.
    fn restore_ring(
        &self,
        ctx: u32,
        arena: u32,
        offset: u64,
        data_size: u32,
        export: &[u8],
    ) -> Result<(), String>;
    /// Every arena and ring of a restore stands.
    fn restore_done(&self);
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
    arenas: HashMap<u32, ContextArena>,
    /// Where the ring is: (arena, offset, data size).
    ring: Option<(u32, u64, u32)>,
}

#[derive(Clone, Copy)]
struct ContextArena {
    arena: ExtArena,
    /// Its window offset, or `None` for one shared through a blob resource.
    window: Option<u64>,
    /// The blob resource it was imported from.
    import: Option<u32>,
}

struct Export {
    arena: ExtArena,
    mapped: Option<u64>,
    /// The context and arena id it was exported from.
    origin: (u32, u32),
}

/// The shape of a surface, as `SURFACE_CREATE` gave it and `SURFACE_IMPORT` answers it.
pub type SurfaceShape = virtio_gpu_limina_resp_surface;

/// One arena in an [`ExtSnapshot`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaSnapshot {
    /// The context that made it, or 0 for one only its export still holds (`arena` is then the
    /// export's resource).
    pub ctx: u32,
    pub arena: u32,
    pub size: u64,
    pub gpu_base: u64,
    pub window: Option<u64>,
    pub contents: Vec<u8>,
}

/// One context in an [`ExtSnapshot`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ContextSnapshot {
    pub ctx: u32,
    pub doorbell: u32,
    /// (arena, offset, data size) and what [`GpuExtension::export_context`] answered.
    pub ring: Option<(u32, u64, u32, Vec<u8>)>,
}

/// The extension across a snapshot: the device's books, the arenas' contents and each
/// context's export.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtSnapshot {
    pub contexts: Vec<ContextSnapshot>,
    /// In address order.
    pub arenas: Vec<ArenaSnapshot>,
    /// (resource, ctx, arena, where its blob is mapped), with `ctx` 0 as in [`ArenaSnapshot`].
    pub exports: Vec<(u32, u32, u32, Option<u64>)>,
    /// (ctx, arena, resource).
    pub imports: Vec<(u32, u32, u32)>,
    /// Surfaces minted for the embedder that still exist: (resource, ctx, shape, rows), `ctx`
    /// being the context the resource was made for. That context may be gone; a surface is given
    /// back only to the contexts in `held`. Filled by the device, which owns the resources: the
    /// extension knows a surface only through who holds it.
    pub made: Vec<(u32, u32, SurfaceShape, Vec<u8>)>,
    /// Every context holding a surface: (ctx, resource).
    pub held: Vec<(u32, u32)>,
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
    /// Surfaces given to contexts, as (context, resource), with the shape each was given as:
    /// one resource is held by the context that made it and by every one that imported it.
    surfaces: HashMap<(u32, u32), SurfaceShape>,
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
            surfaces: HashMap::new(),
        }
    }

    pub fn info(&self) -> virtio_gpu_limina_resp_info {
        virtio_gpu_limina_resp_info {
            wire_version: self.cfg.ext.wire_version(),
            flags: 0,
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
                ring: None,
            },
        );
        Ok(virtio_gpu_limina_resp_ctx {
            doorbell,
            padding: 0,
        })
    }

    pub fn destroy_context(&mut self, ctx: u32) -> Result<(), ExtError> {
        let c = self.contexts.remove(&ctx).ok_or(ExtError::InvalidContext)?;
        let mut reachable = false;
        for a in c.arenas.values() {
            if let Some(off) = a.window {
                if self.unmap(off, a.arena.size) {
                    self.release(off);
                } else {
                    reachable = true;
                }
            }
        }
        if reachable {
            // The context's memory goes with the context in the embedder, so the context stays.
            error!(
                "limina ext: an arena of context {ctx} could not be unmapped; the context is \
                 kept so the guest never reaches freed memory"
            );
        } else {
            for arena in c.arenas.keys() {
                self.cfg.ext.destroy_arena(ctx, *arena);
            }
            self.cfg.ext.destroy_context(ctx);
        }
        // The embedder's context took its surfaces with it; the resources live on.
        self.surfaces.retain(|&(c, _), _| c != ctx);
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
        let shareable = c
            .arenas
            .get(&arena)
            .is_some_and(|a| a.window.is_none() && a.import.is_none());
        if !shareable || self.exports.contains_key(&resource) {
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
                origin: (ctx, arena),
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

    /// Resource `resource`, a surface of shape `shape`, is given to `ctx` through `port`.
    pub fn bind_surface(
        &mut self,
        ctx: u32,
        resource: u32,
        port: u32,
        shape: SurfaceShape,
    ) -> Result<(), ExtError> {
        if !self.contexts.contains_key(&ctx) {
            return Err(ExtError::InvalidContext);
        }
        if self.surfaces.contains_key(&(ctx, resource)) {
            return Err(ExtError::InvalidParameter);
        }
        self.cfg
            .ext
            .bind_surface(ctx, resource, port)
            .map_err(|e| {
                warn!("limina ext: ctx {ctx} surface {resource}: {e}");
                ExtError::Unspec
            })?;
        self.surfaces.insert((ctx, resource), shape);
        Ok(())
    }

    /// The shape `ctx` holds `resource` as, if it holds it.
    pub fn held_surface(&self, ctx: u32, resource: u32) -> Option<SurfaceShape> {
        self.surfaces.get(&(ctx, resource)).copied()
    }

    /// `ctx` let go of the resource; the others holding it keep it.
    pub fn detach_surface(&mut self, ctx: u32, resource: u32) {
        if self.surfaces.remove(&(ctx, resource)).is_some() {
            self.cfg.ext.release_surface(ctx, resource);
        }
    }

    /// The resource is gone; every context holding it gives it back.
    pub fn release_surface(&mut self, resource: u32) {
        let holders: Vec<u32> = self
            .surfaces
            .keys()
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
        self.contexts.get_mut(&ctx).unwrap().arenas.insert(
            req.arena_id,
            ContextArena {
                arena: a,
                window: None,
                import: Some(req.resource_id),
            },
        );
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
    /// Takes `size` bytes at `off` out of guest-physical space. False when the hypervisor did
    /// not, and the guest may still reach the pages: they must not be freed then.
    fn unmap(&self, off: u64, size: u64) -> bool {
        let (tx, rx) = unbounded();
        self.map_sender
            .send(WorkerMessage::GpuRemoveMapping(
                tx,
                self.cfg.arenas_guest_addr + off,
                size,
            ))
            .is_ok()
            && rx.recv().unwrap_or(false)
    }

    #[cfg(not(target_os = "macos"))]
    fn map(&self, _host_addr: u64, _off: u64, _size: u64) -> bool {
        false
    }

    #[cfg(not(target_os = "macos"))]
    fn unmap(&self, _off: u64, _size: u64) -> bool {
        true
    }

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
            self.contexts.get_mut(&ctx).unwrap().arenas.insert(
                req.arena_id,
                ContextArena {
                    arena: a,
                    window: None,
                    import: None,
                },
            );
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
        self.contexts.get_mut(&ctx).unwrap().arenas.insert(
            req.arena_id,
            ContextArena {
                arena: a,
                window: Some(off),
                import: None,
            },
        );
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
        let a = c.arenas.remove(&arena).ok_or(ExtError::InvalidParameter)?;
        if c.ring.is_some_and(|(r, _, _)| r == arena) {
            c.ring = None;
        }
        if let Some(off) = a.window {
            if !self.unmap(off, a.arena.size) {
                error!(
                    "limina ext: arena {ctx}/{arena} could not be unmapped; it is kept so the \
                     guest never reaches freed memory"
                );
                return Ok(());
            }
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
        if !c
            .arenas
            .get(&req.arena_id)
            .is_some_and(|a| a.window.is_some())
        {
            return Err(ExtError::InvalidParameter);
        }
        self.cfg
            .ext
            .attach_ring(ctx, req.arena_id, req.offset, req.data_size)
            .map_err(|e| {
                warn!("limina ext: ring {}/{}: {e}", ctx, req.arena_id);
                ExtError::InvalidParameter
            })?;
        self.contexts.get_mut(&ctx).unwrap().ring = Some((req.arena_id, req.offset, req.data_size));
        Ok(())
    }

    /// The extension as a snapshot carries it. Call with the vCPUs quiesced: each context is
    /// exported, which waits for its committed work, before the arenas' contents are read.
    /// Surfaces' rows are the caller's to fill in.
    pub fn snapshot(&self) -> ExtSnapshot {
        let mut s = ExtSnapshot::default();
        let mut ctxs: Vec<u32> = self.contexts.keys().copied().collect();
        ctxs.sort_unstable();
        for &ctx in &ctxs {
            let c = &self.contexts[&ctx];
            let ring = c.ring.map(|(arena, off, size)| {
                let export = self.cfg.ext.export_context(ctx).unwrap_or_else(|e| {
                    warn!("limina ext: export of context {ctx}: {e}; restored lost");
                    Vec::new()
                });
                (arena, off, size, export)
            });
            s.contexts.push(ContextSnapshot {
                ctx,
                doorbell: c.doorbell,
                ring,
            });
        }
        for &ctx in &ctxs {
            for (&arena, a) in &self.contexts[&ctx].arenas {
                if let Some(resource) = a.import {
                    s.imports.push((ctx, arena, resource));
                    continue;
                }
                s.arenas.push(ArenaSnapshot {
                    ctx,
                    arena,
                    size: a.arena.size,
                    gpu_base: a.arena.gpu_base,
                    window: a.window,
                    contents: read(&a.arena),
                });
            }
        }
        for (&resource, e) in &self.exports {
            let (ctx, arena) = e.origin;
            let alive = self
                .contexts
                .get(&ctx)
                .and_then(|c| c.arenas.get(&arena))
                .is_some_and(|a| a.import.is_none() && a.arena.gpu_base == e.arena.gpu_base);
            let (ctx, arena) = if alive {
                (ctx, arena)
            } else {
                s.arenas.push(ArenaSnapshot {
                    ctx: 0,
                    arena: resource,
                    size: e.arena.size,
                    gpu_base: e.arena.gpu_base,
                    window: None,
                    contents: read(&e.arena),
                });
                (0, resource)
            };
            s.exports.push((resource, ctx, arena, e.mapped));
        }
        s.arenas.sort_by_key(|a| a.gpu_base);
        s.exports.sort_unstable();
        s.imports.sort_unstable();
        s.held = self.surfaces.keys().copied().collect();
        s.held.sort_unstable();
        s
    }

    /// Puts back what [`Self::snapshot`] took, into a fresh device, up to the surfaces: contexts
    /// with their doorbells, arenas at their addresses with their contents (window arenas mapped
    /// where they were), exports and imports. Answers the exports, for the caller to map their
    /// blobs where [`ExtSnapshot::exports`] says. The caller then restores surfaces, replays the
    /// other renderers, and calls [`Self::restore_rings`].
    pub fn restore(&mut self, s: &ExtSnapshot) -> Result<Vec<(u32, ExtArena)>, String> {
        for c in &s.contexts {
            self.cfg.ext.create_context(c.ctx, c.doorbell)?;
            self.contexts.insert(
                c.ctx,
                Context {
                    doorbell: c.doorbell,
                    fences: Vec::new(),
                    arenas: HashMap::new(),
                    ring: c.ring.as_ref().map(|r| (r.0, r.1, r.2)),
                },
            );
        }
        let mut orphans = HashMap::new();
        for a in &s.arenas {
            let got = self
                .cfg
                .ext
                .restore_arena(a.ctx, a.arena, a.size, a.gpu_base)?;
            if got.gpu_base != a.gpu_base || got.size != a.size {
                return Err(format!(
                    "arena {}/{} came back at {:#x}+{:#x}, not {:#x}+{:#x}",
                    a.ctx, a.arena, got.gpu_base, got.size, a.gpu_base, a.size
                ));
            }
            if a.contents.len() as u64 != a.size {
                return Err(format!("arena {}/{}: contents short", a.ctx, a.arena));
            }
            // SAFETY: the embedder maps `size` bytes at `host_addr` for as long as the arena
            // lives, and nothing runs on it before its ring restarts.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    a.contents.as_ptr(),
                    got.host_addr as *mut u8,
                    a.contents.len(),
                )
            };
            if a.ctx == 0 {
                orphans.insert(a.arena, got);
                continue;
            }
            if let Some(off) = a.window {
                if got.host_addr % ARENA_ALIGN != 0 || !self.map(got.host_addr, off, got.size) {
                    return Err(format!("arena {}/{}: window mapping", a.ctx, a.arena));
                }
                self.reserve(off, got.size);
            }
            let c = self
                .contexts
                .get_mut(&a.ctx)
                .ok_or_else(|| format!("arena {}/{}: no such context", a.ctx, a.arena))?;
            c.arenas.insert(
                a.arena,
                ContextArena {
                    arena: got,
                    window: a.window,
                    import: None,
                },
            );
        }
        let mut exports = Vec::new();
        for &(resource, ctx, arena, mapped) in &s.exports {
            let got = self.cfg.ext.export_arena(ctx, arena, resource)?;
            let expected = if ctx == 0 {
                orphans.get(&arena).copied()
            } else {
                self.contexts
                    .get(&ctx)
                    .and_then(|c| c.arenas.get(&arena))
                    .map(|a| a.arena)
            };
            if expected.is_none_or(|e| e.gpu_base != got.gpu_base) {
                return Err(format!("export {resource}: not the arena it was"));
            }
            self.exports.insert(
                resource,
                Export {
                    arena: got,
                    mapped,
                    origin: (ctx, arena),
                },
            );
            exports.push((resource, got));
        }
        for &(ctx, arena, resource) in &s.imports {
            let got = self.cfg.ext.import_arena(ctx, arena, resource)?;
            let c = self
                .contexts
                .get_mut(&ctx)
                .ok_or_else(|| format!("import {resource}: no context {ctx}"))?;
            c.arenas.insert(
                arena,
                ContextArena {
                    arena: got,
                    window: None,
                    import: Some(resource),
                },
            );
        }
        Ok(exports)
    }

    /// Restarts every ring where the snapshot left it, once the surfaces the contexts use are
    /// bound again, and lets the embedder drop what held the arenas' places.
    pub fn restore_rings(&mut self, s: &ExtSnapshot) -> Result<(), String> {
        let r = s.contexts.iter().try_for_each(|c| match &c.ring {
            Some((arena, off, size, export)) => self
                .cfg
                .ext
                .restore_ring(c.ctx, *arena, *off, *size, export),
            None => Ok(()),
        });
        self.cfg.ext.restore_done();
        r
    }
}

const SNAPSHOT_MAGIC: u32 = 0x4c45_5831; // "LEX1"

struct Out(Vec<u8>);

impl Out {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn opt(&mut self, v: Option<u64>) {
        self.u64(v.unwrap_or(u64::MAX));
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
}

struct In<'a>(&'a [u8]);

impl In<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        (self.0.len() >= n).then(|| {
            let (a, b) = self.0.split_at(n);
            self.0 = b;
            a
        })
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn opt(&mut self) -> Option<Option<u64>> {
        let v = self.u64()?;
        Some((v != u64::MAX).then_some(v))
    }
    fn bytes(&mut self) -> Option<Vec<u8>> {
        let n = usize::try_from(self.u64()?).ok()?;
        Some(self.take(n)?.to_vec())
    }
    /// A count of items, refused if the rest cannot hold that many of at least `min` bytes.
    fn count(&mut self, min: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n.checked_mul(min)? <= self.0.len()).then_some(n)
    }
}

impl ExtSnapshot {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut o = Out(Vec::new());
        o.u32(SNAPSHOT_MAGIC);
        o.u32(self.contexts.len() as u32);
        for c in &self.contexts {
            o.u32(c.ctx);
            o.u32(c.doorbell);
            match &c.ring {
                Some((arena, off, size, export)) => {
                    o.u32(1);
                    o.u32(*arena);
                    o.u64(*off);
                    o.u32(*size);
                    o.bytes(export);
                }
                None => o.u32(0),
            }
        }
        o.u32(self.arenas.len() as u32);
        for a in &self.arenas {
            o.u32(a.ctx);
            o.u32(a.arena);
            o.u64(a.size);
            o.u64(a.gpu_base);
            o.opt(a.window);
            o.bytes(&a.contents);
        }
        o.u32(self.exports.len() as u32);
        for &(resource, ctx, arena, mapped) in &self.exports {
            o.u32(resource);
            o.u32(ctx);
            o.u32(arena);
            o.opt(mapped);
        }
        o.u32(self.imports.len() as u32);
        for &(ctx, arena, resource) in &self.imports {
            o.u32(ctx);
            o.u32(arena);
            o.u32(resource);
        }
        o.u32(self.made.len() as u32);
        for (resource, ctx, shape, rows) in &self.made {
            o.u32(*resource);
            o.u32(*ctx);
            o.u32(shape.width);
            o.u32(shape.height);
            o.u32(shape.drm_fourcc);
            o.u32(shape.stride);
            o.bytes(rows);
        }
        o.u32(self.held.len() as u32);
        for &(ctx, resource) in &self.held {
            o.u32(ctx);
            o.u32(resource);
        }
        o.0
    }

    /// `None` for anything malformed.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let mut i = In(data);
        if i.u32()? != SNAPSHOT_MAGIC {
            return None;
        }
        let mut s = Self::default();
        for _ in 0..i.count(12)? {
            let ctx = i.u32()?;
            let doorbell = i.u32()?;
            let ring = match i.u32()? {
                0 => None,
                1 => Some((i.u32()?, i.u64()?, i.u32()?, i.bytes()?)),
                _ => return None,
            };
            s.contexts.push(ContextSnapshot {
                ctx,
                doorbell,
                ring,
            });
        }
        for _ in 0..i.count(40)? {
            s.arenas.push(ArenaSnapshot {
                ctx: i.u32()?,
                arena: i.u32()?,
                size: i.u64()?,
                gpu_base: i.u64()?,
                window: i.opt()?,
                contents: i.bytes()?,
            });
        }
        for _ in 0..i.count(20)? {
            s.exports.push((i.u32()?, i.u32()?, i.u32()?, i.opt()?));
        }
        for _ in 0..i.count(12)? {
            s.imports.push((i.u32()?, i.u32()?, i.u32()?));
        }
        for _ in 0..i.count(8 + 16 + 8)? {
            let resource = i.u32()?;
            let ctx = i.u32()?;
            let shape = SurfaceShape {
                width: i.u32()?,
                height: i.u32()?,
                drm_fourcc: i.u32()?,
                stride: i.u32()?,
            };
            s.made.push((resource, ctx, shape, i.bytes()?));
        }
        for _ in 0..i.count(8)? {
            s.held.push((i.u32()?, i.u32()?));
        }
        i.0.is_empty().then_some(s)
    }
}

/// An arena's bytes, as its mapping in this process holds them.
fn read(a: &ExtArena) -> Vec<u8> {
    // SAFETY: the embedder keeps `size` bytes mapped at `host_addr` for as long as the arena
    // lives; the guest is quiesced and the arena's context drained.
    unsafe { std::slice::from_raw_parts(a.host_addr as *const u8, a.size as usize) }.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Fake {
        log: Mutex<Vec<String>>,
        /// Fences the "GPU" has not completed: (ctx, seqno, signal).
        fences: Mutex<Vec<(u32, u64, Signal)>>,
        /// Arenas by (ctx, arena), and exports by resource, in real memory.
        arenas: Mutex<HashMap<(u32, u32), ExtArena>>,
        exports: Mutex<HashMap<u32, ExtArena>>,
        next_base: Mutex<u64>,
    }

    /// `size` bytes of zeroed, page-aligned memory that lives as long as the test.
    fn pages(size: u64) -> u64 {
        let layout = std::alloc::Layout::from_size_align(size as usize, ARENA_ALIGN as usize);
        // SAFETY: a non-zero size and a power-of-two alignment.
        unsafe { std::alloc::alloc_zeroed(layout.unwrap()) as u64 }
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
        fn wire_version(&self) -> u32 {
            7
        }
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
        fn create_arena(&self, ctx: u32, arena: u32, size: u64) -> Result<ExtArena, String> {
            let size = size.next_multiple_of(ARENA_ALIGN);
            let mut next = self.next_base.lock().unwrap();
            let gpu_base = 0x15_0000_0000 + *next;
            *next += size + 0x8000;
            let a = ExtArena {
                host_addr: pages(size),
                size,
                gpu_base,
            };
            self.arenas.lock().unwrap().insert((ctx, arena), a);
            Ok(a)
        }
        fn destroy_arena(&self, ctx: u32, arena: u32) {
            self.arenas.lock().unwrap().remove(&(ctx, arena));
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
            let a = *self
                .arenas
                .lock()
                .unwrap()
                .get(&(ctx, arena))
                .ok_or("no such arena")?;
            self.exports.lock().unwrap().insert(export, a);
            Ok(a)
        }
        fn import_arena(&self, ctx: u32, arena: u32, export: u32) -> Result<ExtArena, String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("import {export} as {ctx}/{arena}"));
            let a = *self
                .exports
                .lock()
                .unwrap()
                .get(&export)
                .ok_or("no such export")?;
            self.arenas.lock().unwrap().insert((ctx, arena), a);
            Ok(a)
        }
        fn release_export(&self, export: u32) {
            self.exports.lock().unwrap().remove(&export);
            self.log.lock().unwrap().push(format!("-export {export}"));
        }
        fn export_context(&self, ctx: u32) -> Result<Vec<u8>, String> {
            // An odd length, as real exports have: what follows it in a snapshot is unaligned.
            Ok(format!("journal of {ctx}!").into_bytes())
        }
        fn restore_arena(
            &self,
            ctx: u32,
            arena: u32,
            size: u64,
            gpu_base: u64,
        ) -> Result<ExtArena, String> {
            self.log
                .lock()
                .unwrap()
                .push(format!("restore arena {ctx}/{arena} at {gpu_base:#x}"));
            let a = ExtArena {
                host_addr: pages(size),
                size,
                gpu_base,
            };
            self.arenas.lock().unwrap().insert((ctx, arena), a);
            Ok(a)
        }
        fn restore_ring(
            &self,
            ctx: u32,
            arena: u32,
            offset: u64,
            data_size: u32,
            export: &[u8],
        ) -> Result<(), String> {
            self.log.lock().unwrap().push(format!(
                "ring {ctx} at {arena}+{offset} ({data_size}): {}",
                String::from_utf8_lossy(export)
            ));
            Ok(())
        }
        fn restore_done(&self) {
            self.log.lock().unwrap().push("restore done".into());
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
        state_unmapping(true)
    }

    /// A state whose hypervisor answers every unmap with `unmaps`.
    fn state_unmapping(unmaps: bool) -> (Arc<Fake>, ExtState) {
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
                    WorkerMessage::GpuAddMapping(r, ..) => {
                        let _ = r.send(true);
                    }
                    WorkerMessage::GpuRemoveMapping(r, ..) => {
                        let _ = r.send(unmaps);
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

    fn fill(a: &ExtArena, byte: u8) {
        // SAFETY: the fake's arenas are real memory of `size` bytes.
        unsafe { std::ptr::write_bytes(a.host_addr as *mut u8, byte, a.size as usize) };
    }

    /// Everything the extension holds, carried into a fresh device: two contexts with doorbells,
    /// window arenas around a hole, a ring, an export imported elsewhere, an export whose arena
    /// only it still holds, and a surface made by one context and held by another.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_snapshot_restores_into_a_fresh_device_as_it_was() {
        let (f, mut s) = state();
        let window = |arena_id, size| virtio_gpu_limina_arena_create {
            arena_id,
            flags: 0,
            size,
        };
        let blob = |arena_id| virtio_gpu_limina_arena_create {
            arena_id,
            flags: ARENA_FLAG_BLOB,
            size: 1 << 16,
        };
        s.create_context(1).unwrap();
        s.create_context(2).unwrap();
        s.destroy_context(1).unwrap();
        s.create_context(3).unwrap();
        s.create_arena(3, window(1, 1 << 16)).unwrap();
        s.create_arena(3, window(2, 1 << 15)).unwrap();
        s.create_arena(3, window(3, 1 << 16)).unwrap();
        s.destroy_arena(3, 2).unwrap();
        s.attach_ring(
            3,
            virtio_gpu_limina_ring_attach {
                arena_id: 1,
                data_size: 1 << 14,
                offset: 0,
            },
        )
        .unwrap();
        s.create_arena(3, blob(4)).unwrap();
        s.export_arena(3, 4, 40, 1 << 16).unwrap();
        s.set_mapped(40, Some(0x20_0000));
        s.import_arena(
            2,
            virtio_gpu_limina_arena_import {
                arena_id: 9,
                resource_id: 40,
            },
        )
        .unwrap();
        s.create_arena(2, blob(5)).unwrap();
        s.export_arena(2, 5, 41, 1 << 16).unwrap();
        for (&(ctx, arena), a) in f.arenas.lock().unwrap().iter() {
            fill(a, (ctx * 16 + arena) as u8);
        }
        s.destroy_arena(2, 5).unwrap();
        let shape = SurfaceShape {
            width: 64,
            height: 32,
            drm_fourcc: 0x34325258,
            stride: 256,
        };
        s.bind_surface(3, 50, 7, shape).unwrap();
        s.bind_surface(2, 50, 8, SurfaceShape::default()).unwrap();

        let mut snap = s.snapshot();
        assert!(snap.made.is_empty(), "the device fills in the surfaces");
        snap.made = vec![(50, 3, shape, vec![1, 2, 3])];
        assert!(
            snap.arenas
                .windows(2)
                .all(|w| w[0].gpu_base < w[1].gpu_base)
        );
        assert!(
            snap.arenas
                .iter()
                .any(|a| (a.ctx, a.arena) == (0, 41) && a.contents.iter().all(|&b| b == 2 * 16 + 5)),
            "the orphaned export is carried with its bytes"
        );
        assert_eq!(snap.held, vec![(2, 50), (3, 50)]);

        let (g, mut r) = state();
        let exports = r.restore(&snap).unwrap();
        assert_eq!(
            exports.iter().map(|e| e.0).collect::<Vec<_>>(),
            vec![40, 41]
        );
        r.bind_surface(3, 50, 17, shape).unwrap();
        r.bind_surface(2, 50, 18, SurfaceShape::default()).unwrap();
        r.restore_rings(&snap).unwrap();
        let log = g.log.lock().unwrap().clone();
        let restored: Vec<&String> = log
            .iter()
            .filter(|l| l.starts_with("restore arena"))
            .collect();
        assert_eq!(restored.len(), snap.arenas.len(), "{log:?}");
        assert!(
            log.contains(&"ring 3 at 1+0 (16384): journal of 3!".to_string()),
            "{log:?}"
        );
        assert_eq!(log.last().map(String::as_str), Some("restore done"));
        let bytes = snap.to_bytes();
        assert_eq!(ExtSnapshot::from_bytes(&bytes).as_ref(), Some(&snap));
        for n in [0, 3, bytes.len() / 2, bytes.len() - 1] {
            assert_eq!(ExtSnapshot::from_bytes(&bytes[..n]), None, "{n}");
        }

        let again = r.snapshot();
        assert_eq!(again.contexts, snap.contexts, "doorbells and rings");
        assert_eq!(again.arenas, snap.arenas, "addresses, windows and contents");
        assert_eq!(again.exports, snap.exports);
        assert_eq!(again.imports, snap.imports);
        assert_eq!(again.held, snap.held);
    }

    /// A surface outlives the context that made it while another holds it, and a restore binds
    /// every context a snapshot names, so a snapshot must name only contexts it carries.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_surface_whose_maker_is_gone_is_snapshot_without_it() {
        let (_f, mut s) = state();
        s.create_context(3).unwrap();
        s.create_context(5).unwrap();
        s.bind_surface(3, 50, 7, SurfaceShape::default()).unwrap();
        s.bind_surface(5, 50, 8, SurfaceShape::default()).unwrap();
        s.destroy_context(3).unwrap();
        let snap = s.snapshot();
        let live: Vec<u32> = snap.contexts.iter().map(|c| c.ctx).collect();
        assert_eq!(snap.held, vec![(5, 50)]);
        for (resource, ctx, ..) in &snap.made {
            assert!(
                live.contains(ctx),
                "surface {resource} is restored for context {ctx}, which is gone"
            );
        }
    }

    #[test]
    fn info_carries_the_embedders_wire_version() {
        let (_, state) = state();
        assert_eq!(
            state.info(),
            virtio_gpu_limina_resp_info {
                wire_version: 7,
                flags: 0
            }
        );
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

    /// An arena the guest may still reach is never freed: when the hypervisor fails to take it out
    /// of guest-physical space, the arena, or the whole context, is kept, and so is its place.
    #[cfg(target_os = "macos")]
    #[test]
    fn an_arena_the_hypervisor_would_not_unmap_is_kept() {
        let (f, mut s) = state_unmapping(false);
        let req = |arena_id| virtio_gpu_limina_arena_create {
            arena_id,
            flags: 0,
            size: 1 << 19,
        };
        s.create_context(1).unwrap();
        let a = s.create_arena(1, req(1)).unwrap();
        s.destroy_arena(1, 1).unwrap();
        s.create_context(2).unwrap();
        s.create_arena(2, req(1)).unwrap();
        s.destroy_context(2).unwrap();
        let log = f.log.lock().unwrap().clone();
        assert!(!log.contains(&"-arena 1/1".to_string()), "{log:?}");
        assert!(!log.contains(&"-arena 2/1".to_string()), "{log:?}");
        assert!(!log.contains(&"-ctx 2".to_string()), "{log:?}");
        assert!(f.arenas.lock().unwrap().contains_key(&(1, 1)));
        assert_eq!(
            s.create_arena(1, req(3)).err(),
            Some(ExtError::OutOfMemory),
            "the kept arenas still hold their places (at {:#x})",
            a.shm_offset
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
            s.bind_surface(1, 50, 7, SurfaceShape::default()).err(),
            Some(ExtError::InvalidContext),
            "no context to give it to"
        );
        s.create_context(1).unwrap();
        s.bind_surface(1, 50, 7, SurfaceShape::default()).unwrap();
        assert!(log(&f, "surface 50 port 7 to 1"));
        assert_eq!(
            s.bind_surface(1, 50, 8, SurfaceShape::default()).err(),
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

        s.bind_surface(1, 51, 9, SurfaceShape::default()).unwrap();
        s.destroy_context(1).unwrap();
        s.release_surface(51);
        assert!(
            !log(&f, "-surface 51 of 1"),
            "the context took its surfaces with it"
        );
    }

    #[test]
    fn a_held_surface_answers_its_shape_until_it_goes() {
        let (_f, mut s) = state();
        let shape = SurfaceShape {
            width: 718,
            height: 1166,
            drm_fourcc: 0x3432_5241,
            stride: 2880,
        };
        s.create_context(1).unwrap();
        assert_eq!(s.held_surface(1, 70), None);
        s.bind_surface(1, 70, 7, shape).unwrap();
        assert_eq!(s.held_surface(1, 70), Some(shape), "as it was given");
        assert_eq!(s.held_surface(2, 70), None, "to that context only");
        s.release_surface(70);
        assert_eq!(s.held_surface(1, 70), None, "until the resource goes");
    }

    #[test]
    fn an_imported_surface_has_a_holder_per_context() {
        let (f, mut s) = state();
        let gone =
            |f: &Arc<Fake>, line: &str| f.log.lock().unwrap().iter().filter(|l| *l == line).count();
        s.create_context(1).unwrap();
        s.create_context(2).unwrap();
        s.create_context(3).unwrap();
        s.bind_surface(1, 60, 7, SurfaceShape::default()).unwrap();
        s.bind_surface(2, 60, 8, SurfaceShape::default()).unwrap();
        s.bind_surface(3, 60, 9, SurfaceShape::default()).unwrap();

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
