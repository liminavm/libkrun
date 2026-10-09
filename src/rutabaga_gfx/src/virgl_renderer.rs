// Copyright 2020 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! The virtio-gpu 3D backend, on virglrs's Rust API.
//!
//! This used to reach virglrenderer through its C ABI, and the ABI is where every awkwardness in
//! this file came from: an implicit process-global renderer with no handle, `int` returns that
//! flattened a dozen distinct refusals into `-EINVAL`, out-params that had to be checked before
//! they could be believed, and `malloc`ed blobs to copy and free. None of that was virglrenderer's
//! design; it was C's. The renderer's own API has an owned root, typed ids and errors that say
//! what happened, so this file holds one `Renderer` and calls it.
//!
//! The component is `Send + Sync` because `Renderer` is `Send` and the `Mutex` supplies the rest.
//! It remains thread-affine in the way the C was -- vrend's GL context belongs to whichever thread
//! called `init` -- and rutabaga's contract of keeping every call on the renderer thread is what
//! upholds that, exactly as it did before.

#![cfg(feature = "virgl_renderer")]

use std::io::Error as SysError;
use std::io::IoSliceMut;
use std::mem::ManuallyDrop;
use std::mem::size_of;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use log::error;
use log::warn;

use virglrenderer::abi::{GuestIov, VmmPtr};
use virglrenderer::config::{CapsetId, Config, PipelineCacheKey};
use virglrenderer::fence::FenceSink;
use virglrenderer::ids::{
    BlobId, ClientFenceId, ContextId, FenceId, ResourceHandle, RingId, RingIdx,
};
use virglrenderer::renderer::{BlobDesc, BlobSource, Caching, Error as RendererError, Renderer};
use virglrenderer::venus::context::{Submitted, Wait};
use virglrenderer::venus::driver::Answered;
use virglrenderer::vrend::pipe::TextureTarget;
use virglrenderer::vrend::proto::{Box3, Format};
use virglrenderer::vrend::resource::{Args as ClassicArgs, Bind, ResourceFlags};
use virglrenderer::vrend::transfer;

use crate::rutabaga_core::RutabagaComponent;
use crate::rutabaga_core::RutabagaContext;
use crate::rutabaga_core::RutabagaResource;
use crate::rutabaga_os::SafeDescriptor;
use crate::rutabaga_utils::*;

/// One renderer, shared by the component and every context it hands out.
///
/// `Arc` because a `RutabagaContext` outlives the call that created it and must still reach the
/// renderer; the C had the same sharing and spelled it as a global.
type Shared = std::sync::Arc<Guarded<Renderer>>;

/// The renderer behind its lock, and the one way in: [`Guarded::with`].
///
/// A panic inside virglrs ends the renderer, not the VM. The worker is built to unwind, so the
/// panic is caught at this boundary, the renderer is marked dead, and every later call into it
/// is refused with `EIO` without touching it. What a panic leaves behind is renderer state
/// half-updated under the lock -- tables, the GL context, a venus decoder mid-command -- and
/// nothing here can say which context that state belongs to: the renderer's tables are shared by
/// every context, and vrend's GL context is one per process. So the containment is the whole
/// renderer, never one context. The guest loses 3D (its fences behind the dead renderer never
/// retire), and the VM process, its disks and its network stay up.
///
/// A dead renderer is also never dropped. Its teardown would run over the same half-updated
/// state, and the host memory behind its mappable blobs is mapped into the guest, which still
/// runs and may still touch it.
struct Guarded<T> {
    lock: ManuallyDrop<Mutex<T>>,
    dead: AtomicBool,
}

/// What every call into a dead renderer returns.
fn dead() -> RutabagaError {
    RutabagaError::ComponentError(-libc::EIO)
}

impl<T> Guarded<T> {
    fn new(inner: T) -> Self {
        Guarded {
            lock: ManuallyDrop::new(Mutex::new(inner)),
            dead: AtomicBool::new(false),
        }
    }

    /// Run `f` on the renderer under its lock; `what` names the call for the log.
    ///
    /// A panic in `f` unwinds through the guard, which poisons the lock; a poisoned lock is read
    /// as dead too, though nothing but a caught panic can poison it.
    fn with<R>(&self, what: &str, f: impl FnOnce(&mut T) -> R) -> RutabagaResult<R> {
        self.contain(what, || match self.lock.lock() {
            Ok(mut renderer) => Ok(f(&mut renderer)),
            Err(_) => Err(dead()),
        })?
    }

    /// Run `f`, which reaches into virglrs without the lock -- a wait the renderer handed back --
    /// so that a panic there ends the renderer as one under the lock would.
    fn contain<R>(&self, what: &str, f: impl FnOnce() -> R) -> RutabagaResult<R> {
        if self.dead.load(Ordering::Acquire) {
            return Err(dead());
        }
        // Unwind safety: whatever `f` leaves half-updated is reached only through `self`, and a
        // caught panic marks `self` dead before anything can reach it again.
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(v) => Ok(v),
            Err(_) => {
                if !self.dead.swap(true, Ordering::AcqRel) {
                    error!(
                        "virglrs panicked in {what}; the 3D renderer is dead for the rest of \
                         this VM's life and every call into it is refused (the panic message \
                         is above, and in the panic log)"
                    );
                }
                Err(dead())
            }
        }
    }
}

impl<T> Drop for Guarded<T> {
    fn drop(&mut self) {
        if !*self.dead.get_mut() {
            // SAFETY: `lock` is dropped here once, and `self` is never used again.
            unsafe { ManuallyDrop::drop(&mut self.lock) };
        }
    }
}

/// The virtio-gpu backend state tracker which supports accelerated rendering.
pub struct VirglRenderer {
    r: Shared,
}

struct VirglRendererContext {
    ctx_id: ContextId,
    r: Shared,
}

/// Where retired fences go: rutabaga's handler, behind the renderer's own sink trait.
///
/// Two entry points and no `Option` on either. The C table let a VMM supply neither and have that
/// kind of fence silently dropped; rutabaga always supplies one, so the optionality was never
/// anything but a way to lose a fence.
struct Fences(RutabagaFenceHandler);

impl FenceSink for Fences {
    fn context_fence(&mut self, ctx: ContextId, ring: RingIdx, fence: FenceId) {
        self.0.call(RutabagaFence {
            flags: RUTABAGA_FLAG_FENCE | RUTABAGA_FLAG_INFO_RING_IDX,
            fence_id: fence.0,
            ctx_id: ctx.get(),
            ring_idx: ring.0 as u8,
        });
    }

    fn global_fence(&mut self, fence: ClientFenceId) {
        self.0.call(RutabagaFence {
            flags: RUTABAGA_FLAG_FENCE,
            fence_id: fence.0 as u64,
            ctx_id: 0,
            ring_idx: 0,
        });
    }

    /// A present fence: the cookie a parked frame is waiting on, and no context or ring, because
    /// this fence was asked for about a resource rather than created on a guest stream.
    fn present_fence(&mut self, fence: FenceId) {
        self.0.call(RutabagaFence {
            flags: RUTABAGA_FLAG_FENCE | RUTABAGA_FLAG_PRESENT,
            fence_id: fence.0,
            ctx_id: 0,
            ring_idx: 0,
        });
    }
}

/// A renderer refusal as rutabaga's error, with the reason kept where a reader can see it.
///
/// `ComponentError` carries an `i32` and nothing else, so the reason has to be logged rather than
/// returned -- the same loss the C ABI's errno inflicted, one layer further out. Widening
/// rutabaga's error to carry the renderer's is the next thing this migration is for; until then
/// the number stays `-EINVAL` so a log a human has read before still reads the same.
fn refused(what: &str, e: RendererError) -> RutabagaError {
    error!("virglrs: {what}: {e}");
    RutabagaError::ComponentError(-libc::EINVAL)
}

fn ctx(ctx_id: u32) -> RutabagaResult<ContextId> {
    ContextId::new(ctx_id).ok_or(RutabagaError::InvalidContextId)
}

fn res(resource_id: u32) -> RutabagaResult<ResourceHandle> {
    ResourceHandle::new(resource_id).ok_or(RutabagaError::InvalidResourceId)
}

/// The guest pages behind a resource, as the renderer describes them.
///
/// A copy of the descriptions, never of the pages: `GuestIov` is a base and a length, and the
/// renderer holds the pair rather than a pointer and a count that could come apart.
fn iovs(v: &[RutabagaIovec]) -> Vec<GuestIov> {
    v.iter()
        .map(|e| GuestIov {
            base: VmmPtr(e.base),
            len: e.len,
        })
        .collect()
}

fn info_of(t: &Transfer3D, synchronized: bool) -> transfer::Info {
    transfer::Info {
        level: t.level,
        stride: t.stride,
        layer_stride: t.layer_stride,
        offset: t.offset,
        region: Box3 {
            x: t.x as i32,
            y: t.y as i32,
            z: t.z as i32,
            width: t.w as i32,
            height: t.h as i32,
            depth: t.d as i32,
        },
        synchronized,
    }
}

impl RutabagaContext for VirglRendererContext {
    fn submit_cmd(&mut self, commands: &mut [u8], fence_ids: &[u64]) -> RutabagaResult<()> {
        if !fence_ids.is_empty() {
            return Err(RutabagaError::Unsupported);
        }
        if !commands.len().is_multiple_of(size_of::<u32>()) {
            return Err(RutabagaError::InvalidCommandSize(commands.len()));
        }
        submit_all(&self.r, self.ctx_id, commands)
    }

    fn attach(&mut self, resource: &mut RutabagaResource) {
        // Nothing to import: virglrs has no dmabuf path, and every backing it can adopt already
        // reached it through create/attach_backing.
        resource.component_mask |= 1 << (RutabagaComponentType::VirglRenderer as u8);
        if let Ok(handle) = res(resource.resource_id) {
            let id = self.ctx_id;
            let _ = self
                .r
                .with("ctx_attach_resource", |r| r.ctx_attach_resource(id, handle));
        }
    }

    fn detach(&mut self, resource: &RutabagaResource) {
        if let Ok(handle) = res(resource.resource_id) {
            let id = self.ctx_id;
            let _ = self
                .r
                .with("ctx_detach_resource", |r| r.ctx_detach_resource(id, handle));
        }
    }

    fn component_type(&self) -> RutabagaComponentType {
        RutabagaComponentType::VirglRenderer
    }

    fn context_create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        let id = ctx(fence.ctx_id)?;
        self.r
            .with("context_create_fence", |r| {
                r.context_create_fence(id, RingIdx(fence.ring_idx as u32), FenceId(fence.fence_id))
            })?
            .map_err(|e| refused("context_create_fence", e))
    }
}

impl Drop for VirglRendererContext {
    /// Cannot panic: a panic in the destroy is caught like any other, and a dead renderer has no
    /// context left to destroy.
    fn drop(&mut self) {
        let id = self.ctx_id;
        let _ = self.r.with("context_destroy", |r| r.context_destroy(id));
    }
}

/// Run a whole submission, sleeping outside the renderer lock for any wait it contains.
///
/// The loop is the point, and it is the one thing this file cannot simplify away. The lock is held
/// for the renderer's own call, so a `vkWaitRingSeqnoMESA` that blocked inside it would hold it
/// against the ring thread that has to advance the very head being waited for, and a
/// `vkWaitForFences` the driver cannot answer at once would hold it against every ring of the
/// context. The renderer hands the wait back instead: it says how much of the buffer ran and what
/// to wait for, this drops the guard, waits, and comes back with the remainder -- through
/// `submit_cmd` after a ring wait, through `resume_cmd` with the driver's answer after a driver
/// wait, because the remainder begins with the wait command and a `submit_cmd` would ask the
/// driver again.
fn submit_all(r: &Shared, id: ContextId, buf: &[u8]) -> RutabagaResult<()> {
    let mut at = 0usize;
    // What the driver wait the remainder begins with answered, on the pass after one ran.
    let mut answer: Option<Answered> = None;
    loop {
        let out = match answer.take() {
            None => r.with("submit_cmd", |r| r.submit_cmd(id, &buf[at..]))?,
            Some(a) => r.with("resume_cmd", |r| r.resume_cmd(id, &buf[at..], a))?,
        };
        let waiter = match out {
            Err(e) => return Err(refused("submit_cmd", e)),
            Ok(Submitted::Done) => return Ok(()),
            Ok(Submitted::Poisoned) => return Err(refused("submit_cmd", RendererError::Poisoned)),
            Ok(Submitted::Waiting {
                consumed,
                on: Wait::Ring { ring, seqno },
            }) => {
                at += consumed;
                // Fetched under the lock and waited on after it: the waiter holds only `Arc`s to
                // things the ring thread also holds, so from here the renderer is not involved.
                match r.with("ring_waiter", |r| r.ring_waiter(id, ring, seqno))? {
                    Ok(w) => w,
                    Err(e) => return Err(refused("ring_waiter", e)),
                }
            }
            // A blocking driver call, run here with the renderer lock dropped: the guest's own
            // thread is what is spent on it, and nothing else in the process waits behind it.
            Ok(Submitted::Waiting {
                consumed,
                on: Wait::Driver(wait),
            }) => {
                at += consumed;
                answer = Some(r.contain("a driver wait", || {
                    wait.run(|| true)
                        .expect("nothing stops a virtqueue-side wait")
                })?);
                continue;
            }
            // A virtqueue wait is legal only on a ring's own stream, and this is the context's.
            // Its handler refuses that origin, so the stream poisons rather than arriving here.
            Ok(Submitted::Waiting {
                on: Wait::Virtqueue(_),
                ..
            }) => {
                unreachable!(
                    "a context stream's vkWaitVirtqueueSeqnoMESA is refused by its handler"
                )
            }
        };
        if !r.contain("a ring wait", || waiter.wait())? {
            return Err(refused("ring wait", RendererError::Poisoned));
        }
    }
}

impl VirglRenderer {
    pub fn init(
        virglrenderer_flags: VirglRendererFlags,
        fence_handler: RutabagaFenceHandler,
        _render_server_fd: Option<SafeDescriptor>,
        pipeline_cache_key: Option<PipelineCacheKey>,
    ) -> RutabagaResult<Box<dyn RutabagaComponent>> {
        if cfg!(debug_assertions) {
            let ret = unsafe { libc::dup2(libc::STDOUT_FILENO, libc::STDERR_FILENO) };
            if ret == -1 {
                warn!(
                    "unable to dup2 stdout to stderr: {}",
                    SysError::last_os_error()
                );
            }
        }

        // One renderer per process. Not because of a global any more -- the root below is owned --
        // but because there is one GL context, one Vulkan loader and one VideoToolbox registration
        // behind it, and a second instance has never been tried.
        static INIT_ONCE: AtomicBool = AtomicBool::new(false);
        if INIT_ONCE
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err(RutabagaError::AlreadyInUse);
        }

        let flags: i32 = virglrenderer_flags.into();
        let config = Config {
            venus: flags & virglrenderer::abi::VENUS != 0,
            vrend: flags & virglrenderer::abi::NO_VIRGL == 0,
            guest_vram: flags & virglrenderer::abi::USE_GUEST_VRAM != 0,
            video: flags & virglrenderer::abi::USE_VIDEO != 0,
            pipeline_cache_key,
            // The renderer's own defaults for everything the flags do not carry.
            ..Config::default()
        };

        // No context factory: limina has no GL of its own to lend, so vrend opens the renderer's
        // own surfaceless display. The factory arm is for an embedder that draws the scanout
        // itself and needs the texture names to be names in its own share group.
        match Renderer::new(Box::new(Fences(fence_handler)), config, None) {
            Ok(r) => Ok(Box::new(VirglRenderer {
                r: std::sync::Arc::new(Guarded::new(r)),
            })),
            Err(e) => {
                error!("virglrs: init: {e}");
                INIT_ONCE.store(false, Ordering::Release);
                Err(RutabagaError::ComponentError(-libc::EINVAL))
            }
        }
    }

    /// What the guest is told about a mappable resource, and where its pages are.
    ///
    /// One lookup answering both, because the renderer answers both from one thing: a resource
    /// that has no host mapping has neither a pointer nor a caching mode, and the C's two
    /// entry points could disagree about that only by asking twice.
    fn mapping(&self, resource_id: u32) -> Option<(u64, u32)> {
        let handle = ResourceHandle::new(resource_id)?;
        let m = self
            .r
            .with("resource_host_mapping", |r| r.resource_host_mapping(handle))
            .ok()?
            .ok()?;
        let info = match m.caching {
            Caching::Cached => RUTABAGA_MAP_CACHE_CACHED,
            Caching::WriteCombining => RUTABAGA_MAP_CACHE_WC,
        };
        Some((m.addr as u64, info | RUTABAGA_MAP_ACCESS_RW))
    }

    /// The export metadata rutabaga asks for. virglrs exports no descriptors, so the answer is
    /// the same for every resource that exists -- which is why it is built here rather than asked
    /// for: `virgl_renderer_execute`'s export query was an ABI ceremony around these constants.
    fn query(&self, resource_id: u32) -> Option<Resource3DInfo> {
        let handle = ResourceHandle::new(resource_id)?;
        self.r
            .with("with_resource", |r| r.with_resource(handle, |_| ()))
            .ok()??;
        Some(Resource3DInfo {
            width: 0,
            height: 0,
            drm_fourcc: 0,
            strides: [0; 4],
            offsets: [0; 4],
            modifier: virglrenderer::abi::DRM_FORMAT_MOD_INVALID,
        })
    }
}

impl RutabagaComponent for VirglRenderer {
    #[cfg(target_os = "macos")]
    fn iosurface_id(&self, resource_id: u32) -> RutabagaResult<u32> {
        let handle = res(resource_id)?;
        self.r
            .with("resource_surface_id", |r| r.resource_surface_id(handle))?
            .map(|s| s.0)
            .ok_or(RutabagaError::Unsupported)
    }

    /// The version and size the guest sizes its capset buffer from.
    ///
    /// Asked of the renderer, not of `Capset::layout`. A layout exists for every capset this
    /// build can name, served or not, and this answer has to agree with `get_capset` below --
    /// which hands back an empty vec for a capset no renderer is behind. Advertising VIRGL2 at
    /// 1408 bytes in a venus-only configuration and then filling none of them would give the
    /// guest whatever was in that buffer as a capset. Two values that must agree are one value:
    /// both come from the blob `get_capset` returns.
    ///
    /// Diverging from virglrs's own C shim here is deliberate, not an oversight. That shim
    /// answers from `Capset::layout` because QEMU calls `virgl_renderer_get_cap_set` before
    /// `virgl_renderer_init`, when there is no renderer to ask -- and it zeroes the buffer it
    /// will not fill, which is what makes answering from a layout safe there. rutabaga
    /// constructs the component before anything asks, so there is always a renderer, and there
    /// is no fill on this path to do the zeroing. Should a before-init caller ever appear here,
    /// the fix is `layout` AND the zeroing together, never `layout` alone.
    fn get_capset_info(&self, capset_id: u32) -> (u32, u32) {
        let set = CapsetId::from_raw((capset_id & 0xff) as u8);
        self.r
            .with("capset", |r| {
                r.capset(set)
                    .map(|c| (c.version(), c.as_bytes().len() as u32))
            })
            .ok()
            .flatten()
            .unwrap_or((0, 0))
    }

    fn get_capset(&self, capset_id: u32, _version: u32) -> Vec<u8> {
        let set = CapsetId::from_raw((capset_id & 0xff) as u8);
        self.r
            .with("capset", |r| {
                r.capset(set).map(|caps| caps.as_bytes().to_vec())
            })
            .ok()
            .flatten()
            .unwrap_or_default()
    }

    /// The C ABI's implicit current context. There is no such thing here, which is the whole
    /// reason the rewrite exists.
    fn force_ctx_0(&self) {}

    fn limina_settle_video(&self) {
        let _ = self.r.with("settle_video", |r| r.settle_video());
    }

    fn limina_dump_state(&self) {
        if self.r.with("dump_state", dump_state).is_err() {
            error!("virglrs: the renderer is dead; nothing to dump");
        }
    }

    /// `None` is a context with nothing retained, which is deliberately not an empty blob: a VMM
    /// that stored zero bytes and restored them later would have rebuilt nothing and been told it
    /// succeeded. Which renderer keeps the journal is the renderer's question, not this one's.
    fn limina_journal_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        self.r
            .with("journal_export", |r| r.journal_export(id))
            .ok()
            .flatten()
    }

    fn limina_journal_seq(&self, ctx_id: u32) -> u64 {
        ContextId::new(ctx_id)
            .and_then(|id| {
                self.r
                    .with("venus_journal_seq", |r| r.venus_journal_seq(id))
                    .ok()
                    .flatten()
            })
            .unwrap_or(0)
    }

    /// Nothing is pinned. The venus journal keys an entry by a generation rather than holding the
    /// object it names, so there is no pin for the VMM to release.
    fn limina_journal_unpin(&self, _ctx_id: u32, _key: u64) {}

    fn limina_replay_begin(&self, ctx_id: u32) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        matches!(
            self.r.with("replay_begin", |r| r.replay_begin(id)),
            Ok(Ok(_))
        )
    }

    fn limina_journal_restore(&self, ctx_id: u32, blob: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        // A refused journal is reported by the renderer, which names the half that refused it.
        matches!(
            self.r
                .with("journal_restore", |r| r.journal_restore(id, blob)),
            Ok(Ok(_))
        )
    }

    fn limina_journal_replay_upto(&self, ctx_id: u32, upto: u64) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        match self.r.with("replay_upto", |r| r.replay_upto(id, upto)) {
            Ok(Ok(())) => true,
            Err(_) => false,
            Ok(Err(why)) => {
                error!("virglrs: ctx {ctx_id}: replay to {upto} failed: {why}");
                false
            }
        }
    }

    fn limina_replay_submit(&self, ctx_id: u32, cmd: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        matches!(
            self.r
                .with("venus_replay_cmd", |r| r.venus_replay_cmd(id, cmd)),
            Ok(Ok(_))
        )
    }

    fn limina_replay_ring_cmd(&self, ctx_id: u32, ring_id: u64, cmd: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        // Ring 0 is the context's own stream, which `limina_replay_submit` feeds; a journal entry
        // never names it here, and one that does is refused rather than replayed on the wrong
        // decoder.
        let Some(ring) = RingId::new(ring_id) else {
            return false;
        };
        matches!(
            self.r.with("venus_replay_ring_cmd", |r| r
                .venus_replay_ring_cmd(id, ring, cmd)),
            Ok(Ok(_))
        )
    }

    fn limina_replay_end(&self, ctx_id: u32) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        matches!(self.r.with("replay_end", |r| r.replay_end(id)), Ok(Ok(_)))
    }

    fn limina_sync_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        self.r
            .with("venus_sync_export", |r| r.venus_sync_export(id))
            .ok()?
            .ok()
    }

    fn limina_sync_restore(&self, ctx_id: u32, data: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        matches!(
            self.r
                .with("venus_sync_restore", |r| r.venus_sync_restore(id, data)),
            Ok(Ok(_))
        )
    }

    fn limina_memory_census(&self, ctx_id: u32) -> Option<Vec<(u64, u64)>> {
        let id = ContextId::new(ctx_id)?;
        let census = self
            .r
            .with("venus_memory_census", |r| r.venus_memory_census(id))
            .ok()?
            .ok()?;
        Some(census.into_iter().map(|a| (a.id.0, a.size)).collect())
    }

    fn limina_memory_read(&self, ctx_id: u32, mem_id: u64, buf: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mem = virglrenderer::venus::cs::ObjectId(mem_id);
        matches!(
            self.r
                .with("venus_memory_read", |r| r.venus_memory_read(id, mem, buf)),
            Ok(Ok(_))
        )
    }

    fn limina_memory_write(&self, ctx_id: u32, mem_id: u64, buf: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mem = virglrenderer::venus::cs::ObjectId(mem_id);
        matches!(
            self.r
                .with("venus_memory_write", |r| r.venus_memory_write(id, mem, buf)),
            Ok(Ok(_))
        )
    }

    fn limina_classic_content_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        self.r
            .with("vrend_content_export", |r| r.vrend_content_export(id))
            .ok()
            .flatten()
            .map(|(bytes, _)| bytes)
    }

    fn limina_classic_content_restore(&self, ctx_id: u32, data: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        matches!(
            self.r.with("vrend_content_restore", |r| r
                .vrend_content_restore(id, data)),
            Ok(Ok(_))
        )
    }

    fn create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        // ctx_id names the context whose work this fence is for. The renderer takes its GL sync on
        // that context, so dropping it here would leave it syncing on whichever context happened
        // to be current -- right by luck, and wrong whenever the last thing the worker did was a
        // ctx0 operation. Zero is the global ring naming no context, which the renderer answers by
        // finishing everything instead.
        self.r.with("create_fence", |r| {
            r.create_fence(
                ClientFenceId(fence.fence_id as u32),
                ContextId::new(fence.ctx_id),
            )
        })
    }

    /// Fences retire on the renderer's own thread and are delivered through [`Fences`], except a
    /// context fence behind a GL query that was not ready when the guest asked for it. The guest
    /// reads that result by re-reading the query buffer until the host marks it done, and never
    /// asks again, so the renderer holds the fence until this call, on the renderer thread, has
    /// written the answer. Draining the descriptor is the renderer's job, and a spurious call is
    /// cheap.
    fn event_poll(&self) {
        let _ = self.r.with("poll", |r| r.poll());
    }

    /// Asking for the descriptor is the promise to pump [`Self::event_poll`] whenever it is
    /// readable. Without one, a fence behind a parked query finishes its work inline when it is
    /// taken: still correct, but a GPU wait on the submitting thread.
    fn poll_descriptor(&self) -> Option<SafeDescriptor> {
        match self
            .r
            .with("poll_descriptor", |r| r.poll_descriptor())
            .ok()?
        {
            Ok(fd) => fd.map(|fd| SafeDescriptor::from(std::fs::File::from(fd))),
            Err(e) => {
                error!("virglrs: no poll descriptor, so GL queries finish at fence time: {e}");
                None
            }
        }
    }

    fn create_3d(&self, resource_id: u32, c: ResourceCreate3D) -> RutabagaResult<RutabagaResource> {
        let handle = res(resource_id)?;
        // A target or a format the wire has no name for is the guest's parse failing, and is
        // refused here rather than inside: these are the only two of the eleven fields that are
        // not simply numbers.
        let args = ClassicArgs {
            target: TextureTarget::from_wire(c.target).ok_or(RutabagaError::Unsupported)?,
            format: Format::from_wire(c.format).ok_or(RutabagaError::Unsupported)?,
            bind: Bind(c.bind),
            width: c.width,
            height: c.height,
            depth: c.depth,
            array_size: c.array_size,
            last_level: c.last_level,
            nr_samples: c.nr_samples,
            flags: ResourceFlags(c.flags),
        };
        self.r
            .with("resource_create", |r| {
                r.resource_create(handle, args, Vec::new())
            })?
            .map_err(|e| refused("create_3d", e))?;

        Ok(RutabagaResource {
            resource_id,
            handle: None,
            blob: false,
            blob_mem: 0,
            blob_flags: 0,
            map_info: None,
            #[cfg(target_os = "macos")]
            map_ptr: None,
            info_2d: None,
            info_3d: self.query(resource_id),
            vulkan_info: None,
            backing_iovecs: None,
            component_mask: 1 << (RutabagaComponentType::VirglRenderer as u8),
            size: 0,
            mapping: None,
        })
    }

    fn attach_backing(
        &self,
        resource_id: u32,
        vecs: &mut Vec<RutabagaIovec>,
    ) -> RutabagaResult<()> {
        let handle = res(resource_id)?;
        let iov = iovs(vecs);
        self.r
            .with("resource_attach_iov", |r| {
                r.resource_attach_iov(handle, iov)
            })?
            .map_err(|e| refused("attach_backing", e))
    }

    fn detach_backing(&self, resource_id: u32) {
        if let Ok(handle) = res(resource_id) {
            let _ = self
                .r
                .with("resource_detach_iov", |r| r.resource_detach_iov(handle));
        }
    }

    fn unref_resource(&self, resource_id: u32) {
        if let Ok(handle) = res(resource_id) {
            let _ = self.r.with("resource_unref", |r| r.resource_unref(handle));
        }
    }

    fn transfer_write(
        &self,
        ctx_id: u32,
        resource: &mut RutabagaResource,
        t: Transfer3D,
    ) -> RutabagaResult<()> {
        if t.is_empty() {
            return Ok(());
        }
        let handle = res(resource.resource_id)?;
        self.r
            .with("transfer_write", |r| {
                r.transfer(
                    handle,
                    ContextId::new(ctx_id),
                    transfer::Direction::ToHost,
                    &info_of(&t, false),
                    Vec::new(),
                )
            })?
            .map_err(|e| refused("transfer_write", e))
    }

    /// limina: read a venus/virgl scanout's presented IOSurface into `dst`. Venus zero-copy blobs
    /// have no CPU transfer_read, so this is the only way the headless capture sink recovers a
    /// frame.
    #[cfg(target_os = "macos")]
    fn read_iosurface(
        &self,
        resource_id: u32,
        dst: &mut [u8],
        dst_stride: u32,
        height: u32,
    ) -> RutabagaResult<()> {
        let handle = res(resource_id)?;
        match self.r.with("resource_read_surface", |r| {
            r.resource_read_surface(handle, dst, dst_stride as usize, height)
        })? {
            Some(rows) if rows == height => Ok(()),
            other => {
                error!(
                    "virglrs: read_iosurface: resource {resource_id} gave {other:?} of {height} \
                     rows at {dst_stride} bytes"
                );
                Err(RutabagaError::ComponentError(-libc::EINVAL))
            }
        }
    }

    /// limina vrend zero-copy scanout: complete what was rendered into the surface before it is
    /// presented. An error is the VMM's cue to read the pixels back instead.
    #[cfg(target_os = "macos")]
    fn sync_iosurface(&self, resource_id: u32) -> RutabagaResult<()> {
        let handle = res(resource_id)?;
        if self
            .r
            .with("resource_sync_surface", |r| r.resource_sync_surface(handle))?
        {
            Ok(())
        } else {
            Err(RutabagaError::ComponentError(-libc::EINVAL))
        }
    }

    /// limina vrend zero-copy scanout: which context's completion the surface's contents wait on.
    ///
    /// `EINVAL` is not a failure. It is the renderer saying this present cannot be answered by one
    /// fence -- no surface, nothing attached, or several contexts attached -- and the caller should
    /// use `sync_iosurface`.
    #[cfg(target_os = "macos")]
    fn present_waits_on(&self, resource_id: u32) -> RutabagaResult<u32> {
        let handle = res(resource_id)?;
        self.r
            .with("resource_present_waits_on", |r| {
                r.resource_present_waits_on(handle)
            })?
            .map(|ctx| ctx.get())
            .ok_or(RutabagaError::ComponentError(-libc::EINVAL))
    }

    /// limina fence-accurate present: fence the work behind a flushed resource's contents, and
    /// retire `cookie` as a present fence once it has finished on the host.
    ///
    /// `EINVAL` is not a failure: it says this present cannot be answered by a fence -- nothing
    /// has the resource attached, several contexts have, or the one that does has no queue to
    /// fence -- and the caller should present the frame rather than park it.
    #[cfg(target_os = "macos")]
    fn present_fence(&self, resource_id: u32, cookie: u64) -> RutabagaResult<()> {
        let handle = res(resource_id)?;
        if self.r.with("resource_present_fence", |r| {
            r.resource_present_fence(handle, FenceId(cookie))
        })? {
            Ok(())
        } else {
            Err(RutabagaError::ComponentError(-libc::EINVAL))
        }
    }

    /// limina: fence a flushed venus scanout and copy it on its context's queue, so a guest that
    /// is not held off the scanout cannot draw into it before it is read. `Ok` is the surface to
    /// present. Nothing is fenced on an error: `EBUSY` is every copy surface still in use, which
    /// passes and is best met by dropping the frame, and `EINVAL` is "no ordered copy here".
    #[cfg(target_os = "macos")]
    fn present_copy(&self, resource_id: u32, scanout_id: u32, cookie: u64) -> RutabagaResult<u32> {
        let handle = res(resource_id)?;
        let scanout = virglrenderer::ids::ScanoutId(scanout_id);
        match self.r.with("resource_present_copy", |r| {
            r.resource_present_copy(handle, scanout, FenceId(cookie))
        })? {
            Ok(id) => Ok(id.0),
            Err(virglrenderer::venus::present_copy::CopyRefused::Busy) => {
                Err(RutabagaError::ComponentError(-libc::EBUSY))
            }
            Err(virglrenderer::venus::present_copy::CopyRefused::NotOrderable) => {
                Err(RutabagaError::ComponentError(-libc::EINVAL))
            }
        }
    }

    fn transfer_read(
        &self,
        ctx_id: u32,
        resource: &mut RutabagaResource,
        t: Transfer3D,
        buf: Option<IoSliceMut>,
    ) -> RutabagaResult<()> {
        if t.is_empty() {
            return Ok(());
        }
        let handle = res(resource.resource_id)?;
        // The readback buffer, when there is one, is one iovec describing itself. A pointer and a
        // length that came from the same slice cannot disagree.
        let iov = match buf {
            Some(mut b) => {
                vec![GuestIov {
                    base: VmmPtr(b.as_mut_ptr().cast()),
                    len: b.len(),
                }]
            }
            None => Vec::new(),
        };
        self.r
            .with("transfer_read", |r| {
                r.transfer(
                    handle,
                    ContextId::new(ctx_id),
                    transfer::Direction::ToGuest,
                    &info_of(&t, false),
                    iov,
                )
            })?
            .map_err(|e| refused("transfer_read", e))
    }

    fn create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        c: ResourceCreateBlob,
        iovec_opt: Option<Vec<RutabagaIovec>>,
        _handle_opt: Option<RutabagaHandle>,
    ) -> RutabagaResult<RutabagaResource> {
        let handle = res(resource_id)?;
        // A nonzero blob id names something a context already holds, and it means nothing without
        // the context whose table it is an id in. Zero is the guest asking the host for memory it
        // does not yet have, which the context pays for. Anything but host3d is the guest's own
        // pages, and needs no context at all.
        let source = match (c.blob_mem, c.blob_id) {
            (RUTABAGA_BLOB_MEM_HOST3D, id) if id != 0 => BlobSource::InContext {
                ctx: ctx(ctx_id)?,
                id: BlobId(id),
            },
            (RUTABAGA_BLOB_MEM_HOST3D, _) => BlobSource::HostMinted { ctx: ctx(ctx_id)? },
            _ => BlobSource::Guest,
        };
        let desc = BlobDesc {
            blob_flags: c.blob_flags,
            source,
            size: c.size,
        };
        let iov = iovec_opt.as_deref().map(iovs).unwrap_or_default();
        self.r
            .with("resource_create_blob", |r| {
                r.resource_create_blob(handle, desc, iov)
            })?
            .map_err(|e| refused("create_blob", e))?;

        let mapping = self.mapping(resource_id);
        Ok(RutabagaResource {
            resource_id,
            handle: None,
            blob: true,
            blob_mem: c.blob_mem,
            blob_flags: c.blob_flags,
            map_info: mapping.map(|(_, info)| info),
            #[cfg(target_os = "macos")]
            map_ptr: mapping.map(|(ptr, _)| ptr),
            info_2d: None,
            info_3d: self.query(resource_id),
            vulkan_info: None,
            backing_iovecs: iovec_opt,
            component_mask: 1 << (RutabagaComponentType::VirglRenderer as u8),
            size: c.size,
            mapping: None,
        })
    }

    fn resource_map(
        &self,
        _resource_id: u32,
        _addr: u64,
        _size: u64,
        _prot: i32,
        _flags: i32,
    ) -> RutabagaResult<()> {
        Err(RutabagaError::Unsupported)
    }

    fn map(&self, resource_id: u32) -> RutabagaResult<RutabagaMapping> {
        let handle = res(resource_id)?;
        let m = self
            .r
            .with("resource_host_mapping", |r| r.resource_host_mapping(handle))?
            .map_err(|e| refused("map", e))?;
        Ok(RutabagaMapping {
            ptr: m.addr as u64,
            size: m.size,
        })
    }

    /// Nothing to undo. The renderer's mappings live exactly as long as the storage behind them,
    /// so there is no unmap for the VMM to balance a map with -- which is what the C's
    /// "harmless -EINVAL at unref" comment was working around.
    fn unmap(&self, _resource_id: u32) -> RutabagaResult<()> {
        Ok(())
    }

    fn export_fence(&self, _fence_id: u32) -> RutabagaResult<RutabagaHandle> {
        Err(RutabagaError::Unsupported)
    }

    fn create_context(
        &self,
        ctx_id: u32,
        context_init: u32,
        context_name: Option<&str>,
        _fence_handler: RutabagaFenceHandler,
    ) -> RutabagaResult<Box<dyn RutabagaContext>> {
        let id = ctx(ctx_id)?;
        let name = context_name
            .filter(|s| !s.is_empty())
            .unwrap_or("gpu_renderer");
        // A flagless create is the classic one, which the wire spells as a VIRGL2 context.
        let capset = match context_init {
            0 => CapsetId::Virgl2,
            other => CapsetId::from_raw((other & 0xff) as u8),
        };
        self.r
            .with("context_create", |r| {
                r.context_create(id, capset, name.to_string())
            })?
            .map_err(|e| refused("create_context", e))?;
        Ok(Box::new(VirglRendererContext {
            ctx_id: id,
            r: self.r.clone(),
        }))
    }
}

/// The renderer's census, for the VMM's state dump.
fn dump_state(r: &mut Renderer) {
    let (resources, contexts) = r.counts();
    error!("virglrs: {resources} resources, {contexts} contexts");
    error!("virglrs: vrend journal: {}", r.journal_census());
    for (ctx, bytes, entries) in r.vrend_journal_report() {
        match entries {
            Ok(n) => error!("virglrs:   ctx {ctx}: {bytes} bytes, {n} entries"),
            Err(e) => error!("virglrs:   ctx {ctx}: {bytes} bytes, UNREADABLE: {e}"),
        }
    }
    for (name, n) in r.venus_journal_transient() {
        error!("virglrs:   journal kept nothing for {n:>8} x {name}");
    }
    for (name, n) in r.venus_todo() {
        error!("virglrs:   unserved {n:>8} x {name}");
    }
}

/// limina: hand a previously-published scanout IOSurface back to the supervisor.
///
/// virglrs keeps the registry of the surfaces it published, and answers through the same
/// publisher -- so on the same Mach queue as every other publish and release, whose ordering is
/// what keeps a recycled id from naming the wrong surface. `Unsupported` when the renderer does
/// not hold the id, which is the caller's cue to ask the display backend instead.
#[cfg(target_os = "macos")]
pub fn republish_iosurface(iosurface_id: u32) -> RutabagaResult<()> {
    if virglrenderer::metal::republish(iosurface_id) {
        Ok(())
    } else {
        Err(RutabagaError::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A renderer stand-in that records whether it was dropped.
    struct Probe(Arc<AtomicBool>);

    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn probe() -> (Guarded<Probe>, Arc<AtomicBool>) {
        let dropped = Arc::new(AtomicBool::new(false));
        (Guarded::new(Probe(dropped.clone())), dropped)
    }

    fn is_dead<R>(r: &RutabagaResult<R>) -> bool {
        matches!(r, Err(RutabagaError::ComponentError(e)) if *e == -libc::EIO)
    }

    /// A panic under the lock is an error for that call and for every call after it, none of
    /// which reaches the renderer again, and the dead renderer is leaked rather than dropped.
    #[test]
    fn a_panic_under_the_lock_ends_the_renderer_and_not_the_caller() {
        let (g, dropped) = probe();
        assert_eq!(g.with("a live call", |_| 7).ok(), Some(7));

        let r: RutabagaResult<u32> = g.with("a panicking call", |_| panic!("deliberate"));
        assert!(is_dead(&r), "the panic came back as an error");

        let mut reached = false;
        assert!(is_dead(&g.with("a later call", |_| reached = true)));
        assert!(!reached, "a dead renderer is not touched again");

        drop(g);
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a dead renderer is never dropped"
        );
    }

    /// A panic in virglrs code run outside the lock -- a wait the renderer handed back -- ends
    /// the renderer the same way.
    #[test]
    fn a_panic_outside_the_lock_ends_the_renderer_too() {
        let (g, dropped) = probe();
        let r: RutabagaResult<bool> = g.contain("a ring wait", || panic!("deliberate"));
        assert!(is_dead(&r));
        assert!(is_dead(&g.with("a later call", |_| ())));
        drop(g);
        assert!(!dropped.load(Ordering::SeqCst));
    }

    /// Nothing changes for a renderer that never panicked: it is dropped with the last owner.
    #[test]
    fn a_live_renderer_is_dropped() {
        let (g, dropped) = probe();
        assert!(g.with("a live call", |_| ()).is_ok());
        drop(g);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
