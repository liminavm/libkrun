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
use std::mem::size_of;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use log::error;
use log::warn;

use virglrenderer::abi::{GuestIov, VmmPtr};
use virglrenderer::config::{CapsetId, Config};
use virglrenderer::fence::FenceSink;
use virglrenderer::ids::{
    BlobId, ClientFenceId, ContextId, FenceId, ResourceHandle, RingId, RingIdx,
};
use virglrenderer::renderer::{BlobDesc, BlobSource, Caching, Error as RendererError, Renderer};
use virglrenderer::venus::context::{Submitted, Wait};
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
type Shared = std::sync::Arc<Mutex<Renderer>>;

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
            self.r
                .lock()
                .unwrap()
                .ctx_attach_resource(self.ctx_id, handle);
        }
    }

    fn detach(&mut self, resource: &RutabagaResource) {
        if let Ok(handle) = res(resource.resource_id) {
            self.r
                .lock()
                .unwrap()
                .ctx_detach_resource(self.ctx_id, handle);
        }
    }

    fn component_type(&self) -> RutabagaComponentType {
        RutabagaComponentType::VirglRenderer
    }

    fn context_create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        let id = ctx(fence.ctx_id)?;
        self.r
            .lock()
            .unwrap()
            .context_create_fence(id, RingIdx(fence.ring_idx as u32), FenceId(fence.fence_id))
            .map_err(|e| refused("context_create_fence", e))
    }
}

impl Drop for VirglRendererContext {
    fn drop(&mut self) {
        self.r.lock().unwrap().context_destroy(self.ctx_id);
    }
}

/// Run a whole submission, sleeping outside the renderer lock for any wait it contains.
///
/// The loop is the point, and it is the one thing this file cannot simplify away. The lock is held
/// for the renderer's own call, so a `vkWaitRingSeqnoMESA` that blocked inside it would hold it
/// against the ring thread that has to advance the very head being waited for. The renderer hands
/// the wait back instead: it says how much of the buffer ran and what to wait for, this drops the
/// guard, waits, and comes back with the remainder.
fn submit_all(r: &Shared, id: ContextId, buf: &[u8]) -> RutabagaResult<()> {
    let mut at = 0usize;
    loop {
        let out = r.lock().unwrap().submit_cmd(id, &buf[at..]);
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
                match r.lock().unwrap().ring_waiter(id, ring, seqno) {
                    Ok(w) => w,
                    Err(e) => return Err(refused("ring_waiter", e)),
                }
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
        if !waiter.wait() {
            return Err(refused("ring wait", RendererError::Poisoned));
        }
    }
}

impl VirglRenderer {
    pub fn init(
        virglrenderer_flags: VirglRendererFlags,
        fence_handler: RutabagaFenceHandler,
        _render_server_fd: Option<SafeDescriptor>,
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
        };

        match Renderer::new(Box::new(Fences(fence_handler)), config) {
            Ok(r) => Ok(Box::new(VirglRenderer {
                r: std::sync::Arc::new(Mutex::new(r)),
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
        let m = self.r.lock().unwrap().resource_host_mapping(handle).ok()?;
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
        self.r.lock().unwrap().with_resource(handle, |_| ())?;
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
            .lock()
            .unwrap()
            .resource_iosurface_id(handle)
            .map(|s| s.0)
            .ok_or(RutabagaError::Unsupported)
    }

    fn get_capset_info(&self, capset_id: u32) -> (u32, u32) {
        self.r
            .lock()
            .unwrap()
            .capset_max(CapsetId::from_raw((capset_id & 0xff) as u8))
            .unwrap_or((0, 0))
    }

    fn get_capset(&self, capset_id: u32, _version: u32) -> Vec<u8> {
        let set = CapsetId::from_raw((capset_id & 0xff) as u8);
        match self.r.lock().unwrap().capset(set) {
            Some(caps) => caps.as_bytes().to_vec(),
            None => Vec::new(),
        }
    }

    /// The C ABI's implicit current context. There is no such thing here, which is the whole
    /// reason the rewrite exists.
    fn force_ctx_0(&self) {}

    fn limina_dump_state(&self) {
        let r = self.r.lock().unwrap();
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

    /// A context belongs to one renderer and each keeps its own journal, in its own format.
    /// `None` is a context with nothing retained, which is deliberately not an empty blob: a VMM
    /// that stored zero bytes and restored them later would have rebuilt nothing and been told it
    /// succeeded.
    fn limina_journal_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        let r = self.r.lock().unwrap();
        if r.is_classic(id) {
            r.vrend_journal_export(id)
        } else {
            r.venus_journal_export(id)
        }
    }

    fn limina_journal_seq(&self, ctx_id: u32) -> u64 {
        ContextId::new(ctx_id)
            .and_then(|id| self.r.lock().unwrap().venus_journal_seq(id))
            .unwrap_or(0)
    }

    /// Nothing is pinned. The venus journal keys an entry by a generation rather than holding the
    /// object it names, so there is no pin for the VMM to release.
    fn limina_journal_unpin(&self, _ctx_id: u32, _key: u64) {}

    fn limina_replay_begin(&self, ctx_id: u32) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mut r = self.r.lock().unwrap();
        if r.is_classic(id) {
            r.vrend_replay_begin(id)
        } else {
            r.venus_replay_begin(id).is_ok()
        }
    }

    fn limina_journal_restore(&self, ctx_id: u32, blob: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mut r = self.r.lock().unwrap();
        let classic = r.is_classic(id);
        let restored = if classic {
            r.vrend_journal_restore(id, blob)
        } else {
            r.venus_journal_restore(id, blob)
        };
        match restored {
            Ok(_) => true,
            Err(why) => {
                // The blob has been through a snapshot file since we wrote it. Saying which way it
                // is wrong is the difference between a bug we can find and a resume that is merely
                // black.
                let which = if classic { "vrend" } else { "venus" };
                error!("virglrs: {which}: ctx {ctx_id}: journal refused: {why}");
                false
            }
        }
    }

    fn limina_journal_replay_upto(&self, ctx_id: u32, upto: u64) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mut r = self.r.lock().unwrap();
        if r.is_classic(id) {
            return r.vrend_replay_upto(id, upto);
        }
        match r.venus_replay_upto(id, upto) {
            Ok(()) => true,
            Err(why) => {
                error!("virglrs: venus: ctx {ctx_id}: replay to {upto} failed: {why}");
                false
            }
        }
    }

    fn limina_replay_submit(&self, ctx_id: u32, cmd: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        self.r.lock().unwrap().venus_replay_cmd(id, cmd).is_ok()
    }

    fn limina_replay_ring_cmd(&self, ctx_id: u32, ring_id: u64, cmd: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        self.r
            .lock()
            .unwrap()
            .venus_replay_ring_cmd(id, RingId(ring_id), cmd)
            .is_ok()
    }

    fn limina_replay_end(&self, ctx_id: u32) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mut r = self.r.lock().unwrap();
        if r.is_classic(id) {
            r.vrend_replay_end(id)
        } else {
            r.venus_replay_end(id).is_ok()
        }
    }

    fn limina_sync_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        self.r.lock().unwrap().venus_sync_export(id).ok()
    }

    fn limina_sync_restore(&self, ctx_id: u32, data: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        self.r.lock().unwrap().venus_sync_restore(id, data).is_ok()
    }

    fn limina_memory_census(&self, ctx_id: u32) -> Option<Vec<(u64, u64)>> {
        let id = ContextId::new(ctx_id)?;
        let census = self.r.lock().unwrap().venus_memory_census(id).ok()?;
        Some(census.into_iter().map(|a| (a.id.0, a.size)).collect())
    }

    fn limina_memory_read(&self, ctx_id: u32, mem_id: u64, buf: &mut [u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mem = virglrenderer::venus::cs::ObjectId(mem_id);
        self.r
            .lock()
            .unwrap()
            .venus_memory_read(id, mem, buf)
            .is_ok()
    }

    fn limina_memory_write(&self, ctx_id: u32, mem_id: u64, buf: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        let mem = virglrenderer::venus::cs::ObjectId(mem_id);
        self.r
            .lock()
            .unwrap()
            .venus_memory_write(id, mem, buf)
            .is_ok()
    }

    fn limina_classic_content_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let id = ContextId::new(ctx_id)?;
        self.r
            .lock()
            .unwrap()
            .vrend_content_export(id)
            .map(|(bytes, _)| bytes)
    }

    fn limina_classic_content_restore(&self, ctx_id: u32, data: &[u8]) -> bool {
        let Some(id) = ContextId::new(ctx_id) else {
            return false;
        };
        self.r
            .lock()
            .unwrap()
            .vrend_content_restore(id, data)
            .is_ok()
    }

    fn create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        self.r
            .lock()
            .unwrap()
            .create_fence(ClientFenceId(fence.fence_id as u32));
        Ok(())
    }

    /// Fences retire on the renderer's own thread and are delivered through [`Fences`], so there
    /// is nothing here to poll and no descriptor to poll on.
    fn event_poll(&self) {}

    fn poll_descriptor(&self) -> Option<SafeDescriptor> {
        None
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
            .lock()
            .unwrap()
            .resource_create(handle, args, Vec::new())
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
        self.r
            .lock()
            .unwrap()
            .resource_attach_iov(handle, iovs(vecs))
            .map_err(|e| refused("attach_backing", e))
    }

    fn detach_backing(&self, resource_id: u32) {
        if let Ok(handle) = res(resource_id) {
            self.r.lock().unwrap().resource_detach_iov(handle);
        }
    }

    fn unref_resource(&self, resource_id: u32) {
        if let Ok(handle) = res(resource_id) {
            self.r.lock().unwrap().resource_unref(handle);
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
            .lock()
            .unwrap()
            .transfer(
                handle,
                ContextId::new(ctx_id),
                true,
                &info_of(&t, false),
                Vec::new(),
            )
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
        match self.r.lock().unwrap().resource_read_iosurface(
            handle,
            dst,
            dst_stride as usize,
            height,
        ) {
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
        if self.r.lock().unwrap().resource_sync_iosurface(handle) {
            Ok(())
        } else {
            Err(RutabagaError::ComponentError(-libc::EINVAL))
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
            .lock()
            .unwrap()
            .transfer(
                handle,
                ContextId::new(ctx_id),
                false,
                &info_of(&t, false),
                iov,
            )
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
        // the context whose table it is an id in. Zero is the other operation entirely: the guest
        // asking the host for memory it does not yet have.
        let source = match (c.blob_mem, c.blob_id) {
            (RUTABAGA_BLOB_MEM_HOST3D, id) if id != 0 => BlobSource::InContext {
                ctx: ctx(ctx_id)?,
                id: BlobId(id),
            },
            _ => BlobSource::HostMinted,
        };
        let desc = BlobDesc {
            blob_mem: c.blob_mem,
            blob_flags: c.blob_flags,
            source,
            size: c.size,
        };
        let iov = iovec_opt.as_deref().map(iovs).unwrap_or_default();
        self.r
            .lock()
            .unwrap()
            .resource_create_blob(handle, desc, iov)
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
            .lock()
            .unwrap()
            .resource_host_mapping(handle)
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
            .lock()
            .unwrap()
            .context_create(id, capset, name.to_string())
            .map_err(|e| refused("create_context", e))?;
        Ok(Box::new(VirglRendererContext {
            ctx_id: id,
            r: self.r.clone(),
        }))
    }
}

/// limina: hand a previously-published scanout IOSurface back to the supervisor.
///
/// virglrs keeps no process-wide surface registry to republish out of -- a surface is owned by the
/// resource holding it -- so there is nothing here to answer with. The C's registry is what this
/// stood on.
#[cfg(target_os = "macos")]
pub fn republish_iosurface(_iosurface_id: u32) -> RutabagaResult<()> {
    Err(RutabagaError::Unsupported)
}
