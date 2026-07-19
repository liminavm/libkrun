// Copyright 2020 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! virgl_renderer: Handles 3D virtio-gpu hypercalls using virglrenderer.
//! External code found at <https://gitlab.freedesktop.org/virgl/virglrenderer/>.

#![cfg(feature = "virgl_renderer")]

use std::cmp::min;
use std::convert::TryFrom;
use std::io::Error as SysError;
use std::io::IoSliceMut;
use std::mem::size_of;
use std::os::raw::c_char;
use std::os::raw::c_int;
use std::os::raw::c_void;
use std::os::unix::io::AsRawFd;
use std::panic::catch_unwind;
use std::process::abort;
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use log::debug;
use log::error;
use log::info;
use log::warn;

use crate::generated::virgl_debug_callback_bindings::*;
use crate::generated::virgl_renderer_bindings::*;
use crate::renderer_utils::*;
use crate::rutabaga_core::RutabagaComponent;
use crate::rutabaga_core::RutabagaContext;
use crate::rutabaga_core::RutabagaResource;
use crate::rutabaga_os::FromRawDescriptor;
use crate::rutabaga_os::IntoRawDescriptor;
use crate::rutabaga_os::SafeDescriptor;
use crate::rutabaga_utils::*;

type Query = virgl_renderer_export_query;

unsafe extern "C" {
    // limina M9.3 (virglrenderer patch: venus context-table dump): log the live vkr
    // context list — per context: rings, object/resource table sizes by type. Only in
    // our virglrenderer fork; limina builds always link it (see limina-virgl-link-trap).
    fn virgl_renderer_limina_dump_state();

    // limina M9.3 P1 (virglrenderer patch 0035): venus journal export + replay for
    // snapshot restore. See virglrenderer.h for the contract (mutable replay buffers,
    // one journal entry per submit/ring_cmd call, ring threads start at replay_end).
    fn virgl_renderer_limina_journal_export(
        ctx_id: u32,
        out_buf: *mut *mut c_void,
        out_size: *mut u64,
    ) -> i32;
    fn virgl_renderer_limina_journal_seq(ctx_id: u32) -> u64;
    fn virgl_renderer_limina_journal_unpin(ctx_id: u32, key: u64);
    fn virgl_renderer_limina_replay_begin(ctx_id: u32) -> i32;
    fn virgl_renderer_limina_replay_submit(ctx_id: u32, cmd: *mut c_void, size: u32) -> i32;
    fn virgl_renderer_limina_replay_ring_cmd(
        ctx_id: u32,
        ring_id: u64,
        cmd: *mut c_void,
        size: u32,
    ) -> i32;
    fn virgl_renderer_limina_replay_end(ctx_id: u32) -> i32;

    // limina M9.3 P2 (virglrenderer patch: device-memory content capture): census
    // returns a malloc'd (id, size) u64-pair array of the context's capturable
    // VkDeviceMemory objects; read/write copy bytes out of / into the memory's
    // host mapping. See virglrenderer.h for capturable + restore-ordering rules.
    fn virgl_renderer_limina_sync_export(
        ctx_id: u32,
        out_buf: *mut *mut c_void,
        out_size: *mut u64,
    ) -> i32;
    fn virgl_renderer_limina_sync_restore(ctx_id: u32, data: *const c_void, size: u64) -> i32;
    fn virgl_renderer_limina_memory_census(
        ctx_id: u32,
        out_pairs: *mut *mut u64,
        out_count: *mut u32,
    ) -> i32;
    fn virgl_renderer_limina_memory_read(
        ctx_id: u32,
        mem_id: u64,
        buf: *mut c_void,
        size: u64,
    ) -> i32;
    fn virgl_renderer_limina_memory_write(
        ctx_id: u32,
        mem_id: u64,
        buf: *const c_void,
        size: u64,
    ) -> i32;
}

/// The virtio-gpu backend state tracker which supports accelerated rendering.
pub struct VirglRenderer {}

struct VirglRendererContext {
    ctx_id: u32,
}

fn import_resource(resource: &mut RutabagaResource) -> RutabagaResult<()> {
    if (resource.component_mask & (1 << (RutabagaComponentType::VirglRenderer as u8))) != 0 {
        return Ok(());
    }

    if let Some(_handle) = &resource.handle {
        #[cfg(target_os = "linux")]
        if _handle.handle_type == RUTABAGA_MEM_HANDLE_TYPE_DMABUF {
            let dmabuf_fd = _handle.os_handle.try_clone()?.into_raw_descriptor();
            // Safe because we are being passed a valid fd
            unsafe {
                let dmabuf_size = libc::lseek64(dmabuf_fd, 0, libc::SEEK_END);
                libc::lseek64(dmabuf_fd, 0, libc::SEEK_SET);
                let args = virgl_renderer_resource_import_blob_args {
                    res_handle: resource.resource_id,
                    blob_mem: resource.blob_mem,
                    fd_type: VIRGL_RENDERER_BLOB_FD_TYPE_DMABUF,
                    fd: dmabuf_fd,
                    size: dmabuf_size as u64,
                };
                let ret = virgl_renderer_resource_import_blob(&args);
                if ret != 0 {
                    // import_blob can fail if we've previously imported this resource,
                    // but in any case virglrenderer does not take ownership of the fd
                    // in error paths
                    //
                    // Because of the re-import case we must still fall through to the
                    // virgl_renderer_ctx_attach_resource() call.
                    libc::close(dmabuf_fd);
                    return Ok(());
                }
                resource.component_mask |= 1 << (RutabagaComponentType::VirglRenderer as u8);
            }
        }
    }

    Ok(())
}

impl RutabagaContext for VirglRendererContext {
    fn submit_cmd(&mut self, commands: &mut [u8], fence_ids: &[u64]) -> RutabagaResult<()> {
        if !fence_ids.is_empty() {
            return Err(RutabagaError::Unsupported);
        }
        if !commands.len().is_multiple_of(size_of::<u32>()) {
            return Err(RutabagaError::InvalidCommandSize(commands.len()));
        }
        let dword_count = (commands.len() / size_of::<u32>()) as i32;
        // Safe because the context and buffer are valid and virglrenderer will have been
        // initialized if there are Context instances.
        let ret = unsafe {
            virgl_renderer_submit_cmd(
                commands.as_mut_ptr() as *mut c_void,
                self.ctx_id as i32,
                dword_count,
            )
        };
        ret_to_res(ret)
    }

    fn attach(&mut self, resource: &mut RutabagaResource) {
        match import_resource(resource) {
            Ok(()) => (),
            Err(e) => error!("importing resource failing with {e}"),
        }

        // The context id and resource id must be valid because the respective instances ensure
        // their lifetime.
        unsafe {
            virgl_renderer_ctx_attach_resource(self.ctx_id as i32, resource.resource_id as i32);
        }
    }

    fn detach(&mut self, resource: &RutabagaResource) {
        // The context id and resource id must be valid because the respective instances ensure
        // their lifetime.
        unsafe {
            virgl_renderer_ctx_detach_resource(self.ctx_id as i32, resource.resource_id as i32);
        }
    }

    fn component_type(&self) -> RutabagaComponentType {
        RutabagaComponentType::VirglRenderer
    }

    fn context_create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        // RutabagaFence::flags are not compatible with virglrenderer's fencing API and currently
        // virglrenderer context's assume all fences on a single timeline are MERGEABLE, and enforce
        // this assumption.
        let flags: u32 = VIRGL_RENDERER_FENCE_FLAG_MERGEABLE;

        let ret = unsafe {
            virgl_renderer_context_create_fence(
                fence.ctx_id,
                flags,
                fence.ring_idx as u64,
                fence.fence_id,
            )
        };
        ret_to_res(ret)
    }
}

impl Drop for VirglRendererContext {
    fn drop(&mut self) {
        // The context is safe to destroy because nothing else can be referencing it.
        unsafe {
            virgl_renderer_context_destroy(self.ctx_id);
        }
    }
}

/// Level-aware virgl log sink (virgl_set_log_callback). Unlike the legacy
/// va_list callback this receives the severity, so virgl/vkr ERROR lines —
/// e.g. "ring FATAL set at ..." when a venus context is poisoned — surface at
/// a level production deployments (default `warn`) actually record.
extern "C" fn log_callback(
    log_level: u32,
    message: *const ::std::os::raw::c_char,
    _user_data: *mut ::std::os::raw::c_void,
) {
    if message.is_null() {
        return;
    }
    let msg = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
    let msg = msg.trim_end();
    match log_level {
        VIRGL_LOG_LEVEL_ERROR => error!("virgl: {msg}"),
        VIRGL_LOG_LEVEL_WARNING => warn!("virgl: {msg}"),
        VIRGL_LOG_LEVEL_INFO => info!("virgl: {msg}"),
        _ => debug!("virgl: {msg}"),
    }
}

#[allow(dead_code)] // superseded by the level-aware log_callback below; kept for reference
extern "C" fn debug_callback(fmt: *const ::std::os::raw::c_char, ap: stdio::va_list) {
    const BUF_LEN: usize = 256;
    let mut v = [b' '; BUF_LEN];

    let printed_len = unsafe {
        let ptr = v.as_mut_ptr() as *mut ::std::os::raw::c_char;
        #[cfg(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        ))]
        let size = BUF_LEN as ::std::os::raw::c_ulong;
        #[cfg(target_arch = "arm")]
        let size = BUF_LEN as ::std::os::raw::c_uint;

        stdio::vsnprintf(ptr, size, fmt, ap)
    };

    if printed_len < 0 {
        debug!("rutabaga_gfx::virgl_renderer::debug_callback: vsnprintf returned {printed_len}");
    } else {
        // vsnprintf returns the number of chars that *would* have been printed
        let len = min(printed_len as usize, BUF_LEN - 1);
        debug!("{}", String::from_utf8_lossy(&v[..len]));
    }
}

/// TODO(ryanneph): re-evaluate if "ring_idx: u8" can be used instead so we can drop this in favor
/// of the common write_context_fence() from renderer_utils before promoting to
/// cfg(feature = "virgl_renderer").
#[cfg(feature = "virgl_renderer_next")]
extern "C" fn write_context_fence(cookie: *mut c_void, ctx_id: u32, ring_idx: u64, fence_id: u64) {
    catch_unwind(|| {
        assert!(!cookie.is_null());
        let cookie = unsafe { &*(cookie as *mut RutabagaCookie) };

        // Call fence completion callback
        if let Some(handler) = &cookie.fence_handler {
            handler.call(RutabagaFence {
                flags: RUTABAGA_FLAG_FENCE | RUTABAGA_FLAG_INFO_RING_IDX,
                fence_id,
                ctx_id,
                ring_idx: ring_idx as u8,
            });
        }
    })
    .unwrap_or_else(|_| abort())
}

unsafe extern "C" fn write_fence(cookie: *mut c_void, fence: u32) {
    catch_unwind(|| {
        assert!(!cookie.is_null());
        let cookie = unsafe { &*(cookie as *mut RutabagaCookie) };

        // Call fence completion callback
        if let Some(handler) = &cookie.fence_handler {
            handler.call(RutabagaFence {
                flags: RUTABAGA_FLAG_FENCE,
                fence_id: fence as u64,
                ctx_id: 0,
                ring_idx: 0,
            });
        }
    })
    .unwrap_or_else(|_| abort())
}

#[cfg(feature = "virgl_renderer_next")]
unsafe extern "C" fn get_server_fd(cookie: *mut c_void, version: u32) -> c_int {
    catch_unwind(|| {
        assert!(!cookie.is_null());
        let cookie = unsafe { &mut *(cookie as *mut RutabagaCookie) };

        if version != 0 {
            return -1;
        }

        // Transfer the fd ownership to virglrenderer.
        cookie
            .render_server_fd
            .take()
            .map(SafeDescriptor::into_raw_descriptor)
            .unwrap_or(-1)
    })
    .unwrap_or_else(|_| abort())
}

const VIRGL_RENDERER_CALLBACKS: &virgl_renderer_callbacks = &virgl_renderer_callbacks {
    #[cfg(not(feature = "virgl_renderer_next"))]
    version: 1,
    #[cfg(feature = "virgl_renderer_next")]
    version: 3,
    write_fence: Some(write_fence),
    create_gl_context: None,
    destroy_gl_context: None,
    make_current: None,
    get_drm_fd: None,
    #[cfg(not(feature = "virgl_renderer_next"))]
    write_context_fence: None,
    #[cfg(feature = "virgl_renderer_next")]
    write_context_fence: Some(write_context_fence),
    #[cfg(not(feature = "virgl_renderer_next"))]
    get_server_fd: None,
    #[cfg(feature = "virgl_renderer_next")]
    get_server_fd: Some(get_server_fd),
};

/// Retrieves metadata suitable for export about this resource. If "export_fd" is true,
/// performs an export of this resource so that it may be imported by other processes.
fn export_query(resource_id: u32) -> RutabagaResult<Query> {
    let mut query: Query = Default::default();
    query.hdr.stype = VIRGL_RENDERER_STRUCTURE_TYPE_EXPORT_QUERY;
    query.hdr.stype_version = 0;
    query.hdr.size = size_of::<Query>() as u32;
    query.in_resource_id = resource_id;
    query.in_export_fds = 0;

    // Safe because the image parameters are stack variables of the correct type.
    let ret =
        unsafe { virgl_renderer_execute(&mut query as *mut _ as *mut c_void, query.hdr.size) };

    ret_to_res(ret)?;
    Ok(query)
}

impl VirglRenderer {
    pub fn init(
        virglrenderer_flags: VirglRendererFlags,
        fence_handler: RutabagaFenceHandler,
        render_server_fd: Option<SafeDescriptor>,
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

        // virglrenderer is a global state backed library that uses thread bound OpenGL contexts.
        // Initialize it only once and use the non-send/non-sync Renderer struct to keep things tied
        // to whichever thread called this function first.
        static INIT_ONCE: AtomicBool = AtomicBool::new(false);
        if INIT_ONCE
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Acquire)
            .is_err()
        {
            return Err(RutabagaError::AlreadyInUse);
        }

        unsafe { virgl_set_log_callback(Some(log_callback), null_mut(), None) };

        // Cookie is intentionally never freed because virglrenderer never gets uninitialized.
        // Otherwise, Resource and Context would become invalid because their lifetime is not tied
        // to the Renderer instance. Doing so greatly simplifies the ownership for users of this
        // library.
        let cookie = Box::into_raw(Box::new(RutabagaCookie {
            render_server_fd,
            fence_handler: Some(fence_handler),
            debug_handler: None,
        }));

        // Safe because a valid cookie and set of callbacks is used and the result is checked for
        // error.
        let ret = unsafe {
            virgl_renderer_init(
                cookie as *mut c_void,
                virglrenderer_flags.into(),
                VIRGL_RENDERER_CALLBACKS as *const virgl_renderer_callbacks
                    as *mut virgl_renderer_callbacks,
            )
        };

        // If the initialization failed, allow the users to try again.
        if ret != 0 {
            INIT_ONCE.store(false, Ordering::Release);
        }

        ret_to_res(ret)?;
        Ok(Box::new(VirglRenderer {}))
    }

    #[allow(unused_variables)]
    fn map_info(&self, resource_id: u32) -> RutabagaResult<u32> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            let mut map_info = 0;
            let ret = unsafe { virgl_renderer_resource_get_map_info(resource_id, &mut map_info) };
            ret_to_res(ret)?;

            Ok(map_info | RUTABAGA_MAP_ACCESS_RW)
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }

    #[cfg(target_os = "macos")]
    fn map_ptr(&self, resource_id: u32) -> RutabagaResult<u64> {
        let mut map_ptr = 0;
        let ret = unsafe { virgl_renderer_resource_get_map_ptr(resource_id, &mut map_ptr) };
        ret_to_res(ret)?;

        Ok(map_ptr)
    }

    // limina tier-2: the global IOSurface id backing a scanout resource, or 0 if not
    // IOSurface-backed. Cached on the RutabagaResource for zero-copy SET_SCANOUT_BLOB present.
    #[cfg(target_os = "macos")]
    fn iosurface_id(&self, resource_id: u32) -> RutabagaResult<u32> {
        let mut iosurface_id = 0;
        let ret =
            unsafe { virgl_renderer_resource_get_iosurface_id(resource_id, &mut iosurface_id) };
        ret_to_res(ret)?;

        Ok(iosurface_id)
    }

    fn query(&self, resource_id: u32) -> RutabagaResult<Resource3DInfo> {
        let query = export_query(resource_id)?;
        if query.out_num_fds == 0 {
            return Err(RutabagaError::Unsupported);
        }

        // virglrenderer unfortunately doesn't return the width or height, so map to zero.
        Ok(Resource3DInfo {
            width: 0,
            height: 0,
            drm_fourcc: query.out_fourcc,
            strides: query.out_strides,
            offsets: query.out_offsets,
            modifier: query.out_modifier,
        })
    }

    #[allow(unused_variables)]
    fn export_blob(&self, resource_id: u32) -> RutabagaResult<Arc<RutabagaHandle>> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            let mut fd_type = 0;
            let mut fd = -1;
            let ret =
                unsafe { virgl_renderer_resource_export_blob(resource_id, &mut fd_type, &mut fd) };
            ret_to_res(ret)?;

            // Safe because the FD was just returned by a successful virglrenderer
            // call so it must be valid and owned by us.
            let handle = unsafe { SafeDescriptor::from_raw_descriptor(fd) };

            let handle_type = match fd_type {
                VIRGL_RENDERER_BLOB_FD_TYPE_DMABUF => RUTABAGA_MEM_HANDLE_TYPE_DMABUF,
                VIRGL_RENDERER_BLOB_FD_TYPE_SHM => RUTABAGA_MEM_HANDLE_TYPE_SHM,
                VIRGL_RENDERER_BLOB_FD_TYPE_OPAQUE => RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD,
                VIRGL_RENDERER_BLOB_FD_TYPE_APPLE => RUTABAGA_MEM_HANDLE_TYPE_APPLE,
                _ => {
                    return Err(RutabagaError::Unsupported);
                }
            };

            Ok(Arc::new(RutabagaHandle {
                os_handle: handle,
                handle_type,
            }))
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }
}

impl Drop for VirglRenderer {
    fn drop(&mut self) {
        // Safe because virglrenderer is initialized.
        //
        // This invalidates all context ids and resource ids.  It is fine because struct Rutabaga
        // makes sure contexts and resources are dropped before this is reached.  Even if it did
        // not, virglrenderer is designed to deal with invalid ids safely.
        unsafe {
            virgl_renderer_cleanup(null_mut());
        }
    }
}

impl RutabagaComponent for VirglRenderer {
    fn get_capset_info(&self, capset_id: u32) -> (u32, u32) {
        let mut version = 0;
        let mut size = 0;
        // Safe because virglrenderer is initialized by now and properly size stack variables are
        // used for the pointers.
        unsafe {
            virgl_renderer_get_cap_set(capset_id, &mut version, &mut size);
        }
        (version, size)
    }

    fn get_capset(&self, capset_id: u32, version: u32) -> Vec<u8> {
        let (_, max_size) = self.get_capset_info(capset_id);
        let mut buf = vec![0u8; max_size as usize];
        // Safe because virglrenderer is initialized by now and the given buffer is sized properly
        // for the given cap id/version.
        unsafe {
            virgl_renderer_fill_caps(capset_id, version, buf.as_mut_ptr() as *mut c_void);
        }
        buf
    }

    fn force_ctx_0(&self) {
        unsafe { virgl_renderer_force_ctx_0() };
    }

    fn limina_dump_state(&self) {
        // Safe: no arguments, no return; virglrenderer is initialized for as long as
        // this component exists, and the caller contract puts us on the renderer thread.
        unsafe { virgl_renderer_limina_dump_state() };
    }

    fn limina_journal_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let mut buf: *mut c_void = std::ptr::null_mut();
        let mut size: u64 = 0;
        // Safe: out-params are written only on success; the returned buffer is
        // malloc'd by virglrenderer and owned by us until the libc::free below.
        let ret = unsafe { virgl_renderer_limina_journal_export(ctx_id, &mut buf, &mut size) };
        if ret != 0 || buf.is_null() {
            return None;
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(buf as *const u8, size as usize) }.to_vec();
        unsafe { libc::free(buf) };
        Some(bytes)
    }

    fn limina_journal_seq(&self, ctx_id: u32) -> u64 {
        unsafe { virgl_renderer_limina_journal_seq(ctx_id) }
    }

    fn limina_journal_unpin(&self, ctx_id: u32, key: u64) {
        unsafe { virgl_renderer_limina_journal_unpin(ctx_id, key) }
    }

    fn limina_replay_begin(&self, ctx_id: u32) -> bool {
        unsafe { virgl_renderer_limina_replay_begin(ctx_id) == 0 }
    }

    fn limina_replay_submit(&self, ctx_id: u32, cmd: &mut [u8]) -> bool {
        unsafe {
            virgl_renderer_limina_replay_submit(
                ctx_id,
                cmd.as_mut_ptr() as *mut c_void,
                cmd.len() as u32,
            ) == 0
        }
    }

    fn limina_replay_ring_cmd(&self, ctx_id: u32, ring_id: u64, cmd: &mut [u8]) -> bool {
        unsafe {
            virgl_renderer_limina_replay_ring_cmd(
                ctx_id,
                ring_id,
                cmd.as_mut_ptr() as *mut c_void,
                cmd.len() as u32,
            ) == 0
        }
    }

    fn limina_replay_end(&self, ctx_id: u32) -> bool {
        unsafe { virgl_renderer_limina_replay_end(ctx_id) == 0 }
    }

    fn limina_sync_export(&self, ctx_id: u32) -> Option<Vec<u8>> {
        let mut buf: *mut c_void = std::ptr::null_mut();
        let mut size: u64 = 0;
        // Safe: out-params written only on success; blob is malloc'd by
        // virglrenderer, freed below.
        let ret = unsafe { virgl_renderer_limina_sync_export(ctx_id, &mut buf, &mut size) };
        if ret != 0 || buf.is_null() {
            return None;
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(buf as *const u8, size as usize) }.to_vec();
        unsafe { libc::free(buf) };
        Some(bytes)
    }

    fn limina_sync_restore(&self, ctx_id: u32, data: &[u8]) -> bool {
        unsafe {
            virgl_renderer_limina_sync_restore(
                ctx_id,
                data.as_ptr() as *const c_void,
                data.len() as u64,
            ) == 0
        }
    }

    fn limina_memory_census(&self, ctx_id: u32) -> Option<Vec<(u64, u64)>> {
        let mut pairs: *mut u64 = std::ptr::null_mut();
        let mut count: u32 = 0;
        // Safe: out-params are written only on success; the pair array is malloc'd
        // by virglrenderer and owned by us until the libc::free below (NULL with
        // count 0 is the legitimate empty census).
        let ret = unsafe { virgl_renderer_limina_memory_census(ctx_id, &mut pairs, &mut count) };
        if ret != 0 {
            return None;
        }
        if pairs.is_null() {
            return Some(Vec::new());
        }
        let out = unsafe { std::slice::from_raw_parts(pairs, count as usize * 2) }
            .chunks_exact(2)
            .map(|p| (p[0], p[1]))
            .collect();
        unsafe { libc::free(pairs as *mut c_void) };
        Some(out)
    }

    fn limina_memory_read(&self, ctx_id: u32, mem_id: u64, buf: &mut [u8]) -> bool {
        unsafe {
            virgl_renderer_limina_memory_read(
                ctx_id,
                mem_id,
                buf.as_mut_ptr() as *mut c_void,
                buf.len() as u64,
            ) == 0
        }
    }

    fn limina_memory_write(&self, ctx_id: u32, mem_id: u64, buf: &[u8]) -> bool {
        unsafe {
            virgl_renderer_limina_memory_write(
                ctx_id,
                mem_id,
                buf.as_ptr() as *const c_void,
                buf.len() as u64,
            ) == 0
        }
    }

    fn create_fence(&mut self, fence: RutabagaFence) -> RutabagaResult<()> {
        let ret = unsafe { virgl_renderer_create_fence(fence.fence_id as i32, fence.ctx_id) };
        ret_to_res(ret)
    }

    fn event_poll(&self) {
        unsafe { virgl_renderer_poll() };
    }

    fn poll_descriptor(&self) -> Option<SafeDescriptor> {
        // Safe because it can be called anytime and returns -1 in the event of an error.
        let fd = unsafe { virgl_renderer_get_poll_fd() };
        if fd >= 0
            && let Ok(dup_fd) = SafeDescriptor::try_from(&fd as &dyn AsRawFd)
        {
            return Some(dup_fd);
        }
        None
    }

    fn create_3d(
        &self,
        resource_id: u32,
        resource_create_3d: ResourceCreate3D,
    ) -> RutabagaResult<RutabagaResource> {
        let mut args = virgl_renderer_resource_create_args {
            handle: resource_id,
            target: resource_create_3d.target,
            format: resource_create_3d.format,
            bind: resource_create_3d.bind,
            width: resource_create_3d.width,
            height: resource_create_3d.height,
            depth: resource_create_3d.depth,
            array_size: resource_create_3d.array_size,
            last_level: resource_create_3d.last_level,
            nr_samples: resource_create_3d.nr_samples,
            flags: resource_create_3d.flags,
        };

        // Safe because virglrenderer is initialized by now, and the return value is checked before
        // returning a new resource. The backing buffers are not supplied with this call.
        let ret = unsafe { virgl_renderer_resource_create(&mut args, null_mut(), 0) };
        ret_to_res(ret)?;

        Ok(RutabagaResource {
            resource_id,
            handle: self.export_blob(resource_id).ok(),
            blob: false,
            blob_mem: 0,
            blob_flags: 0,
            map_info: None,
            #[cfg(target_os = "macos")]
            map_ptr: None,
            #[cfg(target_os = "macos")]
            iosurface_id: None,
            info_2d: None,
            info_3d: self.query(resource_id).ok(),
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
        // Safe because the backing is into guest memory that we store a reference count for.
        let ret = unsafe {
            virgl_renderer_resource_attach_iov(
                resource_id as i32,
                vecs.as_mut_ptr() as *mut iovec,
                vecs.len() as i32,
            )
        };
        ret_to_res(ret)
    }

    fn detach_backing(&self, resource_id: u32) {
        // Safe as we don't need the old backing iovecs returned and the reference to the guest
        // memory can be dropped as it will no longer be needed for this resource.
        unsafe {
            virgl_renderer_resource_detach_iov(resource_id as i32, null_mut(), null_mut());
        }
    }

    fn unref_resource(&self, resource_id: u32) {
        // The resource is safe to unreference destroy because no user of these bindings can still
        // be holding a reference.
        unsafe {
            // limina (macOS): create_blob eagerly maps every mappable blob via map_ptr() (the
            // host pointer feeds hv_vm_map), and virglrenderer's destroy path does not munmap a
            // live mapping — balance it here. Harmless -EINVAL when the resource was never
            // mapped; the #28 borrowed-driver-pointer case returns early without munmapping.
            #[cfg(target_os = "macos")]
            virgl_renderer_resource_unmap(resource_id);
            virgl_renderer_resource_unref(resource_id);
        }
    }

    fn transfer_write(
        &self,
        ctx_id: u32,
        resource: &mut RutabagaResource,
        transfer: Transfer3D,
    ) -> RutabagaResult<()> {
        if transfer.is_empty() {
            return Ok(());
        }

        let mut transfer_box = VirglBox {
            x: transfer.x,
            y: transfer.y,
            z: transfer.z,
            w: transfer.w,
            h: transfer.h,
            d: transfer.d,
        };

        // Safe because only stack variables of the appropriate type are used.
        let ret = unsafe {
            virgl_renderer_transfer_write_iov(
                resource.resource_id,
                ctx_id,
                transfer.level as i32,
                transfer.stride,
                transfer.layer_stride,
                &mut transfer_box as *mut VirglBox as *mut virgl_box,
                transfer.offset,
                null_mut(),
                0,
            )
        };
        ret_to_res(ret)
    }

    // limina: read a venus/virgl scanout's presented IOSurface (shared-storage, CPU-readable)
    // into `dst`. Venus zero-copy blobs have no CPU transfer_read, so this is the only way the
    // headless capture display sink can recover the frame.
    #[cfg(target_os = "macos")]
    fn read_iosurface(
        &self,
        resource_id: u32,
        dst: &mut [u8],
        dst_stride: u32,
        height: u32,
    ) -> RutabagaResult<()> {
        let ret = unsafe {
            virgl_renderer_resource_read_iosurface(
                resource_id,
                dst.as_mut_ptr() as *mut std::os::raw::c_void,
                dst_stride,
                height,
            )
        };
        ret_to_res(ret)
    }

    fn transfer_read(
        &self,
        ctx_id: u32,
        resource: &mut RutabagaResource,
        transfer: Transfer3D,
        buf: Option<IoSliceMut>,
    ) -> RutabagaResult<()> {
        if transfer.is_empty() {
            return Ok(());
        }

        let mut transfer_box = VirglBox {
            x: transfer.x,
            y: transfer.y,
            z: transfer.z,
            w: transfer.w,
            h: transfer.h,
            d: transfer.d,
        };

        let mut iov = RutabagaIovec {
            base: null_mut(),
            len: 0,
        };

        let (iovecs, num_iovecs) = match buf {
            Some(mut buf) => {
                iov.base = buf.as_mut_ptr() as *mut c_void;
                iov.len = buf.len();
                (&mut iov as *mut RutabagaIovec as *mut iovec, 1)
            }
            None => (null_mut(), 0),
        };

        // Safe because only stack variables of the appropriate type are used.
        let ret = unsafe {
            virgl_renderer_transfer_read_iov(
                resource.resource_id,
                ctx_id,
                transfer.level,
                transfer.stride,
                transfer.layer_stride,
                &mut transfer_box as *mut VirglBox as *mut virgl_box,
                transfer.offset,
                iovecs,
                num_iovecs,
            )
        };
        ret_to_res(ret)
    }

    #[allow(unused_variables)]
    fn create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        mut iovec_opt: Option<Vec<RutabagaIovec>>,
        _handle_opt: Option<RutabagaHandle>,
    ) -> RutabagaResult<RutabagaResource> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            let mut iovec_ptr = null_mut();
            let mut num_iovecs = 0;
            if let Some(ref mut iovecs) = iovec_opt {
                iovec_ptr = iovecs.as_mut_ptr();
                num_iovecs = iovecs.len();
            }

            let resource_create_args = virgl_renderer_resource_create_blob_args {
                res_handle: resource_id,
                ctx_id,
                blob_mem: resource_create_blob.blob_mem,
                blob_flags: resource_create_blob.blob_flags,
                blob_id: resource_create_blob.blob_id,
                size: resource_create_blob.size,
                iovecs: iovec_ptr as *const iovec,
                num_iovs: num_iovecs as u32,
            };

            let ret = unsafe { virgl_renderer_resource_create_blob(&resource_create_args) };
            ret_to_res(ret)?;

            // TODO(b/244591751): assign vulkan_info to support opaque_fd mapping via Vulkano when
            // sandboxing (hence external_blob) is enabled.
            Ok(RutabagaResource {
                resource_id,
                handle: self.export_blob(resource_id).ok(),
                blob: true,
                blob_mem: resource_create_blob.blob_mem,
                blob_flags: resource_create_blob.blob_flags,
                map_info: self.map_info(resource_id).ok(),
                #[cfg(target_os = "macos")]
                map_ptr: self.map_ptr(resource_id).ok(),
                #[cfg(target_os = "macos")]
                iosurface_id: self.iosurface_id(resource_id).ok().filter(|&id| id != 0),
                info_2d: None,
                info_3d: self.query(resource_id).ok(),
                vulkan_info: None,
                backing_iovecs: iovec_opt,
                component_mask: 1 << (RutabagaComponentType::VirglRenderer as u8),
                size: resource_create_blob.size,
                mapping: None,
            })
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }

    fn resource_map(
        &self,
        _resource_id: u32,
        _addr: u64,
        _size: u64,
        _prot: i32,
        _flags: i32,
    ) -> RutabagaResult<()> {
        #[cfg(feature = "virgl_resource_map2")]
        {
            let ret = unsafe {
                virgl_renderer_resource_map2(
                    _resource_id,
                    _addr as *mut libc::c_void,
                    _size,
                    _prot,
                    _flags,
                )
            };
            if ret != 0 {
                return Err(RutabagaError::MappingFailed(ret));
            }

            Ok(())
        }
        #[cfg(not(feature = "virgl_resource_map2"))]
        Err(RutabagaError::Unsupported)
    }

    fn map(&self, resource_id: u32) -> RutabagaResult<RutabagaMapping> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            let mut map: *mut c_void = null_mut();
            let mut size: u64 = 0;
            // Safe because virglrenderer wraps and validates use of GL/VK.
            let ret = unsafe { virgl_renderer_resource_map(resource_id, &mut map, &mut size) };
            if ret != 0 {
                return Err(RutabagaError::MappingFailed(ret));
            }

            Ok(RutabagaMapping {
                ptr: map as u64,
                size,
            })
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }

    fn unmap(&self, resource_id: u32) -> RutabagaResult<()> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            // Safe because virglrenderer is initialized by now.
            let ret = unsafe { virgl_renderer_resource_unmap(resource_id) };
            ret_to_res(ret)
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }

    #[allow(unused_variables)]
    fn export_fence(&self, fence_id: u32) -> RutabagaResult<RutabagaHandle> {
        #[cfg(feature = "virgl_renderer_next")]
        {
            // Safe because the parameters are stack variables of the correct type.
            let mut fd: i32 = 0;
            let ret = unsafe { virgl_renderer_export_fence(fence_id, &mut fd) };
            ret_to_res(ret)?;

            // Safe because the FD was just returned by a successful virglrenderer call so it must
            // be valid and owned by us.
            let fence = unsafe { SafeDescriptor::from_raw_descriptor(fd) };
            Ok(RutabagaHandle {
                os_handle: fence,
                handle_type: RUTABAGA_FENCE_HANDLE_TYPE_SYNC_FD,
            })
        }
        #[cfg(not(feature = "virgl_renderer_next"))]
        Err(RutabagaError::Unsupported)
    }

    #[allow(unused_variables)]
    fn create_context(
        &self,
        ctx_id: u32,
        context_init: u32,
        context_name: Option<&str>,
        _fence_handler: RutabagaFenceHandler,
    ) -> RutabagaResult<Box<dyn RutabagaContext>> {
        let mut name: &str = "gpu_renderer";
        if let Some(name_string) = context_name.filter(|s| !s.is_empty()) {
            name = name_string;
        }

        // Safe because virglrenderer is initialized by now and the context name is statically
        // allocated. The return value is checked before returning a new context.
        let ret = unsafe {
            #[cfg(feature = "virgl_renderer_next")]
            match context_init {
                0 => virgl_renderer_context_create(
                    ctx_id,
                    name.len() as u32,
                    name.as_ptr() as *const c_char,
                ),
                _ => virgl_renderer_context_create_with_flags(
                    ctx_id,
                    context_init,
                    name.len() as u32,
                    name.as_ptr() as *const c_char,
                ),
            }
            #[cfg(not(feature = "virgl_renderer_next"))]
            virgl_renderer_context_create(ctx_id, name.len() as u32, name.as_ptr() as *const c_char)
        };
        ret_to_res(ret)?;
        Ok(Box::new(VirglRendererContext { ctx_id }))
    }
}
