use std::collections::BTreeMap;
use std::env;
use std::io::IoSliceMut;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::super::Queue as VirtQueue;
use super::protocol::GpuResponse::*;
use super::protocol::{
    GpuResponse, GpuResponsePlaneInfo, VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE,
    VIRTIO_GPU_BLOB_MEM_HOST3D, VIRTIO_GPU_MAX_SCANOUTS, VirtioGpuResult,
};
#[cfg(target_os = "macos")]
use crossbeam_channel::{Sender, unbounded};
use krun_display::{
    DisplayBackend, DisplayBackendBasicFramebuffer, DisplayBackendError, DisplayBackendInstance,
    Rect, ResourceFormat,
};
use libc::c_void;
#[cfg(all(feature = "virgl_resource_map2", target_os = "linux"))]
use rutabaga_gfx::RUTABAGA_MEM_HANDLE_TYPE_DMABUF;
#[cfg(all(not(feature = "virgl_resource_map2"), target_os = "linux"))]
use rutabaga_gfx::RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD;
#[cfg(all(feature = "virgl_resource_map2", target_os = "linux"))]
use rutabaga_gfx::RUTABAGA_MEM_HANDLE_TYPE_SHM;
#[cfg(target_os = "linux")]
use rutabaga_gfx::{
    RUTABAGA_CHANNEL_TYPE_PW, RUTABAGA_CHANNEL_TYPE_X11, RUTABAGA_MAP_ACCESS_MASK,
    RUTABAGA_MAP_ACCESS_READ, RUTABAGA_MAP_ACCESS_RW, RUTABAGA_MAP_ACCESS_WRITE,
};
use rutabaga_gfx::{
    RUTABAGA_CHANNEL_TYPE_WAYLAND, RUTABAGA_MAP_CACHE_MASK, ResourceCreate3D, ResourceCreateBlob,
    Rutabaga, RutabagaBuilder, RutabagaChannel, RutabagaFence, RutabagaFenceHandler, RutabagaIovec,
    Transfer3D,
};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap, VolatileSlice};

use super::{GpuError, Result};
use crate::display::DisplayInfo;
use crate::virtio::fs::ExportTable;
use crate::virtio::gpu::protocol::VIRTIO_GPU_FLAG_INFO_RING_IDX;
use crate::virtio::{InterruptTransport, VirtioShmRegion};

fn sglist_to_rutabaga_iovecs(
    vecs: &[(GuestAddress, usize)],
    mem: &GuestMemoryMmap,
) -> Result<Vec<RutabagaIovec>> {
    if vecs
        .iter()
        .any(|&(addr, len)| mem.get_slice(addr, len).is_err())
    {
        return Err(GpuError::GuestMemory);
    }

    let mut rutabaga_iovecs: Vec<RutabagaIovec> = Vec::new();
    for &(addr, len) in vecs {
        let slice = mem.get_slice(addr, len).unwrap();
        rutabaga_iovecs.push(RutabagaIovec {
            base: slice.ptr_guard_mut().as_ptr() as *mut c_void,
            len,
        });
    }
    Ok(rutabaga_iovecs)
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum VirtioGpuRing {
    Global,
    ContextSpecific { ctx_id: u32, ring_idx: u8 },
}

struct FenceDescriptor {
    ring: VirtioGpuRing,
    fence_id: u64,
    desc_index: u16,
    len: u32,
}

#[derive(Default)]
pub struct FenceState {
    descs: Vec<FenceDescriptor>,
    completed_fences: BTreeMap<VirtioGpuRing, u64>,
}

/// Mark a fence as already completed without a renderer (software-2D mode).
///
/// Bumps the ring's completed-fence watermark to at least `fence.fence_id`, so a
/// following `process_fence()` sees the fence as already signaled and retires the
/// descriptor immediately. Used when `rutabaga` is `None`: there is no async fence
/// callback to ever signal it otherwise, and 2D commands are synchronous anyway.
fn mark_fence_completed_sync(fence_state: &Mutex<FenceState>, fence: &RutabagaFence) {
    let ring = match fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
        0 => VirtioGpuRing::Global,
        _ => VirtioGpuRing::ContextSpecific {
            ctx_id: fence.ctx_id,
            ring_idx: fence.ring_idx,
        },
    };
    let mut fence_state = fence_state.lock().unwrap();
    let entry = fence_state.completed_fences.entry(ring).or_insert(0);
    *entry = (*entry).max(fence.fence_id);
}

#[derive(Copy, Clone, Debug, Default)]
struct AssociatedScanouts(u32);

impl AssociatedScanouts {
    fn enable(&mut self, scanout_id: u32) {
        self.0 |= 1 << scanout_id;
    }

    fn disable(&mut self, scanout_id: u32) {
        self.0 ^= 1 << scanout_id;
    }

    const fn has_any_enabled(self) -> bool {
        self.0 != 0
    }

    fn iter_enabled(self) -> impl Iterator<Item = u32> {
        (0..VIRTIO_GPU_MAX_SCANOUTS).filter(move |i| ((self.0 >> i) & 1) == 1)
    }
}

#[derive(Copy, Clone)]
struct VirtioGpuResource {
    id: u32,
    width: u32,
    height: u32,
    scanouts: AssociatedScanouts,
    format: Option<ResourceFormat>,
    size: u64, // only for blob resources
    shmem_offset: Option<u64>,
    rutabaga_external_mapping: bool,
}

impl VirtioGpuResource {
    /// Creates a new VirtioGpuResource with the given metadata.  Width and height are used by the
    /// display, while size is useful for hypervisor mapping.
    pub fn new(
        resource_id: u32,
        width: u32,
        height: u32,
        format: Option<ResourceFormat>,
        size: u64,
    ) -> VirtioGpuResource {
        VirtioGpuResource {
            id: resource_id,
            width,
            height,
            scanouts: Default::default(),
            size,
            format,
            shmem_offset: None,
            rutabaga_external_mapping: false,
        }
    }
}

pub struct VirtioGpuScanout {
    resource_id: u32,
    /// limina tier-2: if `Some`, this scanout's resource is backed by a global IOSurface
    /// (venus SET_SCANOUT_BLOB) and `flush_resource` presents it zero-copy via
    /// `present_surface` instead of the readback + `present_frame` path.
    #[cfg(target_os = "macos")]
    iosurface_id: Option<u32>,
}

/// A host-side software 2D resource (limina patch).
///
/// libkrun normally routes `RESOURCE_CREATE_2D` through virglrenderer as a GL render
/// target — which has no host context on macOS, so creation fails and nothing ever
/// reaches the display. To give a working *software* scanout (the degraded-but-correct
/// baseline tier, e.g. fbcon, EFI GOP, simpledrm), limina shadows 2D resources entirely in
/// host CPU memory, never touching rutabaga:
///   CREATE_2D -> allocate `host`; ATTACH_BACKING -> remember the guest `backing`
///   iovecs; TRANSFER_TO_HOST_2D -> copy backing -> `host`; FLUSH -> hand `host` to the
///   display backend. No GL/Metal involved. The accelerated path (Venus/blob, 3D
///   resources) is untouched and still goes through rutabaga.
struct Sw2dResource {
    /// `width * height * BYTES_PER_PIXEL`, in the resource's pixel format. (Geometry +
    /// format live in the matching `resources` entry.)
    host: Vec<u8>,
    /// Guest backing as host pointers (from `sglist_to_rutabaga_iovecs`), valid for the
    /// lifetime of the guest memory mapping; only read on the GPU worker thread.
    backing: Vec<RutabagaIovec>,
}

impl Sw2dResource {
    /// Gather the guest backing into the host buffer (the guest holds the full current
    /// framebuffer in its backing, so copying it whole satisfies any transfer rect).
    fn copy_from_backing(&mut self) {
        let mut off = 0usize;
        for iov in &self.backing {
            if off >= self.host.len() {
                break;
            }
            let n = iov.len.min(self.host.len() - off);
            // SAFETY: `iov` is a host pointer/len pair derived from the guest memory
            // mapping (sglist_to_rutabaga_iovecs); `host` owns `off..off+n`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    iov.base as *const u8,
                    self.host.as_mut_ptr().add(off),
                    n,
                );
            }
            off += n;
        }
    }
}

pub struct VirtioGpu {
    /// The host 3D renderer. `None` in limina software-2D-only mode, where the device serves
    /// only the 2D scanout path (see [`Sw2dResource`]) and never initializes
    /// virglrenderer/rutabaga — so a GL-less host (e.g. macOS without a usable Metal/GL
    /// context) doesn't pay for, or hang on, renderer init. All renderer-backed commands
    /// (3D/blob/context/capset/fence) degrade to `ErrUnspec` in that mode; the guest sees a
    /// plain 2D virtio-gpu (no VIRGL feature, no capsets — see `Gpu`) and won't issue them.
    rutabaga: Option<Rutabaga>,
    resources: BTreeMap<u32, VirtioGpuResource>,
    /// limina software 2D resources, keyed by resource id (see [`Sw2dResource`]).
    sw2d: BTreeMap<u32, Sw2dResource>,
    fence_state: Arc<Mutex<FenceState>>,
    #[cfg(target_os = "macos")]
    map_sender: Sender<WorkerMessage>,
    scanouts: [Option<VirtioGpuScanout>; VIRTIO_GPU_MAX_SCANOUTS as usize],
    displays: Box<[DisplayInfo]>,
    display_backend: DisplayBackendInstance,
}

impl VirtioGpu {
    fn create_fence_handler(
        mem: GuestMemoryMmap,
        queue_ctl: Arc<Mutex<VirtQueue>>,
        fence_state: Arc<Mutex<FenceState>>,
        interrupt: InterruptTransport,
    ) -> RutabagaFenceHandler {
        RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
            debug!(
                "XXX - fence called: id={}, ring_idx={}",
                completed_fence.fence_id, completed_fence.ring_idx
            );

            let mut queue = queue_ctl.lock().unwrap();
            let mut fence_state = fence_state.lock().unwrap();
            let mut i = 0;

            let ring = match completed_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                0 => VirtioGpuRing::Global,
                _ => VirtioGpuRing::ContextSpecific {
                    ctx_id: completed_fence.ctx_id,
                    ring_idx: completed_fence.ring_idx,
                },
            };

            while i < fence_state.descs.len() {
                debug!("XXX - fence_id: {}", fence_state.descs[i].fence_id);
                if fence_state.descs[i].ring == ring
                    && fence_state.descs[i].fence_id <= completed_fence.fence_id
                {
                    let completed_desc = fence_state.descs.remove(i);
                    debug!(
                        "XXX - found fence: desc_index={}",
                        completed_desc.desc_index
                    );

                    if let Err(e) =
                        queue.add_used(&mem, completed_desc.desc_index, completed_desc.len)
                    {
                        error!("failed to add used elements to the queue: {e:?}");
                    }

                    interrupt.signal_used_queue();
                } else {
                    i += 1;
                }
            }
            // Update the last completed fence for this context.
            // Use max() to avoid a race where an out-of-order completion
            // (e.g., immediate-retire for fence N+1 followed by timeline
            // signal for fence N) would overwrite a higher fence_id with
            // a lower one, causing fence N+1 to be stuck forever.
            let entry = fence_state.completed_fences.entry(ring).or_insert(0);
            *entry = (*entry).max(completed_fence.fence_id);
        })
    }

    pub fn create_rutabaga(
        mem: GuestMemoryMmap,
        queue_ctl: Arc<Mutex<VirtQueue>>,
        interrupt: InterruptTransport,
        fence_state: Arc<Mutex<FenceState>>,
        virgl_flags: u32,
        export_table: Option<ExportTable>,
    ) -> Option<Rutabaga> {
        let xdg_runtime_dir = match env::var("XDG_RUNTIME_DIR") {
            Ok(dir) => dir,
            Err(_) => "/run/user/1000".to_string(),
        };
        let wayland_display = match env::var("WAYLAND_DISPLAY") {
            Ok(display) => display,
            Err(_) => "wayland-0".to_string(),
        };
        let path = PathBuf::from(format!("{xdg_runtime_dir}/{wayland_display}"));

        #[allow(unused_mut)]
        let mut rutabaga_channels: Vec<RutabagaChannel> = vec![RutabagaChannel {
            base_channel: path,
            channel_type: RUTABAGA_CHANNEL_TYPE_WAYLAND,
        }];

        #[cfg(target_os = "linux")]
        if let Ok(x_display) = env::var("DISPLAY")
            && let Some(x_display) = x_display.strip_prefix(":")
        {
            let x_path = PathBuf::from(format!("/tmp/.X11-unix/X{x_display}"));
            rutabaga_channels.push(RutabagaChannel {
                base_channel: x_path,
                channel_type: RUTABAGA_CHANNEL_TYPE_X11,
            });
        }
        #[cfg(target_os = "linux")]
        if let Ok(pw_sock_dir) = env::var("PIPEWIRE_RUNTIME_DIR")
            .or_else(|_| env::var("XDG_RUNTIME_DIR"))
            .or_else(|_| env::var("USERPROFILE"))
        {
            let name = env::var("PIPEWIRE_REMOTE").unwrap_or_else(|_| "pipewire-0".to_string());
            let mut pw_path = PathBuf::from(pw_sock_dir);
            pw_path.push(name);
            rutabaga_channels.push(RutabagaChannel {
                base_channel: pw_path,
                channel_type: RUTABAGA_CHANNEL_TYPE_PW,
            });
        }
        let rutabaga_channels_opt = Some(rutabaga_channels);

        let builder = RutabagaBuilder::new(
            rutabaga_gfx::RutabagaComponentType::VirglRenderer,
            virgl_flags,
            0,
        )
        .set_rutabaga_channels(rutabaga_channels_opt);
        let builder = if let Some(export_table) = export_table {
            builder.set_export_table(export_table)
        } else {
            builder
        };

        let fence =
            Self::create_fence_handler(mem, queue_ctl.clone(), fence_state.clone(), interrupt);
        match builder.clone().build(fence.clone(), None) {
            Ok(r) => Some(r),
            Err(e) => {
                warn!("create_rutabaga(virgl_flags={virgl_flags:#x}) build failed: {e:?}");
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mem: GuestMemoryMmap,
        queue_ctl: Arc<Mutex<VirtQueue>>,
        interrupt: InterruptTransport,
        virgl_flags: u32,
        software_2d: bool,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
        export_table: Option<ExportTable>,
        displays: Box<[DisplayInfo]>,
        display_backend: DisplayBackend,
    ) -> Self {
        let fence_state = Arc::new(Mutex::new(Default::default()));

        // limina software-2D-only mode: skip renderer init entirely (no virglrenderer/Metal).
        // Coexist mode (software_2d == false): try the (Venus) renderer for 3D while the
        // software-2D path keeps serving 2D/scanout. If the renderer fails to init, degrade
        // gracefully to software-2D only (rutabaga = None): 2D keeps working and the guest's
        // 3D commands return ErrUnspec, so Mesa falls back to llvmpipe rather than the worker
        // crashing. We deliberately do NOT fall back to a NO_VIRGL rutabaga — it can't serve
        // 2D either (CREATE_2D -> virgl GL render target, dead on macOS) and just wedges boot.
        let rutabaga = if software_2d {
            None
        } else {
            match Self::create_rutabaga(
                mem.clone(),
                queue_ctl.clone(),
                interrupt.clone(),
                fence_state.clone(),
                virgl_flags,
                export_table.clone(),
            ) {
                Some(rutabaga) => Some(rutabaga),
                None => {
                    warn!("virtio-gpu: renderer init failed; degrading to software-2D (no 3D)");
                    None
                }
            }
        };

        let display_backend = display_backend
            .create_instance()
            .expect("Failed to create display backend instance!");

        Self {
            rutabaga,
            resources: Default::default(),
            sw2d: Default::default(),
            fence_state,
            scanouts: Default::default(),
            displays,
            display_backend,
            #[cfg(target_os = "macos")]
            map_sender,
        }
    }

    // Non-public function -- no doc comment needed!
    fn result_from_query(&mut self, resource_id: u32) -> GpuResponse {
        let Some(rutabaga) = self.rutabaga.as_ref() else {
            return OkNoData;
        };
        match rutabaga.query(resource_id) {
            Ok(query) => {
                let mut plane_info = Vec::with_capacity(4);
                for plane_index in 0..4 {
                    plane_info.push(GpuResponsePlaneInfo {
                        stride: query.strides[plane_index],
                        offset: query.offsets[plane_index],
                    });
                }
                let format_modifier = query.modifier;
                OkResourcePlaneInfo {
                    format_modifier,
                    plane_info,
                }
            }
            Err(_) => OkNoData,
        }
    }

    pub fn force_ctx_0(&self) {
        // Called for every command; a no-op in software-2D-only mode (no rutabaga).
        if let Some(rutabaga) = self.rutabaga.as_ref() {
            rutabaga.force_ctx_0()
        }
    }

    /// Creates a software 2D resource (limina patch) — see [`Sw2dResource`]. Unlike the
    /// stock path (which maps CREATE_2D onto a virgl GL render target and fails on a
    /// GL-less host such as macOS), this allocates a host CPU buffer and never touches
    /// rutabaga. The matching metadata entry in `resources` carries the format/scanout
    /// bookkeeping that `set_scanout`/`flush_resource`/`unref_resource` rely on.
    pub fn resource_create_2d(
        &mut self,
        resource_id: u32,
        format: u32,
        width: u32,
        height: u32,
    ) -> VirtioGpuResult {
        let format = ResourceFormat::try_from(format).map_err(|()| {
            warn!("resource_create_2d: unsupported format {format} for resource {resource_id}");
            ErrUnspec
        })?;
        let len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|px| px.checked_mul(ResourceFormat::BYTES_PER_PIXEL))
            .ok_or(ErrUnspec)?;

        self.sw2d.insert(
            resource_id,
            Sw2dResource {
                host: vec![0u8; len],
                backing: Vec::new(),
            },
        );
        self.resources.insert(
            resource_id,
            VirtioGpuResource::new(resource_id, width, height, Some(format), 0),
        );
        Ok(OkNoData)
    }

    /// Creates a 3D resource with the given properties and resource_id.
    pub fn resource_create_3d(
        &mut self,
        resource_id: u32,
        resource_create_3d: ResourceCreate3D,
    ) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .resource_create_3d(resource_id, resource_create_3d)?;

        let format = ResourceFormat::try_from(resource_create_3d.format).ok();
        if format.is_none() {
            debug!(
                "Unknown format {} for resource {}",
                resource_create_3d.format, resource_id
            );
        }

        let resource = VirtioGpuResource::new(
            resource_id,
            resource_create_3d.width,
            resource_create_3d.height,
            format,
            0,
        );

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Releases guest kernel reference on the resource.
    pub fn unref_resource(&mut self, resource_id: u32) -> VirtioGpuResult {
        let resource = self
            .resources
            .remove(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        if resource.scanouts.has_any_enabled() {
            warn!(
                "The driver requested unref_resource, but resource {resource_id} has \
                     associated scanouts, refusing to delete the resource."
            );
            return Err(ErrUnspec);
        }

        // limina software 2D resources have no rutabaga state.
        if self.sw2d.remove(&resource_id).is_some() {
            return Ok(OkNoData);
        }

        if resource.rutabaga_external_mapping {
            self.rutabaga
                .as_mut()
                .ok_or(ErrUnspec)?
                .unmap(resource_id)?;
        }

        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .unref_resource(resource_id)?;
        Ok(OkNoData)
    }

    pub fn set_scanout(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
    ) -> VirtioGpuResult {
        let scanout = self
            .scanouts
            .get_mut(scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        // If a resource is already associated with this scanout, make sure to disable
        // this scanout for that resource
        if let Some(resource_id) = scanout.as_ref().map(|scanout| scanout.resource_id) {
            let resource = self
                .resources
                .get_mut(&resource_id)
                .ok_or(ErrInvalidResourceId)?;

            resource.scanouts.disable(scanout_id);
        }

        // Virtio spec: "The driver can use resource_id = 0 to disable a scanout."
        if resource_id == 0 {
            debug!("Disabling scanout {scanout_id:?}");
            *scanout = None;
            self.display_backend.disable_scanout(scanout_id)?;
            return Ok(OkNoData);
        }

        // Enable the scanout
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;
        resource.scanouts.enable(scanout_id);

        let Some(format) = resource.format else {
            warn!("Cannot use resource {resource_id} with unknown format for scanout");
            return Err(ErrUnspec);
        };

        let display_info = self
            .displays
            .get(scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        self.display_backend.configure_scanout(
            scanout_id,
            display_info.width,
            display_info.height,
            width,
            height,
            format,
        )?;

        *scanout = Some(VirtioGpuScanout {
            resource_id,
            #[cfg(target_os = "macos")]
            iosurface_id: None,
        });
        Ok(OkNoData)
    }

    /// limina tier-2: VIRTIO_GPU_CMD_SET_SCANOUT_BLOB. The guest (mutter on venus) scans out a
    /// blob resource that is its KMS framebuffer; on macOS that blob's bound VkImage is backed
    /// by a global IOSurface (vkr fix A + the bind linkage), which we present zero-copy.
    ///
    /// Mirrors `set_scanout`, but the format/size come from the command (a blob has no 2D
    /// format of its own) and we resolve + remember the resource's IOSurface id so
    /// `flush_resource` can `present_surface` it without a readback. If the resource is not
    /// IOSurface-backed (e.g. a stock guest), `iosurface_id` stays `None` and flush falls back
    /// to the readback path.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    pub fn set_scanout_blob(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
        format: u32,
    ) -> VirtioGpuResult {
        let scanout = self
            .scanouts
            .get_mut(scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        // Disable this scanout for any resource currently bound to it.
        if let Some(prev) = scanout.as_ref().map(|s| s.resource_id) {
            if let Some(resource) = self.resources.get_mut(&prev) {
                resource.scanouts.disable(scanout_id);
            }
        }

        // resource_id == 0 disables the scanout (virtio spec).
        if resource_id == 0 {
            debug!("Disabling scanout {scanout_id:?} (blob)");
            *scanout = None;
            self.display_backend.disable_scanout(scanout_id)?;
            return Ok(OkNoData);
        }

        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;
        resource.scanouts.enable(scanout_id);
        resource.width = width;
        resource.height = height;

        let res_format = ResourceFormat::try_from(format).unwrap_or(ResourceFormat::BGRA);
        resource.format = Some(res_format);

        let display_info = self
            .displays
            .get(scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        self.display_backend.configure_scanout(
            scanout_id,
            display_info.width,
            display_info.height,
            width,
            height,
            res_format,
        )?;

        // Resolve the resource to its backing IOSurface (0/err -> not IOSurface-backed).
        let iosurface_id = self
            .rutabaga
            .as_ref()
            .and_then(|r| r.iosurface_id(resource_id).ok())
            .filter(|&id| id != 0);
        if let Some(id) = iosurface_id {
            log::info!(
                "SET_SCANOUT_BLOB scanout={scanout_id} res={resource_id} -> IOSurface {id} (zero-copy)"
            );
        } else {
            log::warn!(
                "SET_SCANOUT_BLOB scanout={scanout_id} res={resource_id} not IOSurface-backed; using readback"
            );
        }

        *scanout = Some(VirtioGpuScanout {
            resource_id,
            iosurface_id,
        });
        Ok(OkNoData)
    }

    fn read_2d_resource(
        rutabaga: &mut Rutabaga,
        resource: VirtioGpuResource,
        output: &mut [u8],
    ) -> VirtioGpuResult {
        let transfer = Transfer3D {
            x: 0,
            y: 0,
            z: 0,
            w: resource.width,
            h: resource.height,
            d: 1,
            level: 0,
            stride: resource.width * ResourceFormat::BYTES_PER_PIXEL as u32,
            layer_stride: 0,
            offset: 0,
        };

        if let Err(e) =
            rutabaga.transfer_read(0, resource.id, transfer, Some(IoSliceMut::new(output)))
        {
            // A blob / 3D (Venus) scanout resource has no 2D readback path -> EINVAL.
            // Never panic the GPU worker (that wedges the whole guest); report the failure
            // so the caller logs it and returns an error response for the flush.
            log::warn!(
                "transfer_read failed for scanout resource {} (blob/3D, no 2D readback): {e}",
                resource.id
            );
            return Err(ErrUnspec);
        }

        Ok(OkNoData)
    }

    /// If the resource is the scanout resource, flush it to the display.
    pub fn flush_resource(&mut self, resource_id: u32, rect: Rect) -> VirtioGpuResult {
        if resource_id == 0 {
            return Ok(OkNoData);
        }

        let resource = *self
            .resources
            .get(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        for scanout_id in resource.scanouts.iter_enabled() {
            // limina tier-2: an IOSurface-backed SET_SCANOUT_BLOB scanout is presented zero-copy
            // (venus already rendered into the IOSurface) — no alloc_frame, no readback.
            #[cfg(target_os = "macos")]
            if let Some(iosurface_id) = self
                .scanouts
                .get(scanout_id as usize)
                .and_then(|s| s.as_ref())
                .and_then(|s| s.iosurface_id)
            {
                match self
                    .display_backend
                    .present_surface(scanout_id, iosurface_id, Some(&rect))
                {
                    Ok(()) => continue,
                    Err(DisplayBackendError::MethodNotSupported) => {
                        // Backend has no zero-copy path (e.g. headless capture); fall through
                        // to the readback path below.
                    }
                    Err(e) => {
                        log::error!("present_surface failed for scanout {scanout_id}: {e}");
                        return Err(ErrUnspec);
                    }
                }
            }

            let (frame_id, buffer) = self.display_backend.alloc_frame(scanout_id)?;
            // limina software 2D: the pixels already live in the host buffer; copy them out.
            // Otherwise fall back to the rutabaga readback path (3D/Venus resources).
            if let Some(sw) = self.sw2d.get(&resource_id) {
                let n = buffer.len().min(sw.host.len());
                buffer[..n].copy_from_slice(&sw.host[..n]);
            } else if let Some(rutabaga) = self.rutabaga.as_mut() {
                if let Err(e) = Self::read_2d_resource(rutabaga, resource, buffer) {
                    log::error!(
                        "Failed to read resource {resource_id} for scanout {scanout_id}: {e}"
                    );
                    return Err(ErrUnspec);
                }
            } else {
                // No software-2D buffer and no renderer: nothing to present.
                return Err(ErrUnspec);
            }
            self.display_backend
                .present_frame(scanout_id, frame_id, Some(&rect))?
        }

        #[cfg(windows)]
        if let Some(rutabaga) = self.rutabaga.as_mut() {
            match rutabaga.resource_flush(resource_id) {
                Ok(_) => return Ok(OkNoData),
                Err(RutabagaError::Unsupported) => {}
                Err(e) => return Err(ErrRutabaga(e)),
            }
        }

        Ok(OkNoData)
    }

    /// limina: render the guest hardware cursor as a host overlay (`VIRTIO_GPU_CMD_UPDATE_CURSOR`).
    ///
    /// The cursor image is an ordinary 2D resource (`CREATE_2D` + `TRANSFER_TO_HOST_2D`), so its
    /// pixels already live in the software-2D host buffer. `resource_id == 0` hides the cursor
    /// (virtio-gpu spec). The display backend draws it as an overlay — never into the scanout —
    /// so cursor motion never re-enters the framebuffer present path. A backend without cursor
    /// support (headless capture, stock GTK) returns `MethodNotSupported`, which we treat as a
    /// no-op: the guest keeps whatever software-cursor fallback it had.
    pub fn update_cursor(
        &mut self,
        resource_id: u32,
        hot_x: u32,
        hot_y: u32,
        x: u32,
        y: u32,
    ) -> VirtioGpuResult {
        if resource_id == 0 {
            Self::cursor_ok(self.display_backend.set_cursor(
                0,
                0,
                0,
                0,
                ResourceFormat::BGRA,
                &[],
            ))?;
            return Ok(OkNoData);
        }

        let resource = *self
            .resources
            .get(&resource_id)
            .ok_or(ErrInvalidResourceId)?;
        let format = resource.format.unwrap_or(ResourceFormat::BGRA);
        // The guest kernel creates ALL dumb buffers as XRGB (virtgpu_gem.c hardcodes
        // DRM_FORMAT_HOST_XRGB8888), but cursor images carry real alpha in those X bytes —
        // virtio-gpu treats cursor data as ARGB regardless (QEMU does the same). Promote the
        // X formats to their alpha-carrying counterparts so the overlay keeps the transparent
        // surround instead of compositing an opaque black rectangle around the cursor.
        let format = match format {
            ResourceFormat::BGRX => ResourceFormat::BGRA,
            ResourceFormat::XRGB => ResourceFormat::ARGB,
            ResourceFormat::RGBX => ResourceFormat::RGBA,
            ResourceFormat::XBGR => ResourceFormat::ABGR,
            f => f,
        };
        // Clone the (tiny, ~64x64) cursor pixels so we don't hold a borrow of `self` across the
        // &mut self backend call.
        let Some(pixels) = self.sw2d.get(&resource_id).map(|sw| sw.host.clone()) else {
            warn!("update_cursor: resource {resource_id} has no software-2D pixels");
            return Err(ErrUnspec);
        };
        Self::cursor_ok(self.display_backend.set_cursor(
            resource.width,
            resource.height,
            hot_x,
            hot_y,
            format,
            &pixels,
        ))?;
        Self::cursor_ok(self.display_backend.move_cursor(x, y))?;
        Ok(OkNoData)
    }

    /// limina: reposition the host cursor overlay (`VIRTIO_GPU_CMD_MOVE_CURSOR`).
    pub fn move_cursor(&mut self, x: u32, y: u32) -> VirtioGpuResult {
        Self::cursor_ok(self.display_backend.move_cursor(x, y))?;
        Ok(OkNoData)
    }

    /// Map a cursor backend result to a GPU response, treating `MethodNotSupported` as success
    /// (a backend without a cursor overlay simply ignores cursor commands — the queue still
    /// drains so the guest never stalls).
    fn cursor_ok(r: std::result::Result<(), DisplayBackendError>) -> VirtioGpuResult {
        match r {
            Ok(()) | Err(DisplayBackendError::MethodNotSupported) => Ok(OkNoData),
            Err(e) => {
                warn!("cursor backend error: {e}");
                Err(ErrUnspec)
            }
        }
    }

    pub fn display_info(&self) -> VirtioGpuResult {
        let display_info = self
            .displays
            .iter()
            .map(|d| (d.width, d.height, true))
            .collect();

        Ok(OkDisplayInfo(display_info))
    }

    pub fn get_edid(&self, scanout_id: u32) -> VirtioGpuResult {
        let display = self
            .displays
            .get(scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        Ok(OkEdid(display.edid_bytes()))
    }

    /// Copies data to host resource from the attached iovecs. Can also be used to flush caches.
    pub fn transfer_write(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        transfer: Transfer3D,
    ) -> VirtioGpuResult {
        // limina software 2D: copy the guest backing into our host buffer.
        if let Some(sw) = self.sw2d.get_mut(&resource_id) {
            sw.copy_from_backing();
            return Ok(OkNoData);
        }
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .transfer_write(ctx_id, resource_id, transfer)?;
        Ok(OkNoData)
    }

    /// Copies data from the host resource to:
    ///    1) To the optional volatile slice
    ///    2) To the host resource's attached iovecs
    ///
    /// Can also be used to invalidate caches.
    pub fn transfer_read(
        &mut self,
        _ctx_id: u32,
        _resource_id: u32,
        _transfer: Transfer3D,
        _buf: Option<VolatileSlice>,
    ) -> VirtioGpuResult {
        panic!("virtio_gpu: transfer_read unimplemented");
    }

    /// Attaches backing memory to the given resource, represented by a `Vec` of `(address, size)`
    /// tuples in the guest's physical address space. Converts to RutabagaIovec from the memory
    /// mapping.
    pub fn attach_backing(
        &mut self,
        resource_id: u32,
        mem: &GuestMemoryMmap,
        vecs: Vec<(GuestAddress, usize)>,
    ) -> VirtioGpuResult {
        let rutabaga_iovecs = sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?;
        // limina software 2D: keep the backing host pointers; don't involve rutabaga.
        if let Some(sw) = self.sw2d.get_mut(&resource_id) {
            sw.backing = rutabaga_iovecs;
            return Ok(OkNoData);
        }
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .attach_backing(resource_id, rutabaga_iovecs)?;
        Ok(OkNoData)
    }

    /// Detaches any previously attached iovecs from the resource.
    pub fn detach_backing(&mut self, resource_id: u32) -> VirtioGpuResult {
        // limina software 2D: drop the backing pointers.
        if let Some(sw) = self.sw2d.get_mut(&resource_id) {
            sw.backing.clear();
            return Ok(OkNoData);
        }
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .detach_backing(resource_id)?;
        Ok(OkNoData)
    }

    /// Returns a uuid for the resource.
    pub fn resource_assign_uuid(&self, resource_id: u32) -> VirtioGpuResult {
        if !self.resources.contains_key(&resource_id) {
            return Err(ErrInvalidResourceId);
        }

        // TODO(stevensd): use real uuids once the virtio wayland protocol is updated to
        // handle more than 32 bits. For now, the virtwl driver knows that the uuid is
        // actually just the resource id.
        let mut uuid: [u8; 16] = [0; 16];
        for (idx, byte) in resource_id.to_be_bytes().iter().enumerate() {
            uuid[12 + idx] = *byte;
        }
        Ok(OkResourceUuid { uuid })
    }

    /// Gets rutabaga's capset information associated with `index`.
    pub fn get_capset_info(&self, index: u32) -> VirtioGpuResult {
        let (capset_id, version, size) = self
            .rutabaga
            .as_ref()
            .ok_or(ErrUnspec)?
            .get_capset_info(index)?;
        Ok(OkCapsetInfo {
            capset_id,
            version,
            size,
        })
    }

    /// Gets a capset from rutabaga.
    pub fn get_capset(&self, capset_id: u32, version: u32) -> VirtioGpuResult {
        let capset = self
            .rutabaga
            .as_ref()
            .ok_or(ErrUnspec)?
            .get_capset(capset_id, version)?;
        Ok(OkCapset(capset))
    }

    /// Creates a rutabaga context.
    pub fn create_context(
        &mut self,
        ctx_id: u32,
        context_init: u32,
        context_name: Option<&str>,
    ) -> VirtioGpuResult {
        self.rutabaga.as_mut().ok_or(ErrUnspec)?.create_context(
            ctx_id,
            context_init,
            context_name,
        )?;
        Ok(OkNoData)
    }

    /// Destroys a rutabaga context.
    pub fn destroy_context(&mut self, ctx_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .destroy_context(ctx_id)?;
        Ok(OkNoData)
    }

    /// Attaches a resource to a rutabaga context.
    pub fn context_attach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .context_attach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Detaches a resource from a rutabaga context.
    pub fn context_detach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .context_detach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Submits a command buffer to a rutabaga context.
    pub fn submit_command(
        &mut self,
        ctx_id: u32,
        commands: &mut [u8],
        fence_ids: &[u64],
    ) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .submit_command(ctx_id, commands, fence_ids)?;
        Ok(OkNoData)
    }

    /// Creates a fence with the RutabagaFence that can be used to determine when the previous
    /// command completed.
    pub fn create_fence(&mut self, rutabaga_fence: RutabagaFence) -> VirtioGpuResult {
        // Route the fence by ring. Software-2D (Global-ring) commands finish synchronously
        // before their response is encoded, so the fence is already signaled by the time we
        // get here: record it as completed up-front and let process_fence() retire the
        // descriptor immediately instead of parking it forever (which would hang any guest
        // that fences a 2D command, e.g. GTK4, or the EDK2 firmware GOP).
        //
        // Only context-specific fences belong to a real 3D context and go to rutabaga. In the
        // coexist device (software-2D 2D + VENUS|NO_VIRGL 3D) a venus rutabaga is present but
        // cannot fence the Global ring (ctx 0 isn't a venus context) — routing a 2D fence there
        // fails with ComponentError and wedges the firmware. So: Global ring -> sync, always;
        // context ring -> rutabaga (falling back to sync if somehow there is no renderer).
        let context_ring = rutabaga_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX != 0;
        match self.rutabaga.as_mut() {
            Some(rutabaga) if context_ring => rutabaga.create_fence(rutabaga_fence)?,
            _ => mark_fence_completed_sync(&self.fence_state, &rutabaga_fence),
        }
        Ok(OkNoData)
    }

    pub fn process_fence(
        &mut self,
        ring: VirtioGpuRing,
        fence_id: u64,
        desc_index: u16,
        len: u32,
    ) -> bool {
        // In case the fence is signaled immediately after creation, don't add a return
        // FenceDescriptor.
        let mut fence_state = self.fence_state.lock().unwrap();
        if fence_id > *fence_state.completed_fences.get(&ring).unwrap_or(&0) {
            fence_state.descs.push(FenceDescriptor {
                ring,
                fence_id,
                desc_index,
                len,
            });

            false
        } else {
            true
        }
    }

    /// Creates a blob resource using rutabaga.
    pub fn resource_create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        vecs: Vec<(GuestAddress, usize)>,
        mem: &GuestMemoryMmap,
    ) -> VirtioGpuResult {
        let mut rutabaga_iovecs = None;

        if resource_create_blob.blob_flags & VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE != 0 {
            panic!("GUEST_HANDLE unimplemented");
        } else if resource_create_blob.blob_mem != VIRTIO_GPU_BLOB_MEM_HOST3D {
            rutabaga_iovecs =
                Some(sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?);
        }

        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .resource_create_blob(
                ctx_id,
                resource_id,
                resource_create_blob,
                rutabaga_iovecs,
                None,
            )?;

        let resource = VirtioGpuResource::new(resource_id, 0, 0, None, resource_create_blob.size);

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Uses the hypervisor to map the rutabaga blob resource.
    ///
    /// When sandboxing is disabled, external_blob is unset and opaque fds are mapped by
    /// rutabaga as ExternalMapping.
    /// When sandboxing is enabled, external_blob is set and opaque fds must be mapped in the
    /// hypervisor process by Vulkano using metadata provided by Rutabaga::vulkan_info().
    #[cfg(all(not(feature = "virgl_resource_map2"), target_os = "linux"))]
    pub fn resource_map_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
        offset: u64,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let rutabaga = self.rutabaga.as_ref().ok_or(ErrUnspec)?;
        let map_info = rutabaga.map_info(resource_id).map_err(|_| ErrUnspec)?;

        if let Ok(export) = rutabaga.export_blob(resource_id) {
            if export.handle_type != RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD {
                let prot = match map_info & RUTABAGA_MAP_ACCESS_MASK {
                    RUTABAGA_MAP_ACCESS_READ => libc::PROT_READ,
                    RUTABAGA_MAP_ACCESS_WRITE => libc::PROT_WRITE,
                    RUTABAGA_MAP_ACCESS_RW => libc::PROT_READ | libc::PROT_WRITE,
                    _ => return Err(ErrUnspec),
                };

                let addr = checked_blob_map_addr(
                    shm_region.host_addr,
                    offset,
                    resource.size,
                    shm_region.size as u64,
                )
                .ok_or(ErrUnspec)?;
                debug!(
                    "mapping: host_addr={:x}, addr={:x}, size={}",
                    shm_region.host_addr, addr, resource.size
                );
                let ret = unsafe {
                    libc::mmap(
                        addr as *mut libc::c_void,
                        resource.size as usize,
                        prot,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        export.os_handle.as_raw_fd(),
                        0 as libc::off_t,
                    )
                };
                if ret == libc::MAP_FAILED {
                    return Err(ErrUnspec);
                }
            } else {
                return Err(ErrUnspec);
            }
        } else {
            return Err(ErrUnspec);
        }

        resource.shmem_offset = Some(offset);
        // Access flags not a part of the virtio-gpu spec.
        Ok(OkMapInfo {
            map_info: map_info & RUTABAGA_MAP_CACHE_MASK,
        })
    }
    #[cfg(all(feature = "virgl_resource_map2", target_os = "linux"))]
    pub fn resource_map_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
        offset: u64,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let map_info = self
            .rutabaga
            .as_ref()
            .ok_or(ErrUnspec)?
            .map_info(resource_id)
            .map_err(|_| ErrUnspec)?;

        let prot = match map_info & RUTABAGA_MAP_ACCESS_MASK {
            RUTABAGA_MAP_ACCESS_READ => libc::PROT_READ,
            RUTABAGA_MAP_ACCESS_WRITE => libc::PROT_WRITE,
            RUTABAGA_MAP_ACCESS_RW => libc::PROT_READ | libc::PROT_WRITE,
            _ => return Err(ErrUnspec),
        };

        let addr = checked_blob_map_addr(
            shm_region.host_addr,
            offset,
            resource.size,
            shm_region.size as u64,
        )
        .ok_or(ErrUnspec)?;

        if let Ok(export) = self
            .rutabaga
            .as_ref()
            .ok_or(ErrUnspec)?
            .export_blob(resource_id)
        {
            // SHM and DMABUF are both regular host fds whose pages can be exposed
            // to the guest by mmap'ing them directly into the virtio shm region.
            // For SHM (memfd) this has always worked. For DMABUF it had been
            // delegated to virgl_renderer_resource_map2, which only handles
            // virglrenderer-allocated GPU memory and silently no-ops for external
            // dma-bufs — leaving the guest blob backed by zero pages. That broke
            // muvm camera capture, where the v4l2 source exports kernel buffers
            // via VIDIOC_EXPBUF as dma-bufs, the muvm bridge forwards the fd
            // across SCM_RIGHTS, libkrun classifies it as DMABUF, and the guest's
            // CREATE_BLOB allocates a host-backed-by-nothing blob. Mapping the
            // dma-buf fd directly here gives the guest real, live pages.
            if export.handle_type == RUTABAGA_MEM_HANDLE_TYPE_SHM
                || export.handle_type == RUTABAGA_MEM_HANDLE_TYPE_DMABUF
            {
                let ret = unsafe {
                    libc::mmap(
                        addr as *mut libc::c_void,
                        resource.size as usize,
                        prot,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        export.os_handle.as_raw_fd(),
                        0 as libc::off_t,
                    )
                };
                if ret == libc::MAP_FAILED {
                    error!(
                        "failed to mmap resource in shm region (handle_type={:#x})",
                        export.handle_type
                    );
                    return Err(ErrUnspec);
                }
            } else {
                self.rutabaga.as_mut().ok_or(ErrUnspec)?.resource_map(
                    resource_id,
                    addr,
                    resource.size,
                    prot,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                )?;
            }
        }

        resource.shmem_offset = Some(offset);
        // Access flags not a part of the virtio-gpu spec.
        Ok(OkMapInfo {
            map_info: map_info & RUTABAGA_MAP_CACHE_MASK,
        })
    }
    #[cfg(target_os = "macos")]
    pub fn resource_map_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
        offset: u64,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let rutabaga = self.rutabaga.as_mut().ok_or(ErrUnspec)?;
        let map_info = rutabaga.map_info(resource_id).map_err(|_| ErrUnspec)?;
        // limina: `map_ptr` maps the (venus host-visible) blob and returns the host pointer; with
        // upstream virglrenderer 1.3.0 this goes through `virgl_renderer_resource_map`. We then
        // hv_vm_map that pointer into the guest's SHM window. The old slp `0.10.4e-krunkit` bottle
        // gated this on `export_blob().handle_type == APPLE` (a krunkit-only blob fd type); upstream
        // 1.3.0 has no APPLE fd type (only DMABUF/OPAQUE/SHM), so we no longer require it — the
        // `map_ptr` call itself is the gate (it errors for a non-mappable resource).
        let map_ptr = rutabaga.map_ptr(resource_id).map_err(|_| ErrUnspec)?;

        let guest_addr = checked_blob_map_addr(
            shm_region.guest_addr,
            offset,
            resource.size,
            shm_region.size as u64,
        )
        .ok_or(ErrUnspec)?;
        debug!(
            "mapping: map_ptr={:x}, guest_addr={:x}, size={}",
            map_ptr, guest_addr, resource.size
        );

        let (reply_sender, reply_receiver) = unbounded();
        self.map_sender
            .send(WorkerMessage::GpuAddMapping(
                reply_sender,
                map_ptr,
                guest_addr,
                resource.size,
            ))
            .unwrap();
        if !reply_receiver.recv().unwrap() {
            return Err(ErrUnspec);
        }

        resource.shmem_offset = Some(offset);
        // Access flags not a part of the virtio-gpu spec.
        Ok(OkMapInfo {
            map_info: map_info & RUTABAGA_MAP_CACHE_MASK,
        })
    }

    /// Uses the hypervisor to unmap the blob resource.
    #[cfg(target_os = "linux")]
    pub fn resource_unmap_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let shmem_offset = resource.shmem_offset.ok_or(ErrUnspec)?;

        let addr = shm_region.host_addr + shmem_offset;

        let ret = unsafe {
            libc::mmap(
                addr as *mut libc::c_void,
                resource.size as usize,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0_i64,
            )
        };
        if ret == libc::MAP_FAILED {
            error!("failed to unmap blob resource");
            return Err(ErrUnspec);
        }

        resource.shmem_offset = None;

        Ok(OkNoData)
    }
    #[cfg(target_os = "macos")]
    pub fn resource_unmap_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        debug!("resource_unmap_blob");
        let shmem_offset = resource.shmem_offset.ok_or(ErrUnspec)?;

        let guest_addr = shm_region.guest_addr + shmem_offset;
        debug!(
            "unmapping: guest_addr={:x}, size={}",
            guest_addr, resource.size
        );

        let (reply_sender, reply_receiver) = unbounded();
        self.map_sender
            .send(WorkerMessage::GpuRemoveMapping(
                reply_sender,
                guest_addr,
                resource.size,
            ))
            .unwrap();
        if !reply_receiver.recv().unwrap() {
            return Err(ErrUnspec);
        }

        resource.shmem_offset = None;

        Ok(OkNoData)
    }
}

// A guest-controlled `offset` that wraps `offset + size` or `base + offset` would
// otherwise pass the size guard and place the mmap(MAP_FIXED) out of bounds.
fn checked_blob_map_addr(base: u64, offset: u64, size: u64, shm_size: u64) -> Option<u64> {
    if offset.checked_add(size)? > shm_size {
        return None;
    }
    base.checked_add(offset)
}

#[cfg(test)]
mod test {
    use crate::virtio::gpu::protocol::VIRTIO_GPU_MAX_SCANOUTS;

    // Software-2D mode (rutabaga == None) has no async fence handler. A fence the
    // guest requests on a 2D command must be retired synchronously, otherwise the
    // response is parked forever and the guest hangs (observed: GTK4/nautilus on
    // the tier-1 software-2D scanout). This guards mark_fence_completed_sync().
    #[test]
    fn test_software_2d_fence_retires_synchronously() {
        use super::{mark_fence_completed_sync, FenceState, RutabagaFence, VirtioGpuRing};
        use std::sync::Mutex;

        let fence_state = Mutex::new(FenceState::default());
        let fence = RutabagaFence {
            flags: 0, // VIRTIO_GPU_FLAG_INFO_RING_IDX clear -> Global ring
            fence_id: 1,
            ctx_id: 0,
            ring_idx: 0,
        };

        // Before: nothing completed, so process_fence() would defer (id > 0) and
        // park the descriptor with no handler to ever wake it.
        {
            let st = fence_state.lock().unwrap();
            let completed = *st.completed_fences.get(&VirtioGpuRing::Global).unwrap_or(&0);
            assert!(fence.fence_id > completed, "precondition: fence not yet complete");
        }

        mark_fence_completed_sync(&fence_state, &fence);

        // After: the watermark covers the fence, so process_fence() retires it now.
        let st = fence_state.lock().unwrap();
        let completed = *st.completed_fences.get(&VirtioGpuRing::Global).unwrap_or(&0);
        assert!(
            fence.fence_id <= completed,
            "software-2D fence must be marked completed synchronously"
        );
    }

    #[test]
    fn checked_blob_map_addr_rejects_out_of_range_and_wrapping_offsets() {
        use super::checked_blob_map_addr;

        let base = 0x1_0000_u64;
        let shm = 0x1_0000_u64;

        assert_eq!(
            checked_blob_map_addr(base, 0x1000, 0x2000, shm),
            Some(base + 0x1000)
        );
        assert_eq!(checked_blob_map_addr(base, 0, shm, shm), Some(base));
        assert!(checked_blob_map_addr(base, shm, 1, shm).is_none());

        let size = 0x1000_u64;
        let wrapping_offset = u64::MAX - size + 1;
        assert!(wrapping_offset.wrapping_add(size) <= shm);
        assert!(checked_blob_map_addr(base, wrapping_offset, size, shm).is_none());

        assert!(checked_blob_map_addr(u64::MAX - 5, 10, 0, u64::MAX).is_none());
    }

    #[test]
    fn test_virtio_gpu_associated_scanouts() {
        use super::AssociatedScanouts;

        let mut scanouts = AssociatedScanouts::default();

        assert!(!scanouts.has_any_enabled());
        assert_eq!(scanouts.iter_enabled().next(), None);

        scanouts.enable(1);
        assert!(scanouts.has_any_enabled());
        scanouts.disable(1);
        assert!(!scanouts.has_any_enabled());

        (0..VIRTIO_GPU_MAX_SCANOUTS).for_each(|scanout| scanouts.enable(scanout));
        assert!(scanouts.has_any_enabled());
        assert_eq!(
            scanouts.iter_enabled().collect::<Vec<u32>>(),
            (0..VIRTIO_GPU_MAX_SCANOUTS).collect::<Vec<u32>>()
        );

        (0..VIRTIO_GPU_MAX_SCANOUTS)
            .filter(|&i| i % 2 == 0)
            .for_each(|scanout| scanouts.disable(scanout));
        assert_eq!(
            scanouts.iter_enabled().collect::<Vec<u32>>(),
            (1..VIRTIO_GPU_MAX_SCANOUTS)
                .step_by(2)
                .collect::<Vec<u32>>()
        );

        (0..VIRTIO_GPU_MAX_SCANOUTS)
            .filter(|&i| i % 2 != 0)
            .for_each(|scanout| scanouts.disable(scanout));
        assert!(!scanouts.has_any_enabled());
    }
}
