use std::collections::BTreeMap;
use std::env;
use std::io::IoSliceMut;
use std::os::fd::{AsRawFd, FromRawFd};
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
    RUTABAGA_CHANNEL_TYPE_WAYLAND, RUTABAGA_FLAG_FENCE, RUTABAGA_FLAG_INFO_RING_IDX,
    RUTABAGA_MAP_CACHE_MASK, ResourceCreate3D, ResourceCreateBlob,
    Rutabaga, RutabagaBuilder, RutabagaChannel, RutabagaFence, RutabagaFenceHandler, RutabagaIovec,
    Transfer3D,
};
#[cfg(target_os = "macos")]
use utils::worker_message::WorkerMessage;
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap, VolatileSlice};

use super::journal::{GpuJournal, GpuJournalOp};
use super::trace::GpuTraceStats;
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
    /// limina M9.3 trace: when the descriptor parked (age of an outstanding fence).
    created_at: std::time::Instant,
}

#[derive(Default)]
pub struct FenceState {
    descs: Vec<FenceDescriptor>,
    completed_fences: BTreeMap<VirtioGpuRing, u64>,
}

impl FenceState {
    /// limina M9.3 trace: one-line summary of the outstanding (requested, never
    /// signaled) fences — count, oldest age, and up to 8 entries with ring + age.
    /// A post-restore wedge shows here as entries whose ages only ever climb.
    pub(crate) fn outstanding_summary(&self, now: std::time::Instant) -> String {
        if self.descs.is_empty() {
            return "0".to_string();
        }
        let mut out = format!("{} [", self.descs.len());
        for (i, desc) in self.descs.iter().take(8).enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let ring = match desc.ring {
                VirtioGpuRing::Global => "global".to_string(),
                VirtioGpuRing::ContextSpecific { ctx_id, ring_idx } => {
                    format!("ctx{ctx_id}/r{ring_idx}")
                }
            };
            out.push_str(&format!(
                "{ring}#{}:{}ms",
                desc.fence_id,
                now.duration_since(desc.created_at).as_millis()
            ));
        }
        if self.descs.len() > 8 {
            out.push_str(" …");
        }
        out.push(']');
        out
    }
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
    /// limina (#8): the context that created this resource (blob resources only;
    /// 0 = none). A scanout flush injects its present fence on this context.
    ctx_id: u32,
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
            ctx_id: 0,
        }
    }
}

/// limina fence-accurate presents (#8/#31): state for deferring zero-copy scanout
/// presents until the guest's GPU work completes. A flush parks the frame here and
/// injects a fence on the context's reserved present ring (vkr ring 63); the fence
/// handler pushes the retired cookie + kicks `event`; the worker drains it on its
/// epoll and presents. Parked frames are keyed by cookie and EVERY one presents on
/// its own retirement (retirement is in flush order per context, so presents stay
/// chronological — merely latency-shifted). Never drop a parked frame in favor of
/// a newer one: if GPU latency exceeds the flip interval, dropping would starve
/// the display forever.
struct PresentFenceState {
    /// Cookies retired by the fence handler (vkr sync threads) awaiting present.
    retired: Arc<Mutex<Vec<u64>>>,
    /// Wakes the worker epoll when `retired` gains entries.
    event: utils::eventfd::EventFd,
    next_cookie: u64,
    /// cookie -> parked flush, bounded by in-flight fences (every cookie retires).
    parked: BTreeMap<u64, ParkedFlush>,

    /// limina (#8 half 2) — guest flush-fence holds. The patched guest kernel fences
    /// blob-scanout RESOURCE_FLUSH and dma_fence_wait()s it before completing the
    /// commit (whose fake flip event paces mutter). We hold the flush's virtio
    /// fence descriptor until the parked frame has been shown AND the compositor
    /// latch window passed, then retire it via the regular fence handler — giving
    /// the guest honest flip pacing and race-free buffer reuse.
    ///
    /// Cookies parked by the currently-executing flush command (consumed by the
    /// flush's trailing FLAG_FENCE, cleared on every flush).
    flush_parked_cookies: Vec<u64>,
    /// Held guest fences, each waiting for its flush's cookies to present (+latch).
    guest_holds: Vec<GuestFlushHold>,
    /// Completes held fences after a delay (clone of the rutabaga fence handler —
    /// it owns the desc-retirement bookkeeping). With shown acks this is only the
    /// safety fallback; completion is idempotent.
    latch_tx: std::sync::mpsc::Sender<(std::time::Instant, RutabagaFence)>,
    /// Present -> guest-visible-completion delay when there is NO ack channel:
    /// covers the supervisor's display timer tick plus Core Animation's latch.
    latch_delay: std::time::Duration,

    /// #8 leg 2 — supervisor "shown <id>" acks (the truthful CA latch boundary).
    /// True when the ack reader thread is running; holds then complete on acks
    /// (with `latch_tx` as a generous fallback) instead of the open-loop delay.
    ack_active: bool,
    /// Surface ids acked by the supervisor (reader thread -> worker via `event`).
    shown: Arc<Mutex<Vec<u32>>>,
    /// Presented-but-not-yet-acked frames in apply order: (iosurface_id, cookie).
    /// An ack for id X confirms every entry up to and including the first X —
    /// frames applied before X are certainly off glass once X latched.
    awaiting_shown: std::collections::VecDeque<(u32, u64)>,
    /// Completes ack-confirmed holds immediately (same handler the latch thread uses).
    guest_fence_handler: RutabagaFenceHandler,
}

struct GuestFlushHold {
    fence: RutabagaFence,
    /// Cookies whose frames haven't presented yet.
    unpresented: std::collections::BTreeSet<u64>,
    /// Cookies presented but not yet latch-confirmed (ack or no-ack policy).
    unconfirmed: std::collections::BTreeSet<u64>,
    /// Safety fallback deadline (ack mode), set once all frames presented:
    /// completion is idempotent, so the fallback and the ack path may both fire.
    /// Past the deadline the hold is dropped (the latch thread completed it).
    fallback_at: Option<std::time::Instant>,
    /// Hard ceiling: the latch thread completes the fence at creation+500ms no
    /// matter what (scheduled at creation; completion is idempotent). A display
    /// fence held past that is already pathological — wedging the guest's whole
    /// scanout pipeline on it is never the right outcome. Holds older than the
    /// ceiling are dropped by process_retired_presents.
    created_at: std::time::Instant,
}

struct ParkedFlush {
    scanout_id: u32,
    iosurface_id: u32,
    /// The flushed resource — needed by the deferred readback fallback (a sink
    /// without zero-copy reads the IOSurface's pixels back via the resource).
    resource_id: u32,
    rect: Rect,
}

/// The reserved vkr fence ring for present fences (mirrors VKR_LIMINA_PRESENT_RING in
/// our virglrenderer fork). Guest fences never use it — a guest process would need
/// 63 concurrent VkQueues.
const LIMINA_PRESENT_RING: u8 = 63;

pub struct VirtioGpuScanout {
    resource_id: u32,
    /// limina: the SET_SCANOUT rect dimensions — the visible region the guest scans out, and
    /// the size of the host staging buffer (`configure_scanout` is called with these). The
    /// backing *resource* can be LARGER (mutter pads its framebuffer, e.g. a 1000×708 mode
    /// backed by a 1024×768 resource), so the 2D readback must extract this rect at the
    /// resource's own stride — a flat copy shears. See `flush_resource`.
    width: u32,
    height: u32,
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

/// limina: copy the top-left rect of a (possibly wider) source framebuffer into a tightly-strided
/// destination, row by row — de-shearing a padded resource. The guest's scanout *resource* can be
/// wider than the visible scanout rect (mutter pads its framebuffer, e.g. a 1000×708 mode backed by
/// a 1024×768 resource); the host staging buffer is `dst_stride`-wide, so a flat byte copy would
/// drift every row by `(src_stride − dst_stride)` px. When the strides match (the common case) this
/// is a single flat copy. `src_stride >= dst_stride` is expected; rows that would read/write out of
/// bounds are skipped. See `flush_resource`.
fn blit_scanout_rect(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    rows: usize,
) {
    if src_stride == dst_stride {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    for y in 0..rows {
        let s = y * src_stride;
        let d = y * dst_stride;
        if s + dst_stride > src.len() || d + dst_stride > dst.len() {
            break;
        }
        dst[d..d + dst_stride].copy_from_slice(&src[s..s + dst_stride]);
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
    /// limina (#8): present-fence state; `None` when the renderer is absent.
    present_fence: Option<PresentFenceState>,
    /// limina M9.3: counted GPU health probes (stale-ctx submissions, fence ledger).
    trace: Arc<GpuTraceStats>,
    /// limina M9.3 P0: the rutabaga-layer half of the snapshot-replay journal
    /// (the vkr wire half lives in virglrenderer). Worker-thread-only.
    journal: GpuJournal,
    /// limina M9.3 P3: the last cursor-overlay state handed to the display backend
    /// (pixels included), carried in the snapshot payload so a restored session
    /// keeps its cursor (UPDATE/MOVE_CURSOR are not journaled ops).
    cursor_state: Option<super::journal::CursorSnapshot>,
    /// limina (vrend fence honesty): the renderer's poll eventfd, held for the device's
    /// lifetime (it's a dup that closes on drop). vrend's sync thread parks in `wait_sync`
    /// whenever GL queries are pending until someone pumps `virgl_renderer_poll()` on the
    /// GL thread — the gpu worker registers this fd in its epoll and calls
    /// [`Self::renderer_event_poll`] when it fires. venus never needed the pump (its
    /// fences ride `write_context_fence` from the render server), which is why it was
    /// never wired before Global-ring fences routed through virglrenderer.
    renderer_poll: Option<rutabaga_gfx::RutabagaDescriptor>,
    /// limina: true once any non-venus (vrend/GL) 3D context exists this session. From
    /// that point Global-ring fences route through virglrenderer's GL timeline
    /// (`virgl_renderer_create_fence` → glFenceSync → ASYNC_FENCE_CB → fence handler)
    /// instead of being marked completed at decode: a vrend guest's `glFinish` must wait
    /// for real GL/Metal completion. (Sync-marking every Global fence made vrend fences
    /// loose — crossmark 2026-07-28 measured a stock guest's fenced desktop frame
    /// "faster" than host-native references.) Before any vrend context — firmware GOP,
    /// venus-only sessions, software-2D — the sync mark keeps its wedge-free behavior
    /// and the hot venus scanout-flush path stays untouched. Reset with the session;
    /// snapshot restore re-derives it by replaying CtxCreate through `create_context`.
    vrend_ctx_seen: bool,
}

/// The per-activation transport the fence handler retires guest fences into.
///
/// limina (renderer-singleton fix): `virgl_renderer_init` is a process-global, init-once,
/// thread-bound singleton, so the `Rutabaga`/renderer (and the fence handler registered into
/// it) must outlive any single device activation. But the control queue, guest memory, and
/// interrupt are recreated on every `activate()` (driver rebind, EFI→kernel hand-off, reboot).
/// So the long-lived fence handler reaches them indirectly through a shared cell the worker
/// swaps on each activate and clears on reset; `None` means the device is currently inactive
/// (a fence that completes in that window has nothing to retire into). See `worker.rs`.
#[derive(Clone)]
pub(crate) struct GpuActivation {
    pub(crate) mem: GuestMemoryMmap,
    pub(crate) control_queue: Arc<Mutex<VirtQueue>>,
    pub(crate) interrupt: InterruptTransport,
}

impl VirtioGpu {
    fn create_fence_handler(
        active: Arc<Mutex<Option<GpuActivation>>>,
        fence_state: Arc<Mutex<FenceState>>,
        present_retired: Arc<Mutex<Vec<u64>>>,
        present_event: utils::eventfd::EventFd,
        trace: Arc<GpuTraceStats>,
    ) -> RutabagaFenceHandler {
        // limina wake-trace (LIMINA_WAKE_TRACE=1): guest-fence callback/signal rates,
        // ~5s cadence — see docs/perf/overhead-inventory.md. Shared across the fence
        // threads, hence the mutex (trace-only, env-gated).
        let wake_trace: Option<Arc<Mutex<(std::time::Instant, [u64; 3])>>> =
            std::env::var("LIMINA_WAKE_TRACE")
                .ok()
                .map(|_| Arc::new(Mutex::new((std::time::Instant::now(), [0u64; 3]))));
        RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
            debug!(
                "XXX - fence called: id={}, ring_idx={}",
                completed_fence.fence_id, completed_fence.ring_idx
            );

            // limina (#8): a fence on the reserved present ring is host-injected — it
            // carries a parked-present cookie, not a guest fence id. Hand it to the
            // worker thread (which owns the display backend) and stay clear of the
            // guest fence bookkeeping below.
            if completed_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX != 0
                && completed_fence.ring_idx == LIMINA_PRESENT_RING
            {
                present_retired
                    .lock()
                    .unwrap()
                    .push(completed_fence.fence_id);
                if let Err(e) = present_event.write(1) {
                    error!("present fence eventfd write failed: {e}");
                }
                return;
            }

            // Retire the guest fence into the *current* activation's queue. If the device is
            // inactive (between reset and re-activate), there's nothing to retire into — drop
            // it (the descriptors it would have retired were cleared by `reset_session`).
            let Some(GpuActivation {
                mem,
                control_queue,
                interrupt,
            }) = active.lock().unwrap().clone()
            else {
                return;
            };

            let mut queue = control_queue.lock().unwrap();
            let mut fence_state = fence_state.lock().unwrap();
            let mut i = 0;

            let ring = match completed_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                0 => VirtioGpuRing::Global,
                _ => VirtioGpuRing::ContextSpecific {
                    ctx_id: completed_fence.ctx_id,
                    ring_idx: completed_fence.ring_idx,
                },
            };

            let mut retired_any = false;
            while i < fence_state.descs.len() {
                debug!("XXX - fence_id: {}", fence_state.descs[i].fence_id);
                if fence_state.descs[i].ring == ring
                    && fence_state.descs[i].fence_id <= completed_fence.fence_id
                {
                    let completed_desc = fence_state.descs.remove(i);
                    trace
                        .fences_retired
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    debug!(
                        "XXX - found fence: desc_index={}",
                        completed_desc.desc_index
                    );

                    if let Err(e) =
                        queue.add_used(&mem, completed_desc.desc_index, completed_desc.len)
                    {
                        error!("failed to add used elements to the queue: {e:?}");
                    }
                    retired_any = true;
                } else {
                    i += 1;
                }
            }
            // One interrupt per completion callback, not per retired descriptor — and with
            // EVENT_IDX negotiated only when the guest's used_event says it's waiting
            // (`needs_notification` is always true otherwise, preserving stock behavior).
            let mut signaled = false;
            if retired_any && queue.needs_notification(&mem).unwrap_or(true) {
                interrupt.signal_used_queue();
                signaled = true;
            }
            if let Some(tr) = wake_trace.as_ref() {
                let mut t = tr.lock().unwrap();
                t.1[0] += 1;
                if retired_any {
                    t.1[1] += 1;
                }
                if signaled {
                    t.1[2] += 1;
                }
                let secs = t.0.elapsed().as_secs_f64();
                if secs >= 5.0 {
                    eprintln!(
                        "[WAKETRACE fence] callbacks={:.0}/s retiring={:.0}/s irq_signals={:.0}/s",
                        t.1[0] as f64 / secs,
                        t.1[1] as f64 / secs,
                        t.1[2] as f64 / secs,
                    );
                    t.1 = [0; 3];
                    t.0 = std::time::Instant::now();
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

    /// The rutabaga capsets our renderer actually backs for this `virgl_flags` set.
    ///
    /// Upstream passes `capset_mask = 0` to `RutabagaBuilder`, which registers ALL nine
    /// capsets regardless of the renderer config; the guest then enumerates and probes
    /// capsets we can't serve (e.g. a virgl GL context under `NO_VIRGL`, rejected with a
    /// harmless-but-noisy EINVAL). Advertise only what these flags support: venus, plus
    /// virgl/virgl2 when GL is enabled. The device's `num_capsets` config field is the
    /// pop-count of this mask, so the guest enumerates exactly the real set.
    pub(crate) fn capset_mask_from_virgl_flags(virgl_flags: u32) -> u64 {
        // Mirror of the `VirglRendererFlags` bit layout in rutabaga_utils.rs.
        const VIRGLRENDERER_VENUS: u32 = 1 << 6;
        const VIRGLRENDERER_NO_VIRGL: u32 = 1 << 7;

        let mut mask = 0u64;
        if virgl_flags & VIRGLRENDERER_VENUS != 0 {
            mask |= 1u64 << rutabaga_gfx::RUTABAGA_CAPSET_VENUS;
        }
        if virgl_flags & VIRGLRENDERER_NO_VIRGL == 0 {
            mask |= 1u64 << rutabaga_gfx::RUTABAGA_CAPSET_VIRGL;
            mask |= 1u64 << rutabaga_gfx::RUTABAGA_CAPSET_VIRGL2;
        }
        mask
    }

    pub fn create_rutabaga(
        virgl_flags: u32,
        export_table: Option<ExportTable>,
        fence: RutabagaFenceHandler,
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
            Self::capset_mask_from_virgl_flags(virgl_flags),
        )
        .set_rutabaga_channels(rutabaga_channels_opt);
        let builder = if let Some(export_table) = export_table {
            builder.set_export_table(export_table)
        } else {
            builder
        };

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
        active: Arc<Mutex<Option<GpuActivation>>>,
        virgl_flags: u32,
        software_2d: bool,
        #[cfg(target_os = "macos")] map_sender: Sender<WorkerMessage>,
        export_table: Option<ExportTable>,
        displays: Box<[DisplayInfo]>,
        display_backend: DisplayBackend,
    ) -> Self {
        let fence_state: Arc<Mutex<FenceState>> = Arc::new(Mutex::new(Default::default()));

        // limina M9.3: probe counters, shared with the fence handler (retire counts) and
        // the opt-in tick reporter (LIMINA_GPU_TRACE=1).
        let trace: Arc<GpuTraceStats> = Arc::new(Default::default());
        super::trace::maybe_spawn_reporter(trace.clone(), fence_state.clone());

        // limina (#8): present-fence plumbing — built up front because the fence handler
        // needs its endpoints.
        let present_retired: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let present_event = utils::eventfd::EventFd::new(utils::eventfd::EFD_NONBLOCK)
            .expect("failed to create present-fence eventfd");

        // limina (renderer-singleton fix): the handler is registered into the process-global
        // renderer once and lives as long as it does, so it reaches the per-activation queue/
        // mem/interrupt through `active` rather than capturing them directly.
        let fence_handler = Self::create_fence_handler(
            active,
            fence_state.clone(),
            present_retired.clone(),
            present_event.try_clone().expect("eventfd clone"),
            trace.clone(),
        );

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
            match Self::create_rutabaga(virgl_flags, export_table.clone(), fence_handler.clone()) {
                Some(rutabaga) => Some(rutabaga),
                None => {
                    warn!("virtio-gpu: renderer init failed; degrading to software-2D (no 3D)");
                    None
                }
            }
        };

        // limina (#8 half 2): the latch thread completes held guest flush fences a fixed
        // delay after their frames presented (supervisor tick + CA latch); completion
        // goes through the regular fence handler, which owns desc retirement.
        let present_fence = rutabaga.as_ref().map(|_| {
            let (latch_tx, latch_rx) =
                std::sync::mpsc::channel::<(std::time::Instant, RutabagaFence)>();
            let latch_handler = fence_handler.clone();
            std::thread::Builder::new()
                .name("gpu latch".into())
                .spawn(move || {
                    while let Ok((deadline, fence)) = latch_rx.recv() {
                        let now = std::time::Instant::now();
                        if deadline > now {
                            std::thread::sleep(deadline - now);
                        }
                        latch_handler.call(fence);
                    }
                })
                .expect("failed to spawn gpu latch thread");
            let latch_ms = std::env::var("LIMINA_FENCE_LATCH_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(35);
            // #8 leg 2: read the supervisor's "shown <id>" acks off the control
            // socketpair (rendezvous'd via env by limina-vmm main). The display backend
            // only ever writes that fd; the supervisor's acks are the only inbound
            // bytes. The reader wakes the worker through the same present eventfd.
            let shown: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
            let ack_active = if let Some(fd) = std::env::var("LIMINA_SHOWN_ACK_FD")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .filter(|fd| *fd >= 0)
            {
                let dup = unsafe { libc::dup(fd) };
                if dup >= 0 {
                    let shown = shown.clone();
                    let wake = present_event.try_clone().expect("eventfd clone");
                    std::thread::Builder::new()
                        .name("gpu shown-ack".into())
                        .spawn(move || {
                            use std::io::BufRead;
                            // SAFETY: we own `dup`.
                            let file = unsafe { std::fs::File::from_raw_fd(dup) };
                            for line in std::io::BufReader::new(file).lines() {
                                let Ok(line) = line else { break };
                                let mut parts = line.split_whitespace();
                                if parts.next() == Some("shown") {
                                    if let Some(id) =
                                        parts.next().and_then(|s| s.parse::<u32>().ok())
                                    {
                                        shown.lock().unwrap().push(id);
                                        let _ = wake.write(1);
                                    }
                                }
                            }
                            debug!("gpu shown-ack reader: channel closed");
                        })
                        .is_ok()
                } else {
                    false
                }
            } else {
                false
            };
            PresentFenceState {
                retired: present_retired,
                event: present_event,
                next_cookie: 1,
                parked: BTreeMap::new(),
                flush_parked_cookies: Vec::new(),
                guest_holds: Vec::new(),
                latch_tx,
                latch_delay: std::time::Duration::from_millis(latch_ms),
                ack_active,
                shown,
                awaiting_shown: std::collections::VecDeque::new(),
                guest_fence_handler: fence_handler.clone(),
            }
        });

        let display_backend = display_backend
            .create_instance()
            .expect("Failed to create display backend instance!");

        let journal = GpuJournal::new(trace.clone());

        // See the `renderer_poll` field doc: vrend's query-check pump. Only virglrenderer
        // components expose a descriptor; None everywhere else keeps the worker's epoll
        // unchanged.
        let renderer_poll = rutabaga.as_ref().and_then(|r| r.poll_descriptor());

        Self {
            rutabaga,
            renderer_poll,
            resources: Default::default(),
            sw2d: Default::default(),
            fence_state,
            scanouts: Default::default(),
            displays,
            display_backend,
            #[cfg(target_os = "macos")]
            map_sender,
            present_fence,
            trace,
            journal,
            cursor_state: None,
            vrend_ctx_seen: false,
        }
    }

    /// limina M9.3: the probe counters (shared; incremented from the worker's dispatch
    /// sites and the fence handler, read by the tick reporter).
    pub fn trace(&self) -> &Arc<GpuTraceStats> {
        &self.trace
    }

    /// limina (vrend fence honesty): the renderer's poll eventfd for the worker's epoll,
    /// or `None` when the renderer is absent or isn't virglrenderer. See `renderer_poll`.
    pub fn renderer_poll_fd(&self) -> Option<i32> {
        use rutabaga_gfx::AsRawDescriptor;
        // The raw fd of the descriptor we hold — `renderer_poll` outlives every borrower
        // (device lifetime), so handing out the raw fd is safe; no dup (a dropped clone
        // would close it).
        self.renderer_poll.as_ref().map(|d| d.as_raw_descriptor())
    }

    /// limina (vrend fence honesty): pump `virgl_renderer_poll()` — flushes the poll
    /// eventfd, runs vrend's pending GL query checks on this (the GL) thread, and signals
    /// the sync thread parked in `wait_sync`. Must be called from the gpu worker thread.
    pub fn renderer_event_poll(&self) {
        if let Some(r) = self.rutabaga.as_ref() {
            r.event_poll();
        }
    }

    /// limina M9.3 P0: the rutabaga-layer snapshot-replay journal (worker thread only).
    pub fn journal_mut(&mut self) -> &mut GpuJournal {
        &mut self.journal
    }

    pub fn journal(&self) -> &GpuJournal {
        &self.journal
    }

    /// limina M9.3: the number of outstanding guest fences (requested, not yet retired
    /// to the used ring), plus a summary for logging. The snapshot path drains these to
    /// zero before capture: guest waiters hold sync_files backed by them, and a fence
    /// whose completion misses the snapshot never signals in the restored epoch.
    pub fn outstanding_fences(&self) -> (usize, String) {
        let fs = self.fence_state.lock().unwrap();
        (
            fs.descs.len(),
            fs.outstanding_summary(std::time::Instant::now()),
        )
    }

    /// limina M9.3 P1: the venus wire-journal watermark for `ctx_id`, stamped on
    /// rutabaga journal entries as the cross-layer replay fence.
    pub fn journal_vkr_seq(&self, ctx_id: u32) -> u64 {
        self.rutabaga
            .as_ref()
            .map(|r| r.limina_journal_seq(ctx_id))
            .unwrap_or(0)
    }

    /// limina M9.3: release the vkr journal pin a venus blob create took on its
    /// backing VkDeviceMemory (+ dedicated object). Called at global resource unref.
    pub fn journal_unpin(&self, ctx_id: u32, blob_id: u64) {
        if let Some(r) = self.rutabaga.as_ref() {
            r.limina_journal_unpin(ctx_id, blob_id);
        }
    }

    /// limina M9.3 P1: read a mapped blob's bytes through its host mapping (the
    /// venus vkMapMemory pointer rutabaga resolves). None for unmappable blobs.
    fn blob_content_read(&mut self, resource_id: u32) -> Option<Vec<u8>> {
        let size = self.resources.get(&resource_id)?.size as usize;
        let rutabaga = self.rutabaga.as_mut()?;
        let ptr = rutabaga.map_ptr(resource_id).ok()?;
        // Safe: the mapping is alive as long as the resource exists (worker thread
        // owns both), and `size` is the blob's own size.
        Some(unsafe { std::slice::from_raw_parts(ptr as *const u8, size) }.to_vec())
    }

    fn blob_content_write(&mut self, resource_id: u32, bytes: &[u8]) -> bool {
        let Some(resource) = self.resources.get(&resource_id) else {
            return false;
        };
        let size = (resource.size as usize).min(bytes.len());
        let Some(rutabaga) = self.rutabaga.as_mut() else {
            return false;
        };
        let Ok(ptr) = rutabaga.map_ptr(resource_id) else {
            return false;
        };
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, size) };
        true
    }

    /// limina M9.3 P1 snapshot: assemble the GPU section — the rutabaga journal,
    /// each venus context's vkr wire journal, and every guest-mapped blob's raw
    /// bytes (host allocations, invisible to the guest-RAM dump). Worker thread.
    pub fn snapshot_gpu_payload(&mut self) -> Option<Vec<u8>> {
        if self.rutabaga.is_none() || self.journal.entries().is_empty() {
            return None;
        }

        let mut payload = super::journal::GpuSnapshotPayload {
            ops: Vec::new(),
            vkr_journals: Vec::new(),
            blob_contents: Vec::new(),
            memory_contents: Vec::new(),
            sync_states: Vec::new(),
            cursor: self.cursor_state.clone(),
        };

        let mut ctxs: Vec<u32> = Vec::new();
        let mut mapped: Vec<u32> = Vec::new();
        for e in self.journal.entries() {
            match &e.op {
                GpuJournalOp::CtxCreate { ctx_id, .. } => ctxs.push(*ctx_id),
                GpuJournalOp::MapBlob { resource_id, .. } => mapped.push(*resource_id),
                _ => {}
            }
            payload.ops.push(super::journal::GpuJournalEntry {
                seq: e.seq,
                vkr_seq: e.vkr_seq,
                op: e.op.clone(),
            });
        }

        for ctx_id in ctxs {
            match self
                .rutabaga
                .as_ref()
                .and_then(|r| r.limina_journal_export(ctx_id))
            {
                Some(bytes) => payload.vkr_journals.push((ctx_id, bytes)),
                // Normal for non-venus contexts (the kernel's ctx 1, virgl contexts).
                None => {
                    debug!("gpu snapshot: no vkr journal for ctx {ctx_id}");
                    continue;
                }
            }
            // P2.1: fence status + timeline semaphore counter values (opaque vkr
            // blob) for the restore-time sync fast-forward.
            match self
                .rutabaga
                .as_ref()
                .and_then(|r| r.limina_sync_export(ctx_id))
            {
                Some(bytes) => payload.sync_states.push((ctx_id, bytes)),
                None => warn!("gpu snapshot: sync export failed for venus ctx {ctx_id}"),
            }
            // P2: every capturable VkDeviceMemory's raw bytes. The census excludes
            // map_ptr-exported blobs (the mapped-blob loop below captures those) and
            // imports (they alias another storage); what remains — textures, render
            // targets, non-staging buffers — lives only in host heaps.
            let Some(census) = self
                .rutabaga
                .as_ref()
                .and_then(|r| r.limina_memory_census(ctx_id))
            else {
                warn!("gpu snapshot: memory census failed for venus ctx {ctx_id}");
                continue;
            };
            for (mem_id, size) in census {
                let mut bytes = vec![0u8; size as usize];
                if self
                    .rutabaga
                    .as_ref()
                    .is_some_and(|r| r.limina_memory_read(ctx_id, mem_id, &mut bytes))
                {
                    payload.memory_contents.push((ctx_id, mem_id, bytes));
                } else {
                    warn!(
                        "gpu snapshot: content read failed for ctx {ctx_id} mem {mem_id} ({size} bytes)"
                    );
                }
            }
        }

        for res_id in mapped {
            match self.blob_content_read(res_id) {
                Some(bytes) => payload.blob_contents.push((res_id, bytes)),
                None => warn!("gpu snapshot: mapped blob {res_id} content unreadable"),
            }
        }

        let content_bytes: usize = payload
            .memory_contents
            .iter()
            .map(|(_, _, b)| b.len())
            .sum();
        info!(
            "gpu snapshot: {} ops, {} vkr journals, {} blob contents, {} memory contents ({} MiB)",
            payload.ops.len(),
            payload.vkr_journals.len(),
            payload.blob_contents.len(),
            payload.memory_contents.len(),
            content_bytes >> 20
        );
        Some(payload.to_bytes())
    }

    /// limina M9.3 P1 restore: replay the GPU section into the fresh renderer —
    /// rutabaga ops interleaved with venus wire entries by the recorded vkr_seq
    /// fences, blob contents restored at their creates (before the ring-creates
    /// that read them), ring threads started at the end. Worker thread, before
    /// the guest resumes. Returns false on any replay failure (the session then
    /// restarts, i.e. today's fresh-renderer behavior — never a wedge).
    #[cfg(target_os = "macos")]
    pub fn restore_gpu_payload(
        &mut self,
        data: &[u8],
        shm_region: &VirtioShmRegion,
        mem: &GuestMemoryMmap,
    ) -> bool {
        use super::journal::{parse_vkr_journal, GpuSnapshotPayload, VKR_KLASS_RING_STREAM};
        use std::collections::HashMap;

        let Some(payload) = GpuSnapshotPayload::from_bytes(data) else {
            error!("gpu restore: unparseable GPU snapshot section");
            return false;
        };
        let GpuSnapshotPayload {
            mut ops,
            vkr_journals,
            blob_contents,
            memory_contents,
            sync_states,
            cursor,
        } = payload;

        let mut wire: HashMap<u32, (Vec<super::journal::VkrWireEntry>, usize)> = HashMap::new();
        for (ctx_id, bytes) in &vkr_journals {
            match parse_vkr_journal(bytes) {
                Some(entries) => {
                    wire.insert(*ctx_id, (entries, 0));
                }
                None => {
                    error!("gpu restore: unparseable vkr journal for ctx {ctx_id}");
                    return false;
                }
            }
        }
        let contents: HashMap<u32, &Vec<u8>> =
            blob_contents.iter().map(|(id, b)| (*id, b)).collect();

        // Two failure classes: a wire entry can fail RECOVERABLY (a retained command
        // referencing an object destroyed pre-snapshot — its stale write is dropped, virgl
        // clears the context FATAL, replay continues), while a structural rutabaga failure
        // (context/blob/mapping) leaves guest-visible state missing and fails the replay.
        let mut wire_failed = 0u32;
        let mut op_failed = 0u32;
        // Per-class drop histogram (klass 0..=8), reported in the completion log so a
        // failing run says exactly which entry classes it lost.
        let mut drops_by_klass = [0u32; 9];
        // Feed ctx's wire entries with seq <= fence (0 = none pending). Ring-stream
        // entries go to the target ring's decoder; everything else to the context.
        macro_rules! replay_wire_upto {
            ($ctx:expr, $fence:expr) => {
                if let Some((entries, pos)) = wire.get_mut(&$ctx) {
                    while *pos < entries.len()
                        && ($fence == u64::MAX || entries[*pos].seq <= $fence)
                    {
                        let e = &mut entries[*pos];
                        let ok = if e.klass == VKR_KLASS_RING_STREAM && e.ring_key != 0 {
                            self.rutabaga.as_ref().is_some_and(|r| {
                                r.limina_replay_ring_cmd($ctx, e.ring_key, &mut e.bytes)
                            })
                        } else {
                            self.rutabaga
                                .as_ref()
                                .is_some_and(|r| r.limina_replay_submit($ctx, &mut e.bytes))
                        };
                        if !ok {
                            wire_failed += 1;
                            *drops_by_klass
                                .get_mut(e.klass.min(8) as usize)
                                .unwrap() += 1;
                            // Class matters: TRANSIENT..NOTED drops are the benign
                            // stale-reference kind (a retained vkUpdateDescriptorSets /
                            // vkBind*Memory naming an object destroyed pre-snapshot — its
                            // write was dangling before and stays unwritten after). A
                            // dropped RING class (create/destroy/stream) is load-bearing:
                            // it can skew the ring's command/reply stream and corrupt the
                            // live session after resume.
                            if e.klass >= 6 {
                                warn!(
                                    "gpu restore: dropped LOAD-BEARING ring entry ctx={} seq={} klass={} cmd={} ring={:#x} size={}",
                                    $ctx, e.seq, e.klass, e.cmd_type, e.ring_key, e.bytes.len()
                                );
                            } else {
                                debug!(
                                    "gpu restore: dropped stale wire entry ctx={} seq={} klass={} cmd={} size={}",
                                    $ctx, e.seq, e.klass, e.cmd_type, e.bytes.len()
                                );
                            }
                        }
                        *pos += 1;
                    }
                }
            };
        }

        for entry in &mut ops {
            // GENERATION REBASE (the 2nd-resume disk of the journal): a CreateBlob's
            // vkr_seq fence is meaningful only against the wire journal that recorded it.
            // The fresh context re-records the replayed wire commands from seq 1, so the
            // fence must be rewritten to the NEW journal's watermark at the exact point
            // its old fence was satisfied — otherwise the next generation's merge is
            // cross-epoch garbage (the first old fence, ~1M, drains the entire new wire
            // journal, and every later import replays before its exporter blob exists:
            // the observed fd_type=-999 cascade + KK descriptor assert). Set in the
            // CreateBlob arm below; carrying it as a local keeps entry.op's borrow short.
            let mut rebased_fence: Option<u64> = None;
            match &entry.op {
                GpuJournalOp::CtxCreate {
                    ctx_id,
                    context_init,
                    name,
                } => {
                    if self
                        .create_context(*ctx_id, *context_init, name.as_deref())
                        .is_err()
                    {
                        error!("gpu restore: create_context {ctx_id} failed");
                        return false;
                    }
                    // Only venus contexts (the ones with a wire journal) have vkr state to
                    // replay into; the kernel's virgl/none context has nothing host-side.
                    // The create is proxied to the (same-process) render-server thread and
                    // applies asynchronously, so the vkr context may not exist yet — poll.
                    // Each FFI call takes/releases the renderer lock, letting the server in.
                    if wire.contains_key(ctx_id) {
                        let mut began = false;
                        for _ in 0..2000 {
                            if self
                                .rutabaga
                                .as_ref()
                                .is_some_and(|r| r.limina_replay_begin(*ctx_id))
                            {
                                began = true;
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                        if !began {
                            error!("gpu restore: replay_begin {ctx_id} failed (2s timeout)");
                            return false;
                        }
                    }
                }
                GpuJournalOp::CreateBlob {
                    ctx_id,
                    resource_id,
                    blob_mem,
                    blob_flags,
                    blob_id,
                    size,
                    backing,
                } => {
                    let fed_before = wire.get(ctx_id).map(|(_, p)| *p).unwrap_or(0);
                    replay_wire_upto!(*ctx_id, entry.vkr_seq);
                    let fed_after = wire.get(ctx_id).map(|(_, p)| *p).unwrap_or(0);
                    debug!(
                        "gpu restore: CREATE_BLOB res {} ctx {} blob_id {} fence {} fed {} wire entries (pos {} of {})",
                        resource_id,
                        ctx_id,
                        blob_id,
                        entry.vkr_seq,
                        fed_after - fed_before,
                        fed_after,
                        wire.get(ctx_id).map(|(e, _)| e.len()).unwrap_or(0)
                    );
                    let create = ResourceCreateBlob {
                        blob_mem: *blob_mem,
                        blob_flags: *blob_flags,
                        blob_id: *blob_id,
                        size: *size,
                    };
                    let vecs: Vec<(GuestAddress, usize)> = backing
                        .iter()
                        .map(|(a, l)| (GuestAddress(*a), *l))
                        .collect();
                    // Rebase the fence into the new journal's epoch (see the loop head):
                    // journal_vkr_seq is the new journal's last-assigned seq — everything
                    // this fence fed has just re-recorded, so this IS the old fence's
                    // position translated. Venus contexts only (others have no journal).
                    if wire.contains_key(ctx_id) {
                        rebased_fence = Some(self.journal_vkr_seq(*ctx_id));
                    }
                    if self
                        .resource_create_blob(*ctx_id, *resource_id, create, vecs, mem)
                        .is_err()
                    {
                        error!("gpu restore: CREATE_BLOB res {resource_id} failed");
                        op_failed += 1;
                        continue;
                    }
                    if let Some(bytes) = contents.get(resource_id) {
                        if !self.blob_content_write(*resource_id, bytes) {
                            warn!("gpu restore: content restore for blob {resource_id} failed");
                        }
                    }
                }
                GpuJournalOp::MapBlob {
                    resource_id,
                    offset,
                } => {
                    if self
                        .resource_map_blob(*resource_id, shm_region, *offset)
                        .is_err()
                    {
                        error!("gpu restore: MAP_BLOB res {resource_id} failed");
                        op_failed += 1;
                    }
                }
                GpuJournalOp::CtxAttachResource {
                    ctx_id,
                    resource_id,
                } => {
                    if self.context_attach_resource(*ctx_id, *resource_id).is_err() {
                        error!("gpu restore: ATTACH ctx {ctx_id} res {resource_id} failed");
                        op_failed += 1;
                    }
                }
                GpuJournalOp::SetScanoutBlob {
                    scanout_id,
                    resource_id,
                    width,
                    height,
                    format,
                } => {
                    if self
                        .set_scanout_blob(*scanout_id, *resource_id, *width, *height, *format)
                        .is_err()
                    {
                        warn!("gpu restore: SET_SCANOUT_BLOB scanout {scanout_id} failed");
                    }
                }
            }
            if let Some(fence) = rebased_fence {
                entry.vkr_seq = fence;
            }
        }

        // Drain each context's remaining wire entries, restore its device-memory
        // contents (the allocs exist now, and the rings — which consume parked
        // commands the moment they start — haven't started yet), then start the
        // rings. A failed content write is recoverable in the same sense as a
        // dropped stale wire entry: the memory whose alloc replay was dropped is
        // garbage-if-accessed either way.
        let ctxs: Vec<u32> = wire.keys().copied().collect();
        let mut content_failed = 0u32;
        for ctx_id in ctxs {
            replay_wire_upto!(ctx_id, u64::MAX);
            for (mem_ctx, mem_id, bytes) in &memory_contents {
                if *mem_ctx != ctx_id {
                    continue;
                }
                if !self
                    .rutabaga
                    .as_ref()
                    .is_some_and(|r| r.limina_memory_write(ctx_id, *mem_id, bytes))
                {
                    content_failed += 1;
                    debug!(
                        "gpu restore: content write failed for ctx {ctx_id} mem {mem_id} ({} bytes)",
                        bytes.len()
                    );
                }
            }
            // P2.1: fast-forward the context's sync objects (fences, timeline +
            // binary semaphores) to their retired pre-suspend state BEFORE the
            // rings start — a started ring may immediately consume a guest wait
            // rooted in the pre-suspend epoch (the mutter WSI-semaphore wedge).
            if let Some((_, blob)) = sync_states.iter().find(|(c, _)| *c == ctx_id) {
                if !self
                    .rutabaga
                    .as_ref()
                    .is_some_and(|r| r.limina_sync_restore(ctx_id, blob))
                {
                    warn!("gpu restore: sync fast-forward failed for ctx {ctx_id}");
                }
            }
            if !self
                .rutabaga
                .as_ref()
                .is_some_and(|r| r.limina_replay_end(ctx_id))
            {
                error!("gpu restore: replay_end {ctx_id} failed");
                op_failed += 1;
            }
        }
        if content_failed > 0 {
            warn!(
                "gpu restore: {content_failed} of {} memory-content writes failed (allocs \
                 whose replay was dropped — garbage-if-accessed before and after)",
                memory_contents.len()
            );
        }

        if op_failed > 0 {
            error!(
                "gpu restore: replay FAILED — {op_failed} structural failures ({wire_failed} wire entries also failed)"
            );
            return false;
        }
        if wire_failed > 0 {
            warn!(
                "gpu restore: replay complete with {wire_failed} dropped wire entries (stale references); drops by class [transient,create,recording,noted,free,pool-reset,ring-create,ring-destroy,ring-stream] = {drops_by_klass:?}"
            );
        } else {
            info!("gpu restore: replay complete (session state re-created)");
        }
        // The re-created world is the payload's world (minus dropped stale writes); adopt
        // its journal so the next suspend records from a warm, accurate baseline.
        self.journal.restore_entries(ops);
        // P3: re-apply the captured cursor overlay — the restored guest believes its cursor
        // is set and won't re-send UPDATE_CURSOR until it next changes. Cosmetic: failures
        // (or a backend without cursor support) degrade to the default cursor, never an error.
        if let Some(c) = cursor {
            match ResourceFormat::try_from(c.format) {
                Ok(format) => {
                    let _ = self
                        .display_backend
                        .set_cursor(c.width, c.height, c.hot_x, c.hot_y, format, &c.pixels);
                    let _ = self.display_backend.move_cursor(c.x, c.y);
                    info!(
                        "gpu restore: cursor overlay re-applied ({}x{} at {},{})",
                        c.width, c.height, c.x, c.y
                    );
                    self.cursor_state = Some(c);
                }
                Err(()) => warn!(
                    "gpu restore: captured cursor has unknown format {}; skipped",
                    c.format
                ),
            }
        }
        true
    }

    /// limina M9.3: dump the renderer's live context table (serviced on the worker
    /// thread, the only thread allowed to call into the renderer singleton).
    pub fn dump_renderer_state(&self) {
        match self.rutabaga.as_ref() {
            Some(rutabaga) => rutabaga.limina_dump_state(),
            None => warn!("[GPUTRACE] renderer state dump requested, but no renderer is live"),
        }
    }

    /// limina (renderer-singleton fix): reset the per-VM-session bookkeeping on a device reset,
    /// while **keeping** the process-global renderer (`rutabaga`), the display backend, and the
    /// present-fence infrastructure alive. The guest re-initializes virtio-gpu from scratch
    /// (driver rebind, EFI→kernel hand-off, reboot), so our resource/scanout maps must start
    /// empty; critically, any in-flight fence descriptors index into the now-freed control
    /// queue and MUST NOT be retired into the next activation's queue. (Resources the guest
    /// owned in the renderer are released by its own CTX_DESTROY/RESOURCE_UNREF before the
    /// reset; a guest that resets dirty is a separate, narrower concern.)
    pub fn reset_session(&mut self) {
        self.resources.clear();
        self.sw2d.clear();
        self.scanouts = Default::default();
        // Dirty-reset hardening: a guest that crashed (or a firmware→kernel hand-off) never sent
        // CTX_DESTROY/RESOURCE_UNREF, so its contexts/resources survive in the process-global
        // renderer and collide with the re-initialized guest's reused ids — InvalidContextId /
        // InvalidResourceId, which cascade-crashes the recovering session's GPU clients. Drop that
        // leaked per-session renderer state so the next session starts clean. (A *clean* reset
        // already emptied these via the guest's own teardown, so this is a no-op there.)
        if let Some(rutabaga) = self.rutabaga.as_mut() {
            rutabaga.reset_session_state();
        }
        // The re-creation journal describes the session state just dropped; clear it so the
        // next session's recording (or a pending snapshot replay) starts from empty.
        self.journal.reset();
        // The cursor belonged to the dropped session; the re-initialized guest re-sets it.
        self.cursor_state = None;
        // The dropped session's contexts are gone; Global-ring fence routing returns to
        // the sync mark until the next session creates a vrend context.
        self.vrend_ctx_seen = false;
        {
            let mut fs = self.fence_state.lock().unwrap();
            fs.descs.clear();
            fs.completed_fences.clear();
        }
        if let Some(pf) = self.present_fence.as_mut() {
            // Drop parked frames and held flush fences: their cookies/descriptors belong to the
            // freed queue. Keep the retired/event/latch/ack plumbing (tied to the renderer).
            pf.parked.clear();
            pf.flush_parked_cookies.clear();
            pf.guest_holds.clear();
            pf.awaiting_shown.clear();
            pf.retired.lock().unwrap().clear();
        }
    }

    /// limina (host-sleep s2idle): whether the present-fence plumbing holds NOTHING that
    /// references the current activation's queue — no parked flushes, held guest flush
    /// fences, frames awaiting shown, or completed-but-unretired presents. Together with
    /// an empty fence ledger this makes the session safe to PARK across a device reset
    /// instead of wiping it (see the worker's defer-and-classify path); anything pending
    /// here indexes the about-to-be-freed queue and forces the fail-closed wipe.
    pub fn present_quiescent(&self) -> bool {
        match self.present_fence.as_ref() {
            Some(pf) => {
                pf.parked.is_empty()
                    && pf.flush_parked_cookies.is_empty()
                    && pf.guest_holds.is_empty()
                    && pf.awaiting_shown.is_empty()
                    && pf.retired.lock().unwrap().is_empty()
            }
            None => true,
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

        // limina vrend zero-copy scanout (docs/design/vrend-iosurface-scanout.md): a plain
        // SET_SCANOUT resource may be IOSurface-backed too (vrend allocates the surface for
        // SCANOUT-bound textures at create). Resolve exactly like set_scanout_blob; None
        // keeps the readback path — a stock guest with an ineligible format loses nothing.
        #[cfg(target_os = "macos")]
        let iosurface_id = self
            .rutabaga
            .as_ref()
            .and_then(|r| r.iosurface_id(resource_id).ok())
            .filter(|&id| id != 0);
        #[cfg(target_os = "macos")]
        if let Some(id) = iosurface_id {
            log::debug!("SET_SCANOUT scanout={scanout_id} res={resource_id} -> IOSurface {id} (vrend zero-copy)");
        }

        *scanout = Some(VirtioGpuScanout {
            resource_id,
            width,
            height,
            #[cfg(target_os = "macos")]
            iosurface_id,
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
            // Per-flip (mutter alternates swapchain buffers every frame) — debug, not info.
            log::debug!(
                "SET_SCANOUT_BLOB scanout={scanout_id} res={resource_id} -> IOSurface {id} (zero-copy)"
            );
        } else {
            log::warn!(
                "SET_SCANOUT_BLOB scanout={scanout_id} res={resource_id} not IOSurface-backed; using readback"
            );
        }

        *scanout = Some(VirtioGpuScanout {
            resource_id,
            width,
            height,
            iosurface_id,
        });
        Ok(OkNoData)
    }

    fn read_2d_resource(
        rutabaga: &mut Rutabaga,
        resource: VirtioGpuResource,
        width: u32,
        height: u32,
        output: &mut [u8],
    ) -> VirtioGpuResult {
        // limina: read the top-left `width`×`height` rect (the scanout's visible region) into a
        // TIGHTLY packed `width*4`-stride output — the host staging buffer. The backing resource
        // may be wider (mutter pads its framebuffer), but the box read extracts just the rect, so
        // the output never shears regardless of the resource's own stride.
        let transfer = Transfer3D {
            x: 0,
            y: 0,
            z: 0,
            w: width,
            h: height,
            d: 1,
            level: 0,
            stride: width * ResourceFormat::BYTES_PER_PIXEL as u32,
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
        // limina (#8 half 2): the parked-cookie list is per flush command — it feeds the
        // flush's own trailing FLAG_FENCE and must never leak into a later command.
        if let Some(pf) = self.present_fence.as_mut() {
            pf.flush_parked_cookies.clear();
        }
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
            if log::log_enabled!(log::Level::Debug) {
                let dbg_ios = self
                    .scanouts
                    .get(scanout_id as usize)
                    .and_then(|s| s.as_ref())
                    .and_then(|s| s.iosurface_id);
                log::debug!("[FLUSHDBG] flush res={resource_id} scanout={scanout_id} iosurface_id={dbg_ios:?}");
            }
            #[cfg(target_os = "macos")]
            if let Some(iosurface_id) = self
                .scanouts
                .get(scanout_id as usize)
                .and_then(|s| s.as_ref())
                .and_then(|s| s.iosurface_id)
            {
                // limina vrend zero-copy scanout: non-blob (ctx_id == 0) scanouts are vrend
                // textures — unlike venus blobs they don't render INTO the surface, so blit
                // the current frame into it (GPU-side, sync) before presenting. On failure
                // (vrend poisons the surface and keeps the resource alive) clear the cached
                // id and take the readback path from now on.
                if resource.ctx_id == 0 {
                    if let Err(e) = self
                        .rutabaga
                        .as_ref()
                        .ok_or(ErrUnspec)
                        .and_then(|r| r.sync_iosurface(resource_id).map_err(|_| ErrUnspec))
                    {
                        log::warn!(
                            "vrend iosurface sync failed for res {resource_id} ({e:?}); \
                             falling back to readback"
                        );
                        if let Some(Some(s)) = self.scanouts.get_mut(scanout_id as usize) {
                            s.iosurface_id = None;
                        }
                        // fall through to the readback path below
                    } else {
                        match self.display_backend.present_surface(
                            scanout_id,
                            iosurface_id,
                            Some(&rect),
                        ) {
                            Ok(()) => continue,
                            Err(DisplayBackendError::MethodNotSupported) => {}
                            Err(e) => {
                                log::error!(
                                    "present_surface failed for scanout {scanout_id}: {e}"
                                );
                                return Err(ErrUnspec);
                            }
                        }
                    }
                } else {
                // limina (#8): fence-accurate present — park the frame and inject a
                // present fence on the rendering context; the worker presents when it
                // retires (true GPU completion). Falls through to the immediate
                // present if parking isn't possible.
                if self.try_park_present(
                    scanout_id,
                    iosurface_id,
                    resource_id,
                    &rect,
                    resource.ctx_id,
                ) {
                    continue;
                }
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
            }

            // limina: the scanout's visible rect = the staging-buffer size (`configure_scanout`
            // was called with it). The backing resource may be LARGER — mutter pads its
            // framebuffer (e.g. a 1000×708 mode backed by a 1024×768 resource) — so we must
            // extract this rect at the resource's own stride. A flat copy shears every row by
            // (resource.width − scan_w) px. Falls back to the resource dims if unknown.
            let (scan_w, scan_h) = self
                .scanouts
                .get(scanout_id as usize)
                .and_then(|s| s.as_ref())
                .map(|s| (s.width, s.height))
                .filter(|&(w, h)| w != 0 && h != 0)
                .unwrap_or((resource.width, resource.height));
            let bpp = ResourceFormat::BYTES_PER_PIXEL;
            let dst_stride = scan_w as usize * bpp;
            let src_stride = resource.width as usize * bpp;

            // limina: an IOSurface-backed (venus zero-copy) scanout whose present_surface fell
            // through to readback — i.e. a display sink with no zero-copy path, the headless
            // capture sink. venus blobs have no CPU transfer_read, so read the presented
            // IOSurface's shared storage directly instead of read_2d_resource (which EINVALs).
            #[cfg(target_os = "macos")]
            let scan_iosurface_id = self
                .scanouts
                .get(scanout_id as usize)
                .and_then(|s| s.as_ref())
                .and_then(|s| s.iosurface_id);
            #[cfg(not(target_os = "macos"))]
            let scan_iosurface_id: Option<u32> = None;

            let (frame_id, buffer) = self.display_backend.alloc_frame(scanout_id)?;
            // limina software 2D: the pixels already live in the host buffer; copy them out.
            // Otherwise fall back to the rutabaga readback path (3D/Venus resources).
            if let Some(sw) = self.sw2d.get(&resource_id) {
                blit_scanout_rect(buffer, dst_stride, &sw.host, src_stride, scan_h as usize);
            } else if let Some(iosurface_id) = scan_iosurface_id {
                match self.rutabaga.as_ref() {
                    Some(rutabaga) => {
                        if let Err(e) =
                            rutabaga.read_iosurface(resource_id, buffer, dst_stride as u32, scan_h)
                        {
                            log::error!(
                                "Failed to read IOSurface {iosurface_id} for scanout {scanout_id} (res {resource_id}): {e}"
                            );
                            return Err(ErrUnspec);
                        }
                    }
                    None => return Err(ErrUnspec),
                }
            } else if let Some(rutabaga) = self.rutabaga.as_mut() {
                // limina DIAG (flicker hunt): tunable pre-readback delay. `echo N >
                // /tmp/limina-readback-delay` sleeps N ms before transfer_read, giving the
                // host GPU time to finish rendering this buffer. If the incomplete-frame
                // rate collapses with N>0, the flicker is a readback-vs-render race (the
                // readback path lacks the #8 fence-accurate present the zero-copy path has).
                if let Ok(s) = std::fs::read_to_string("/tmp/limina-readback-delay") {
                    if let Ok(ms) = s.trim().parse::<u64>() {
                        if ms > 0 {
                            std::thread::sleep(std::time::Duration::from_millis(ms));
                        }
                    }
                }
                if let Err(e) = Self::read_2d_resource(rutabaga, resource, scan_w, scan_h, buffer) {
                    log::error!(
                        "Failed to read resource {resource_id} for scanout {scanout_id}: {e}"
                    );
                    return Err(ErrUnspec);
                }
            } else {
                // No software-2D buffer and no renderer: nothing to present.
                return Err(ErrUnspec);
            }
            // limina DIAG (flicker hunt): characterise every readback present. Pins whether the
            // flicker is multi-buffer divergence (resource_id cycles, hash differs per buffer),
            // a large damage rect overwriting good canvas, or a stride mismatch (dims != scanout).
            // FNV-1a over a sparse sample of the staging readback — cheap, but reveals content
            // oscillation: same res with two alternating hashes = the guest re-renders it stale.
            {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let n = buffer.len();
                let mut i = 0usize;
                while i < n {
                    h = (h ^ buffer[i] as u64).wrapping_mul(0x0100_0000_01b3);
                    i += 1021;
                }
                log::trace!(
                    "[FLUSH2] res={resource_id} scan={scanout_id} rect=({},{},{},{}) dims={}x{} hash={h:016x}",
                    rect.x, rect.y, rect.width, rect.height, resource.width, resource.height
                );
            }
            // limina DIAG (`touch /tmp/limina-dump-staging`): dump the RAW readback (staging,
            // BGRA, full frame) BEFORE any swizzle/canvas/ring, so we can run the duplicate-row
            // detector directly on it. Answers: is the tear already in the readback (guest
            // delivered a torn buffer) or introduced downstream (host present path)?
            if resource.width == 1280
                && resource.height == 800
                && std::fs::metadata("/tmp/limina-dump-staging").is_ok()
            {
                use std::sync::atomic::{AtomicU32, Ordering};
                static SEQ: AtomicU32 = AtomicU32::new(0);
                let n = SEQ.fetch_add(1, Ordering::Relaxed);
                if n < 80 {
                    let _ = std::fs::write(format!("/tmp/limina-staging-{n:03}.raw"), &buffer[..]);
                }
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

    /// limina (#8): the fence-present policy, pure for testability.
    ///
    /// Explicit env wins: `LIMINA_FENCE_PRESENT=0`/`off` forces off, any other value
    /// forces on. Unset defaults to **on exactly when the supervisor's shown-ack
    /// channel exists** (`LIMINA_SHOWN_ACK_FD`, set only by windowed workers): the ack
    /// path is what keeps held flush fences at display rate — without it the open-loop
    /// latch delay serializes compositors, and on ack-less sinks (headless capture) a
    /// parked frame's deferred `present_surface` is unsupported and would drop frames.
    fn fence_present_policy(env: Option<&str>, ack_channel: bool) -> bool {
        match env {
            Some(v) => !(v == "0" || v.eq_ignore_ascii_case("off")),
            None => ack_channel,
        }
    }

    /// limina (#8): true when fence-accurate presents are armed. The policy decision is
    /// cached (per-flush getenv serialized the draw path once before — round 24), and
    /// the live force-OFF marker (`touch /tmp/disable-limina-fence-present` → immediate
    /// presents, `rm` → parked again; mid-flight parked frames retire normally either
    /// way) is polled by a dedicated thread every 500 ms into an atomic — NEVER stat'ed
    /// on the flush path itself: a synchronous /tmp I/O there is a present-path stall
    /// source of exactly the hard-to-attribute kind the present-miss work chases. The
    /// flush path pays one relaxed load; the poller (windowed runs only) costs 2
    /// wakes/s. Replaces the pre-default-on force-ON marker of the 0017 era.
    fn fence_present_enabled() -> bool {
        use std::sync::atomic::{AtomicBool, Ordering};
        static POLICY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static FORCE_OFF: AtomicBool = AtomicBool::new(false);
        static POLLER: std::sync::Once = std::sync::Once::new();
        let on = *POLICY.get_or_init(|| {
            Self::fence_present_policy(
                std::env::var("LIMINA_FENCE_PRESENT").ok().as_deref(),
                std::env::var_os("LIMINA_SHOWN_ACK_FD").is_some(),
            )
        });
        if !on {
            return false;
        }
        POLLER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("gpu fence-toggle".into())
                .spawn(|| loop {
                    FORCE_OFF.store(
                        std::fs::metadata("/tmp/disable-limina-fence-present").is_ok(),
                        Ordering::Relaxed,
                    );
                    std::thread::sleep(std::time::Duration::from_millis(500));
                });
        });
        !FORCE_OFF.load(Ordering::Relaxed)
    }

    /// limina (#8): park a zero-copy scanout flush and inject a present fence on the
    /// rendering context's reserved ring. Returns false when the frame must be
    /// presented immediately instead (knob off, no renderer, unknown context, or
    /// the injection failed).
    #[cfg(target_os = "macos")]
    fn try_park_present(
        &mut self,
        scanout_id: u32,
        iosurface_id: u32,
        resource_id: u32,
        rect: &Rect,
        ctx_id: u32,
    ) -> bool {
        if ctx_id == 0 || !Self::fence_present_enabled() {
            return false;
        }
        let (Some(pf), Some(rutabaga)) = (self.present_fence.as_mut(), self.rutabaga.as_mut())
        else {
            return false;
        };

        let cookie = pf.next_cookie;
        pf.next_cookie += 1;
        pf.parked.insert(
            cookie,
            ParkedFlush {
                scanout_id,
                iosurface_id,
                resource_id,
                rect: *rect,
            },
        );
        // The flush's trailing FLAG_FENCE (patched guest kernel) will hold on this.
        pf.flush_parked_cookies.push(cookie);

        let fence = RutabagaFence {
            flags: RUTABAGA_FLAG_FENCE | RUTABAGA_FLAG_INFO_RING_IDX,
            fence_id: cookie,
            ctx_id,
            ring_idx: LIMINA_PRESENT_RING,
        };
        if let Err(e) = rutabaga.create_fence(fence) {
            warn!("present fence injection failed (ctx {ctx_id}): {e}; presenting now");
            // Roll the cookie ALL the way back: leaving it in flush_parked_cookies
            // poisons the next fenced flush's GuestFlushHold with a cookie that can
            // never present (no parked frame, no injected fence) -> the guest's
            // display fence never signals -> hard scanout wedge. Hit in the wild by
            // direct scanout outliving its client: mutter keeps flipping an exited
            // client's buffer, whose owning context is gone, so injection fails on
            // every frame.
            let pf = self.present_fence.as_mut().unwrap();
            pf.parked.remove(&cookie);
            pf.flush_parked_cookies.retain(|c| *c != cookie);
            return false;
        }
        true
    }

    /// limina (#8): drain retired present-fence cookies and present their parked
    /// frames. Runs on the worker thread (epoll on the present eventfd).
    pub fn process_retired_presents(&mut self) {
        let Some(pf) = self.present_fence.as_mut() else {
            return;
        };
        // Drain the eventfd (level cleared) before the cookies so a cookie pushed
        // after the swap re-raises the event rather than getting lost.
        let _ = pf.event.read();
        let cookies = std::mem::take(&mut *pf.retired.lock().unwrap());
        let shown_ids = std::mem::take(&mut *pf.shown.lock().unwrap());
        let mut hits: Vec<(u32, u32, u32, Rect)> = Vec::new();
        for cookie in &cookies {
            if let Some(p) = pf.parked.remove(cookie) {
                hits.push((p.scanout_id, p.iosurface_id, p.resource_id, p.rect));
                // limina (#8 half 2): the frame presents below (this same thread, before
                // anything sleeps). With acks: confirmation comes from the supervisor's
                // "shown"; without: presenting IS the confirmation (the open-loop latch
                // delay supplies the margin).
                for hold in pf.guest_holds.iter_mut() {
                    hold.unpresented.remove(cookie);
                    if !pf.ack_active {
                        hold.unconfirmed.remove(cookie);
                    }
                }
                if pf.ack_active {
                    pf.awaiting_shown.push_back((p.iosurface_id, *cookie));
                }
            }
        }
        // Ack path: a shown(X) confirms every awaited frame up to and including the
        // first X — anything applied before X is off glass once X latched.
        for id in &shown_ids {
            if !pf.awaiting_shown.iter().any(|(i, _)| i == id) {
                continue; // ack for a frame we didn't park (2D path, toggle transitions)
            }
            while let Some((i, cookie)) = pf.awaiting_shown.pop_front() {
                for hold in pf.guest_holds.iter_mut() {
                    hold.unconfirmed.remove(&cookie);
                }
                if i == *id {
                    break;
                }
            }
        }
        // Complete / arm holds. Completion is idempotent (watermark max + desc scan),
        // so the ack path and the fallback may both fire harmlessly.
        let now = std::time::Instant::now();
        let holds = std::mem::take(&mut pf.guest_holds);
        for mut hold in holds {
            // Past the creation-time ceiling the latch thread already completed the
            // fence unconditionally — drop the hold (covers frames that can never
            // present, e.g. the owning context died while its buffer was on scanout).
            if now > hold.created_at + std::time::Duration::from_millis(500) {
                continue;
            }
            if hold.unconfirmed.is_empty() {
                if pf.ack_active {
                    // Acked = latched: complete immediately.
                    pf.guest_fence_handler.call(hold.fence);
                } else if let Err(e) = pf.latch_tx.send((now + pf.latch_delay, hold.fence)) {
                    error!("latch thread gone: {e}");
                }
                continue;
            }
            match hold.fallback_at {
                // Fallback elapsed with acks still missing: the latch thread already
                // completed the fence — drop the hold (lost-ack cleanup).
                Some(t) if now > t => continue,
                Some(_) => {}
                None => {
                    if pf.ack_active && hold.unpresented.is_empty() {
                        // All frames presented, acks outstanding: arm the safety
                        // fallback in case an ack is lost (gen collapse, dropped write).
                        let fallback = now + std::time::Duration::from_millis(150);
                        hold.fallback_at = Some(fallback);
                        if let Err(e) = pf.latch_tx.send((fallback, hold.fence)) {
                            error!("latch thread gone: {e}");
                        }
                    }
                }
            }
            pf.guest_holds.push(hold);
        }
        for (scanout_id, iosurface_id, resource_id, rect) in hits {
            // Engagement oracle: proves the fence-accurate path is live (a silent
            // fallback to immediate presents would otherwise look identical). The FIRST
            // deferred present logs at INFO — one line per boot, so a production log
            // (default warn is quiet, but RUST_LOG=info costs nothing per-frame) can
            // confirm the mode without enabling the per-fence debug/trace firehose,
            // which measurably destabilizes frame pacing (~2k sync writes/s under a
            // benchmark — observed as 40-60 fps ping-pong, 2026-07-27). The periodic
            // counter stays at trace.
            static PRESENTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = PRESENTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if n == 0 {
                log::info!(
                    "virtio-gpu: fence-accurate presents ENGAGED (first deferred present, scanout {scanout_id}, iosurface {iosurface_id})"
                );
            } else if n % 512 == 0 {
                log::trace!("[FENCEPRESENT] deferred presents={n} (scanout {scanout_id}, iosurface {iosurface_id})");
            }
            match self
                .display_backend
                .present_surface(scanout_id, iosurface_id, Some(&rect))
            {
                Ok(()) => {}
                // A sink without zero-copy (headless capture): read the pixels back and
                // present them as a software frame — the deferred twin of the immediate
                // path's fallback in `flush_resource`. Before this, arming fence-present
                // on such a sink silently dropped every parked frame.
                #[cfg(target_os = "macos")]
                Err(DisplayBackendError::MethodNotSupported) => {
                    self.deferred_readback_present(scanout_id, resource_id, &rect);
                }
                Err(e) => {
                    error!("deferred present_surface failed for scanout {scanout_id}: {e}");
                }
            }
        }
    }

    /// limina (#8): readback fallback for a retired parked frame on a sink without
    /// zero-copy support. Mirrors the venus-IOSurface readback in `flush_resource`
    /// (the immediate path): pull the presented IOSurface's pixels through the
    /// renderer into a staging frame and present that.
    #[cfg(target_os = "macos")]
    fn deferred_readback_present(&mut self, scanout_id: u32, resource_id: u32, rect: &Rect) {
        let Some((scan_w, scan_h)) = self
            .scanouts
            .get(scanout_id as usize)
            .and_then(|s| s.as_ref())
            .map(|s| (s.width, s.height))
            .filter(|&(w, h)| w != 0 && h != 0)
        else {
            error!("deferred readback: scanout {scanout_id} has no geometry");
            return;
        };
        let dst_stride = scan_w as usize * ResourceFormat::BYTES_PER_PIXEL;
        let Some(rutabaga) = self.rutabaga.as_ref() else {
            return;
        };
        let (frame_id, buffer) = match self.display_backend.alloc_frame(scanout_id) {
            Ok(fb) => fb,
            Err(e) => {
                error!("deferred readback: alloc_frame failed for scanout {scanout_id}: {e}");
                return;
            }
        };
        if let Err(e) = rutabaga.read_iosurface(resource_id, buffer, dst_stride as u32, scan_h) {
            error!("deferred readback: read_iosurface failed for resource {resource_id}: {e}");
            return;
        }
        if let Err(e) = self
            .display_backend
            .present_frame(scanout_id, frame_id, Some(rect))
        {
            error!("deferred readback: present_frame failed for scanout {scanout_id}: {e}");
        }
    }

    /// limina (#8): the eventfd the worker should poll for retired present fences.
    pub fn present_event_fd(&self) -> Option<std::os::fd::RawFd> {
        self.present_fence.as_ref().map(|pf| pf.event.as_raw_fd())
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
            self.cursor_state = None;
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
        self.cursor_state = Some(super::journal::CursorSnapshot {
            width: resource.width,
            height: resource.height,
            hot_x,
            hot_y,
            format: format as u32,
            x,
            y,
            pixels,
        });
        Ok(OkNoData)
    }

    /// limina: reposition the host cursor overlay (`VIRTIO_GPU_CMD_MOVE_CURSOR`).
    pub fn move_cursor(&mut self, x: u32, y: u32) -> VirtioGpuResult {
        Self::cursor_ok(self.display_backend.move_cursor(x, y))?;
        if let Some(c) = &mut self.cursor_state {
            c.x = x;
            c.y = y;
        }
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

    /// limina runtime resize: update a scanout's preferred mode (host window resize). The EDID
    /// regenerates from these dimensions on the next `GET_EDID` (for the `Generated` case). The
    /// caller (worker) raises a config-change interrupt so the guest re-reads `display_info()`
    /// and re-modesets to the new `width`×`height`. See `docs/design/runtime-display-resize.md`.
    pub fn set_display_size(&mut self, display_id: u32, width: u32, height: u32) {
        match self.displays.get_mut(display_id as usize) {
            Some(d) => {
                debug!("set_display_size: scanout {display_id} -> {width}x{height}");
                d.width = width;
                d.height = height;
            }
            None => error!("set_display_size: invalid display id {display_id}"),
        }
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
    ///
    /// limina: this is the virgl/vrend copy-model readback path (`TRANSFER_FROM_HOST_3D`).
    /// venus (zero-copy, host-visible blobs) never needs it, so upstream left it a `panic!` —
    /// but a stock 4 KiB guest on the coexist device drives vrend, and any GL readback
    /// (`glReadPixels`, `glxinfo`, WebGL) issues `TRANSFER_FROM_HOST_3D`. Panicking here killed
    /// the GPU worker thread and wedged the whole guest (every later command blocks on a fence
    /// that never completes). Delegate to rutabaga like `transfer_write` does, and never panic.
    pub fn transfer_read(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        transfer: Transfer3D,
        buf: Option<VolatileSlice>,
    ) -> VirtioGpuResult {
        // limina software 2D: the guest backing already mirrors our host buffer (we copy it on
        // write/flush), so a host->guest readback is a no-op for these resources.
        if self.sw2d.contains_key(&resource_id) {
            return Ok(OkNoData);
        }
        // The common `TRANSFER_FROM_HOST_3D` path passes `buf = None`: rutabaga copies from the
        // host resource into the resource's attached guest iovecs. When a destination slice is
        // given instead, hand it to rutabaga as an `IoSliceMut`.
        // SAFETY: `s` is a writable guest-memory slice the descriptor chain keeps valid for the
        // duration of this command; `IoSliceMut` only borrows it (same pattern as line ~62/968).
        let mut dst = buf.map(|s| unsafe {
            IoSliceMut::new(std::slice::from_raw_parts_mut(
                s.ptr_guard_mut().as_ptr(),
                s.len(),
            ))
        });
        self.rutabaga.as_mut().ok_or(ErrUnspec)?.transfer_read(
            ctx_id,
            resource_id,
            transfer,
            dst.take(),
        )?;
        Ok(OkNoData)
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
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .create_context(ctx_id, context_init, context_name)
            .map_err(|e| {
                warn!("CTX_CREATE failed ctx={ctx_id} init={context_init:#x} name={context_name:?}: {e}");
                e
            })?;
        info!("CTX_CREATE ctx={ctx_id} init={context_init:#x} name={context_name:?}");
        // A non-venus context means a vrend/GL guest is live: from here on, Global-ring
        // fences must retire through virglrenderer's GL timeline, not the decode-time
        // sync mark (see `vrend_ctx_seen`). capset id lives in the low byte of
        // context_init; 0 = the pre-CONTEXT_INIT default, which is also virgl.
        let capset_id = context_init & 0xff;
        if capset_id != rutabaga_gfx::RUTABAGA_CAPSET_VENUS && !self.vrend_ctx_seen {
            self.vrend_ctx_seen = true;
            info!(
                "vrend context ctx={ctx_id} (capset {capset_id}): \
                 Global-ring fences now retire through virglrenderer"
            );
        }
        Ok(OkNoData)
    }

    /// Destroys a rutabaga context.
    pub fn destroy_context(&mut self, ctx_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .destroy_context(ctx_id)
            .map_err(|e| {
                warn!("CTX_DESTROY failed ctx={ctx_id}: {e}");
                e
            })?;
        info!("CTX_DESTROY ctx={ctx_id}");
        Ok(OkNoData)
    }

    /// Attaches a resource to a rutabaga context.
    pub fn context_attach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .context_attach_resource(ctx_id, resource_id)
            .map_err(|e| {
                warn!("CTX_ATTACH_RESOURCE failed ctx={ctx_id} res={resource_id}: {e}");
                e
            })?;
        Ok(OkNoData)
    }

    /// Detaches a resource from a rutabaga context.
    pub fn context_detach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga
            .as_mut()
            .ok_or(ErrUnspec)?
            .context_detach_resource(ctx_id, resource_id)
            .map_err(|e| {
                warn!("CTX_DETACH_RESOURCE failed ctx={ctx_id} res={resource_id}: {e}");
                e
            })?;
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
    /// command completed. `is_flush` marks the fence as trailing a RESOURCE_FLUSH —
    /// the only command whose fence may be held for fence-accurate presents.
    pub fn create_fence(
        &mut self,
        rutabaga_fence: RutabagaFence,
        is_flush: bool,
    ) -> VirtioGpuResult {
        // limina M9.3 trace: every guest fence request, whatever path it takes below.
        self.trace
            .fences_requested
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // limina (#8 half 2): a fenced flush whose frames were parked — hold the guest
        // fence until those frames have presented + the latch delay. The descriptor
        // parks in process_fence (watermark untouched); the latch thread retires it
        // through the regular fence handler. The patched guest kernel's
        // dma_fence_wait on this fence is what holds the commit (and the fake flip
        // event), pacing mutter honestly and keeping the buffer unwritten while CA
        // still samples it.
        let context_ring = rutabaga_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX != 0;
        if is_flush && !context_ring {
            if let Some(pf) = self.present_fence.as_mut() {
                if !pf.flush_parked_cookies.is_empty() {
                    let cookies = std::mem::take(&mut pf.flush_parked_cookies);
                    let set: std::collections::BTreeSet<u64> = cookies.into_iter().collect();
                    let now = std::time::Instant::now();
                    // Wedge-proof ceiling: whatever happens to the parked frames
                    // (lost ack, dead context, any future leak class), the guest's
                    // display fence completes by now+500ms. Completion is
                    // idempotent with the ack/fallback paths.
                    let ceiling = now + std::time::Duration::from_millis(500);
                    if let Err(e) = pf.latch_tx.send((ceiling, rutabaga_fence)) {
                        error!("latch thread gone: {e}");
                    }
                    pf.guest_holds.push(GuestFlushHold {
                        fence: rutabaga_fence,
                        unpresented: set.clone(),
                        unconfirmed: set,
                        fallback_at: None,
                        created_at: now,
                    });
                    return Ok(OkNoData);
                }
            }
        }
        self.create_fence_inner(rutabaga_fence)
    }

    fn create_fence_inner(&mut self, rutabaga_fence: RutabagaFence) -> VirtioGpuResult {
        // Route the fence by ring. Software-2D (Global-ring) commands finish synchronously
        // before their response is encoded, so the fence is already signaled by the time we
        // get here: record it as completed up-front and let process_fence() retire the
        // descriptor immediately instead of parking it forever (which would hang any guest
        // that fences a 2D command, e.g. GTK4, or the EDK2 firmware GOP).
        //
        // Context-specific fences belong to a real 3D context and go to rutabaga. Global-ring
        // fences are two different animals depending on the session:
        //  - Before any vrend context exists (firmware GOP, venus-only sessions, software-2D):
        //    every Global-ring command completes synchronously before its response is encoded,
        //    so mark the fence completed up-front. Routing it to a venus-only rutabaga would
        //    fail with ComponentError and wedge the firmware (ctx 0 isn't a venus context).
        //  - Once a vrend context exists (`vrend_ctx_seen`): the guest's GL work retires
        //    asynchronously on the host GPU, so Global-ring fences MUST go through
        //    virglrenderer (`virgl_renderer_create_fence` → glFenceSync on the sync thread →
        //    ASYNC_FENCE_CB → fence handler), or a stock guest's glFinish returns at decode
        //    time — loose fences, measured "faster than host-native" (crossmark 2026-07-28).
        //    All Global fences must then take the one GL timeline: mixing sync-marked and
        //    async fences on a single ring could retire a younger fence's watermark past an
        //    older in-flight one. 2D fences riding the GL timeline is upstream semantics
        //    (their work already completed synchronously; the sync only adds ordering).
        let context_ring = rutabaga_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX != 0;
        match self.rutabaga.as_mut() {
            Some(rutabaga) if context_ring => rutabaga.create_fence(rutabaga_fence)?,
            Some(rutabaga) if self.vrend_ctx_seen => {
                if let Err(e) = rutabaga.create_fence(rutabaga_fence) {
                    // Never wedge on a renderer refusal: fall back to the sync mark (the
                    // pre-vrend behavior) and say so once per incident class in the log.
                    warn!(
                        "global fence {} (ctx {}) refused by renderer ({e}); completing sync",
                        rutabaga_fence.fence_id, rutabaga_fence.ctx_id
                    );
                    mark_fence_completed_sync(&self.fence_state, &rutabaga_fence);
                }
            }
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
                created_at: std::time::Instant::now(),
            });

            false
        } else {
            // Already signaled at creation — retired without ever parking.
            self.trace
                .fences_retired
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

        let mut resource =
            VirtioGpuResource::new(resource_id, 0, 0, None, resource_create_blob.size);
        // limina (#8): remember the rendering context so a scanout flush of this blob
        // can inject its present fence there.
        resource.ctx_id = ctx_id;

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

    // limina (#8): fence-accurate presents default ON exactly when the supervisor's
    // shown-ack channel exists (windowed runs). Ack-less sinks (headless capture, GTK)
    // must stay off — a parked frame's deferred present_surface is unsupported there
    // and the no-ack fallback is the compositor-serializing open-loop latch. Explicit
    // env always wins, `0`/`off` meaning off.
    #[test]
    fn fence_present_defaults_on_only_with_ack_channel() {
        use super::VirtioGpu;
        assert!(VirtioGpu::fence_present_policy(None, true));
        assert!(!VirtioGpu::fence_present_policy(None, false));
        assert!(VirtioGpu::fence_present_policy(Some("1"), false));
        assert!(!VirtioGpu::fence_present_policy(Some("0"), true));
        assert!(!VirtioGpu::fence_present_policy(Some("off"), true));
        assert!(!VirtioGpu::fence_present_policy(Some("OFF"), true));
        // The historical arming value ("any set value = on") keeps working.
        assert!(VirtioGpu::fence_present_policy(Some(""), false));
    }

    // limina: a window-resize to a non-stride-aligned width gives a scanout whose visible rect is
    // narrower than the guest's (padded) framebuffer resource (e.g. a 1000-wide mode backed by a
    // 1024-wide resource). The 2D present must extract the rect at the *resource's* stride; a flat
    // copy shears every row. This guards `blit_scanout_rect` against that regression (the real bug:
    // a resized GNOME desktop rendered as diagonal stripes, pixel-confirmed 2026-06-23).
    #[test]
    fn blit_scanout_rect_de_shears_padded_resource() {
        use super::blit_scanout_rect;
        // 4px-wide visible scanout from a 6px-wide padded resource, 3 rows. Tag each pixel's first
        // byte with row*10 + col so a shear (wrong row offset) is detectable.
        let (src_w, dst_w, rows) = (6usize, 4usize, 3usize);
        let (src_stride, dst_stride) = (src_w * 4, dst_w * 4);
        let mut src = vec![0u8; src_stride * rows];
        for y in 0..rows {
            for x in 0..src_w {
                src[y * src_stride + x * 4] = (y * 10 + x) as u8;
            }
        }
        let mut dst = vec![0u8; dst_stride * rows];
        blit_scanout_rect(&mut dst, dst_stride, &src, src_stride, rows);
        // Each dst row must hold exactly the top-left dst_w pixels of the matching src row.
        for y in 0..rows {
            for x in 0..dst_w {
                assert_eq!(
                    dst[y * dst_stride + x * 4],
                    (y * 10 + x) as u8,
                    "sheared at row {y} col {x}"
                );
            }
        }
    }

    // The matching-stride fast path (resource width == scanout width) is a plain flat copy.
    #[test]
    fn blit_scanout_rect_flat_copies_when_strides_match() {
        use super::blit_scanout_rect;
        let src: Vec<u8> = (0..(4 * 4 * 2) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        blit_scanout_rect(&mut dst, 4 * 4, &src, 4 * 4, 2);
        assert_eq!(dst, src);
    }

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
            let completed = *st
                .completed_fences
                .get(&VirtioGpuRing::Global)
                .unwrap_or(&0);
            assert!(
                fence.fence_id > completed,
                "precondition: fence not yet complete"
            );
        }

        mark_fence_completed_sync(&fence_state, &fence);

        // After: the watermark covers the fence, so process_fence() retires it now.
        let st = fence_state.lock().unwrap();
        let completed = *st
            .completed_fences
            .get(&VirtioGpuRing::Global)
            .unwrap_or(&0);
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
