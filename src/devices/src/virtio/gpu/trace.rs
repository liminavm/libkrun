// limina M9.3 instrumentation: counted, tick-visible GPU health probes.
//
// The M9.3 restore wedge presents as "gnome-shell D-hangs after snapshot restore",
// which is merely *consistent with* host-side venus context loss (a fresh renderer
// has none of the guest's contexts/resources/fences). These probes turn that into
// counted facts:
//
//   - stale-context submissions: the guest driving ctx ids the renderer never saw
//     (SUBMIT_3D / fence creation -> InvalidContextId)
//   - unknown-resource references (map/transfer -> InvalidResourceId)
//   - a fence ledger: fences requested vs retired, plus the live outstanding set
//     with ages (requested-but-never-signaled fences ARE the wedge)
//
// Counting is always on (uncontended atomic increments). Reporting is opt-in:
// `LIMINA_GPU_TRACE=1` starts a reporter thread that logs one aggregate line per
// tick (2s). Independently of the env, the FIRST unknown-context event requests a
// one-shot renderer state dump (serviced on the worker thread, which owns the
// renderer singleton) so a wedged run is never silent about the cause.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::protocol::GpuResponse;
use super::virtio_gpu::FenceState;
use rutabaga_gfx::RutabagaError;

#[derive(Default)]
pub struct GpuTraceStats {
    /// Total SUBMIT_3D commands seen (denominator for unknown_ctx).
    pub submits: AtomicU64,
    /// Commands rejected because the renderer doesn't know the context id.
    pub unknown_ctx: AtomicU64,
    /// Commands rejected because the renderer doesn't know the resource id.
    pub unknown_resource: AtomicU64,
    /// Any other command error (the per-event warn carries the detail).
    pub errors_other: AtomicU64,
    /// Guest fences requested (VIRTIO_GPU_FLAG_FENCE control commands).
    pub fences_requested: AtomicU64,
    /// Guest fence descriptors retired (signaled or completed-at-creation).
    pub fences_retired: AtomicU64,
    /// Set on the first unknown-ctx event (and periodically by the reporter when
    /// `LIMINA_GPU_TRACE_VKR=1`); the worker thread services it by dumping the
    /// renderer's context table, then clears it. Only the worker thread may call
    /// into the renderer, so this is a request flag, not a direct call.
    pub dump_requested: AtomicBool,
    /// M9.3 P0 gpu-journal gauges (absolute, mirrored by GpuJournal::sync_trace;
    /// the journal itself is worker-thread-only, these make it tick-visible).
    pub journal_live: AtomicU64,
    pub journal_recorded: AtomicU64,
    pub journal_pruned: AtomicU64,
}

impl GpuTraceStats {
    /// Classify a failed control-queue command into the probe counters. The first
    /// unknown-context event also requests a renderer state dump (see field doc).
    pub fn classify_error(&self, resp: &GpuResponse) {
        match resp {
            GpuResponse::ErrRutabaga(RutabagaError::InvalidContextId)
            | GpuResponse::ErrInvalidContextId => {
                if self.unknown_ctx.fetch_add(1, Ordering::Relaxed) == 0 {
                    self.dump_requested.store(true, Ordering::Relaxed);
                }
            }
            GpuResponse::ErrRutabaga(RutabagaError::InvalidResourceId)
            | GpuResponse::ErrInvalidResourceId => {
                self.unknown_resource.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.errors_other.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// One tick's worth of deltas, plus the absolute outstanding-fence picture.
struct TickSnapshot {
    submits: u64,
    unknown_ctx: u64,
    unknown_resource: u64,
    errors_other: u64,
    fences_requested: u64,
    fences_retired: u64,
}

fn delta(counter: &AtomicU64, prev: &mut u64) -> u64 {
    let now = counter.load(Ordering::Relaxed);
    let d = now - *prev;
    *prev = now;
    d
}

/// Spawn the tick reporter when `LIMINA_GPU_TRACE=1`. One aggregate warn-level
/// line per 2s tick (opt-in, so warn is the right visibility), plus the
/// outstanding-fence set whenever it is non-empty. `LIMINA_GPU_TRACE_VKR=1`
/// additionally requests a renderer context-table dump every 10th tick.
pub fn maybe_spawn_reporter(stats: Arc<GpuTraceStats>, fence_state: Arc<Mutex<FenceState>>) {
    if std::env::var("LIMINA_GPU_TRACE").map(|v| v == "1") != Ok(true) {
        return;
    }
    let vkr_ticks = std::env::var("LIMINA_GPU_TRACE_VKR").map(|v| v == "1") == Ok(true);

    std::thread::Builder::new()
        .name("gpu trace".into())
        .spawn(move || {
            let mut prev = TickSnapshot {
                submits: 0,
                unknown_ctx: 0,
                unknown_resource: 0,
                errors_other: 0,
                fences_requested: 0,
                fences_retired: 0,
            };
            let mut tick: u64 = 0;
            loop {
                std::thread::sleep(Duration::from_secs(2));
                tick += 1;

                let submits = delta(&stats.submits, &mut prev.submits);
                let unknown_ctx = delta(&stats.unknown_ctx, &mut prev.unknown_ctx);
                let unknown_res = delta(&stats.unknown_resource, &mut prev.unknown_resource);
                let errs = delta(&stats.errors_other, &mut prev.errors_other);
                let freq = delta(&stats.fences_requested, &mut prev.fences_requested);
                let fret = delta(&stats.fences_retired, &mut prev.fences_retired);

                // The absolute outstanding set: (ring, fence_id, age). This is the
                // wedge signature — post-restore, requested keeps growing while
                // retired stays flat and these ages climb.
                let outstanding = {
                    let fs = fence_state.lock().unwrap();
                    fs.outstanding_summary(Instant::now())
                };

                let jl = stats.journal_live.load(Ordering::Relaxed);
                let jr = stats.journal_recorded.load(Ordering::Relaxed);
                let jp = stats.journal_pruned.load(Ordering::Relaxed);

                warn!(
                    "[GPUTRACE] tick={tick} submits=+{submits} unknown_ctx=+{unknown_ctx} \
                     unknown_res=+{unknown_res} errs=+{errs} fences_req=+{freq} \
                     fences_ret=+{fret} outstanding={outstanding} journal={jl}/{jr}/{jp}"
                );

                if vkr_ticks && tick % 10 == 0 {
                    stats.dump_requested.store(true, Ordering::Relaxed);
                }
            }
        })
        .expect("failed to spawn gpu trace thread");
}
