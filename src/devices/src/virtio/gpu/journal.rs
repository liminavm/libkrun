// limina M9.3 (venus snapshot-replay, P0): the rutabaga/virtio-gpu-layer half of
// the two-layer re-creation journal (design: limina docs/design/venus-snapshot-replay.md).
//
// The vkr wire journal (virglrenderer vkr_journal.c) retains the venus commands
// that rebuild the in-renderer object world. This journal retains the ops that
// arrive on the virtio-gpu control queue *around* that world: context create,
// blob-resource create, blob map (guest PA), context-resource attach, and blob
// scanout binding. Replay at restore walks both journals; a CREATE_BLOB is
// ordered after the vkAllocateMemory that backs it because the guest flushes the
// ring before issuing CREATE_BLOB (the blob id lookup would fail otherwise), so
// per-layer order plus that flush guarantee is sufficient until the P1
// serializer stamps entries with the vkr sequence for an explicit fence.
//
// Same tombstone semantics as the vkr side: destroys prune, they are not
// retained. Live size is bounded by live contexts/resources, not uptime.
// Recording happens on the worker thread only (the control queue is serial), so
// no locking; the tick-visible counters go through the shared GpuTraceStats.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::trace::GpuTraceStats;

// The op payloads and `entries()` are consumed by the P1 snapshot serializer;
// until it lands, only the census reads them.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum GpuJournalOp {
    CtxCreate {
        ctx_id: u32,
        context_init: u32,
        name: Option<String>,
    },
    CtxAttachResource {
        ctx_id: u32,
        resource_id: u32,
    },
    CreateBlob {
        ctx_id: u32,
        resource_id: u32,
        blob_mem: u32,
        blob_flags: u32,
        blob_id: u64,
        size: u64,
        backing: Vec<(u64, usize)>,
    },
    /// Latest-wins per resource; the guest PA window is `shm base + offset`.
    MapBlob {
        resource_id: u32,
        offset: u64,
    },
    /// Latest-wins per scanout.
    SetScanoutBlob {
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
        format: u32,
    },
}

pub struct GpuJournalEntry {
    #[allow(dead_code)]
    pub seq: u64,
    pub op: GpuJournalOp,
}

pub struct GpuJournal {
    entries: Vec<GpuJournalEntry>,
    seq_next: u64,
    recorded: u64,
    pruned: u64,
    trace: Arc<GpuTraceStats>,
}

impl GpuJournal {
    pub fn new(trace: Arc<GpuTraceStats>) -> Self {
        Self {
            entries: Vec::new(),
            seq_next: 1,
            recorded: 0,
            pruned: 0,
            trace,
        }
    }

    fn push(&mut self, op: GpuJournalOp) {
        let seq = self.seq_next;
        self.seq_next += 1;
        self.entries.push(GpuJournalEntry { seq, op });
        self.recorded += 1;
        self.sync_trace();
    }

    fn prune<F: Fn(&GpuJournalOp) -> bool>(&mut self, dead: F) {
        let before = self.entries.len();
        self.entries.retain(|e| !dead(&e.op));
        self.pruned += (before - self.entries.len()) as u64;
        self.sync_trace();
    }

    fn sync_trace(&self) {
        self.trace
            .journal_live
            .store(self.entries.len() as u64, Ordering::Relaxed);
        self.trace
            .journal_recorded
            .store(self.recorded, Ordering::Relaxed);
        self.trace
            .journal_pruned
            .store(self.pruned, Ordering::Relaxed);
    }

    pub fn ctx_create(&mut self, ctx_id: u32, context_init: u32, name: Option<String>) {
        self.push(GpuJournalOp::CtxCreate {
            ctx_id,
            context_init,
            name,
        });
    }

    /// A context destroy prunes the context and its attaches; blob resources are
    /// lifetime-independent (they die by unref).
    pub fn ctx_destroy(&mut self, ctx_id: u32) {
        self.prune(|op| match op {
            GpuJournalOp::CtxCreate { ctx_id: c, .. } => *c == ctx_id,
            GpuJournalOp::CtxAttachResource { ctx_id: c, .. } => *c == ctx_id,
            _ => false,
        });
    }

    pub fn ctx_attach_resource(&mut self, ctx_id: u32, resource_id: u32) {
        self.push(GpuJournalOp::CtxAttachResource {
            ctx_id,
            resource_id,
        });
    }

    pub fn ctx_detach_resource(&mut self, ctx_id: u32, resource_id: u32) {
        self.prune(|op| {
            matches!(op, GpuJournalOp::CtxAttachResource { ctx_id: c, resource_id: r }
                if *c == ctx_id && *r == resource_id)
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        blob_mem: u32,
        blob_flags: u32,
        blob_id: u64,
        size: u64,
        backing: Vec<(u64, usize)>,
    ) {
        self.push(GpuJournalOp::CreateBlob {
            ctx_id,
            resource_id,
            blob_mem,
            blob_flags,
            blob_id,
            size,
            backing,
        });
    }

    /// An unref prunes the blob, its map, its attaches, and any scanout binding
    /// still pointing at it.
    pub fn resource_unref(&mut self, resource_id: u32) {
        self.prune(|op| match op {
            GpuJournalOp::CreateBlob { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::MapBlob { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::CtxAttachResource { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::SetScanoutBlob { resource_id: r, .. } => *r == resource_id,
            _ => false,
        });
    }

    pub fn map_blob(&mut self, resource_id: u32, offset: u64) {
        self.prune(
            |op| matches!(op, GpuJournalOp::MapBlob { resource_id: r, .. } if *r == resource_id),
        );
        self.push(GpuJournalOp::MapBlob {
            resource_id,
            offset,
        });
    }

    pub fn unmap_blob(&mut self, resource_id: u32) {
        self.prune(
            |op| matches!(op, GpuJournalOp::MapBlob { resource_id: r, .. } if *r == resource_id),
        );
    }

    pub fn set_scanout_blob(
        &mut self,
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
        format: u32,
    ) {
        self.prune(
            |op| matches!(op, GpuJournalOp::SetScanoutBlob { scanout_id: s, .. } if *s == scanout_id),
        );
        if resource_id != 0 {
            self.push(GpuJournalOp::SetScanoutBlob {
                scanout_id,
                resource_id,
                width,
                height,
                format,
            });
        }
    }

    /// Live-entry census for the GPUTRACE state dump.
    pub fn dump(&self) {
        let mut ctxs = 0u32;
        let mut blobs = 0u32;
        let mut maps = 0u32;
        let mut attaches = 0u32;
        let mut scanouts = 0u32;
        let mut blob_bytes = 0u64;
        for e in &self.entries {
            match &e.op {
                GpuJournalOp::CtxCreate { .. } => ctxs += 1,
                GpuJournalOp::CreateBlob { size, .. } => {
                    blobs += 1;
                    blob_bytes += size;
                }
                GpuJournalOp::MapBlob { .. } => maps += 1,
                GpuJournalOp::CtxAttachResource { .. } => attaches += 1,
                GpuJournalOp::SetScanoutBlob { .. } => scanouts += 1,
            }
        }
        warn!(
            "[GPUTRACE] gpu journal: {} live ops (recorded={} pruned={}): ctxs={} \
             blobs={} ({} KiB) maps={} attaches={} scanouts={}",
            self.entries.len(),
            self.recorded,
            self.pruned,
            ctxs,
            blobs,
            blob_bytes / 1024,
            maps,
            attaches,
            scanouts
        );
    }

    /// The ordered live entries — the P1 serializer's input.
    #[allow(dead_code)]
    pub fn entries(&self) -> &[GpuJournalEntry] {
        &self.entries
    }
}
