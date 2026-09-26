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

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::trace::GpuTraceStats;
pub use crate::virtio::gpu_snapshot::{
    CursorSnapshot, GpuJournalEntry, GpuJournalOp, GpuSnapshotPayload, ScanoutFrame,
};

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
        self.push_fenced(op, 0);
    }

    fn push_fenced(&mut self, op: GpuJournalOp, vkr_seq: u64) {
        let seq = self.seq_next;
        self.seq_next += 1;
        self.entries.push(GpuJournalEntry { seq, vkr_seq, op });
        self.recorded += 1;
        self.sync_trace();
    }

    fn prune<F: Fn(&GpuJournalOp) -> bool>(&mut self, dead: F) {
        let before = self.entries.len();
        self.entries.retain(|e| !dead(&e.op));
        self.pruned += (before - self.entries.len()) as u64;
        self.sync_trace();
    }

    /// M9.3 restore: replace the journal wholesale with the replayed snapshot's ops.
    /// The re-created world IS the payload's world, so its re-creation journal is the
    /// payload's op list; recording continues from past its highest seq.
    pub fn restore_entries(&mut self, entries: Vec<GpuJournalEntry>) {
        self.seq_next = entries.iter().map(|e| e.seq).max().unwrap_or(0) + 1;
        self.recorded += entries.len() as u64;
        self.entries = entries;
        self.sync_trace();
    }

    /// Session reset (guest device reset / dirty reset): `reset_session_state` drops the
    /// renderer's per-session contexts/resources, so nothing remains to re-create.
    pub fn reset(&mut self) {
        self.pruned += self.entries.len() as u64;
        self.entries.clear();
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
        // A re-attach reuses the first one's place, so a guest that attaches every frame does
        // not grow the journal.
        if let Some(d) = self.attach_mut(ctx_id, resource_id) {
            *d = false;
            return;
        }
        self.push(GpuJournalOp::CtxAttachResource {
            ctx_id,
            resource_id,
            detached: false,
        });
    }

    pub fn ctx_detach_resource(&mut self, ctx_id: u32, resource_id: u32) {
        if let Some(d) = self.attach_mut(ctx_id, resource_id) {
            *d = true;
        }
    }

    /// The `detached` mark of the attach record for this context and resource.
    fn attach_mut(&mut self, ctx_id: u32, resource_id: u32) -> Option<&mut bool> {
        self.entries.iter_mut().find_map(|e| match &mut e.op {
            GpuJournalOp::CtxAttachResource {
                ctx_id: c,
                resource_id: r,
                detached,
            } if *c == ctx_id && *r == resource_id => Some(detached),
            _ => None,
        })
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
        vkr_seq: u64,
    ) {
        self.push_fenced(
            GpuJournalOp::CreateBlob {
                ctx_id,
                resource_id,
                blob_mem,
                blob_flags,
                blob_id,
                size,
                backing,
            },
            vkr_seq,
        );
    }

    /// An unref prunes the blob, its map, its attaches, and any scanout binding
    /// still pointing at it. Returns the pruned blob's `(ctx_id, blob_id)` when it
    /// was a venus blob (blob_id != 0) — the caller must release the vkr journal
    /// pin its create took (the pin lives until this GLOBAL unref, not any
    /// per-context detach: cross-context shares outlive the exporter's attach).
    pub fn resource_unref(&mut self, resource_id: u32) -> Option<(u32, u64)> {
        let mut pinned = None;
        for e in &self.entries {
            if let GpuJournalOp::CreateBlob {
                ctx_id,
                resource_id: r,
                blob_id,
                ..
            } = &e.op
                && *r == resource_id
                && *blob_id != 0
            {
                pinned = Some((*ctx_id, *blob_id));
            }
        }
        self.prune(|op| match op {
            GpuJournalOp::CreateBlob { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::MapBlob { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::CtxAttachResource { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::SetScanoutBlob { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::ResourceCreate3d { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::ResourceCreate2d { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::AttachBacking { resource_id: r, .. } => *r == resource_id,
            GpuJournalOp::SetScanout { resource_id: r, .. } => *r == resource_id,
            _ => false,
        });
        pinned
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

    #[allow(clippy::too_many_arguments)]
    pub fn resource_create_3d(
        &mut self,
        resource_id: u32,
        target: u32,
        format: u32,
        bind: u32,
        width: u32,
        height: u32,
        depth: u32,
        array_size: u32,
        last_level: u32,
        nr_samples: u32,
        flags: u32,
    ) {
        self.push(GpuJournalOp::ResourceCreate3d {
            resource_id,
            target,
            format,
            bind,
            width,
            height,
            depth,
            array_size,
            last_level,
            nr_samples,
            flags,
        });
    }

    pub fn resource_create_2d(&mut self, resource_id: u32, format: u32, width: u32, height: u32) {
        self.push(GpuJournalOp::ResourceCreate2d {
            resource_id,
            format,
            width,
            height,
        });
    }

    pub fn attach_backing(&mut self, resource_id: u32, backing: Vec<(u64, usize)>) {
        self.prune(
            |op| matches!(op, GpuJournalOp::AttachBacking { resource_id: r, .. } if *r == resource_id),
        );
        self.push(GpuJournalOp::AttachBacking {
            resource_id,
            backing,
        });
    }

    pub fn detach_backing(&mut self, resource_id: u32) {
        self.prune(
            |op| matches!(op, GpuJournalOp::AttachBacking { resource_id: r, .. } if *r == resource_id),
        );
    }

    pub fn set_scanout(&mut self, scanout_id: u32, resource_id: u32, width: u32, height: u32) {
        self.prune(
            |op| matches!(op, GpuJournalOp::SetScanout { scanout_id: s, .. } if *s == scanout_id),
        );
        if resource_id != 0 {
            self.push(GpuJournalOp::SetScanout {
                scanout_id,
                resource_id,
                width,
                height,
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
        let mut classic = 0u32;
        let mut backings = 0u32;
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
                GpuJournalOp::SetScanoutBlob { .. } | GpuJournalOp::SetScanout { .. } => {
                    scanouts += 1
                }
                GpuJournalOp::ResourceCreate3d { .. } | GpuJournalOp::ResourceCreate2d { .. } => {
                    classic += 1
                }
                GpuJournalOp::AttachBacking { .. } => backings += 1,
            }
        }
        warn!(
            "[GPUTRACE] gpu journal: {} live ops (recorded={} pruned={}): ctxs={} \
             blobs={} ({} KiB) maps={} attaches={} scanouts={} classic={} backings={}",
            self.entries.len(),
            self.recorded,
            self.pruned,
            ctxs,
            blobs,
            blob_bytes / 1024,
            maps,
            attaches,
            scanouts,
            classic,
            backings
        );
    }

    /// The ordered live entries — the P1 serializer's input.
    #[allow(dead_code)]
    pub fn entries(&self) -> &[GpuJournalEntry] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Firefox's compositor attaches a video frame, makes its plane views, and detaches the
    /// frame a frame later, while the views live on. A replay must find the frame attached where
    /// the views were made, so a detach cannot erase the attach.
    #[test]
    fn a_detach_keeps_the_attach_the_replay_needs() {
        let mut j = GpuJournal::new(Arc::new(GpuTraceStats::default()));
        j.ctx_create(9, 0x2, Some("Renderer".into()));
        j.ctx_attach_resource(9, 1239);
        j.ctx_detach_resource(9, 1239);
        let attaches = j
            .entries()
            .iter()
            .filter(|e| {
                matches!(
                    e.op,
                    GpuJournalOp::CtxAttachResource {
                        ctx_id: 9,
                        resource_id: 1239,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(attaches, 1, "the detach erased the attach");
        // Attached and detached every frame: still one record, and it says what is true now.
        j.ctx_attach_resource(9, 1239);
        assert!(matches!(
            j.entries().last().map(|e| &e.op),
            Some(GpuJournalOp::CtxAttachResource {
                detached: false,
                ..
            })
        ));
        j.ctx_detach_resource(9, 1239);
        assert_eq!(j.entries().len(), 2);
        assert!(matches!(
            j.entries()[1].op,
            GpuJournalOp::CtxAttachResource { detached: true, .. }
        ));
    }
}
