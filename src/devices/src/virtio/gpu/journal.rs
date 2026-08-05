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
    /// Task #19 (classic vrend replay): a classic 3D resource create. Replay
    /// re-creates it before the context's wire journal (which may bind it) is fed.
    ResourceCreate3d {
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
    },
    /// A software-2D resource create (cursor/fbdev planes).
    ResourceCreate2d {
        resource_id: u32,
        format: u32,
        width: u32,
        height: u32,
    },
    /// Latest-wins per resource: the guest iovec backing store. Replay re-attaches
    /// it (the bytes are in the restored RAM) and then re-uploads content with a
    /// full-box transfer — classic resources' canonical storage IS this backing.
    AttachBacking {
        resource_id: u32,
        backing: Vec<(u64, usize)>,
    },
    /// Latest-wins per scanout (the classic, non-blob scanout binding).
    SetScanout {
        scanout_id: u32,
        resource_id: u32,
        width: u32,
        height: u32,
    },
}

pub struct GpuJournalEntry {
    #[allow(dead_code)]
    pub seq: u64,
    /// Cross-layer ordering fence: the owning context's vkr wire-journal
    /// watermark when this op executed. Replay must feed all wire entries with
    /// seq <= this before executing this op (a CREATE_BLOB's backing
    /// vkAllocateMemory is below the fence; the ring-create that reads the blob
    /// is above it). 0 = no wire dependency.
    pub vkr_seq: u64,
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
            {
                if *r == resource_id && *blob_id != 0 {
                    pinned = Some((*ctx_id, *blob_id));
                }
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

// --- P1 snapshot payload: the GPU section of the VM snapshot file -----------
//
// One opaque byte blob assembled by the worker at snapshot and consumed by the
// worker at restore: the rutabaga-layer journal above, plus each venus
// context's serialized vkr wire journal (virglrenderer's VKJR format), plus
// the raw contents of every guest-mapped blob (rings, reply shmems, staging —
// their bytes live in HOST allocations, not in the guest-RAM dump).

const PAYLOAD_MAGIC: u32 = 0x5550_474c; // 'LGPU' LE
                                        // v2 (M9.3 P2): + memory_contents — every capturable VkDeviceMemory's raw bytes
                                        // (not just guest-mapped blobs). v3 (P2.1): + sync_states — per-context opaque
                                        // vkr sync blobs (fence status + timeline counter values) for the restore-time
                                        // sync fast-forward. v4 (P3): + cursor — the last cursor-overlay state
                                        // (UPDATE/MOVE_CURSOR are not journaled ops; without this the restored session
                                        // shows the default dot cursor until the guest next changes it). Snapshots are
                                        // single-use against their exact post-suspend disk, so no cross-version parse
                                        // compatibility is kept.
// v5 (task #19): + classic-vrend ops (ResourceCreate3d/2d, AttachBacking,
// SetScanout, tags 6..=9) and classic contexts' wire journals riding
// vkr_journals in the same VKJR format — the compositor's GL world.
const PAYLOAD_VERSION: u32 = 5;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.data.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

#[derive(Default)]
pub struct GpuSnapshotPayload {
    pub ops: Vec<GpuJournalEntry>,
    /// per venus context: (ctx_id, serialized vkr journal — VKJR format)
    pub vkr_journals: Vec<(u32, Vec<u8>)>,
    /// per guest-mapped blob: (resource_id, raw bytes at snapshot)
    pub blob_contents: Vec<(u32, Vec<u8>)>,
    /// per capturable VkDeviceMemory: (ctx_id, vkr object id, raw bytes at
    /// snapshot). The P2 class: never-mapped host allocations — textures,
    /// render targets, non-staging buffers — invisible to both the guest-RAM
    /// dump and the mapped-blob capture above.
    pub memory_contents: Vec<(u32, u64, Vec<u8>)>,
    /// per venus context: opaque vkr sync-state blob (fence signaled status +
    /// timeline semaphore counter values) applied by the restore-time sync
    /// fast-forward — see vkr_renderer_sync_export/restore in the fork.
    pub sync_states: Vec<(u32, Vec<u8>)>,
    /// v4: the last cursor-overlay state, re-applied to the display backend at
    /// restore. `None` = cursor hidden (or never set) at snapshot.
    pub cursor: Option<CursorSnapshot>,
}

/// The rendered cursor-overlay state as last handed to the display backend —
/// pixels included (≤ ~64×64×4), so restore needs no resource lookup at all.
#[derive(Clone)]
pub struct CursorSnapshot {
    pub width: u32,
    pub height: u32,
    pub hot_x: u32,
    pub hot_y: u32,
    /// `ResourceFormat` as its `repr(u32)` value (already alpha-promoted).
    pub format: u32,
    pub x: u32,
    pub y: u32,
    pub pixels: Vec<u8>,
}

impl GpuSnapshotPayload {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        put_u32(&mut buf, PAYLOAD_MAGIC);
        put_u32(&mut buf, PAYLOAD_VERSION);

        put_u32(&mut buf, self.ops.len() as u32);
        for e in &self.ops {
            put_u64(&mut buf, e.vkr_seq);
            match &e.op {
                GpuJournalOp::CtxCreate {
                    ctx_id,
                    context_init,
                    name,
                } => {
                    buf.push(1);
                    put_u32(&mut buf, *ctx_id);
                    put_u32(&mut buf, *context_init);
                    let name = name.as_deref().unwrap_or("");
                    put_u32(&mut buf, name.len() as u32);
                    buf.extend_from_slice(name.as_bytes());
                }
                GpuJournalOp::CtxAttachResource {
                    ctx_id,
                    resource_id,
                } => {
                    buf.push(2);
                    put_u32(&mut buf, *ctx_id);
                    put_u32(&mut buf, *resource_id);
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
                    buf.push(3);
                    put_u32(&mut buf, *ctx_id);
                    put_u32(&mut buf, *resource_id);
                    put_u32(&mut buf, *blob_mem);
                    put_u32(&mut buf, *blob_flags);
                    put_u64(&mut buf, *blob_id);
                    put_u64(&mut buf, *size);
                    put_u32(&mut buf, backing.len() as u32);
                    for (addr, len) in backing {
                        put_u64(&mut buf, *addr);
                        put_u64(&mut buf, *len as u64);
                    }
                }
                GpuJournalOp::MapBlob {
                    resource_id,
                    offset,
                } => {
                    buf.push(4);
                    put_u32(&mut buf, *resource_id);
                    put_u64(&mut buf, *offset);
                }
                GpuJournalOp::SetScanoutBlob {
                    scanout_id,
                    resource_id,
                    width,
                    height,
                    format,
                } => {
                    buf.push(5);
                    put_u32(&mut buf, *scanout_id);
                    put_u32(&mut buf, *resource_id);
                    put_u32(&mut buf, *width);
                    put_u32(&mut buf, *height);
                    put_u32(&mut buf, *format);
                }
                GpuJournalOp::ResourceCreate3d {
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
                } => {
                    buf.push(6);
                    for v in [
                        *resource_id,
                        *target,
                        *format,
                        *bind,
                        *width,
                        *height,
                        *depth,
                        *array_size,
                        *last_level,
                        *nr_samples,
                        *flags,
                    ] {
                        put_u32(&mut buf, v);
                    }
                }
                GpuJournalOp::ResourceCreate2d {
                    resource_id,
                    format,
                    width,
                    height,
                } => {
                    buf.push(7);
                    for v in [*resource_id, *format, *width, *height] {
                        put_u32(&mut buf, v);
                    }
                }
                GpuJournalOp::AttachBacking {
                    resource_id,
                    backing,
                } => {
                    buf.push(8);
                    put_u32(&mut buf, *resource_id);
                    put_u32(&mut buf, backing.len() as u32);
                    for (addr, len) in backing {
                        put_u64(&mut buf, *addr);
                        put_u64(&mut buf, *len as u64);
                    }
                }
                GpuJournalOp::SetScanout {
                    scanout_id,
                    resource_id,
                    width,
                    height,
                } => {
                    buf.push(9);
                    for v in [*scanout_id, *resource_id, *width, *height] {
                        put_u32(&mut buf, v);
                    }
                }
            }
        }

        put_u32(&mut buf, self.vkr_journals.len() as u32);
        for (ctx_id, bytes) in &self.vkr_journals {
            put_u32(&mut buf, *ctx_id);
            put_u64(&mut buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }

        put_u32(&mut buf, self.blob_contents.len() as u32);
        for (res_id, bytes) in &self.blob_contents {
            put_u32(&mut buf, *res_id);
            put_u64(&mut buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }

        put_u32(&mut buf, self.memory_contents.len() as u32);
        for (ctx_id, mem_id, bytes) in &self.memory_contents {
            put_u32(&mut buf, *ctx_id);
            put_u64(&mut buf, *mem_id);
            put_u64(&mut buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }

        put_u32(&mut buf, self.sync_states.len() as u32);
        for (ctx_id, bytes) in &self.sync_states {
            put_u32(&mut buf, *ctx_id);
            put_u64(&mut buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }

        // v4 cursor section: presence byte + geometry + pixels.
        match &self.cursor {
            Some(c) => {
                buf.push(1);
                for v in [c.width, c.height, c.hot_x, c.hot_y, c.format, c.x, c.y] {
                    put_u32(&mut buf, v);
                }
                put_u64(&mut buf, c.pixels.len() as u64);
                buf.extend_from_slice(&c.pixels);
            }
            None => buf.push(0),
        }

        buf
    }

    pub fn from_bytes(data: &[u8]) -> Option<GpuSnapshotPayload> {
        let mut c = Cursor { data, pos: 0 };
        if c.u32()? != PAYLOAD_MAGIC || c.u32()? != PAYLOAD_VERSION {
            return None;
        }

        let mut payload = GpuSnapshotPayload::default();
        let nops = c.u32()?;
        for i in 0..nops {
            let vkr_seq = c.u64()?;
            let tag = *c.take(1)?.first()?;
            let op = match tag {
                1 => {
                    let ctx_id = c.u32()?;
                    let context_init = c.u32()?;
                    let nlen = c.u32()? as usize;
                    let name = std::str::from_utf8(c.take(nlen)?).ok()?.to_string();
                    GpuJournalOp::CtxCreate {
                        ctx_id,
                        context_init,
                        name: if name.is_empty() { None } else { Some(name) },
                    }
                }
                2 => GpuJournalOp::CtxAttachResource {
                    ctx_id: c.u32()?,
                    resource_id: c.u32()?,
                },
                3 => {
                    let ctx_id = c.u32()?;
                    let resource_id = c.u32()?;
                    let blob_mem = c.u32()?;
                    let blob_flags = c.u32()?;
                    let blob_id = c.u64()?;
                    let size = c.u64()?;
                    let nbacking = c.u32()?;
                    let mut backing = Vec::with_capacity(nbacking as usize);
                    for _ in 0..nbacking {
                        let addr = c.u64()?;
                        let len = c.u64()? as usize;
                        backing.push((addr, len));
                    }
                    GpuJournalOp::CreateBlob {
                        ctx_id,
                        resource_id,
                        blob_mem,
                        blob_flags,
                        blob_id,
                        size,
                        backing,
                    }
                }
                4 => GpuJournalOp::MapBlob {
                    resource_id: c.u32()?,
                    offset: c.u64()?,
                },
                5 => GpuJournalOp::SetScanoutBlob {
                    scanout_id: c.u32()?,
                    resource_id: c.u32()?,
                    width: c.u32()?,
                    height: c.u32()?,
                    format: c.u32()?,
                },
                6 => GpuJournalOp::ResourceCreate3d {
                    resource_id: c.u32()?,
                    target: c.u32()?,
                    format: c.u32()?,
                    bind: c.u32()?,
                    width: c.u32()?,
                    height: c.u32()?,
                    depth: c.u32()?,
                    array_size: c.u32()?,
                    last_level: c.u32()?,
                    nr_samples: c.u32()?,
                    flags: c.u32()?,
                },
                7 => GpuJournalOp::ResourceCreate2d {
                    resource_id: c.u32()?,
                    format: c.u32()?,
                    width: c.u32()?,
                    height: c.u32()?,
                },
                8 => {
                    let resource_id = c.u32()?;
                    let nbacking = c.u32()?;
                    let mut backing = Vec::with_capacity(nbacking as usize);
                    for _ in 0..nbacking {
                        let addr = c.u64()?;
                        let len = c.u64()? as usize;
                        backing.push((addr, len));
                    }
                    GpuJournalOp::AttachBacking {
                        resource_id,
                        backing,
                    }
                }
                9 => GpuJournalOp::SetScanout {
                    scanout_id: c.u32()?,
                    resource_id: c.u32()?,
                    width: c.u32()?,
                    height: c.u32()?,
                },
                _ => return None,
            };
            payload.ops.push(GpuJournalEntry {
                seq: (i + 1) as u64,
                vkr_seq,
                op,
            });
        }

        let nvkr = c.u32()?;
        for _ in 0..nvkr {
            let ctx_id = c.u32()?;
            let len = c.u64()? as usize;
            payload.vkr_journals.push((ctx_id, c.take(len)?.to_vec()));
        }

        let nblobs = c.u32()?;
        for _ in 0..nblobs {
            let res_id = c.u32()?;
            let len = c.u64()? as usize;
            payload.blob_contents.push((res_id, c.take(len)?.to_vec()));
        }

        let nmems = c.u32()?;
        for _ in 0..nmems {
            let ctx_id = c.u32()?;
            let mem_id = c.u64()?;
            let len = c.u64()? as usize;
            payload
                .memory_contents
                .push((ctx_id, mem_id, c.take(len)?.to_vec()));
        }

        let nsyncs = c.u32()?;
        for _ in 0..nsyncs {
            let ctx_id = c.u32()?;
            let len = c.u64()? as usize;
            payload.sync_states.push((ctx_id, c.take(len)?.to_vec()));
        }

        // v4 cursor section.
        payload.cursor = match *c.take(1)?.first()? {
            0 => None,
            1 => {
                let width = c.u32()?;
                let height = c.u32()?;
                let hot_x = c.u32()?;
                let hot_y = c.u32()?;
                let format = c.u32()?;
                let x = c.u32()?;
                let y = c.u32()?;
                let len = c.u64()? as usize;
                Some(CursorSnapshot {
                    width,
                    height,
                    hot_x,
                    hot_y,
                    format,
                    x,
                    y,
                    pixels: c.take(len)?.to_vec(),
                })
            }
            _ => return None,
        };

        Some(payload)
    }
}

/// One parsed entry of a serialized vkr wire journal (virglrenderer's VKJR
/// format — see vkr_journal.h in the fork).
pub struct VkrWireEntry {
    pub seq: u64,
    pub cmd_type: u32,
    pub klass: u8,
    pub ring_key: u64,
    pub bytes: Vec<u8>,
}

/// vkr_journal.h: klass value of ring-scoped reply-stream entries, which must
/// replay on the target ring's own decoder.
pub const VKR_KLASS_RING_STREAM: u8 = 8;

const VKJR_MAGIC: u32 = 0x524a_4b56; // 'VKJR' LE

pub fn parse_vkr_journal(data: &[u8]) -> Option<Vec<VkrWireEntry>> {
    let mut c = Cursor { data, pos: 0 };
    if c.u32()? != VKJR_MAGIC || c.u32()? != 1 {
        return None;
    }
    let count = c.u32()?;
    let _reserved = c.u32()?;
    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let seq = c.u64()?;
        let cmd_type = c.u32()?;
        let klass = *c.take(1)?.first()?;
        c.take(3)?; // pad
        let ring_key = c.u64()?;
        let size = c.u32()? as usize;
        let bytes = c.take(size)?.to_vec();
        let padding = (4 - (size % 4)) % 4;
        c.take(padding)?;
        entries.push(VkrWireEntry {
            seq,
            cmd_type,
            klass,
            ring_key,
            bytes,
        });
    }
    Some(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_snapshot_payload_round_trips() {
        let payload = GpuSnapshotPayload {
            ops: vec![
                GpuJournalEntry {
                    seq: 1,
                    vkr_seq: 0,
                    op: GpuJournalOp::CtxCreate {
                        ctx_id: 3,
                        context_init: 0x2,
                        name: Some("gnome-shell".into()),
                    },
                },
                GpuJournalEntry {
                    seq: 2,
                    vkr_seq: 41,
                    op: GpuJournalOp::CreateBlob {
                        ctx_id: 3,
                        resource_id: 7,
                        blob_mem: 3,
                        blob_flags: 0x3,
                        blob_id: 12,
                        size: 0x10000,
                        backing: vec![(0x4000_0000, 0x8000), (0x5000_0000, 0x8000)],
                    },
                },
                GpuJournalEntry {
                    seq: 3,
                    vkr_seq: 41,
                    op: GpuJournalOp::MapBlob {
                        resource_id: 7,
                        offset: 0x2_0000,
                    },
                },
                GpuJournalEntry {
                    seq: 4,
                    vkr_seq: 41,
                    op: GpuJournalOp::CtxAttachResource {
                        ctx_id: 3,
                        resource_id: 7,
                    },
                },
                GpuJournalEntry {
                    seq: 5,
                    vkr_seq: 44,
                    op: GpuJournalOp::SetScanoutBlob {
                        scanout_id: 0,
                        resource_id: 7,
                        width: 1280,
                        height: 800,
                        format: 67,
                    },
                },
            ],
            vkr_journals: vec![(3, vec![0xaa; 24])],
            blob_contents: vec![(7, vec![0x5a; 64])],
            memory_contents: vec![(3, 21, vec![0xc3; 48])],
            sync_states: vec![(3, vec![0x11; 16])],
            cursor: Some(CursorSnapshot {
                width: 64,
                height: 64,
                hot_x: 4,
                hot_y: 6,
                format: 2,
                x: 800,
                y: 450,
                pixels: vec![0x7e; 64 * 64 * 4],
            }),
        };
        let bytes = payload.to_bytes();
        let got = GpuSnapshotPayload::from_bytes(&bytes).expect("parse");
        assert_eq!(got.ops.len(), 5);
        assert_eq!(got.ops[1].vkr_seq, 41);
        match &got.ops[1].op {
            GpuJournalOp::CreateBlob {
                resource_id,
                blob_id,
                backing,
                ..
            } => {
                assert_eq!(*resource_id, 7);
                assert_eq!(*blob_id, 12);
                assert_eq!(backing, &vec![(0x4000_0000, 0x8000), (0x5000_0000, 0x8000)]);
            }
            other => panic!("wrong op: {other:?}"),
        }
        match &got.ops[0].op {
            GpuJournalOp::CtxCreate { name, .. } => {
                assert_eq!(name.as_deref(), Some("gnome-shell"))
            }
            other => panic!("wrong op: {other:?}"),
        }
        assert_eq!(got.vkr_journals, vec![(3, vec![0xaa; 24])]);
        assert_eq!(got.blob_contents, vec![(7, vec![0x5a; 64])]);
        assert_eq!(got.memory_contents, vec![(3, 21, vec![0xc3; 48])]);
        assert_eq!(got.sync_states, vec![(3, vec![0x11; 16])]);
        let cur = got.cursor.expect("cursor state present");
        assert_eq!(
            (cur.width, cur.height, cur.hot_x, cur.hot_y),
            (64, 64, 4, 6)
        );
        assert_eq!((cur.format, cur.x, cur.y), (2, 800, 450));
        assert_eq!(cur.pixels, vec![0x7e; 64 * 64 * 4]);
    }

    #[test]
    fn gpu_snapshot_payload_rejects_garbage() {
        assert!(GpuSnapshotPayload::from_bytes(b"not a payload").is_none());
        // Truncation mid-stream fails closed, not a panic.
        let bytes = GpuSnapshotPayload {
            ops: vec![GpuJournalEntry {
                seq: 1,
                vkr_seq: 0,
                op: GpuJournalOp::MapBlob {
                    resource_id: 1,
                    offset: 0,
                },
            }],
            ..Default::default()
        }
        .to_bytes();
        assert!(GpuSnapshotPayload::from_bytes(&bytes[..bytes.len() - 3]).is_none());
    }

    #[test]
    fn vkr_journal_parses_padded_entries() {
        // Hand-build a 2-entry VKJR blob: sizes 6 (pad 2) and 8 (pad 0), one ring-stream.
        let mut v = Vec::new();
        v.extend_from_slice(&VKJR_MAGIC.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes()); // version
        v.extend_from_slice(&2u32.to_le_bytes()); // count
        v.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // entry 1: seq=5, cmd_type=17, klass=1 (CREATE), ring 0, 6 bytes + 2 pad
        v.extend_from_slice(&5u64.to_le_bytes());
        v.extend_from_slice(&17u32.to_le_bytes());
        v.push(1);
        v.extend_from_slice(&[0; 3]);
        v.extend_from_slice(&0u64.to_le_bytes());
        v.extend_from_slice(&6u32.to_le_bytes());
        v.extend_from_slice(&[1, 2, 3, 4, 5, 6, 0, 0]);
        // entry 2: seq=6, cmd_type=99, klass=8 (RING_STREAM), ring key, 8 bytes
        v.extend_from_slice(&6u64.to_le_bytes());
        v.extend_from_slice(&99u32.to_le_bytes());
        v.push(VKR_KLASS_RING_STREAM);
        v.extend_from_slice(&[0; 3]);
        v.extend_from_slice(&0xdead_beefu64.to_le_bytes());
        v.extend_from_slice(&8u32.to_le_bytes());
        v.extend_from_slice(&[9, 8, 7, 6, 5, 4, 3, 2]);

        let entries = parse_vkr_journal(&v).expect("parse");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 5);
        assert_eq!(entries[0].bytes, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(entries[1].klass, VKR_KLASS_RING_STREAM);
        assert_eq!(entries[1].ring_key, 0xdead_beef);
        assert_eq!(entries[1].bytes.len(), 8);
        assert!(parse_vkr_journal(&v[..v.len() - 1]).is_none());
    }
}
