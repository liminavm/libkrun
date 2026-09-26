// The GPU section of the VM snapshot file, as bytes: the journal's ops, each venus context's
// wire journal, the contents of mapped blobs and device memory, and the display state, written
// at snapshot and read back at restore (limina docs/design/venus-snapshot-replay.md).
//
// Split out of `gpu/journal.rs` so it builds without the `gpu` feature: the reader takes bytes a
// file hands back after a crash, a partial write or a version skew, and the fuzz target that
// holds it to "read or refuse, never panic" should not have to build a renderer to get here.

use crate::display::{
    DetailedMode, DisplayInfo, DisplayInfoEdid, EdidIdentity, EdidParams, PhysicalSize,
    RefreshRange, StandardTiming, StandardTimings,
};

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
    /// One per context and resource, kept at the first attach. A detach only marks it:
    /// what the context made while the resource was attached (a sampler view) outlives the
    /// detach, and its replay needs the resource attached again. Restore re-attaches, replays,
    /// and then applies the detach.
    CtxAttachResource {
        ctx_id: u32,
        resource_id: u32,
        detached: bool,
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
    MapBlob { resource_id: u32, offset: u64 },
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
// v6 (task #19 P2): + classic_contents — per classic context, the opaque vrend
// host-side resource content blob (icon atlases, glyph caches: textures whose
// only copy lives in the host GL object; the guest-shadow re-upload can't
// restore those).
// v7: + displays — the scanout/EDID configuration the device was answering
// GET_DISPLAY_INFO/GET_EDID with at snapshot. The restored worker builds its
// virtio-gpu with the default display config, and the already-running guest
// re-probes as soon as it resumes: without this it briefly sees virtio's
// default 10" panel, picks a 250% scale for it, and constrains every window to
// the resulting tiny logical screen — geometry the guest keeps after the host's
// real EDID lands a moment later.
// v8: + scanout_frames — the pixels the host was actually presenting for each
// enabled scanout. A Vulkan compositor's framebuffers are device-local, so
// `vkr_device_memory_content_copy`'s vkMapMemory refuses them and the snapshot
// carries no copy of the screen; the guest then resumes with a damage-tracking
// compositor that has no reason to repaint, and the desktop stays blank until
// something unrelated forces a full recomposite. These are the one copy of those
// pixels that certainly exists — the host is displaying them — and they need no
// guest cooperation to capture or to put back.
const PAYLOAD_VERSION: u32 = 8;

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_opt<T>(buf: &mut Vec<u8>, v: &Option<T>, f: impl FnOnce(&mut Vec<u8>, &T)) {
    match v {
        Some(v) => {
            buf.push(1);
            f(buf, v);
        }
        None => buf.push(0),
    }
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// `n` is often a 64-bit length out of the payload, so the end is checked, not added.
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.data.get(self.pos..self.pos.checked_add(n)?)?;
        self.pos += n;
        Some(s)
    }
    /// A count of entries of `each` bytes that sizes an allocation before they are read, refused
    /// when the bytes left cannot hold that many. Trusted as read, one count asked for 38.8 GB.
    fn count(&mut self, each: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n <= (self.data.len() - self.pos) / each).then_some(n)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> Option<u8> {
        Some(*self.take(1)?.first()?)
    }
    fn opt<T>(&mut self, f: impl FnOnce(&mut Self) -> Option<T>) -> Option<Option<T>> {
        match self.u8()? {
            0 => Some(None),
            1 => Some(Some(f(self)?)),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct GpuSnapshotPayload {
    pub ops: Vec<GpuJournalEntry>,
    /// per journaled context: (ctx_id, the renderer's serialized journal — opaque here,
    /// and not one format: the classic and venus renderers each write their own)
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
    /// v6: per classic context, the opaque vrend resource-content blob
    /// (virgl_renderer_limina_classic_content_export), uploaded back after the
    /// context's wire replay.
    pub classic_contents: Vec<(u32, Vec<u8>)>,
    /// v7: the per-scanout display configuration (size, position, EDID, connected)
    /// in force at snapshot, re-applied to the restored device before the guest
    /// resumes and re-probes.
    pub displays: Vec<DisplayInfo>,
    /// v8: the frame the host was presenting on each enabled scanout, re-presented
    /// after the restore's scanout flips so the first frame the user sees is the
    /// desktop they left rather than whatever the fresh renderer starts with.
    pub scanout_frames: Vec<ScanoutFrame>,
}

/// One scanout's presented pixels, tightly packed BGRA/RGBA at `width * 4` stride —
/// whatever the display backend's staging buffer holds, copied verbatim, because it
/// goes straight back into that same buffer on restore.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScanoutFrame {
    pub scanout_id: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// The rendered cursor-overlay state as last handed to the display backend —
/// pixels included (≤ ~64×64×4), so restore needs no resource lookup at all.
#[derive(Clone)]
pub struct CursorSnapshot {
    /// Which scanout the cursor was on. A guest enables the cursor plane on one CRTC at a time,
    /// so a single snapshot plus its scanout is the whole state.
    pub scanout_id: u32,
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
                    detached,
                } => {
                    // A detached attach takes its own tag, so snapshots written before the mark
                    // existed still read.
                    buf.push(if *detached { 10 } else { 2 });
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
        //
        // The presence byte carries the layout: 1 is the original (no scanout id, implicitly 0),
        // 2 adds it in front. Extending the tag rather than the section's version keeps every
        // v4-and-later snapshot readable — a cursor is the one piece of state where being wrong
        // is merely cosmetic, so refusing to restore an older one would cost more than it saves.
        match &self.cursor {
            Some(c) => {
                buf.push(2);
                put_u32(&mut buf, c.scanout_id);
                for v in [c.width, c.height, c.hot_x, c.hot_y, c.format, c.x, c.y] {
                    put_u32(&mut buf, v);
                }
                put_u64(&mut buf, c.pixels.len() as u64);
                buf.extend_from_slice(&c.pixels);
            }
            None => buf.push(0),
        }

        // v6 classic-contents section.
        put_u32(&mut buf, self.classic_contents.len() as u32);
        for (ctx_id, bytes) in &self.classic_contents {
            put_u32(&mut buf, *ctx_id);
            put_u64(&mut buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }

        // v7 displays section.
        put_u32(&mut buf, self.displays.len() as u32);
        for d in &self.displays {
            put_display(&mut buf, d);
        }

        // v8 scanout-frames section.
        put_u32(&mut buf, self.scanout_frames.len() as u32);
        for f in &self.scanout_frames {
            put_u32(&mut buf, f.scanout_id);
            put_u32(&mut buf, f.width);
            put_u32(&mut buf, f.height);
            put_u64(&mut buf, f.pixels.len() as u64);
            buf.extend_from_slice(&f.pixels);
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
                2 | 10 => GpuJournalOp::CtxAttachResource {
                    ctx_id: c.u32()?,
                    resource_id: c.u32()?,
                    detached: tag == 10,
                },
                3 => {
                    let ctx_id = c.u32()?;
                    let resource_id = c.u32()?;
                    let blob_mem = c.u32()?;
                    let blob_flags = c.u32()?;
                    let blob_id = c.u64()?;
                    let size = c.u64()?;
                    let nbacking = c.count(16)?;
                    let mut backing = Vec::with_capacity(nbacking);
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
                    let nbacking = c.count(16)?;
                    let mut backing = Vec::with_capacity(nbacking);
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
            tag @ (1 | 2) => {
                // Tag 1 predates multi-head: its cursor was always scanout 0.
                let scanout_id = if tag == 2 { c.u32()? } else { 0 };
                let width = c.u32()?;
                let height = c.u32()?;
                let hot_x = c.u32()?;
                let hot_y = c.u32()?;
                let format = c.u32()?;
                let x = c.u32()?;
                let y = c.u32()?;
                let len = c.u64()? as usize;
                Some(CursorSnapshot {
                    scanout_id,
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

        // v6 classic-contents section.
        let n = c.u32()?;
        for _ in 0..n {
            let ctx_id = c.u32()?;
            let len = c.u64()? as usize;
            payload
                .classic_contents
                .push((ctx_id, c.take(len)?.to_vec()));
        }

        // v7 displays section.
        let n = c.u32()?;
        for _ in 0..n {
            payload.displays.push(get_display(&mut c)?);
        }

        // v8 scanout-frames section.
        let n = c.u32()?;
        for _ in 0..n {
            let scanout_id = c.u32()?;
            let width = c.u32()?;
            let height = c.u32()?;
            let len = c.u64()? as usize;
            let pixels = c.take(len)?.to_vec();
            payload.scanout_frames.push(ScanoutFrame {
                scanout_id,
                width,
                height,
                pixels,
            });
        }

        Some(payload)
    }
}

fn put_display(buf: &mut Vec<u8>, d: &DisplayInfo) {
    put_u32(buf, d.width);
    put_u32(buf, d.height);
    put_u32(buf, d.position.0);
    put_u32(buf, d.position.1);
    buf.push(d.connected as u8);
    match &d.edid {
        DisplayInfoEdid::Generated(p) => {
            buf.push(0);
            put_edid_params(buf, p);
        }
        DisplayInfoEdid::Provided(bytes) => {
            buf.push(1);
            put_u64(buf, bytes.len() as u64);
            buf.extend_from_slice(bytes);
        }
    }
}

fn put_edid_params(buf: &mut Vec<u8>, p: &EdidParams) {
    put_u32(buf, p.refresh_rate);
    match p.physical_size {
        PhysicalSize::Dpi(dpi) => {
            buf.push(0);
            put_u32(buf, dpi);
        }
        PhysicalSize::DimensionsMillimeters(w, h) => {
            buf.push(1);
            put_u16(buf, w);
            put_u16(buf, h);
        }
    }
    put_opt(buf, &p.identity, |buf, id| {
        buf.extend_from_slice(&id.manufacturer);
        put_u16(buf, id.product_id);
        put_u32(buf, id.serial);
        buf.extend_from_slice(&id.product_name);
        put_opt(buf, &id.serial_string, |buf, s| buf.extend_from_slice(s));
    });
    put_opt(buf, &p.range, |buf, r| {
        buf.push(r.min_vertical_hz);
        buf.push(r.max_vertical_hz);
        put_u16(buf, r.min_horizontal_khz);
        put_u16(buf, r.max_horizontal_khz);
        put_u32(buf, r.max_pixel_clock_mhz);
    });
    put_opt(buf, &p.standard_timings, |buf, timings| {
        for t in timings.iter() {
            put_opt(buf, t, |buf, t| {
                put_u16(buf, t.width);
                put_u16(buf, t.height);
                put_u16(buf, t.refresh_hz);
            });
        }
    });
    put_opt(buf, &p.alt_mode, |buf, m| {
        put_u32(buf, m.width);
        put_u32(buf, m.height);
        put_u32(buf, m.refresh_hz);
    });
}

fn get_display(c: &mut Cursor) -> Option<DisplayInfo> {
    let width = c.u32()?;
    let height = c.u32()?;
    let position = (c.u32()?, c.u32()?);
    let connected = match c.u8()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    let edid = match c.u8()? {
        0 => DisplayInfoEdid::Generated(get_edid_params(c)?),
        1 => {
            let len = c.u64()? as usize;
            DisplayInfoEdid::Provided(c.take(len)?.to_vec().into_boxed_slice())
        }
        _ => return None,
    };
    Some(DisplayInfo {
        width,
        height,
        position,
        edid,
        connected,
    })
}

fn get_edid_params(c: &mut Cursor) -> Option<EdidParams> {
    let refresh_rate = c.u32()?;
    let physical_size = match c.u8()? {
        0 => PhysicalSize::Dpi(c.u32()?),
        1 => PhysicalSize::DimensionsMillimeters(c.u16()?, c.u16()?),
        _ => return None,
    };
    let identity = c.opt(|c| {
        let manufacturer: [u8; 3] = c.take(3)?.try_into().unwrap();
        let product_id = c.u16()?;
        let serial = c.u32()?;
        let product_name: [u8; 13] = c.take(13)?.try_into().unwrap();
        let serial_string = c.opt(|c| Some(<[u8; 13]>::try_from(c.take(13)?).unwrap()))?;
        Some(EdidIdentity {
            manufacturer,
            product_id,
            serial,
            product_name,
            serial_string,
        })
    })?;
    let range = c.opt(|c| {
        Some(RefreshRange {
            min_vertical_hz: c.u8()?,
            max_vertical_hz: c.u8()?,
            min_horizontal_khz: c.u16()?,
            max_horizontal_khz: c.u16()?,
            max_pixel_clock_mhz: c.u32()?,
        })
    })?;
    let standard_timings = c.opt(|c| {
        let mut timings: StandardTimings = [None; 8];
        for slot in timings.iter_mut() {
            *slot = c.opt(|c| {
                Some(StandardTiming {
                    width: c.u16()?,
                    height: c.u16()?,
                    refresh_hz: c.u16()?,
                })
            })?;
        }
        Some(timings)
    })?;
    let alt_mode = c.opt(|c| {
        Some(DetailedMode {
            width: c.u32()?,
            height: c.u32()?,
            refresh_hz: c.u32()?,
        })
    })?;
    Some(EdidParams {
        refresh_rate,
        physical_size,
        identity,
        range,
        standard_timings,
        alt_mode,
    })
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
                        detached: false,
                    },
                },
                GpuJournalEntry {
                    seq: 6,
                    vkr_seq: 41,
                    op: GpuJournalOp::CtxAttachResource {
                        ctx_id: 4,
                        resource_id: 7,
                        detached: true,
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
                scanout_id: 0,
                width: 64,
                height: 64,
                hot_x: 4,
                hot_y: 6,
                format: 2,
                x: 800,
                y: 450,
                pixels: vec![0x7e; 64 * 64 * 4],
            }),
            classic_contents: vec![(9, vec![0x42; 32])],
            scanout_frames: vec![ScanoutFrame {
                scanout_id: 1,
                width: 3,
                height: 2,
                pixels: (0..24u8).collect(),
            }],
            displays: vec![
                DisplayInfo {
                    width: 2560,
                    height: 1440,
                    position: (100, 200),
                    edid: DisplayInfoEdid::Generated(EdidParams {
                        refresh_rate: 120,
                        physical_size: PhysicalSize::DimensionsMillimeters(600, 340),
                        identity: Some(EdidIdentity {
                            manufacturer: *b"LMN",
                            product_id: 0x1234,
                            serial: 0xdead_beef,
                            product_name: *b"BenQ LCD\n    ",
                            serial_string: Some(*b"SN0123456789\n"),
                        }),
                        range: Some(RefreshRange {
                            min_vertical_hz: 48,
                            max_vertical_hz: 120,
                            min_horizontal_khz: 30,
                            max_horizontal_khz: 300,
                            max_pixel_clock_mhz: 650,
                        }),
                        standard_timings: Some([
                            Some(StandardTiming {
                                width: 1920,
                                height: 1080,
                                refresh_hz: 60,
                            }),
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            Some(StandardTiming {
                                width: 1280,
                                height: 720,
                                refresh_hz: 75,
                            }),
                        ]),
                        alt_mode: Some(DetailedMode {
                            width: 2560,
                            height: 1440,
                            refresh_hz: 60,
                        }),
                    }),
                    connected: true,
                },
                DisplayInfo {
                    width: 800,
                    height: 600,
                    position: (0, 0),
                    edid: DisplayInfoEdid::Provided(vec![0x9c; 128].into_boxed_slice()),
                    connected: false,
                },
            ],
        };
        let bytes = payload.to_bytes();
        let got = GpuSnapshotPayload::from_bytes(&bytes).expect("parse");
        assert_eq!(got.ops.len(), 6);
        assert_eq!(got.ops[1].vkr_seq, 41);
        let attaches: Vec<(u32, bool)> = got
            .ops
            .iter()
            .filter_map(|e| match e.op {
                GpuJournalOp::CtxAttachResource {
                    ctx_id, detached, ..
                } => Some((ctx_id, detached)),
                _ => None,
            })
            .collect();
        assert_eq!(attaches, vec![(3, false), (4, true)]);
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
        assert_eq!(got.classic_contents, vec![(9, vec![0x42; 32])]);
        // v8: the presented pixels are the only copy of a Vulkan compositor's screen, so a
        // silent truncation here is a blank desktop on restore, not a smaller payload.
        assert_eq!(got.scanout_frames, payload.scanout_frames);

        assert_eq!(got.displays.len(), 2);
        let d0 = &got.displays[0];
        assert_eq!(
            (d0.width, d0.height, d0.position, d0.connected),
            (2560, 1440, (100, 200), true)
        );
        let DisplayInfoEdid::Generated(p0) = &d0.edid else {
            panic!("display 0 lost its generated EDID params")
        };
        let DisplayInfoEdid::Generated(want) = &payload.displays[0].edid else {
            unreachable!()
        };
        assert_eq!(p0, want);
        let d1 = &got.displays[1];
        assert_eq!((d1.width, d1.height, d1.connected), (800, 600, false));
        let DisplayInfoEdid::Provided(bytes) = &d1.edid else {
            panic!("display 1 lost its provided EDID blob")
        };
        assert_eq!(&bytes[..], &[0x9c; 128][..]);
    }

    /// A backing count past what the payload holds is refused before it sizes anything, and a
    /// length that would run the reader past the end of the address space is refused, not added.
    #[test]
    fn gpu_snapshot_payload_refuses_counts_and_lengths_past_its_end() {
        let mut bytes = Vec::new();
        put_u32(&mut bytes, PAYLOAD_MAGIC);
        put_u32(&mut bytes, PAYLOAD_VERSION);
        put_u32(&mut bytes, 1); // one op
        put_u64(&mut bytes, 0);
        bytes.push(8); // AttachBacking
        put_u32(&mut bytes, 1);
        put_u32(&mut bytes, u32::MAX); // backing entries
        put_u64(&mut bytes, 0x1000);
        put_u64(&mut bytes, 0x1000);
        let mut c = Cursor {
            data: &bytes,
            pos: bytes.len() - 20,
        };
        assert_eq!(c.count(16), None, "u32::MAX entries in 16 bytes");
        assert!(GpuSnapshotPayload::from_bytes(&bytes).is_none());

        let mut c = Cursor {
            data: &bytes,
            pos: 1,
        };
        assert_eq!(c.take(usize::MAX), None);
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
}
