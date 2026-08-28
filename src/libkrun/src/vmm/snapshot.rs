// SPDX-License-Identifier: Apache-2.0
//! M9 VM snapshot file format + (de)serialization.
//!
//! Mechanism only — limina (the worker/supervisor) decides the path and when to snapshot. One
//! self-describing, CRC32-checked file holds every vCPU's [`VcpuState`], the VM-wide in-kernel
//! GIC blob, and the guest RAM regions (the caller skips the GPU/fs SHM window). Little-endian
//! throughout. The version is bumped on any layout change; a mismatched magic/version/CRC is a
//! hard, fail-closed error (Firecracker's "no cross-version migration" stance).

use std::fs;
use std::io;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Mutex;

use devices::legacy::GpioState;
use devices::usb_state::{
    XhciEpState, XhciEventRingState, XhciPortState, XhciSlotState, XhciState,
};
use hvf::{VcpuPower, VcpuState};
use vm_memory::{Bytes, GuestAddress, GuestMemoryMmap};

const MAGIC: &[u8; 8] = b"LIMINAS1";
// v2 adds the legacy-device (PL061 GPIO) register section after the GIC blob (M9.2 restore wake).
// v3 widens every byte-section length prefix from u32 to u64: a guest-RAM region can be ≥ 4 GiB, and
// a u32 length silently truncated it (4 GiB → 0), so restore wrote back an empty region and the guest
// resumed into blank RAM. See `put_bytes`/`Reader::bytes`.
// v4 adds (a) a memory-layout identity header (fail closed if the restore worker's RAM/SHM layout
// differs from the captured one — else the restored PTEs/device mappings silently diverge) and (b) a
// per-device virtio-transport section (M9.3): devices the guest left `DRIVER_OK` across suspend (the
// GPU, which has no s2idle PM ops so it never resets/re-negotiates) need their transport re-activated
// on restore, or the fresh worker's device stays in INIT with dead queues and the guest wedges.
// v5 adds an optional GPU re-creation section (M9.3 venus snapshot-replay): the rutabaga-layer
// journal + per-context venus wire journals + mapped-blob contents, serialized by the GPU worker
// ('LGPU' payload). Replayed into the fresh renderer before the guest resumes so the guest's
// surviving venus world-view finds its host objects again instead of a dead ring.
// v6 (M9.4 snapshot speed) replaces the raw whole-RAM section + trailing whole-file CRC with:
// a CRC32 over the head (everything before the RAM section), then per-region CHUNKED FRAMES —
// all-zero chunks become data-less hole frames (a ballooned/idle guest is mostly zeros), the rest
// are lz4 block frames (or raw, if lz4 would grow them), each with its own CRC32 over the stored
// bytes. Save streams frames from a worker pool reading guest memory directly (no whole-RAM copy —
// the old path transiently held ~2× guest RAM and CRC'd ~9 GB serially, ~54 s for an 8 GiB guest);
// restore decompresses frames back into guest memory in parallel. Same fail-closed stance: bad
// magic/version, a head CRC mismatch, or any frame CRC/shape mismatch is a hard error.
// v7 adds an optional xHCI USB controller section (the M14 suspend/resume fix). A
// snapshot-suspend tears the worker down, so the controller's host-side state — its registers, the
// command-ring walker, the event-ring producer, and the per-slot / per-endpoint ring positions —
// is lost, while the guest resumes believing xHCI's own CSS/CRS save-restore preserved it: it
// light-resumes, its first command walks a blank controller, and USB dies. See
// docs/design/usb-xhci-snapshot/ in the limina repo.
// v8 adds each vCPU's PSCI power state. Deep suspend (PSCI SYSTEM_SUSPEND, which is what
// `systemctl suspend` means once we advertise PSCI 1.0) offlines every secondary through CPU_OFF
// and parks the caller in SYSTEM_SUSPEND, so a snapshot taken through the suspend bracket now
// captures parked vCPUs as a matter of course. Their saved PC is the instruction after a PSCI
// call that never returns; resuming them there makes a secondary return from CPU_OFF and the boot
// CPU return from a suspend it did not take, and the guest never executes anything again.
const VERSION: u32 = 8;

/// v6 RAM chunk size: 4 MiB — large enough to amortize per-frame overhead, small enough to spread
/// across the worker pool and bound per-worker scratch memory.
const CHUNK_SIZE: usize = 4 << 20;
/// Frame kinds. ZERO carries no bytes (restore writes zeros); LZ4 is an lz4 block of the chunk;
/// RAW is the chunk verbatim (only when lz4 would not shrink it).
const FRAME_ZERO: u8 = 0;
const FRAME_LZ4: u8 = 1;
const FRAME_RAW: u8 = 2;

/// Worker-pool width for the RAM save/restore paths: leave headroom for the VMM's own threads,
/// and past ~8 workers the memcpy/lz4 pipeline is memory-bandwidth-bound anyway.
fn ram_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(2).clamp(2, 8))
        .unwrap_or(4)
}

fn is_all_zero(b: &[u8]) -> bool {
    // u128 lanes; the unaligned head/tail are byte-checked. (A byte-wise `all()` over 4 MiB is the
    // hot path of the whole save — this vectorizes, plain bytes don't reliably.)
    let (head, mid, tail) = unsafe { b.align_to::<u128>() };
    head.iter().all(|&x| x == 0) && mid.iter().all(|&x| x == 0) && tail.iter().all(|&x| x == 0)
}

/// Memory-layout identity of the captured VM. Restore fails closed unless the fresh worker computes an
/// identical layout — the guest's page tables and the device SHM-window mappings are baked into the
/// snapshotted RAM at these exact addresses, so a mismatch (e.g. a different `--ram-mib`) is silent
/// corruption, not a migration.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LayoutInfo {
    pub ram_start: u64,
    pub ram_last: u64,
    pub shm_start: u64,
    pub firmware: bool,
}

/// One virtqueue's negotiated register file, as the driver programmed it (guest-physical ring
/// addresses + size/ready). `next_avail`/`next_used` are deliberately NOT carried: after the
/// capture-time drain they equal `avail.idx == used.idx` in guest RAM, and the restore path derives
/// them from the restored rings (which also self-enforces the drain contract).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct QueueRegs {
    pub size: u16,
    pub ready: bool,
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
}

/// A virtio-mmio device's transport state, captured for every transport still `device_status != 0` at
/// quiesce (today exactly virtio-gpu). Restore replays this as real negotiation writes, ending in the
/// genuine `DRIVER_OK`-triggered activation. `type_id`/`mmio_base`/`irq` are identity: restore fails
/// closed if the fresh worker registered the device differently.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DeviceTransportState {
    pub type_id: u32,
    pub mmio_base: u64,
    pub irq: u32,
    pub device_status: u32,
    pub acked_features: u64,
    pub config_generation: u32,
    pub isr: u32,
    pub queues: Vec<QueueRegs>,
}

/// Everything in a snapshot except the guest RAM (v6: RAM is streamed separately as chunked
/// frames — see [`write_streaming`] / [`SnapshotFile::apply_ram`]).
pub struct SnapshotHead {
    /// Per-vCPU architectural state, in vCPU-index order.
    pub vcpus: Vec<VcpuState>,
    /// The VM-wide in-kernel GICv3 distributor/redistributor blob.
    pub gic: Vec<u8>,
    /// The PL061 GPIO register file, if the VM has a GPIO device. Restored before the guest resumes
    /// so the injected wake demuxes (M9.2). `None` when there is no GPIO device (non-macOS/aarch64).
    pub gpio: Option<GpioState>,
    /// Memory-layout identity — checked against the restore worker's layout, fail-closed.
    pub layout: LayoutInfo,
    /// Per-device virtio-transport state for devices left `DRIVER_OK` across suspend (M9.3).
    pub devices: Vec<DeviceTransportState>,
    /// v5: the GPU re-creation section ('LGPU' payload from the GPU worker), if the VM had a
    /// venus renderer with recorded state. `None` = nothing to replay (software-2D / stock guest).
    pub gpu: Option<Vec<u8>>,
    /// v7: the emulated xHCI USB controller's host-side state, restored into the fresh
    /// controller before the guest resumes so its USB devices survive the suspend transparently.
    /// `None` when the VM has no USB controller.
    pub usb: Option<XhciState>,
}

/// CRC-32 (IEEE 802.3, reflected) — a small dependency-free integrity check over the payload.
///
/// Table-driven (one byte per step, not one bit): the payload spans the whole guest RAM, so the
/// naive bit-by-bit form costs 8 iterations/byte — tens of seconds over a multi-GiB VM, enough to
/// look like a hang. The 256-entry table is built once (same reflected `0xEDB88320` polynomial, so
/// it produces byte-for-byte the same CRC as the bit-by-bit form).
fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        let mut i = 0usize;
        while i < 256 {
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 != 0 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    });
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

// --- encode ---------------------------------------------------------------------------------

fn put_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_u64(v: &mut Vec<u8>, x: u64) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_u128(v: &mut Vec<u8>, x: u128) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_bytes(v: &mut Vec<u8>, b: &[u8]) {
    // u64, not u32: a guest-RAM region routinely exceeds 4 GiB and a u32 length truncated it to 0.
    put_u64(v, b.len() as u64);
    v.extend_from_slice(b);
}
fn put_u64_slice(v: &mut Vec<u8>, s: &[u64]) {
    put_u32(v, s.len() as u32);
    for &x in s {
        put_u64(v, x);
    }
}

fn encode_vcpu(v: &mut Vec<u8>, s: &VcpuState) {
    for &x in &s.x {
        put_u64(v, x);
    }
    put_u64(v, s.pc);
    put_u64(v, s.cpsr);
    put_u64(v, s.fpcr);
    put_u64(v, s.fpsr);
    for &q in &s.q {
        put_u128(v, q);
    }
    put_u64_slice(v, &s.sysregs);
    put_u64_slice(v, &s.icc);
    put_u64(v, s.vtimer_offset);
    v.push(s.vtimer_masked as u8);
    v.push(s.pending_irq as u8);
    v.push(s.pending_fiq as u8);
    // v8: PSCI power state. Tag 0 = Running, 1 = Offline, 2 = SystemSuspended (+ entry,
    // context_id). A parked vCPU's saved PC sits after a PSCI call that must not return, so
    // without this the restore resumes it into a state the architecture forbids.
    match s.power {
        VcpuPower::Running => v.push(0),
        VcpuPower::Offline => v.push(1),
        VcpuPower::SystemSuspended { entry, context_id } => {
            v.push(2);
            put_u64(v, entry);
            put_u64(v, context_id);
        }
    }
}

fn encode_head(head: &SnapshotHead) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(MAGIC);
    put_u32(&mut v, VERSION);
    put_u32(&mut v, head.vcpus.len() as u32);
    for s in &head.vcpus {
        encode_vcpu(&mut v, s);
    }
    put_bytes(&mut v, &head.gic);
    // Legacy-device (PL061 GPIO) register section: a presence byte, then the 8 registers.
    match &head.gpio {
        Some(g) => {
            v.push(1);
            for x in [
                g.data, g.dir, g.isense, g.ibe, g.iev, g.im, g.istate, g.afsel,
            ] {
                put_u32(&mut v, x);
            }
        }
        None => v.push(0),
    }
    // v4 memory-layout identity header.
    put_u64(&mut v, head.layout.ram_start);
    put_u64(&mut v, head.layout.ram_last);
    put_u64(&mut v, head.layout.shm_start);
    v.push(head.layout.firmware as u8);
    // v4 per-device virtio-transport section.
    put_u32(&mut v, head.devices.len() as u32);
    for d in &head.devices {
        put_u32(&mut v, d.type_id);
        put_u64(&mut v, d.mmio_base);
        put_u32(&mut v, d.irq);
        put_u32(&mut v, d.device_status);
        put_u64(&mut v, d.acked_features);
        put_u32(&mut v, d.config_generation);
        put_u32(&mut v, d.isr);
        put_u32(&mut v, d.queues.len() as u32);
        for q in &d.queues {
            put_u16(&mut v, q.size);
            v.push(q.ready as u8);
            put_u64(&mut v, q.desc);
            put_u64(&mut v, q.avail);
            put_u64(&mut v, q.used);
        }
    }
    // v5 GPU re-creation section: presence byte + bytes. v6 compresses it — on a lived-in venus
    // guest the 'LGPU' payload (journal + blob + memory contents) dominates the head (hundreds of
    // MB) and lz4s well. Kind 2 = lz4 with the uncompressed size prefixed; kind 1 = raw fallback
    // when lz4 would not shrink it (tiny payloads).
    match &head.gpu {
        Some(g) => {
            let mut buf = vec![0u8; lz4_flex::block::get_maximum_output_size(g.len())];
            match lz4_flex::block::compress_into(g, &mut buf) {
                Ok(n) if n < g.len() => {
                    v.push(2);
                    put_u64(&mut v, g.len() as u64);
                    put_bytes(&mut v, &buf[..n]);
                }
                _ => {
                    v.push(1);
                    put_bytes(&mut v, g);
                }
            }
        }
        None => v.push(0),
    }
    // v7 xHCI USB controller section: a presence byte, then the register file, the ring positions
    // and the per-slot/per-endpoint state. Small and fixed-shape (no compression).
    match &head.usb {
        Some(u) => {
            v.push(1);
            encode_usb(&mut v, u);
        }
        None => v.push(0),
    }
    v
}

/// Encode a ring position (dequeue pointer + Consumer Cycle State), presence-prefixed.
fn put_ring_pos(v: &mut Vec<u8>, p: &Option<(u64, bool)>) {
    match p {
        Some((ptr, ccs)) => {
            v.push(1);
            put_u64(v, *ptr);
            v.push(*ccs as u8);
        }
        None => v.push(0),
    }
}

fn encode_usb(v: &mut Vec<u8>, u: &XhciState) {
    put_u32(v, u.usbcmd);
    put_u32(v, u.usbsts);
    put_u32(v, u.dnctrl);
    put_u64(v, u.crcr);
    put_u64(v, u.dcbaap);
    put_u32(v, u.config);
    put_u32(v, u.iman);
    put_u32(v, u.imod);
    put_u32(v, u.erstsz);
    put_u64(v, u.erstba);
    put_u64(v, u.erdp);
    v.push(u.next_address);
    v.push(u.crcr_stop_write as u8);
    // Ports: PORTSC + the attached gadget's (idVendor, idProduct) identity.
    put_u32(v, u.ports.len() as u32);
    for p in &u.ports {
        put_u32(v, p.portsc);
        match p.model_id {
            Some((vid, pid)) => {
                v.push(1);
                put_u16(v, vid);
                put_u16(v, pid);
            }
            None => v.push(0),
        }
    }
    put_ring_pos(v, &u.cmd_ring);
    match &u.event_ring {
        Some(er) => {
            v.push(1);
            put_u64(v, er.seg_base);
            put_u32(v, er.seg_size);
            put_u32(v, er.enqueue_idx);
            v.push(er.pcs as u8);
        }
        None => v.push(0),
    }
    put_u32(v, u.slots.len() as u32);
    for s in &u.slots {
        v.push(s.slot_id);
        v.push(s.port);
        v.push(s.address);
        put_u32(v, s.state);
        v.push(s.config_value);
        put_ring_pos(v, &s.ep0_ring);
        put_u32(v, s.eps.len() as u32);
        for e in &s.eps {
            v.push(e.dci);
            put_u64(v, e.ring.0);
            v.push(e.ring.1 as u8);
            v.push(e.dir_in as u8);
            put_u32(v, e.ep_type);
            put_u32(v, e.state);
            put_u64(v, e.generation);
        }
    }
    v.push(u.work_run_started as u8);
    v.push(u.work_cmd_doorbell as u8);
    v.push(u.work_cmd_abort as u8);
    put_u32(v, u.work_ep_doorbells.len() as u32);
    for (slot, dci) in &u.work_ep_doorbells {
        v.push(*slot);
        v.push(*dci);
    }
    put_u32(v, u.work_port_events.len() as u32);
    for p in &u.work_port_events {
        v.push(*p);
    }
}

/// Save-side counters, for the operator-visible summary log line.
#[derive(Default)]
pub struct SaveStats {
    /// Total guest-RAM bytes covered (the uncompressed size).
    pub ram_bytes: u64,
    /// Bytes actually written to the file.
    pub written_bytes: u64,
    pub zero_frames: u64,
    pub lz4_frames: u64,
    pub raw_frames: u64,
}

/// One produced RAM frame, in flight from a compressor worker to the writer.
struct Frame {
    off: u64,
    kind: u8,
    crc: u32,
    data: Vec<u8>,
}

/// Serialize a snapshot to `path`: the head (+ its CRC32), then each RAM region as chunked frames
/// produced by a worker pool reading `mem` directly — no whole-RAM intermediate copy. `regions` is
/// the caller's (gpa, len) list (the GPU/fs SHM window already excluded). vCPUs must be parked:
/// the workers read live guest memory.
pub fn write_streaming(
    path: &Path,
    head: &SnapshotHead,
    mem: &GuestMemoryMmap,
    regions: &[(u64, u64)],
) -> io::Result<SaveStats> {
    let file = fs::File::create(path)?;
    let mut out = io::BufWriter::with_capacity(4 << 20, file);
    let mut hv = encode_head(head);
    let hcrc = crc32(&hv);
    put_u32(&mut hv, hcrc);
    put_u32(&mut hv, regions.len() as u32);
    out.write_all(&hv)?;

    let mut stats = SaveStats {
        written_bytes: hv.len() as u64,
        ..Default::default()
    };
    let workers = ram_workers();
    for &(gpa, len) in regions {
        stats.ram_bytes += len;
        let nchunks = len.div_ceil(CHUNK_SIZE as u64) as usize;
        let mut rh = Vec::new();
        put_u64(&mut rh, gpa);
        put_u64(&mut rh, len);
        put_u32(&mut rh, CHUNK_SIZE as u32);
        put_u32(&mut rh, nchunks as u32);
        out.write_all(&rh)?;
        stats.written_bytes += rh.len() as u64;

        let next = AtomicUsize::new(0);
        std::thread::scope(|s| -> io::Result<()> {
            let (tx, rx) = sync_channel::<io::Result<Frame>>(workers * 4);
            for _ in 0..workers {
                let tx = tx.clone();
                let next = &next;
                s.spawn(move || {
                    let mut inbuf = vec![0u8; CHUNK_SIZE];
                    let mut outbuf =
                        vec![0u8; lz4_flex::block::get_maximum_output_size(CHUNK_SIZE)];
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= nchunks {
                            return;
                        }
                        let off = idx as u64 * CHUNK_SIZE as u64;
                        let clen = (len - off).min(CHUNK_SIZE as u64) as usize;
                        let res = (|| -> io::Result<Frame> {
                            mem.read_slice(&mut inbuf[..clen], GuestAddress(gpa + off))
                                .map_err(|e| {
                                    io::Error::other(format!(
                                        "snapshot: reading RAM at 0x{:x}: {e:?}",
                                        gpa + off
                                    ))
                                })?;
                            if is_all_zero(&inbuf[..clen]) {
                                return Ok(Frame {
                                    off,
                                    kind: FRAME_ZERO,
                                    crc: 0,
                                    data: Vec::new(),
                                });
                            }
                            let n = lz4_flex::block::compress_into(&inbuf[..clen], &mut outbuf)
                                .map_err(|e| io::Error::other(format!("snapshot: lz4: {e}")))?;
                            let (kind, data) = if n < clen {
                                (FRAME_LZ4, outbuf[..n].to_vec())
                            } else {
                                (FRAME_RAW, inbuf[..clen].to_vec())
                            };
                            let crc = crc32(&data);
                            Ok(Frame {
                                off,
                                kind,
                                crc,
                                data,
                            })
                        })();
                        let failed = res.is_err();
                        if tx.send(res).is_err() || failed {
                            return; // writer gone (error abort), or our own error is now delivered
                        }
                    }
                });
            }
            drop(tx);
            // Frames arrive in completion order; each carries its offset, so order is immaterial.
            for _ in 0..nchunks {
                let frame = rx
                    .recv()
                    .map_err(|_| io::Error::other("snapshot: compressor workers died"))??;
                let mut fh = Vec::with_capacity(21);
                put_u64(&mut fh, frame.off);
                fh.push(frame.kind);
                match frame.kind {
                    FRAME_ZERO => stats.zero_frames += 1,
                    _ => {
                        put_u32(&mut fh, frame.crc);
                        put_u64(&mut fh, frame.data.len() as u64);
                        if frame.kind == FRAME_LZ4 {
                            stats.lz4_frames += 1;
                        } else {
                            stats.raw_frames += 1;
                        }
                    }
                }
                out.write_all(&fh)?;
                out.write_all(&frame.data)?;
                stats.written_bytes += (fh.len() + frame.data.len()) as u64;
            }
            Ok(())
        })?;
    }
    out.flush()?;
    Ok(stats)
}

// --- decode ---------------------------------------------------------------------------------

/// A bounds-checked little-endian cursor; every read fails closed on underrun.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

fn corrupt(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("snapshot: {what}"))
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| corrupt("overflow"))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| corrupt("truncated"))?;
        self.pos = end;
        Ok(s)
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u128(&mut self) -> io::Result<u128> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().unwrap()))
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let n = self.u64()? as usize;
        Ok(self.take(n)?.to_vec())
    }
}

fn decode_u64_vec(r: &mut Reader) -> io::Result<Vec<u64>> {
    let n = r.u32()? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(r.u64()?);
    }
    Ok(out)
}

fn decode_vcpu(r: &mut Reader) -> io::Result<VcpuState> {
    let mut x = [0u64; 31];
    for slot in &mut x {
        *slot = r.u64()?;
    }
    let pc = r.u64()?;
    let cpsr = r.u64()?;
    let fpcr = r.u64()?;
    let fpsr = r.u64()?;
    let mut q = [0u128; 32];
    for slot in &mut q {
        *slot = r.u128()?;
    }
    let sysregs = decode_u64_vec(r)?;
    let icc = decode_u64_vec(r)?;
    let vtimer_offset = r.u64()?;
    let vtimer_masked = r.u8()? != 0;
    let pending_irq = r.u8()? != 0;
    let pending_fiq = r.u8()? != 0;
    let power = match r.u8()? {
        0 => VcpuPower::Running,
        1 => VcpuPower::Offline,
        2 => VcpuPower::SystemSuspended {
            entry: r.u64()?,
            context_id: r.u64()?,
        },
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown vCPU power tag {other}"),
            ));
        }
    };
    Ok(VcpuState {
        x,
        pc,
        cpsr,
        fpcr,
        fpsr,
        q,
        sysregs,
        icc,
        vtimer_offset,
        vtimer_masked,
        pending_irq,
        pending_fiq,
        power,
    })
}

/// A parsed-and-verified snapshot: the deserialized head plus the raw file, from which
/// [`Self::apply_ram`] later streams the RAM frames straight into guest memory.
pub struct SnapshotFile {
    pub head: SnapshotHead,
    raw: Vec<u8>,
    /// Byte offset of the RAM section (the u32 region count) within `raw`.
    ram_off: usize,
}

/// Restore-side counters, for the operator-visible summary log line.
#[derive(Default)]
pub struct ApplyStats {
    /// Total guest-RAM bytes reconstructed (holes included).
    pub ram_bytes: u64,
    pub zero_frames: u64,
    pub data_frames: u64,
}

/// One parsed RAM frame, referencing its stored bytes inside `SnapshotFile::raw`.
struct FrameRef {
    gpa: u64,
    ulen: usize,
    kind: u8,
    crc: u32,
    data_off: usize,
    data_len: usize,
}

impl SnapshotFile {
    /// Decompress every RAM frame into `mem` (holes are written as zeros — the fresh worker
    /// already wrote a boot payload + FDT into RAM, which the snapshot must fully overwrite).
    /// Parallel across a worker pool; frames target disjoint ranges. Fails closed on any frame
    /// CRC/shape mismatch or an out-of-range write.
    pub fn apply_ram(&self, mem: &GuestMemoryMmap) -> io::Result<ApplyStats> {
        let mut r = Reader {
            buf: &self.raw,
            pos: self.ram_off,
        };
        let mut stats = ApplyStats::default();
        let region_count = r.u32()? as usize;
        let mut frames: Vec<FrameRef> = Vec::new();
        let mut max_chunk = 0usize;
        for _ in 0..region_count {
            let gpa = r.u64()?;
            let len = r.u64()?;
            let chunk = r.u32()? as u64;
            let nframes = r.u32()? as usize;
            if chunk == 0 {
                return Err(corrupt("zero chunk size"));
            }
            if nframes as u64 != len.div_ceil(chunk) {
                return Err(corrupt("frame count does not cover the region"));
            }
            stats.ram_bytes += len;
            max_chunk = max_chunk.max(chunk as usize);
            for _ in 0..nframes {
                let off = r.u64()?;
                let kind = r.u8()?;
                if off >= len || off % chunk != 0 {
                    return Err(corrupt("frame offset out of range"));
                }
                let ulen = (len - off).min(chunk) as usize;
                let (crc, data_off, data_len) = match kind {
                    FRAME_ZERO => (0, 0, 0),
                    FRAME_LZ4 | FRAME_RAW => {
                        let crc = r.u32()?;
                        let dlen = r.u64()? as usize;
                        let start = r.pos;
                        r.take(dlen)?;
                        (crc, start, dlen)
                    }
                    _ => return Err(corrupt("bad frame kind")),
                };
                frames.push(FrameRef {
                    gpa: gpa + off,
                    ulen,
                    kind,
                    crc,
                    data_off,
                    data_len,
                });
            }
        }
        if r.pos != self.raw.len() {
            return Err(corrupt("trailing garbage after RAM section"));
        }
        stats.zero_frames = frames.iter().filter(|f| f.kind == FRAME_ZERO).count() as u64;
        stats.data_frames = frames.len() as u64 - stats.zero_frames;

        let next = AtomicUsize::new(0);
        let first_err: Mutex<Option<io::Error>> = Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..ram_workers() {
                let next = &next;
                let first_err = &first_err;
                let frames = &frames;
                let raw = &self.raw;
                s.spawn(move || {
                    let mut outbuf = vec![0u8; max_chunk];
                    let zeros = vec![0u8; max_chunk];
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= frames.len() || first_err.lock().unwrap().is_some() {
                            return;
                        }
                        let f = &frames[idx];
                        let res = (|| -> io::Result<()> {
                            let write = |buf: &[u8]| {
                                mem.write_slice(buf, GuestAddress(f.gpa)).map_err(|e| {
                                    io::Error::other(format!(
                                        "snapshot: writing RAM at 0x{:x}: {e:?}",
                                        f.gpa
                                    ))
                                })
                            };
                            match f.kind {
                                FRAME_ZERO => write(&zeros[..f.ulen]),
                                kind => {
                                    let data = &raw[f.data_off..f.data_off + f.data_len];
                                    if crc32(data) != f.crc {
                                        return Err(corrupt("frame CRC mismatch"));
                                    }
                                    if kind == FRAME_RAW {
                                        if data.len() != f.ulen {
                                            return Err(corrupt("raw frame length mismatch"));
                                        }
                                        write(data)
                                    } else {
                                        let n = lz4_flex::block::decompress_into(
                                            data,
                                            &mut outbuf[..f.ulen],
                                        )
                                        .map_err(|e| {
                                            corrupt(&format!("frame decompress failed: {e}"))
                                        })?;
                                        if n != f.ulen {
                                            return Err(corrupt(
                                                "frame decompressed size mismatch",
                                            ));
                                        }
                                        write(&outbuf[..f.ulen])
                                    }
                                }
                            }
                        })();
                        if let Err(e) = res {
                            first_err.lock().unwrap().get_or_insert(e);
                            return;
                        }
                    }
                });
            }
        });
        match first_err.into_inner().unwrap() {
            Some(e) => Err(e),
            None => Ok(stats),
        }
    }
}

/// Decode a presence-prefixed ring position.
fn decode_ring_pos(r: &mut Reader) -> io::Result<Option<(u64, bool)>> {
    match r.u8()? {
        0 => Ok(None),
        1 => {
            let ptr = r.u64()?;
            Ok(Some((ptr, decode_bool(r)?)))
        }
        _ => Err(corrupt("bad ring-position presence byte")),
    }
}

fn decode_bool(r: &mut Reader) -> io::Result<bool> {
    match r.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(corrupt("bad boolean byte")),
    }
}

/// Bound a decoded element count before it drives an allocation. The head CRC is only checked
/// *after* the whole head parses, so a corrupt length must not be trusted this far — and every
/// count here has a small hard ceiling from the controller's geometry anyway.
fn bounded_count(r: &mut Reader, max: u32, what: &str) -> io::Result<usize> {
    let n = r.u32()?;
    if n > max {
        return Err(corrupt(&format!("implausible {what} count {n}")));
    }
    Ok(n as usize)
}

fn decode_usb(r: &mut Reader) -> io::Result<XhciState> {
    let usbcmd = r.u32()?;
    let usbsts = r.u32()?;
    let dnctrl = r.u32()?;
    let crcr = r.u64()?;
    let dcbaap = r.u64()?;
    let config = r.u32()?;
    let iman = r.u32()?;
    let imod = r.u32()?;
    let erstsz = r.u32()?;
    let erstba = r.u64()?;
    let erdp = r.u64()?;
    let next_address = r.u8()?;
    let crcr_stop_write = decode_bool(r)?;

    let port_count = bounded_count(r, 256, "usb port")?;
    let mut ports = Vec::with_capacity(port_count);
    for _ in 0..port_count {
        let portsc = r.u32()?;
        let model_id = match r.u8()? {
            0 => None,
            1 => Some((r.u16()?, r.u16()?)),
            _ => return Err(corrupt("bad usb model-id presence byte")),
        };
        ports.push(XhciPortState { portsc, model_id });
    }

    let cmd_ring = decode_ring_pos(r)?;
    let event_ring = match r.u8()? {
        0 => None,
        1 => Some(XhciEventRingState {
            seg_base: r.u64()?,
            seg_size: r.u32()?,
            enqueue_idx: r.u32()?,
            pcs: decode_bool(r)?,
        }),
        _ => return Err(corrupt("bad usb event-ring presence byte")),
    };

    let slot_count = bounded_count(r, 256, "usb slot")?;
    let mut slots = Vec::with_capacity(slot_count);
    for _ in 0..slot_count {
        let slot_id = r.u8()?;
        let port = r.u8()?;
        let address = r.u8()?;
        let state = r.u32()?;
        let config_value = r.u8()?;
        let ep0_ring = decode_ring_pos(r)?;
        let ep_count = bounded_count(r, 32, "usb endpoint")?;
        let mut eps = Vec::with_capacity(ep_count);
        for _ in 0..ep_count {
            let dci = r.u8()?;
            let ptr = r.u64()?;
            eps.push(XhciEpState {
                dci,
                ring: (ptr, decode_bool(r)?),
                dir_in: decode_bool(r)?,
                ep_type: r.u32()?,
                state: r.u32()?,
                generation: r.u64()?,
            });
        }
        slots.push(XhciSlotState {
            slot_id,
            port,
            address,
            state,
            config_value,
            ep0_ring,
            eps,
        });
    }

    let work_run_started = decode_bool(r)?;
    let work_cmd_doorbell = decode_bool(r)?;
    let work_cmd_abort = decode_bool(r)?;
    let db_count = bounded_count(r, 8192, "usb doorbell")?;
    let mut work_ep_doorbells = Vec::with_capacity(db_count);
    for _ in 0..db_count {
        work_ep_doorbells.push((r.u8()?, r.u8()?));
    }
    let pe_count = bounded_count(r, 8192, "usb port event")?;
    let mut work_port_events = Vec::with_capacity(pe_count);
    for _ in 0..pe_count {
        work_port_events.push(r.u8()?);
    }

    Ok(XhciState {
        usbcmd,
        usbsts,
        dnctrl,
        crcr,
        dcbaap,
        config,
        ports,
        iman,
        imod,
        erstsz,
        erstba,
        erdp,
        cmd_ring,
        event_ring,
        slots,
        next_address,
        crcr_stop_write,
        work_run_started,
        work_cmd_doorbell,
        work_cmd_abort,
        work_ep_doorbells,
        work_port_events,
    })
}

/// Read + verify a snapshot's head from `path` (magic, version, head CRC). The RAM frames are
/// verified per-frame when [`SnapshotFile::apply_ram`] runs.
pub fn read(path: &Path) -> io::Result<SnapshotFile> {
    let raw = fs::read(path)?;
    if raw.len() < 16 {
        return Err(corrupt("too small"));
    }
    let mut r = Reader { buf: &raw, pos: 0 };
    if r.take(8)? != MAGIC {
        return Err(corrupt("bad magic"));
    }
    if r.u32()? != VERSION {
        return Err(corrupt("unsupported version"));
    }
    let vcpu_count = r.u32()? as usize;
    let mut vcpus = Vec::with_capacity(vcpu_count);
    for _ in 0..vcpu_count {
        vcpus.push(decode_vcpu(&mut r)?);
    }
    let gic = r.bytes()?;
    let gpio = match r.u8()? {
        0 => None,
        1 => Some(GpioState {
            data: r.u32()?,
            dir: r.u32()?,
            isense: r.u32()?,
            ibe: r.u32()?,
            iev: r.u32()?,
            im: r.u32()?,
            istate: r.u32()?,
            afsel: r.u32()?,
        }),
        _ => return Err(corrupt("bad gpio presence byte")),
    };
    // v4 memory-layout identity header.
    let layout = LayoutInfo {
        ram_start: r.u64()?,
        ram_last: r.u64()?,
        shm_start: r.u64()?,
        firmware: match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err(corrupt("bad layout firmware byte")),
        },
    };
    // v4 per-device virtio-transport section.
    let dev_count = r.u32()? as usize;
    let mut devices = Vec::with_capacity(dev_count);
    for _ in 0..dev_count {
        let type_id = r.u32()?;
        let mmio_base = r.u64()?;
        let irq = r.u32()?;
        let device_status = r.u32()?;
        let acked_features = r.u64()?;
        let config_generation = r.u32()?;
        let isr = r.u32()?;
        let q_count = r.u32()? as usize;
        let mut queues = Vec::with_capacity(q_count);
        for _ in 0..q_count {
            let size = r.u16()?;
            let ready = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err(corrupt("bad queue ready byte")),
            };
            queues.push(QueueRegs {
                size,
                ready,
                desc: r.u64()?,
                avail: r.u64()?,
                used: r.u64()?,
            });
        }
        devices.push(DeviceTransportState {
            type_id,
            mmio_base,
            irq,
            device_status,
            acked_features,
            config_generation,
            isr,
            queues,
        });
    }
    // v5 GPU re-creation section (v6: kind 2 = lz4 with uncompressed-size prefix).
    let gpu = match r.u8()? {
        0 => None,
        1 => Some(r.bytes()?),
        2 => {
            let ulen = r.u64()? as usize;
            let comp = r.bytes()?;
            // lz4's max ratio is 255:1 — a larger claimed size is corruption, not data (and would
            // otherwise drive a huge allocation before the head CRC is checked below).
            if ulen > comp.len().saturating_mul(256) {
                return Err(corrupt("gpu section size implausible"));
            }
            let mut out = vec![0u8; ulen];
            let n = lz4_flex::block::decompress_into(&comp, &mut out)
                .map_err(|e| corrupt(&format!("gpu section decompress failed: {e}")))?;
            if n != ulen {
                return Err(corrupt("gpu section decompressed size mismatch"));
            }
            Some(out)
        }
        _ => return Err(corrupt("bad gpu presence byte")),
    };
    // v7 xHCI USB controller section.
    let usb = match r.u8()? {
        0 => None,
        1 => Some(decode_usb(&mut r)?),
        _ => return Err(corrupt("bad usb presence byte")),
    };
    // v6: the head is covered by its own CRC (the RAM frames each carry theirs).
    let head_end = r.pos;
    let stored = r.u32()?;
    if crc32(&raw[..head_end]) != stored {
        return Err(corrupt("head CRC mismatch"));
    }
    let ram_off = r.pos;
    Ok(SnapshotFile {
        head: SnapshotHead {
            vcpus,
            gic,
            gpio,
            layout,
            devices,
            gpu,
            usb,
        },
        raw,
        ram_off,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vcpu(seed: u64) -> VcpuState {
        VcpuState {
            x: std::array::from_fn(|i| seed + i as u64),
            pc: seed + 100,
            cpsr: seed + 200,
            fpcr: seed + 300,
            fpsr: seed + 400,
            q: std::array::from_fn(|i| (seed as u128) << 64 | i as u128),
            sysregs: (0..113).map(|i| seed * 1000 + i).collect(),
            icc: (0..9).map(|i| seed * 10 + i).collect(),
            vtimer_offset: seed + 500,
            vtimer_masked: seed % 2 == 1,
            pending_irq: seed % 2 == 0,
            pending_fiq: seed % 3 == 0,
            // Vary by seed so the round-trip covers all three encodings, including the
            // variable-length one (SystemSuspended carries two u64s; the others carry none, so a
            // mis-sized tag would desync every field after it).
            power: match seed % 3 {
                0 => VcpuPower::Running,
                1 => VcpuPower::Offline,
                _ => VcpuPower::SystemSuspended {
                    entry: seed + 600,
                    context_id: seed + 700,
                },
            },
        }
    }

    fn sample_layout() -> LayoutInfo {
        LayoutInfo {
            ram_start: 0x4000_0000,
            ram_last: 0x1_4000_0000,
            shm_start: 0x1_8000_0000,
            firmware: true,
        }
    }

    fn sample_gpu_device() -> DeviceTransportState {
        DeviceTransportState {
            type_id: 16, // virtio-gpu
            mmio_base: 0x0a00_8000,
            irq: 47,
            device_status: 0xf,
            acked_features: 0x1_0000_012b,
            config_generation: 3,
            isr: 0,
            queues: vec![
                QueueRegs {
                    size: 64,
                    ready: true,
                    desc: 0x1_2340_0000,
                    avail: 0x1_2340_0400,
                    used: 0x1_2340_0800,
                },
                QueueRegs {
                    size: 16,
                    ready: true,
                    desc: 0x1_2350_0000,
                    avail: 0x1_2350_0100,
                    used: 0x1_2350_0200,
                },
            ],
        }
    }

    /// Two-region test memory: 8 MiB at 0x4000_0000 and 5 MiB at 0x8000_0000 (a non-chunk-multiple
    /// tail, so the short-final-frame path is exercised).
    const REGION_A: (u64, u64) = (0x4000_0000, 8 << 20);
    const REGION_B: (u64, u64) = (0x8000_0000, 5 << 20);

    fn test_mem() -> GuestMemoryMmap {
        GuestMemoryMmap::from_ranges(&[
            (GuestAddress(REGION_A.0), REGION_A.1 as usize),
            (GuestAddress(REGION_B.0), REGION_B.1 as usize),
        ])
        .expect("test guest memory")
    }

    /// Fill the test memory with: a whole-chunk hole (zeros), a compressible pattern, and
    /// LCG pseudorandom bytes (incompressible → exercises the RAW frame path).
    fn fill_test_mem(mem: &GuestMemoryMmap) {
        // Region A: first 4 MiB stay zero (a hole frame), second 4 MiB compressible.
        let compressible: Vec<u8> = (0..=255u8).cycle().take(4 << 20).collect();
        mem.write_slice(&compressible, GuestAddress(REGION_A.0 + (4 << 20)))
            .unwrap();
        // Region B: 5 MiB of LCG noise.
        let mut x = 0x12345678u64;
        let noise: Vec<u8> = (0..(5 << 20))
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 33) as u8
            })
            .collect();
        mem.write_slice(&noise, GuestAddress(REGION_B.0)).unwrap();
    }

    fn read_back(mem: &GuestMemoryMmap, (gpa, len): (u64, u64)) -> Vec<u8> {
        let mut buf = vec![0u8; len as usize];
        mem.read_slice(&mut buf, GuestAddress(gpa)).unwrap();
        buf
    }

    /// A lived-in xHCI controller: two populated ports (one with a gadget that has since gone),
    /// an addressed+configured slot with two data endpoints, a mid-ring command walker and a
    /// mid-segment event-ring producer.
    fn sample_usb() -> XhciState {
        XhciState {
            usbcmd: 0x5,
            usbsts: 0x8,
            dnctrl: 0x5,
            crcr: 0x8400_4001,
            dcbaap: 0x8400_6000,
            config: 8,
            ports: vec![
                XhciPortState {
                    portsc: 0x0e01_0203,
                    model_id: Some((0x04f3, 0x0c7d)),
                },
                XhciPortState {
                    portsc: 0x0000_02a0,
                    model_id: None,
                },
            ],
            iman: 0x2,
            imod: 0x40,
            erstsz: 16,
            erstba: 0x8400_8000,
            erdp: 0x8401_0030,
            cmd_ring: Some((0x8400_4020, true)),
            event_ring: Some(XhciEventRingState {
                seg_base: 0x8401_0000,
                seg_size: 16,
                enqueue_idx: 5,
                pcs: false,
            }),
            slots: vec![XhciSlotState {
                slot_id: 1,
                port: 1,
                address: 2,
                state: 3,
                config_value: 1,
                ep0_ring: Some((0x8402_0000, true)),
                eps: vec![
                    XhciEpState {
                        dci: 2,
                        ring: (0x8403_0000, true),
                        dir_in: false,
                        ep_type: 2,
                        state: 1,
                        generation: 7,
                    },
                    XhciEpState {
                        dci: 3,
                        ring: (0x8404_0010, false),
                        dir_in: true,
                        ep_type: 6,
                        state: 3,
                        generation: 9,
                    },
                ],
            }],
            next_address: 3,
            crcr_stop_write: true,
            work_run_started: true,
            work_cmd_doorbell: false,
            work_cmd_abort: true,
            work_ep_doorbells: vec![(1, 3), (1, 2)],
            work_port_events: vec![1, 2],
        }
    }

    fn sample_head() -> SnapshotHead {
        SnapshotHead {
            vcpus: vec![sample_vcpu(1), sample_vcpu(2)],
            gic: vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0x11],
            gpio: Some(GpioState {
                data: 0x40,
                dir: 0x1,
                isense: 0x2,
                ibe: 0x3,
                iev: 0x4,
                im: 0x78,
                istate: 0x40,
                afsel: 0x5,
            }),
            layout: sample_layout(),
            devices: vec![sample_gpu_device()],
            gpu: Some(vec![0x4c, 0x47, 0x50, 0x55, 9, 9]),
            usb: Some(sample_usb()),
        }
    }

    fn write_sample(path: &Path) -> (SaveStats, Vec<u8>, Vec<u8>) {
        let mem = test_mem();
        fill_test_mem(&mem);
        let stats = write_streaming(path, &sample_head(), &mem, &[REGION_A, REGION_B])
            .expect("write_streaming");
        (stats, read_back(&mem, REGION_A), read_back(&mem, REGION_B))
    }

    #[test]
    fn snapshot_file_round_trips() {
        let path =
            std::env::temp_dir().join(format!("limina-snap-test-{}.bin", std::process::id()));
        let (stats, want_a, want_b) = write_sample(&path);
        assert_eq!(stats.ram_bytes, REGION_A.1 + REGION_B.1);
        assert!(
            stats.zero_frames >= 1,
            "the zeroed chunk must become a hole"
        );
        assert!(stats.lz4_frames >= 1, "the pattern chunk must compress");
        // The hole + compression must shrink the file well below the raw RAM size.
        assert!(stats.written_bytes < stats.ram_bytes);

        let got = read(&path).expect("read");
        let head = &got.head;
        let want = sample_head();
        assert_eq!(head.vcpus.len(), 2);
        assert_eq!(head.vcpus[0].x, want.vcpus[0].x);
        assert_eq!(head.vcpus[1].q, want.vcpus[1].q);
        assert_eq!(head.vcpus[0].sysregs, want.vcpus[0].sysregs);
        assert_eq!(head.vcpus[1].icc, want.vcpus[1].icc);
        assert_eq!(head.vcpus[0].pending_irq, want.vcpus[0].pending_irq);
        // The power tag is variable-length (SystemSuspended carries two u64s, the others none),
        // so a wrong tag desyncs every field decoded after it. seed 1 is Offline, seed 2 is
        // SystemSuspended: both encodings, and the pair pins that the second vCPU decodes
        // correctly after the first one's shorter tag.
        assert_eq!(head.vcpus[0].power, VcpuPower::Offline);
        assert_eq!(
            head.vcpus[1].power,
            VcpuPower::SystemSuspended {
                entry: 2 + 600,
                context_id: 2 + 700
            }
        );
        assert_eq!(head.gic, want.gic);
        let gpio = head.gpio.as_ref().expect("gpio present");
        assert_eq!(gpio.im, 0x78);
        assert_eq!(gpio.istate, 0x40);
        assert_eq!(gpio.data, 0x40);
        assert_eq!(head.layout, want.layout);
        assert_eq!(head.devices, want.devices);
        assert_eq!(head.devices[0].queues[0].avail, 0x1_2340_0400);
        assert_eq!(head.gpu, want.gpu);
        // v7: the whole xHCI controller state, field for field (the `PartialEq` covers every
        // register, ring position, slot and endpoint — a dropped field fails here).
        assert_eq!(head.usb, want.usb);

        // Restore into memory pre-filled with garbage: data frames AND holes must both overwrite.
        let mem2 = test_mem();
        let junk = vec![0xAAu8; REGION_A.1 as usize];
        mem2.write_slice(&junk, GuestAddress(REGION_A.0)).unwrap();
        mem2.write_slice(&junk[..REGION_B.1 as usize], GuestAddress(REGION_B.0))
            .unwrap();
        let astats = got.apply_ram(&mem2).expect("apply_ram");
        let _ = fs::remove_file(&path);
        assert_eq!(astats.ram_bytes, REGION_A.1 + REGION_B.1);
        assert!(astats.zero_frames >= 1);
        assert_eq!(read_back(&mem2, REGION_A), want_a);
        assert_eq!(read_back(&mem2, REGION_B), want_b);
    }

    #[test]
    fn v4_device_transport_section_survives_no_devices() {
        // The common case (no sticky device) must encode an empty section and read back empty — not
        // desync the stream. Also pins the layout header round-trip in isolation.
        let head = SnapshotHead {
            vcpus: vec![sample_vcpu(3)],
            gic: vec![1, 2, 3, 4],
            gpio: None,
            layout: sample_layout(),
            devices: vec![],
            gpu: None,
            usb: None,
        };
        let mem = test_mem();
        let path =
            std::env::temp_dir().join(format!("limina-snap-v4-empty-{}.bin", std::process::id()));
        write_streaming(&path, &head, &mem, &[REGION_A]).expect("write");
        let got = read(&path).expect("read");
        let _ = fs::remove_file(&path);
        assert!(got.head.devices.is_empty());
        assert!(got.head.gpio.is_none());
        assert!(got.head.gpu.is_none());
        assert!(
            got.head.usb.is_none(),
            "an absent USB section must decode as None"
        );
        assert_eq!(got.head.layout, sample_layout());
    }

    /// A corrupt element count in the USB section must fail closed, not drive a huge allocation:
    /// the head CRC is only verified once the whole head has parsed, so the counts cannot be
    /// trusted while decoding.
    #[test]
    fn usb_section_rejects_implausible_counts() {
        let mut v = Vec::new();
        put_u32(&mut v, 0xffff_ffff); // a "slot count" of 4 billion
        let mut r = Reader { buf: &v, pos: 0 };
        let err = bounded_count(&mut r, 256, "usb slot").expect_err("must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn byte_section_length_is_u64_not_u32() {
        // Regression (M9.3 floor spike): a guest-RAM region ≥ 4 GiB has a length that does not fit
        // in a u32. The old encoder wrote `len as u32`, truncating a 4 GiB region's length to 0 while
        // still appending all 4 GiB of data — so the CRC matched, `write` "succeeded", and the file
        // was ~4 GiB, but on restore `bytes()` read length 0 and returned an EMPTY region. The guest
        // resumed into blank RAM and stalled. The length prefix MUST be a 64-bit value.
        let mut v = Vec::new();
        put_bytes(&mut v, &[0xABu8; 5]);
        assert_eq!(
            v.len(),
            8 + 5,
            "byte-section length must be a 64-bit prefix"
        );
        assert_eq!(
            &v[..8],
            &5u64.to_le_bytes(),
            "length encoded little-endian u64"
        );
        // And a length that would overflow u32 must survive the u64 round-trip in the header.
        let mut h = Vec::new();
        put_u64(&mut h, 0x1_0000_0000u64); // exactly 4 GiB — the value a u32 truncates to 0
        let mut r = Reader { buf: &h, pos: 0 };
        assert_eq!(r.u64().unwrap(), 0x1_0000_0000u64);
    }

    #[test]
    fn gpu_section_compresses_and_round_trips() {
        // A realistic 'LGPU' payload is large and compressible — it must take the lz4 (kind 2)
        // path and decode byte-identical. (sample_head()'s 6-byte payload covers the raw fallback.)
        let payload: Vec<u8> = (0..=255u8).cycle().take(2 << 20).collect();
        let head = SnapshotHead {
            gpu: Some(payload.clone()),
            ..sample_head()
        };
        let mem = test_mem();
        let path =
            std::env::temp_dir().join(format!("limina-snap-gpu-lz4-{}.bin", std::process::id()));
        let stats = write_streaming(&path, &head, &mem, &[REGION_B]).expect("write");
        let got = read(&path).expect("read");
        let _ = fs::remove_file(&path);
        assert_eq!(got.head.gpu.as_deref(), Some(payload.as_slice()));
        // The compressible 2 MiB payload must not have been stored raw.
        assert!(
            stats.written_bytes < payload.len() as u64,
            "gpu section should compress (file {} bytes vs {} payload)",
            stats.written_bytes,
            payload.len()
        );
    }

    #[test]
    fn snapshot_rejects_head_corruption() {
        let path = std::env::temp_dir().join(format!(
            "limina-snap-head-corrupt-{}.bin",
            std::process::id()
        ));
        let _ = write_sample(&path);
        // Flip a head byte (offset 12 is inside the vCPU section); the head CRC must catch it.
        let mut raw = fs::read(&path).unwrap();
        raw[12] ^= 0xff;
        fs::write(&path, &raw).unwrap();
        let err = read(&path)
            .err()
            .expect("head-corrupted snapshot must be rejected");
        let _ = fs::remove_file(&path);
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn snapshot_rejects_frame_corruption() {
        // v6: RAM frames carry their own CRC, verified at apply time. Flip the file's last byte —
        // region B's frames are LCG noise, so the tail is always inside a data frame's bytes.
        let path = std::env::temp_dir().join(format!(
            "limina-snap-frame-corrupt-{}.bin",
            std::process::id()
        ));
        let _ = write_sample(&path);
        let mut raw = fs::read(&path).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        fs::write(&path, &raw).unwrap();
        let got = read(&path).expect("head is intact; read must succeed");
        let err = got
            .apply_ram(&test_mem())
            .err()
            .expect("frame-corrupted snapshot must be rejected at apply");
        let _ = fs::remove_file(&path);
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn snapshot_rejects_truncation() {
        let path =
            std::env::temp_dir().join(format!("limina-snap-trunc-{}.bin", std::process::id()));
        let _ = write_sample(&path);
        let raw = fs::read(&path).unwrap();
        fs::write(&path, &raw[..raw.len() - 10]).unwrap();
        let got = read(&path).expect("head is intact; read must succeed");
        let err = got
            .apply_ram(&test_mem())
            .err()
            .expect("truncated snapshot must be rejected at apply");
        let _ = fs::remove_file(&path);
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
