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
use std::path::Path;

use devices::legacy::GpioState;
use hvf::VcpuState;

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
const VERSION: u32 = 4;

/// One guest-RAM region: its guest-physical base and raw bytes.
pub struct RamRegion {
    pub gpa: u64,
    pub data: Vec<u8>,
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

/// The full deserialized contents of a snapshot file.
pub struct Snapshot {
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
    /// Guest RAM regions (SHM window excluded by the caller).
    pub ram: Vec<RamRegion>,
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
}

/// Serialize a snapshot to `path` (payload + trailing CRC32).
pub fn write(path: &Path, snap: &Snapshot) -> io::Result<()> {
    let mut v = Vec::new();
    v.extend_from_slice(MAGIC);
    put_u32(&mut v, VERSION);
    put_u32(&mut v, snap.vcpus.len() as u32);
    for s in &snap.vcpus {
        encode_vcpu(&mut v, s);
    }
    put_bytes(&mut v, &snap.gic);
    // Legacy-device (PL061 GPIO) register section: a presence byte, then the 8 registers.
    match &snap.gpio {
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
    put_u64(&mut v, snap.layout.ram_start);
    put_u64(&mut v, snap.layout.ram_last);
    put_u64(&mut v, snap.layout.shm_start);
    v.push(snap.layout.firmware as u8);
    // v4 per-device virtio-transport section.
    put_u32(&mut v, snap.devices.len() as u32);
    for d in &snap.devices {
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
    put_u32(&mut v, snap.ram.len() as u32);
    for r in &snap.ram {
        put_u64(&mut v, r.gpa);
        put_bytes(&mut v, &r.data);
    }
    let crc = crc32(&v);
    put_u32(&mut v, crc);
    fs::write(path, &v)
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
        let s = self.buf.get(self.pos..end).ok_or_else(|| corrupt("truncated"))?;
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
    })
}

/// Read + verify a snapshot from `path`. Fails closed on a bad magic, version, or CRC.
pub fn read(path: &Path) -> io::Result<Snapshot> {
    let raw = fs::read(path)?;
    if raw.len() < 12 {
        return Err(corrupt("too small"));
    }
    let (payload, crc_bytes) = raw.split_at(raw.len() - 4);
    let stored = u32::from_le_bytes(crc_bytes.try_into().unwrap());
    if crc32(payload) != stored {
        return Err(corrupt("CRC mismatch"));
    }
    let mut r = Reader {
        buf: payload,
        pos: 0,
    };
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
    let ram_count = r.u32()? as usize;
    let mut ram = Vec::with_capacity(ram_count);
    for _ in 0..ram_count {
        let gpa = r.u64()?;
        let data = r.bytes()?;
        ram.push(RamRegion { gpa, data });
    }
    Ok(Snapshot {
        vcpus,
        gic,
        gpio,
        layout,
        devices,
        ram,
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

    #[test]
    fn snapshot_file_round_trips() {
        let snap = Snapshot {
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
            ram: vec![
                RamRegion {
                    gpa: 0x4000_0000,
                    data: vec![7u8; 4096],
                },
                RamRegion {
                    gpa: 0x8000_0000,
                    data: (0..=255u8).cycle().take(9000).collect(),
                },
            ],
        };
        let path = std::env::temp_dir().join(format!("limina-snap-test-{}.bin", std::process::id()));
        write(&path, &snap).expect("write");
        let got = read(&path).expect("read");
        let _ = fs::remove_file(&path);

        assert_eq!(got.vcpus.len(), 2);
        assert_eq!(got.vcpus[0].x, snap.vcpus[0].x);
        assert_eq!(got.vcpus[1].q, snap.vcpus[1].q);
        assert_eq!(got.vcpus[0].sysregs, snap.vcpus[0].sysregs);
        assert_eq!(got.vcpus[1].icc, snap.vcpus[1].icc);
        assert_eq!(got.vcpus[0].pending_irq, snap.vcpus[0].pending_irq);
        assert_eq!(got.gic, snap.gic);
        let gpio = got.gpio.expect("gpio present");
        assert_eq!(gpio.im, 0x78);
        assert_eq!(gpio.istate, 0x40);
        assert_eq!(gpio.data, 0x40);
        // v4: layout identity + device-transport section round-trip byte-for-byte.
        assert_eq!(got.layout, snap.layout);
        assert_eq!(got.devices, snap.devices);
        assert_eq!(got.devices[0].queues[0].avail, 0x1_2340_0400);
        assert_eq!(got.ram.len(), 2);
        assert_eq!(got.ram[0].gpa, 0x4000_0000);
        assert_eq!(got.ram[1].data, snap.ram[1].data);
    }

    #[test]
    fn v4_device_transport_section_survives_no_devices() {
        // The common case (no sticky device) must encode an empty section and read back empty — not
        // desync the stream. Also pins the layout header round-trip in isolation.
        let snap = Snapshot {
            vcpus: vec![sample_vcpu(3)],
            gic: vec![1, 2, 3, 4],
            gpio: None,
            layout: sample_layout(),
            devices: vec![],
            ram: vec![RamRegion {
                gpa: 0x4000_0000,
                data: vec![9u8; 64],
            }],
        };
        let path =
            std::env::temp_dir().join(format!("limina-snap-v4-empty-{}.bin", std::process::id()));
        write(&path, &snap).expect("write");
        let got = read(&path).expect("read");
        let _ = fs::remove_file(&path);
        assert!(got.devices.is_empty());
        assert_eq!(got.layout, sample_layout());
        assert_eq!(got.ram[0].data, vec![9u8; 64]);
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
        assert_eq!(v.len(), 8 + 5, "byte-section length must be a 64-bit prefix");
        assert_eq!(&v[..8], &5u64.to_le_bytes(), "length encoded little-endian u64");
        // And a length that would overflow u32 must survive the u64 round-trip in the header.
        let mut h = Vec::new();
        put_u64(&mut h, 0x1_0000_0000u64); // exactly 4 GiB — the value a u32 truncates to 0
        let mut r = Reader { buf: &h, pos: 0 };
        assert_eq!(r.u64().unwrap(), 0x1_0000_0000u64);
    }

    #[test]
    fn snapshot_rejects_corruption() {
        let snap = Snapshot {
            vcpus: vec![sample_vcpu(9)],
            gic: vec![1, 2, 3],
            gpio: None,
            layout: sample_layout(),
            devices: vec![],
            ram: vec![],
        };
        let path =
            std::env::temp_dir().join(format!("limina-snap-corrupt-{}.bin", std::process::id()));
        write(&path, &snap).expect("write");
        // Flip a payload byte; the trailing CRC must catch it.
        let mut raw = fs::read(&path).unwrap();
        raw[12] ^= 0xff;
        fs::write(&path, &raw).unwrap();
        let err = read(&path).err().expect("corrupted snapshot must be rejected");
        let _ = fs::remove_file(&path);
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
