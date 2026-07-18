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

use hvf::VcpuState;

const MAGIC: &[u8; 8] = b"LIMINAS1";
const VERSION: u32 = 1;

/// One guest-RAM region: its guest-physical base and raw bytes.
pub struct RamRegion {
    pub gpa: u64,
    pub data: Vec<u8>,
}

/// The full deserialized contents of a snapshot file.
pub struct Snapshot {
    /// Per-vCPU architectural state, in vCPU-index order.
    pub vcpus: Vec<VcpuState>,
    /// The VM-wide in-kernel GICv3 distributor/redistributor blob.
    pub gic: Vec<u8>,
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
    put_u32(v, b.len() as u32);
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
        let n = self.u32()? as usize;
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
    let ram_count = r.u32()? as usize;
    let mut ram = Vec::with_capacity(ram_count);
    for _ in 0..ram_count {
        let gpa = r.u64()?;
        let data = r.bytes()?;
        ram.push(RamRegion { gpa, data });
    }
    Ok(Snapshot { vcpus, gic, ram })
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

    #[test]
    fn snapshot_file_round_trips() {
        let snap = Snapshot {
            vcpus: vec![sample_vcpu(1), sample_vcpu(2)],
            gic: vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0x11],
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
        assert_eq!(got.ram.len(), 2);
        assert_eq!(got.ram[0].gpa, 0x4000_0000);
        assert_eq!(got.ram[1].data, snap.ram[1].data);
    }

    #[test]
    fn snapshot_rejects_corruption() {
        let snap = Snapshot {
            vcpus: vec![sample_vcpu(9)],
            gic: vec![1, 2, 3],
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
