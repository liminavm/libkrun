// SPDX-License-Identifier: Apache-2.0

//! The TIS 1.3 FIFO interface (TCG PC Client Platform TPM Profile, §6.5) as an MMIO device.
//!
//! Five localities, one 4 KiB register window each. A locality must be granted (ACCESS) before
//! its STS and FIFO registers do anything; the granted one sends a command by writing it into the
//! FIFO and setting `tpmGo`, then reads the response back out of the same FIFO. The behaviour
//! that the guest drivers depend on follows QEMU's `tpm_tis_common.c`, which Linux and edk2 are
//! tested against:
//!
//! - STS never reads with a `TPM_STS_READ_ZERO` bit set (`tpmGo`, `responseRetry`, bit 0):
//!   Linux treats such a read as a broken TPM.
//! - `dataAvail` clears exactly when the last response byte is read: Linux checks it is clear
//!   afterwards and fails the command as "left over data" otherwise.
//! - A locality that is not the active one reads 0xFF from every register except ACCESS,
//!   INTF_CAPABILITY, INTERFACE_ID, DID_VID and RID, which describe the device rather than a
//!   transaction.
//!
//! Commands execute synchronously, inside the `tpmGo` write. There are no interrupts: the guest
//! polls, and the device-tree node carries no `interrupts` property.
//!
//! **Identity.** DID_VID reads `0x0001_0000`: vendor 0x0000, which no vendor holds, and device 1.
//! This device does not claim to be anyone's TPM.

use crate::bus::BusDevice;

use super::TpmBackend;

/// The number of localities, and of register windows.
pub const LOCALITIES: u8 = 5;
/// The size of one locality's register window.
pub const LOCALITY_SIZE: u64 = 0x1000;
/// The device's MMIO window.
pub const MMIO_LEN: u64 = LOCALITIES as u64 * LOCALITY_SIZE;
/// The largest command the device takes and response it holds.
pub const BUFFER_SIZE: usize = 4096;

/// A locality, 0 to 4: which register window an access came through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Locality(u8);

impl Locality {
    pub const fn get(self) -> u8 {
        self.0
    }

    #[cfg(any(test, kani, fuzzing))]
    pub const fn new(n: u8) -> Option<Locality> {
        if n < LOCALITIES {
            Some(Locality(n))
        } else {
            None
        }
    }

    fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// A register in a locality's window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reg {
    Access,
    IntEnable,
    IntVector,
    IntStatus,
    IntfCapability,
    Sts,
    DataFifo,
    InterfaceId,
    XDataFifo,
    DidVid,
    Rid,
}

impl Reg {
    /// Every register with its offset in the window and its width in bytes.
    const LAYOUT: [(Reg, u64, u64); 11] = [
        (Reg::Access, 0x00, 1),
        (Reg::IntEnable, 0x08, 4),
        (Reg::IntVector, 0x0C, 1),
        (Reg::IntStatus, 0x10, 4),
        (Reg::IntfCapability, 0x14, 4),
        (Reg::Sts, 0x18, 4),
        (Reg::DataFifo, 0x24, 4),
        (Reg::InterfaceId, 0x30, 4),
        (Reg::XDataFifo, 0x80, 64),
        (Reg::DidVid, 0xF00, 4),
        (Reg::Rid, 0xF04, 1),
    ];

    /// Readable from a locality that is not the active one.
    fn describes_device(self) -> bool {
        matches!(
            self,
            Reg::Access | Reg::IntfCapability | Reg::InterfaceId | Reg::DidVid | Reg::Rid
        )
    }

    fn is_fifo(self) -> bool {
        matches!(self, Reg::DataFifo | Reg::XDataFifo)
    }
}

/// Where an MMIO offset lands: its locality, and the register and byte within it, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoded {
    pub locality: Locality,
    /// `None` for an offset between registers.
    pub reg: Option<(Reg, u8)>,
}

/// Decodes an offset into the device's window. `None` outside it.
pub fn decode(offset: u64) -> Option<Decoded> {
    if offset >= MMIO_LEN {
        return None;
    }
    let locality = Locality((offset / LOCALITY_SIZE) as u8);
    let within = offset % LOCALITY_SIZE;
    let reg = Reg::LAYOUT
        .iter()
        .find(|(_, base, width)| (*base..base + width).contains(&within))
        .map(|(reg, base, _)| (*reg, (within - base) as u8));
    Some(Decoded { locality, reg })
}

mod access {
    pub const ESTABLISHMENT: u8 = 1 << 0;
    pub const REQUEST_USE: u8 = 1 << 1;
    pub const PENDING_REQUEST: u8 = 1 << 2;
    pub const SEIZE: u8 = 1 << 3;
    pub const BEEN_SEIZED: u8 = 1 << 4;
    pub const ACTIVE_LOCALITY: u8 = 1 << 5;
    pub const REG_VALID_STS: u8 = 1 << 7;
}

mod sts {
    pub const RESPONSE_RETRY: u32 = 1 << 1;
    pub const EXPECT: u32 = 1 << 3;
    pub const DATA_AVAIL: u32 = 1 << 4;
    pub const TPM_GO: u32 = 1 << 5;
    pub const COMMAND_READY: u32 = 1 << 6;
    pub const VALID: u32 = 1 << 7;
    pub const FAMILY_TPM2: u32 = 1 << 26;
    /// The bits Linux requires to read as zero (`TPM_STS_READ_ZERO`).
    pub const READ_ZERO: u32 = 0x23;
}

/// INTF_CAPABILITY: interface version 1.3 for TPM 2.0, 64-byte transfers, a dynamic burst
/// count, and no interrupt support.
const INTF_CAPABILITY: u32 = (3 << 28) | (3 << 9);
/// INTERFACE_ID: the FIFO interface, version 0, five localities, TIS supported.
const INTERFACE_ID: u32 = (1 << 13) | (1 << 8);
/// See the module documentation.
const DID_VID: u32 = 0x0001_0000;
const RID: u32 = 0x01;

/// Where the command/response transaction is. It belongs to the active locality, and a change
/// of locality returns it to `Idle`.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    /// `commandReady`: waiting for the first command byte.
    Ready,
    /// Taking a command. `expect` is whether the device wants more bytes.
    Reception {
        command: Vec<u8>,
        expect: bool,
    },
    /// Holding a response, of which `read` bytes have been read.
    Completion {
        response: Vec<u8>,
        read: usize,
    },
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct LocalityFlags {
    request_use: bool,
    been_seized: bool,
    int_enable: u32,
}

/// The device.
pub struct TpmTis<B: TpmBackend> {
    backend: B,
    active: Option<Locality>,
    flags: [LocalityFlags; LOCALITIES as usize],
    phase: Phase,
}

impl<B: TpmBackend> TpmTis<B> {
    /// A device at power-on: no locality granted, and the backend told the platform started.
    pub fn new(mut backend: B) -> Self {
        backend.init();
        TpmTis {
            backend,
            active: None,
            flags: Default::default(),
            phase: Phase::Idle,
        }
    }

    #[cfg(any(test, kani, fuzzing))]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn activate(&mut self, l: Option<Locality>) {
        if self.active != l {
            self.phase = Phase::Idle;
        }
        self.active = l;
        if let Some(l) = l {
            self.flags[l.index()].request_use = false;
        }
    }

    /// The highest locality waiting for use, if any.
    fn next_requester(&self) -> Option<Locality> {
        (0..LOCALITIES)
            .rev()
            .map(Locality)
            .find(|l| self.flags[l.index()].request_use)
    }

    fn write_access(&mut self, l: Locality, mut v: u8) {
        if v & access::SEIZE != 0 {
            v &= !(access::REQUEST_USE | access::ACTIVE_LOCALITY);
        }
        if v & access::ACTIVE_LOCALITY != 0 {
            if self.active == Some(l) {
                let next = self.next_requester();
                self.activate(next);
            } else {
                self.flags[l.index()].request_use = false;
            }
        }
        if v & access::BEEN_SEIZED != 0 {
            self.flags[l.index()].been_seized = false;
        }
        if v & access::SEIZE != 0 {
            match self.active {
                None => self.activate(Some(l)),
                Some(a) if l.0 > a.0 => {
                    self.flags[a.index()].been_seized = true;
                    self.activate(Some(l));
                }
                Some(_) => {}
            }
        }
        if v & access::REQUEST_USE != 0 && self.active != Some(l) {
            if self.active.is_some() {
                self.flags[l.index()].request_use = true;
            } else {
                self.activate(Some(l));
            }
        }
    }

    fn read_access(&self, l: Locality) -> u8 {
        let f = self.flags[l.index()];
        let pending = (0..LOCALITIES)
            .filter(|m| *m != l.0)
            .any(|m| self.flags[usize::from(m)].request_use);
        // ESTABLISHMENT reads 1, "not established": nothing on this platform runs a DRTM
        // sequence, which is what would set it.
        access::REG_VALID_STS
            | access::ESTABLISHMENT
            | if self.active == Some(l) {
                access::ACTIVE_LOCALITY
            } else {
                0
            }
            | if f.been_seized {
                access::BEEN_SEIZED
            } else {
                0
            }
            | if pending { access::PENDING_REQUEST } else { 0 }
            | if f.request_use {
                access::REQUEST_USE
            } else {
                0
            }
    }

    /// STS as the active locality reads it, for an access of `width` bytes.
    fn sts(&self, width: usize) -> u32 {
        let (bits, avail) = match &self.phase {
            Phase::Idle => (0, BUFFER_SIZE),
            Phase::Ready => (sts::COMMAND_READY, BUFFER_SIZE),
            Phase::Reception { command, expect } => (
                if *expect { sts::EXPECT } else { 0 },
                BUFFER_SIZE - command.len(),
            ),
            Phase::Completion { response, read } => {
                let left = response.len() - read;
                (if left > 0 { sts::DATA_AVAIL } else { 0 }, left)
            }
        };
        // A byte-wide read of the burst count must not read 0x00 for 0x100 bytes available.
        let avail = if width == 1 {
            avail.min(0xFF)
        } else {
            avail.min(0xFFFF)
        };
        let v = sts::VALID | sts::FAMILY_TPM2 | bits | ((avail as u32) << 8);
        debug_assert_eq!(v & sts::READ_ZERO, 0);
        v
    }

    fn write_sts(&mut self, v: u32) {
        match v & (sts::COMMAND_READY | sts::TPM_GO | sts::RESPONSE_RETRY) {
            sts::COMMAND_READY => self.phase = Phase::Ready,
            sts::TPM_GO => {
                if let Phase::Reception {
                    command,
                    expect: false,
                } = &self.phase
                {
                    let l = self.active.expect("only the active locality writes STS");
                    let response = self.backend.command(l, command);
                    assert!(
                        response.len() <= BUFFER_SIZE,
                        "the TPM backend returned a {}-byte response",
                        response.len()
                    );
                    self.phase = Phase::Completion { response, read: 0 };
                }
            }
            sts::RESPONSE_RETRY => {
                if let Phase::Completion { read, .. } = &mut self.phase {
                    *read = 0;
                }
            }
            // None of the three, or more than one: nothing to do.
            _ => {}
        }
    }

    fn fifo_write(&mut self, b: u8) {
        if self.phase == Phase::Ready {
            self.phase = Phase::Reception {
                command: Vec::with_capacity(BUFFER_SIZE),
                expect: true,
            };
        }
        let Phase::Reception { command, expect } = &mut self.phase else {
            return;
        };
        if !*expect {
            return;
        }
        command.push(b);
        // The header's size is known from the sixth byte on; until then, more is expected.
        // A full buffer expects nothing more, whatever the header claims.
        *expect = command.len() < BUFFER_SIZE
            && (command.len() < 6
                || u32::from_be_bytes([command[2], command[3], command[4], command[5]]) as usize
                    > command.len());
    }

    fn fifo_read(&mut self) -> u8 {
        match &mut self.phase {
            Phase::Completion { response, read } if *read < response.len() => {
                *read += 1;
                response[*read - 1]
            }
            _ => 0xFF,
        }
    }

    /// A register's value as a 32-bit word, for the registers that are not the FIFO.
    fn value(&self, l: Locality, reg: Reg, width: usize) -> u32 {
        match reg {
            Reg::Access => u32::from(self.read_access(l)),
            Reg::IntEnable => self.flags[l.index()].int_enable,
            Reg::IntVector | Reg::IntStatus => 0,
            Reg::IntfCapability => INTF_CAPABILITY,
            Reg::Sts => self.sts(width),
            Reg::InterfaceId => INTERFACE_ID,
            Reg::DidVid => DID_VID,
            Reg::Rid => RID,
            Reg::DataFifo | Reg::XDataFifo => unreachable!("the FIFO has no value"),
        }
    }
}

impl<B: TpmBackend + 'static> BusDevice for TpmTis<B> {
    fn read(&mut self, _vcpuid: u64, offset: u64, data: &mut [u8]) {
        data.fill(0xFF);
        let Some(Decoded {
            locality,
            reg: Some((reg, byte)),
        }) = decode(offset)
        else {
            return;
        };
        if self.active != Some(locality) && !reg.describes_device() {
            return;
        }
        if reg.is_fifo() {
            for d in data.iter_mut() {
                *d = self.fifo_read();
            }
            return;
        }
        let value = self.value(locality, reg, data.len());
        for (i, d) in data.iter_mut().enumerate() {
            let at = usize::from(byte) + i;
            *d = if at < 4 { (value >> (8 * at)) as u8 } else { 0 };
        }
    }

    fn write(&mut self, _vcpuid: u64, offset: u64, data: &[u8]) {
        let Some(Decoded {
            locality,
            reg: Some((reg, byte)),
        }) = decode(offset)
        else {
            return;
        };
        let Some(&first) = data.first() else {
            return;
        };
        // The bytes of a 32-bit register this write covers, in place.
        let word = || {
            data.iter()
                .enumerate()
                .filter(|(i, _)| usize::from(byte) + i < 4)
                .fold(0u32, |w, (i, b)| {
                    w | (u32::from(*b) << (8 * (usize::from(byte) + i)))
                })
        };
        match reg {
            Reg::Access if byte == 0 => self.write_access(locality, first),
            Reg::IntEnable => {
                let mask = data
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| usize::from(byte) + i < 4)
                    .fold(0u32, |m, (i, _)| {
                        m | (0xFF << (8 * (usize::from(byte) + i)))
                    });
                let f = &mut self.flags[locality.index()];
                f.int_enable = (f.int_enable & !mask) | word();
            }
            _ if self.active != Some(locality) => {}
            Reg::Sts => self.write_sts(word()),
            Reg::DataFifo | Reg::XDataFifo => {
                for b in data {
                    self.fifo_write(*b);
                }
            }
            // Read-only, or without effect: interrupt status and vector (no interrupts), the
            // identity registers.
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::echo::{Echo, response};
    use super::*;

    fn base(l: u8) -> u64 {
        u64::from(l) * LOCALITY_SIZE
    }

    fn rd(d: &mut TpmTis<Echo>, l: u8, reg: u64, width: usize) -> u64 {
        let mut b = [0u8; 8];
        d.read(0, base(l) + reg, &mut b[..width]);
        u64::from_le_bytes(b)
    }

    fn wr(d: &mut TpmTis<Echo>, l: u8, reg: u64, v: u64, width: usize) {
        d.write(0, base(l) + reg, &v.to_le_bytes()[..width]);
    }

    fn sts(d: &mut TpmTis<Echo>, l: u8) -> u32 {
        rd(d, l, 0x18, 4) as u32
    }

    /// Linux's `tpm_tis_send_data` + `tpm_tis_recv`, byte by byte through DATA_FIFO.
    pub(super) fn transact(d: &mut TpmTis<Echo>, l: u8, cmd: &[u8]) -> Vec<u8> {
        wr(d, l, 0x00, u64::from(access::REQUEST_USE), 1);
        assert_eq!(
            rd(d, l, 0x00, 1) as u8 & access::ACTIVE_LOCALITY,
            access::ACTIVE_LOCALITY
        );
        wr(d, l, 0x18, u64::from(sts::COMMAND_READY), 4);
        assert_ne!(sts(d, l) & sts::COMMAND_READY, 0);
        for (i, b) in cmd.iter().enumerate() {
            wr(d, l, 0x24, u64::from(*b), 1);
            let expect = sts(d, l) & sts::EXPECT != 0;
            assert_eq!(expect, i + 1 < cmd.len(), "EXPECT after byte {}", i + 1);
        }
        wr(d, l, 0x18, u64::from(sts::TPM_GO), 4);
        let mut rsp = Vec::new();
        while sts(d, l) & sts::DATA_AVAIL != 0 {
            rsp.push(rd(d, l, 0x24, 1) as u8);
        }
        wr(d, l, 0x18, u64::from(sts::COMMAND_READY), 4);
        wr(d, l, 0x00, u64::from(access::ACTIVE_LOCALITY), 1);
        rsp
    }

    pub(super) const CMD: [u8; 12] = [0x80, 0x01, 0, 0, 0, 12, 0, 0, 0x01, 0x44, 0, 0];

    #[test]
    fn a_command_round_trips_through_the_fifo() {
        let mut d = TpmTis::new(Echo::default());
        assert_eq!(d.backend().inits, 1);
        let rsp = transact(&mut d, 0, &CMD);
        assert_eq!(rsp, response(Locality(0), &CMD));
        assert_eq!(d.backend().delivered, vec![(0, CMD.to_vec())]);
        let rsp = transact(&mut d, 3, &CMD);
        assert_eq!(rsp[10], 3, "the command reached the backend at locality 3");
    }

    #[test]
    fn data_avail_clears_exactly_at_the_last_response_byte() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 0, 0x18, u64::from(sts::COMMAND_READY), 4);
        for b in CMD {
            wr(&mut d, 0, 0x24, u64::from(b), 1);
        }
        wr(&mut d, 0, 0x18, u64::from(sts::TPM_GO), 4);
        let len = response(Locality(0), &CMD).len();
        for i in 0..len {
            assert_ne!(sts(&mut d, 0) & sts::DATA_AVAIL, 0, "before byte {i}");
            assert_eq!((sts(&mut d, 0) >> 8) & 0xFFFF, (len - i) as u32);
            rd(&mut d, 0, 0x24, 1);
        }
        assert_eq!(sts(&mut d, 0) & sts::DATA_AVAIL, 0);
        // responseRetry rewinds.
        wr(&mut d, 0, 0x18, u64::from(sts::RESPONSE_RETRY), 4);
        assert_ne!(sts(&mut d, 0) & sts::DATA_AVAIL, 0);
    }

    #[test]
    fn sts_never_reads_a_bit_linux_requires_to_be_zero() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        assert_eq!(sts(&mut d, 0) & sts::READ_ZERO, 0);
        wr(
            &mut d,
            0,
            0x18,
            u64::from(sts::TPM_GO | sts::RESPONSE_RETRY),
            4,
        );
        assert_eq!(sts(&mut d, 0) & sts::READ_ZERO, 0);
        assert_eq!(sts(&mut d, 0) & (3 << 26), sts::FAMILY_TPM2);
    }

    #[test]
    fn a_locality_that_is_not_active_reads_only_the_identity() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        assert_eq!(rd(&mut d, 1, 0x18, 4), 0xFFFF_FFFF);
        assert_eq!(rd(&mut d, 1, 0x24, 1), 0xFF);
        assert_eq!(rd(&mut d, 1, 0xF00, 4), u64::from(DID_VID));
        assert_eq!(rd(&mut d, 1, 0xF04, 1), u64::from(RID));
        assert_eq!(rd(&mut d, 1, 0x14, 4), u64::from(INTF_CAPABILITY));
        assert_eq!(rd(&mut d, 1, 0x30, 4), u64::from(INTERFACE_ID));
        assert_eq!(rd(&mut d, 1, 0x00, 1) as u8 & access::ACTIVE_LOCALITY, 0);
        // And its writes to STS and the FIFO do nothing.
        wr(&mut d, 1, 0x18, u64::from(sts::COMMAND_READY), 4);
        wr(&mut d, 1, 0x24, 0x80, 1);
        assert_eq!(sts(&mut d, 0) & sts::COMMAND_READY, 0);
    }

    #[test]
    fn a_requesting_locality_gets_the_tpm_when_the_active_one_lets_go() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 2, 0x00, u64::from(access::REQUEST_USE), 1);
        assert_eq!(
            rd(&mut d, 0, 0x00, 1) as u8 & access::PENDING_REQUEST,
            access::PENDING_REQUEST
        );
        assert_eq!(
            rd(&mut d, 2, 0x00, 1) as u8 & access::REQUEST_USE,
            access::REQUEST_USE
        );
        wr(&mut d, 0, 0x00, u64::from(access::ACTIVE_LOCALITY), 1);
        assert_eq!(d.active, Some(Locality(2)));
        assert_eq!(rd(&mut d, 2, 0x00, 1) as u8 & access::REQUEST_USE, 0);
    }

    #[test]
    fn a_higher_locality_may_seize_and_the_loser_is_told() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 1, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 1, 0x18, u64::from(sts::COMMAND_READY), 4);
        wr(&mut d, 1, 0x24, 0x80, 1);
        // A lower locality may not.
        wr(&mut d, 0, 0x00, u64::from(access::SEIZE), 1);
        assert_eq!(d.active, Some(Locality(1)));
        wr(&mut d, 4, 0x00, u64::from(access::SEIZE), 1);
        assert_eq!(d.active, Some(Locality(4)));
        assert_eq!(
            d.phase,
            Phase::Idle,
            "the seized locality's command is gone"
        );
        assert_eq!(
            rd(&mut d, 1, 0x00, 1) as u8 & access::BEEN_SEIZED,
            access::BEEN_SEIZED
        );
        wr(&mut d, 1, 0x00, u64::from(access::BEEN_SEIZED), 1);
        assert_eq!(rd(&mut d, 1, 0x00, 1) as u8 & access::BEEN_SEIZED, 0);
    }

    #[test]
    fn a_byte_wide_burst_count_read_never_reads_zero_for_a_full_page() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 0, 0x18, u64::from(sts::COMMAND_READY), 4);
        // edk2 reads the burst count a byte at a time.
        assert_eq!(rd(&mut d, 0, 0x19, 1), 0xFF);
        assert_eq!(rd(&mut d, 0, 0x19, 2), 0x1000);
    }

    #[test]
    fn the_xdata_fifo_takes_wide_accesses() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 0, 0x18, u64::from(sts::COMMAND_READY), 4);
        wr(
            &mut d,
            0,
            0x80,
            u64::from_le_bytes(CMD[..8].try_into().unwrap()),
            8,
        );
        wr(
            &mut d,
            0,
            0x80,
            u64::from(u32::from_le_bytes(CMD[8..].try_into().unwrap())),
            4,
        );
        assert_eq!(sts(&mut d, 0) & sts::EXPECT, 0);
        wr(&mut d, 0, 0x18, u64::from(sts::TPM_GO), 4);
        let mut rsp = [0u8; 8];
        d.read(0, 0x80, &mut rsp);
        assert_eq!(rsp, response(Locality(0), &CMD)[..8]);
    }

    #[test]
    fn a_command_larger_than_the_buffer_stops_expecting_when_it_is_full() {
        let mut d = TpmTis::new(Echo::default());
        wr(&mut d, 0, 0x00, u64::from(access::REQUEST_USE), 1);
        wr(&mut d, 0, 0x18, u64::from(sts::COMMAND_READY), 4);
        let mut cmd: Vec<u8> = vec![0x80, 0x01, 0, 0, 0x20, 0];
        cmd.resize(BUFFER_SIZE + 10, 0);
        for b in &cmd {
            wr(&mut d, 0, 0x24, u64::from(*b), 1);
        }
        assert_eq!(sts(&mut d, 0) & sts::EXPECT, 0);
        wr(&mut d, 0, 0x18, u64::from(sts::TPM_GO), 4);
        assert_eq!(d.backend().delivered[0].1.len(), BUFFER_SIZE);
    }
}

#[cfg(test)]
mod every_sequence {
    //! Every sequence of register operations up to a fixed depth, from power-on, checked
    //! against the invariants the guest drivers rely on — and, after each sequence, a whole
    //! Linux-style transaction, which must deliver exactly its command and return exactly its
    //! response whatever state the sequence left behind.

    use super::super::echo::{Echo, response};
    use super::tests::{CMD, transact};
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Access(u8, u8),
        Sts(u8, u32),
        Fifo(u8, u8),
        ReadFifo(u8),
    }

    fn ops() -> Vec<Op> {
        let mut v = Vec::new();
        for l in [0, 1, 4] {
            for a in [
                access::REQUEST_USE,
                access::ACTIVE_LOCALITY,
                access::SEIZE,
                access::BEEN_SEIZED,
            ] {
                v.push(Op::Access(l, a));
            }
            for s in [sts::COMMAND_READY, sts::TPM_GO, sts::RESPONSE_RETRY] {
                v.push(Op::Sts(l, s));
            }
            v.push(Op::Fifo(l, 0x80));
            v.push(Op::ReadFifo(l));
        }
        v
    }

    fn apply(d: &mut TpmTis<Echo>, op: Op) {
        let base = |l: u8| u64::from(l) * LOCALITY_SIZE;
        match op {
            Op::Access(l, v) => d.write(0, base(l), &[v]),
            Op::Sts(l, v) => d.write(0, base(l) + 0x18, &v.to_le_bytes()),
            Op::Fifo(l, b) => d.write(0, base(l) + 0x24, &[b]),
            Op::ReadFifo(l) => d.read(0, base(l) + 0x24, &mut [0]),
        }
    }

    fn check(d: &mut TpmTis<Echo>, trace: &[Op]) {
        let mut actives = 0;
        for l in 0..LOCALITIES {
            let mut a = [0u8];
            d.read(0, u64::from(l) * LOCALITY_SIZE, &mut a);
            assert_ne!(a[0] & access::REG_VALID_STS, 0, "{trace:?}");
            if a[0] & access::ACTIVE_LOCALITY != 0 {
                actives += 1;
                let mut s = [0u8; 4];
                d.read(0, u64::from(l) * LOCALITY_SIZE + 0x18, &mut s);
                let s = u32::from_le_bytes(s);
                assert_eq!(s & sts::READ_ZERO, 0, "STS {s:#x} after {trace:?}");
                let data_avail = s & sts::DATA_AVAIL != 0;
                let burst = (s >> 8) & 0xFFFF;
                if data_avail {
                    assert!(burst > 0, "dataAvail with nothing to read after {trace:?}");
                }
            }
        }
        assert!(actives <= 1, "{actives} active localities after {trace:?}");
    }

    #[test]
    fn every_sequence_keeps_the_invariants_and_recovers() {
        const DEPTH: u32 = 4;
        let ops = ops();
        let total = ops.len().pow(DEPTH);
        let mut sent_from_reception = false;
        for n in 0..total {
            let mut d = TpmTis::new(Echo::default());
            let mut k = n;
            let mut trace = Vec::new();
            for _ in 0..DEPTH {
                let op = ops[k % ops.len()];
                k /= ops.len();
                trace.push(op);
                apply(&mut d, op);
                check(&mut d, &trace);
            }
            let before = d.backend().delivered.len();
            sent_from_reception |= before > 0;
            // Whoever holds the TPM lets go; then locality 2 runs a whole command.
            for l in 0..LOCALITIES {
                d.write(0, u64::from(l) * LOCALITY_SIZE, &[access::ACTIVE_LOCALITY]);
            }
            let rsp = transact(&mut d, 2, &CMD);
            assert_eq!(rsp, response(Locality(2), &CMD), "after {trace:?}");
            assert_eq!(d.backend().delivered.len(), before + 1);
            assert_eq!(d.backend().delivered[before], (2, CMD.to_vec()));
        }
        // No sequence of this depth writes a whole header, so none may have reached the
        // backend: a command is delivered only once it is complete.
        assert!(
            !sent_from_reception,
            "a partial command reached the backend"
        );
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    /// Every offset in the window decodes to its locality, and to the register whose range
    /// holds it, at the right byte; nothing outside the window decodes.
    #[kani::proof]
    #[kani::unwind(12)]
    fn the_decoder_maps_every_offset_to_its_register() {
        let offset: u64 = kani::any();
        match decode(offset) {
            None => assert!(offset >= MMIO_LEN),
            Some(d) => {
                assert!(offset < MMIO_LEN);
                assert!(u64::from(d.locality.get()) == offset / LOCALITY_SIZE);
                let within = offset % LOCALITY_SIZE;
                match d.reg {
                    Some((reg, byte)) => {
                        let (_, base, width) = *Reg::LAYOUT.iter().find(|r| r.0 == reg).unwrap();
                        assert!(within == base + u64::from(byte));
                        assert!(u64::from(byte) < width);
                    }
                    None => {
                        assert!(
                            Reg::LAYOUT
                                .iter()
                                .all(|(_, base, width)| within < *base || within >= base + width)
                        );
                    }
                }
            }
        }
    }

    /// The register ranges do not overlap, so an offset has one meaning.
    #[kani::proof]
    #[kani::unwind(12)]
    fn registers_do_not_overlap() {
        let i: usize = kani::any_where(|i| *i < Reg::LAYOUT.len());
        let j: usize = kani::any_where(|j| *j < Reg::LAYOUT.len() && *j != i);
        let (_, a, aw) = Reg::LAYOUT[i];
        let (_, b, bw) = Reg::LAYOUT[j];
        assert!(a + aw <= b || b + bw <= a);
    }

    struct Null;
    impl TpmBackend for Null {
        fn command(&mut self, _: Locality, _: &[u8]) -> Vec<u8> {
            Vec::new()
        }
        fn init(&mut self) {}
    }

    /// A read of any width at any offset of the granted locality's window answers every byte
    /// asked for, and STS never shows a bit Linux requires to be zero.
    #[kani::proof]
    #[kani::unwind(13)]
    fn every_read_of_the_active_locality_is_well_formed() {
        let mut d = TpmTis::new(Null);
        d.write(0, 0, &[access::REQUEST_USE]);
        let offset: u64 = kani::any_where(|o| *o < LOCALITY_SIZE);
        let width: usize = kani::any_where(|w| [1usize, 2, 4, 8].contains(w));
        let mut data = [0u8; 8];
        d.read(0, offset, &mut data[..width]);
        if offset == 0x18 && width >= 1 {
            assert!(u32::from(data[0]) & sts::READ_ZERO == 0);
        }
    }
}
