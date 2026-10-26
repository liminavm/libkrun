// SPDX-License-Identifier: Apache-2.0

//! The TPM TIS register file under any sequence of MMIO accesses a guest can make, and any
//! snapshot taken between them (limina).
//!
//! The input is a list of records: an op byte (bit 0: write; bits 1-2: width 1, 2, 4 or 8; bit
//! 3: save and restore first), a 16-bit offset into the device's window, and for a write the
//! bytes it carries. Every access goes to two devices: one that runs straight through, and one
//! that is saved and restored into a new device wherever a record asks; every read must return
//! the same bytes from both. After every access, at most one locality may read as active, and
//! the active one's STS must never show a bit Linux requires to be zero. An input whose first
//! byte is 0xFF is instead handed to `restore_state` as saved bytes. Nothing may panic.

#![no_main]

use devices::BusDevice;
use devices::tpm::echo::Echo;
use devices::tpm::{LOCALITY_SIZE, MMIO_LEN, TpmTis};
use libfuzzer_sys::fuzz_target;

fn check(d: &mut TpmTis<Echo>) {
    let mut active = 0;
    for l in 0..5u64 {
        let mut a = [0u8];
        d.read(0, l * LOCALITY_SIZE, &mut a);
        if a[0] & 0x20 != 0 {
            active += 1;
            let mut s = [0u8; 4];
            d.read(0, l * LOCALITY_SIZE + 0x18, &mut s);
            assert_eq!(s[0] & 0x23, 0, "STS {s:02x?} at locality {l}");
        }
    }
    assert!(active <= 1, "{active} active localities");
}

fuzz_target!(|data: &[u8]| {
    if let Some(saved) = data.strip_prefix(&[0xFF]) {
        let mut d = TpmTis::new(Echo::default());
        if d.restore_state(saved).is_ok() {
            check(&mut d);
        }
        return;
    }
    let mut d = TpmTis::new(Echo::default());
    let mut e = TpmTis::new(Echo::default());
    let mut rest = data;
    while rest.len() >= 3 {
        let op = rest[0];
        let offset = u64::from(u16::from_le_bytes([rest[1], rest[2]])) % MMIO_LEN;
        rest = &rest[3..];
        if op & 8 != 0 {
            let mut restored = TpmTis::new(Echo::default());
            restored
                .restore_state(&e.save_state())
                .expect("a saved device restores");
            e = restored;
        }
        let width = 1usize << ((op >> 1) & 3);
        if op & 1 == 1 {
            let n = width.min(rest.len());
            d.write(0, offset, &rest[..n]);
            e.write(0, offset, &rest[..n]);
            rest = &rest[n..];
        } else {
            let mut a = [0u8; 8];
            let mut b = [0u8; 8];
            d.read(0, offset, &mut a[..width]);
            e.read(0, offset, &mut b[..width]);
            assert_eq!(a, b, "a restored device read differently at {offset:#x}");
        }
        check(&mut d);
    }
});
