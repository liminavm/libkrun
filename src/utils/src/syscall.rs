// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::raw::c_int;

/// Wrapper to interpret syscall exit codes and provide a rustacean `io::Result`
pub struct SyscallReturnCode(pub c_int);
impl SyscallReturnCode {
    /// Returns the last OS error if value is -1 or Ok(value) otherwise.
    pub fn into_result(self) -> std::io::Result<c_int> {
        if self.0 == -1 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(self.0)
        }
    }

    /// Returns the last OS error if value is -1 or Ok(()) otherwise.
    pub fn into_empty_result(self) -> std::io::Result<()> {
        self.into_result().map(|_| ())
    }
}

/// Retry `syscall` while it fails with a transient `EFAULT`, bounded to ~100 ms.
///
/// On macOS the VMM periodically `mprotect(PROT_NONE)`s slices of guest RAM for a few
/// microseconds to settle the task-pmap ledger share (`ReleasedRam::settle_sweep` in the
/// hvf crate). Userspace touches during such a window fault and are transparently retried
/// by a signal handler, but kernel copyio — a syscall reading from or writing into guest
/// buffers — reports `EFAULT` instead of faulting. Syscalls addressing guest RAM wrap
/// here: a sweep-window `EFAULT` resolves within microseconds; a genuine one (otherwise
/// always a bug) still surfaces once the retries expire.
pub fn retry_transient_efault<R: PartialEq + From<i8>, F: FnMut() -> R>(mut syscall: F) -> R {
    let mut tries = 0u32;
    loop {
        let ret = syscall();
        if ret != R::from(-1i8)
            || std::io::Error::last_os_error().raw_os_error() != Some(libc::EFAULT)
            || tries >= 1000
        {
            return ret;
        }
        tries += 1;
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}

/// [`retry_transient_efault`] for operations that already return an `io::Result`.
pub fn retry_transient_efault_io<T, F: FnMut() -> std::io::Result<T>>(
    mut op: F,
) -> std::io::Result<T> {
    let mut tries = 0u32;
    loop {
        match op() {
            Err(e) if e.raw_os_error() == Some(libc::EFAULT) && tries < 1000 => {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            r => return r,
        }
    }
}
