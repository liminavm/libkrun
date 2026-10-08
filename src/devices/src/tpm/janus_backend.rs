// SPDX-License-Identifier: Apache-2.0

//! The janus TPM 2.0 engine as a [`TpmBackend`] (limina).

use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::os::unix::fs::OpenOptionsExt as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::{Locality, TpmBackend};

/// Randomness from the host's CSPRNG, and time from its monotonic clock.
struct Host {
    /// When the device was made: the TPM only uses differences between readings.
    epoch: Instant,
}

impl Host {
    fn new() -> Host {
        Host {
            epoch: Instant::now(),
        }
    }
}

impl janus::Platform for Host {
    fn fill_random(&mut self, buf: &mut [u8]) {
        // The host CSPRNG failing is not something a TPM can answer around: every key, nonce
        // and seed would be predictable. Stop loudly.
        getrandom::fill(buf).expect("the host CSPRNG failed");
    }

    fn now_ms(&mut self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// Where the TPM's NV lives between runs, and what was last written there.
struct StateFile {
    path: PathBuf,
    written: Vec<u8>,
}

impl StateFile {
    /// Replaces the file with `state`: a sibling `.tmp`, readable by its owner only, synced,
    /// then renamed over it, so the path never holds a partial state.
    fn write(&mut self, state: &[u8]) -> io::Result<()> {
        let mut name = self.path.file_name().unwrap_or_default().to_os_string();
        name.push(".tmp");
        let tmp = self.path.with_file_name(name);
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        f.write_all(state)?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        if let Some(dir) = self.path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::File::open(dir)?.sync_all()?;
        }
        self.written.clear();
        self.written.extend_from_slice(state);
        Ok(())
    }
}

/// A janus TPM: in memory for the life of the VM, or kept in a state file across runs.
pub struct JanusBackend {
    tpm: janus::Janus<Host>,
    file: Option<StateFile>,
}

impl JanusBackend {
    /// A freshly manufactured TPM that lives in memory: a new one every run.
    pub fn new() -> Self {
        JanusBackend {
            tpm: janus::Janus::manufacture(Host::new()),
            file: None,
        }
    }

    /// The TPM `path` holds, or, when there is no such file, a freshly manufactured one written
    /// there at once. A file that is there but does not restore is an error, never replaced:
    /// a new TPM in its place would lose every secret sealed to the old one.
    pub fn with_state_file(path: &Path) -> io::Result<Self> {
        let (tpm, existing) = match fs::read(path) {
            Ok(bytes) => {
                let tpm = janus::Janus::restore(Host::new(), &bytes).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}: {e}", path.display()),
                    )
                })?;
                (tpm, Some(bytes))
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                (janus::Janus::manufacture(Host::new()), None)
            }
            Err(e) => return Err(e),
        };
        let mut file = StateFile {
            path: path.to_owned(),
            written: existing.unwrap_or_default(),
        };
        let state = tpm.state();
        if file.written != *state {
            file.write(&state)?;
        }
        Ok(JanusBackend {
            tpm,
            file: Some(file),
        })
    }

    /// Writes the TPM's NV out if it changed. The command's response waits for this, so a
    /// guest never sees a write succeed that a crash could undo.
    fn persist(&mut self) {
        let Some(file) = &mut self.file else { return };
        let state = self.tpm.state();
        if file.written == *state {
            return;
        }
        if let Err(e) = file.write(&state) {
            // The guest already has its answer's effect in the TPM's memory; all the device can
            // do is say, loudly, that it will not outlive this run.
            error!(
                "tpm: writing its state to {} failed: {e}; changes since the last write will \
                 be lost when the VM stops",
                file.path.display()
            );
        }
    }
}

impl Default for JanusBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TpmBackend for JanusBackend {
    fn command(&mut self, locality: Locality, command: &[u8]) -> Vec<u8> {
        let locality =
            janus::Locality::new(locality.get()).expect("the device has five localities");
        let response = self.tpm.execute(locality, command);
        trace!(
            "tpm: command {:#x} -> {:#x}",
            word(command, 6),
            word(&response, 6)
        );
        self.persist();
        response
    }

    fn init(&mut self) {
        self.tpm.init();
    }
}

/// The big-endian `u32` at `at`, or 0 where there is none: a command or response code, for
/// the trace.
fn word(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .map_or(0, |b| u32::from_be_bytes(b.try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt as _;

    use super::*;

    /// TPM2_Startup(CLEAR) and TPM2_Shutdown(CLEAR).
    const STARTUP: [u8; 12] = [0x80, 0x01, 0, 0, 0, 12, 0, 0, 0x01, 0x44, 0, 0];
    const SHUTDOWN: [u8; 12] = [0x80, 0x01, 0, 0, 0, 12, 0, 0, 0x01, 0x45, 0, 0];

    fn rc(r: &[u8]) -> u32 {
        u32::from_be_bytes(r[6..10].try_into().unwrap())
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("krun-tpm-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("tpm.state")
    }

    /// A new file is written at once, owner-only; a second run restores it and finds the
    /// first run's orderly shutdown.
    #[test]
    fn a_state_file_carries_the_tpm_to_the_next_run() {
        let path = scratch("carry");
        let mut first = JanusBackend::with_state_file(&path).unwrap();
        let created = fs::read(&path).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        first.init();
        assert_eq!(rc(&first.command(Locality::new(0).unwrap(), &STARTUP)), 0);
        let started = fs::read(&path).unwrap();
        assert_ne!(
            started, created,
            "Startup's orderly-state change was not written"
        );
        assert_eq!(rc(&first.command(Locality::new(0).unwrap(), &SHUTDOWN)), 0);
        let shut = fs::read(&path).unwrap();
        drop(first);

        let mut second = JanusBackend::with_state_file(&path).unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            shut,
            "restoring rewrote an unchanged state"
        );
        second.init();
        assert_eq!(rc(&second.command(Locality::new(0).unwrap(), &STARTUP)), 0);
        assert!(!path.with_file_name("tpm.state.tmp").exists());
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    /// A file that is not a state is refused, and left as it was.
    #[test]
    fn a_damaged_state_file_is_refused_and_kept() {
        let path = scratch("damaged");
        fs::write(&path, b"not a TPM").unwrap();
        let e = JanusBackend::with_state_file(&path).err().unwrap();
        assert_eq!(e.kind(), io::ErrorKind::InvalidData);
        assert_eq!(fs::read(&path).unwrap(), b"not a TPM");
        fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
