// SPDX-License-Identifier: Apache-2.0

//! The janus TPM 2.0 engine as a [`TpmBackend`] (limina).

use super::{Locality, TpmBackend};

/// Randomness from the host's CSPRNG.
struct HostRandom;

impl janus::Platform for HostRandom {
    fn fill_random(&mut self, buf: &mut [u8]) {
        // The host CSPRNG failing is not something a TPM can answer around: every key, nonce
        // and seed would be predictable. Stop loudly.
        getrandom::fill(buf).expect("the host CSPRNG failed");
    }
}

/// A freshly manufactured janus TPM. Its state lives in memory for the life of the VM.
pub struct JanusBackend(janus::Janus<HostRandom>);

impl JanusBackend {
    pub fn new() -> Self {
        JanusBackend(janus::Janus::manufacture(HostRandom))
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
        self.0.execute(locality, command)
    }

    fn init(&mut self) {
        self.0.init();
    }
}
