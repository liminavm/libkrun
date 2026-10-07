// SPDX-License-Identifier: Apache-2.0

//! A TPM 2.0 device: the TIS FIFO register interface over MMIO, and the backend it hands each
//! command to (limina).
//!
//! The device is mechanism only — registers, localities and the command/response FIFO — and
//! knows nothing about what a TPM does with a command. That is the [`TpmBackend`]'s job. The
//! `tpm` feature adds one backend, [`JanusBackend`], over the janus engine.

mod tis;

#[cfg(feature = "tpm")]
mod janus_backend;

pub use tis::{BUFFER_SIZE, LOCALITY_SIZE, Locality, MMIO_LEN, Reg, TpmTis, decode};

#[cfg(feature = "tpm")]
pub use janus_backend::JanusBackend;

/// What executes TPM commands.
pub trait TpmBackend: Send {
    /// Executes one command, delivered at `locality`, and returns its response. The response
    /// must be a TPM response frame of at most [`BUFFER_SIZE`] bytes: a backend that returns a
    /// larger one is broken, and the device asserts it is not.
    fn command(&mut self, locality: Locality, command: &[u8]) -> Vec<u8>;

    /// `_TPM_Init`: the platform was reset. The device signals it once at power-on; libkrun has
    /// no in-process reset (a guest reboot builds a new VM, and so a new device).
    fn init(&mut self);
}

#[cfg(any(test, kani, fuzzing))]
pub mod echo {
    //! A backend for tests: answers every command with a frame that carries the locality and
    //! the command back, so a test can tell exactly what the device delivered.

    use super::{Locality, TpmBackend};

    #[derive(Default)]
    pub struct Echo {
        /// Every command delivered, with its locality.
        pub delivered: Vec<(u8, Vec<u8>)>,
        pub inits: usize,
    }

    /// The response `Echo` gives to `command` at `locality`: a `TPM_ST_NO_SESSIONS` header, then
    /// the locality and the command's own bytes, capped so the frame fits the buffer.
    pub fn response(locality: Locality, command: &[u8]) -> Vec<u8> {
        let body = &command[..command.len().min(super::BUFFER_SIZE - 11)];
        let size = (11 + body.len()) as u32;
        let mut r = vec![0x80, 0x01];
        r.extend_from_slice(&size.to_be_bytes());
        r.extend_from_slice(&0u32.to_be_bytes());
        r.push(locality.get());
        r.extend_from_slice(body);
        r
    }

    impl TpmBackend for Echo {
        fn command(&mut self, locality: Locality, command: &[u8]) -> Vec<u8> {
            self.delivered.push((locality.get(), command.to_vec()));
            response(locality, command)
        }

        fn init(&mut self) {
            self.inits += 1;
        }
    }
}
