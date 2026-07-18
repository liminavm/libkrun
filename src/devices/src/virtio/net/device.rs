// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.
use crate::Error as DeviceError;
use crate::virtio::net::{Error, Result};
use crate::virtio::net::{NUM_QUEUES, QUEUE_CONFIG};
use crate::virtio::queue::Error as QueueError;
use crate::virtio::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, InterruptTransport, QueueConfig,
    TYPE_NET, VirtioDevice,
};

use super::backend::{NetBackend, ReadError, WriteError};
use super::worker::{connect_backend, NetWorker};

use std::cmp;
use std::io::Write;
use std::os::fd::RawFd;
use std::path::PathBuf;
use std::thread::JoinHandle;
use utils::eventfd::{EventFd, EFD_NONBLOCK};
use virtio_bindings::virtio_net::VIRTIO_NET_F_MAC;
use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
use vm_memory::{ByteValued, GuestMemoryError, GuestMemoryMmap};

const VIRTIO_F_VERSION_1: u32 = 32;

#[derive(Debug)]
pub enum FrontendError {
    DescriptorChainTooSmall,
    EmptyQueue,
    GuestMemory(GuestMemoryError),
    QueueError(QueueError),
    ReadOnlyDescriptor,
}

#[derive(Debug)]
pub enum RxError {
    Backend(ReadError),
    DeviceError(DeviceError),
}

#[derive(Debug)]
pub enum TxError {
    Backend(WriteError),
    DeviceError(DeviceError),
    QueueError(QueueError),
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
struct VirtioNetConfig {
    mac: [u8; 6],
    status: u16,
    max_virtqueue_pairs: u16,
}

// Safe because it only has data and has no implicit padding.
unsafe impl ByteValued for VirtioNetConfig {}

#[derive(Clone)]
pub enum VirtioNetBackend {
    UnixstreamFd(RawFd),
    UnixstreamPath(PathBuf),
    UnixgramFd(RawFd),
    UnixgramPath(PathBuf, bool),
    #[cfg(target_os = "linux")]
    Tap(String),
}

pub struct Net {
    id: String,
    pub cfg_backend: VirtioNetBackend,

    avail_features: u64,
    acked_features: u64,

    pub(crate) device_state: DeviceState,

    config: VirtioNetConfig,

    // Suspend/resume: the running worker's handle + the eventfd that stops it, so `reset` (the
    // virtio reset the guest issues on resume) can tear the old worker down cleanly before the
    // device is re-activated with fresh queues. The join handle yields the backend connection back
    // so it survives the reset (see [`Net::reset`] and `worker::connect_backend`).
    worker_thread: Option<JoinHandle<Box<dyn NetBackend + Send>>>,
    worker_stopfd: EventFd,
    // The gateway connection (e.g. gvproxy socket), preserved across suspend/resume. `Some` while
    // the device is inactive/idle; taken by `activate` (moved into the worker) and handed back by
    // `reset`. Lazily opened on first activate.
    backend: Option<Box<dyn NetBackend + Send>>,
}

impl Net {
    /// Create a new virtio network device using the backend
    pub fn new(
        id: String,
        cfg_backend: VirtioNetBackend,
        mac: [u8; 6],
        features: u32,
    ) -> Result<Self> {
        let avail_features = features as u64
            | (1 << VIRTIO_NET_F_MAC)
            | (1 << VIRTIO_RING_F_EVENT_IDX)
            | (1 << VIRTIO_F_VERSION_1);

        let config = VirtioNetConfig {
            mac,
            status: 0,
            max_virtqueue_pairs: 0,
        };

        Ok(Net {
            id,
            cfg_backend,

            avail_features,
            acked_features: 0u64,

            device_state: DeviceState::Inactive,
            config,

            worker_thread: None,
            worker_stopfd: EventFd::new(EFD_NONBLOCK).map_err(Error::EventFd)?,
            backend: None,
        })
    }

    /// Provides the ID of this net device.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Provides the virtio-net backend of this net device.
    pub fn backend(&self) -> &VirtioNetBackend {
        &self.cfg_backend
    }
}

impl VirtioDevice for Net {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features;
    }

    fn device_type(&self) -> u32 {
        TYPE_NET
    }

    fn device_name(&self) -> &str {
        "net"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_slice = self.config.as_slice();
        let config_len = config_slice.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(&config_slice[offset as usize..cmp::min(end, config_len) as usize])
                .unwrap();
        }
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        log::warn!(
            "Net: guest driver attempted to write device config (offset={:x}, len={:x})",
            offset,
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        let [rx_q, tx_q]: [_; NUM_QUEUES] = queues.try_into().map_err(|_| {
            error!("Cannot perform activate. Expected {} queue(s)", NUM_QUEUES);
            ActivateError::BadActivate
        })?;

        let stop_fd = match self.worker_stopfd.try_clone() {
            Ok(fd) => fd,
            Err(err) => {
                error!(
                    "Cannot clone virtio-net ({}) stop eventfd: {err:?}",
                    self.id()
                );
                return Err(ActivateError::BadActivate);
            }
        };

        // Reuse the existing gateway connection across a suspend/resume cycle; open it only the
        // first time (or if a previous worker failed to hand it back).
        let backend = match self.backend.take() {
            Some(backend) => backend,
            None => match connect_backend(self.cfg_backend.clone(), self.acked_features) {
                Ok(backend) => backend,
                Err(err) => {
                    error!(
                        "Error connecting virtio-net ({}) backend: {err:?}",
                        self.id()
                    );
                    return Err(ActivateError::BadActivate);
                }
            },
        };

        let worker = NetWorker::new(rx_q, tx_q, interrupt.clone(), mem.clone(), backend, stop_fd);
        self.worker_thread = Some(worker.run());
        self.device_state = DeviceState::Activated(mem, interrupt);
        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    /// Deactivate on the virtio reset the guest issues when re-initialising the device — notably
    /// on resume from suspend-to-idle. Stop the worker thread (so a fresh `activate` doesn't race a
    /// stale one on the same queues) and go Inactive; the transport then recreates the queues and
    /// re-activates. Returning `false` here (the trait default) would leave the transport marking
    /// the device FAILED, so the guest's re-init writes get dropped and networking never comes back.
    fn reset(&mut self) -> bool {
        if let Some(worker) = self.worker_thread.take() {
            let _ = self.worker_stopfd.write(1);
            match worker.join() {
                // Preserve the gateway connection so the next activate reuses it (don't reconnect,
                // which would drop gvproxy).
                Ok(backend) => self.backend = Some(backend),
                Err(e) => error!(
                    "error waiting for virtio-net ({}) worker thread: {e:?}",
                    self.id()
                ),
            }
        }
        self.device_state = DeviceState::Inactive;
        true
    }
}
