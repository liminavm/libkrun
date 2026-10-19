// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! virtio-i2c adapter (device ID 34, virtio spec 5.13) backed by emulated slaves.
//!
//! The only slave today is an SBS smart battery at the standard address 0x0b,
//! mirroring the host machine's battery so the guest desktop shows charge state
//! natively: the guest's stock `i2c-virtio` driver binds the adapter, the DT
//! child node (see `fdt/aarch64.rs`) makes the i2c core instantiate an
//! `sbs-battery` client, and UPower picks up the resulting power_supply — no
//! guest-side components required.

mod device;
mod event_handler;

pub use self::defs::uapi::VIRTIO_ID_I2C as TYPE_I2C;
pub use self::device::{BatteryProvider, BatteryState, I2c, SBS_BATTERY_ADDR};

mod defs {
    use crate::virtio::QueueConfig;

    pub const I2C_DEV_ID: &str = "virtio_i2c";
    pub const NUM_QUEUES: usize = 1;
    const QUEUE_SIZE: u16 = 64;
    pub static QUEUE_CONFIG: [QueueConfig; NUM_QUEUES] = [QueueConfig::new(QUEUE_SIZE); NUM_QUEUES];

    pub mod uapi {
        pub const VIRTIO_F_VERSION_1: u32 = 32;
        pub const VIRTIO_ID_I2C: u32 = 34;
        /// The Linux driver refuses to probe without this feature bit.
        pub const VIRTIO_I2C_F_ZERO_LENGTH_REQUEST: u32 = 0;

        /// out_hdr flags (virtio spec 5.13.6.1).
        pub const VIRTIO_I2C_FLAGS_FAIL_NEXT: u32 = 1;
        pub const VIRTIO_I2C_FLAGS_M_RD: u32 = 2;

        /// in_hdr status.
        pub const VIRTIO_I2C_MSG_OK: u8 = 0;
        pub const VIRTIO_I2C_MSG_ERR: u8 = 1;
    }
}

#[derive(Debug)]
pub enum I2cError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
}

type Result<T> = std::result::Result<T, I2cError>;
