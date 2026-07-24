// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Emulated xHCI USB controller (platform `generic-xhci`).
//!
//! Stage A brought up the MMIO register file. **Stage B1** adds the data path:
//! the command/event/transfer ring machinery (`trb`), 32-byte contexts
//! (`context`), the command set + EP0 control-transfer walker + port enumeration
//! (`engine`), the ring `worker` thread, and the device-model gadgets — so a
//! stock guest's `xhci-plat` driver **enumerates** an attached device. See
//! `docs/design/usb-xhci.md`.

mod context;
mod device;
mod engine;
mod trb;
mod worker;

pub use self::device::{XhciDevice, XHCI_MMIO_LEN};
pub use self::worker::spawn as spawn_worker;
