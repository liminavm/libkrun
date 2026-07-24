// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! Emulated xHCI USB controller (platform `generic-xhci`).
//!
//! Stage A is the controller bring-up skeleton: the MMIO register file that a
//! stock guest's `xhci-plat` driver needs to reset the controller, read its
//! parameters, program its rings, and declare the HCD running with the root hub
//! up and no device connected. Ring/TRB processing, device slots, transfers and
//! the `UsbDeviceModel` gadgets are Stage B. See `docs/design/usb-xhci.md`.

mod device;

pub use self::device::{XhciDevice, XHCI_MMIO_LEN};
