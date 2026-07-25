// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! USB support: an emulated xHCI controller (mechanism) that limina drives with
//! software-defined gadgets (policy — FIDO HID, fingerprint reader). See
//! `docs/design/usb-xhci.md`. Only the aarch64 platform controller exists today.
//!
//! The [`UsbDeviceModel`] trait is the mechanism/policy seam: libkrun carries the
//! controller and the trait; limina implements the gadgets. A minimal
//! [`MockUsbDevice`] lives here to prove enumeration and anchor unit tests.

pub mod xhci;

mod bulk_pipe;
mod hid;
mod mock;
mod model;
mod report_pipe;

pub use self::bulk_pipe::{BulkPipe, BulkSink};
pub use self::hid::HidMockDevice;
pub use self::mock::MockUsbDevice;
pub use self::model::{
    ControlTransfer, DeviceDescriptors, EpAddr, SetupPacket, Transfer, UsbDeviceModel, UsbSpeed,
    XferOutcome,
};
pub use self::report_pipe::{HidReportPipe, ReportSink};
pub use self::xhci::{spawn_worker, XhciDevice, XHCI_MMIO_LEN};
