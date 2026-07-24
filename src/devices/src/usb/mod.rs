// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! USB support: an emulated xHCI controller (mechanism) that limina drives with
//! software-defined gadgets (policy — FIDO HID, fingerprint reader). See
//! `docs/design/usb-xhci.md`. Only the aarch64 platform controller exists today.

pub mod xhci;

pub use self::xhci::XhciDevice;
