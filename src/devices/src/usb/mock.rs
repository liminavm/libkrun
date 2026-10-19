// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! A minimal, hardware-free full-speed USB device model. It exists to prove the
//! emulated xHCI controller enumerates an attached device end-to-end (the Stage
//! B1 exit oracle) and to anchor the ring/transfer unit tests. It is a
//! *vendor-specific* device with a single configuration and one interface that
//! carries **zero non-EP0 endpoints**, so a stock guest enumerates it fully
//! (device + config + strings) without binding any class driver.

use super::model::{DeviceDescriptors, UsbDeviceModel, UsbSpeed};

/// A distinctive, clearly-synthetic identity (`0x1d6b` = "Linux Foundation"
/// vendor-neutral VID, as the shipped FIDO/uhid gadget uses).
const VID: u16 = 0x1d6b;
const PID: u16 = 0x0f10;

// Descriptor types.
const DT_STRING: u8 = 0x03;

// Vendor-specific class (no class driver binds).
const CLASS_VENDOR: u8 = 0xff;

/// The 18-byte device descriptor.
fn device_descriptor() -> Vec<u8> {
    vec![
        0x12, // bLength = 18
        0x01, // bDescriptorType = DEVICE
        0x00,
        0x02,         // bcdUSB 2.00
        CLASS_VENDOR, // bDeviceClass (vendor-specific)
        0x00,         // bDeviceSubClass
        0x00,         // bDeviceProtocol
        0x40,         // bMaxPacketSize0 = 64
        (VID & 0xff) as u8,
        (VID >> 8) as u8, // idVendor (LE)
        (PID & 0xff) as u8,
        (PID >> 8) as u8, // idProduct (LE)
        0x00,
        0x01, // bcdDevice 1.00
        0x01, // iManufacturer
        0x02, // iProduct
        0x03, // iSerialNumber
        0x01, // bNumConfigurations
    ]
}

/// The configuration block: config descriptor + one vendor-specific interface
/// with no endpoints. wTotalLength = 9 + 9 = 18.
fn config_descriptor() -> Vec<u8> {
    let total_len: u16 = 18;
    let mut c = vec![
        0x09,
        0x02, // CONFIGURATION
        (total_len & 0xff) as u8,
        (total_len >> 8) as u8, // wTotalLength
        0x01,                   // bNumInterfaces
        0x01,                   // bConfigurationValue
        0x00,                   // iConfiguration
        0x80,                   // bmAttributes (bus-powered)
        0x32,                   // bMaxPower = 100 mA
    ];
    // Interface 0: vendor-specific, zero endpoints.
    c.extend_from_slice(&[
        0x09,
        0x04, // INTERFACE
        0x00, // bInterfaceNumber
        0x00, // bAlternateSetting
        0x00, // bNumEndpoints
        CLASS_VENDOR,
        0x00, // bInterfaceSubClass
        0x00, // bInterfaceProtocol
        0x04, // iInterface
    ]);
    debug_assert_eq!(c.len(), total_len as usize);
    c
}

/// A USB string descriptor (UTF-16LE); index 0 is the LANGID list (English-US).
fn string_descriptor(s: &str) -> Vec<u8> {
    let utf16: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let mut d = vec![(2 + utf16.len()) as u8, DT_STRING];
    d.extend_from_slice(&utf16);
    d
}

/// The full-speed mock device.
pub struct MockUsbDevice;

impl MockUsbDevice {
    pub fn new() -> Self {
        MockUsbDevice
    }
}

impl Default for MockUsbDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl UsbDeviceModel for MockUsbDevice {
    fn descriptors(&self) -> DeviceDescriptors {
        DeviceDescriptors {
            device: device_descriptor(),
            configs: vec![config_descriptor()],
            strings: vec![
                vec![0x04, DT_STRING, 0x09, 0x04], // index 0: LANGID 0x0409
                string_descriptor("limina"),       // 1: iManufacturer
                string_descriptor("limina Mock USB Device"), // 2: iProduct
                string_descriptor("LIMINA-MOCK-0001"), // 3: iSerialNumber
                string_descriptor("limina mock interface"), // 4: iInterface
            ],
        }
    }

    fn speed(&self) -> UsbSpeed {
        UsbSpeed::Full
    }

    // Vendor-specific control requests: none — the default (stall) is correct;
    // the guest only ever issues standard requests, answered by the controller.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_descriptor_is_well_formed() {
        let d = device_descriptor();
        assert_eq!(d.len(), 18);
        assert_eq!(d[0], 18);
        assert_eq!(u16::from_le_bytes([d[8], d[9]]), VID);
        assert_eq!(u16::from_le_bytes([d[10], d[11]]), PID);
        assert_eq!(d[17], 1, "one configuration");
    }

    #[test]
    fn config_total_length_matches() {
        let c = config_descriptor();
        assert_eq!(u16::from_le_bytes([c[2], c[3]]) as usize, c.len());
        assert_eq!(c[4], 1, "one interface");
        // Interface descriptor advertises zero endpoints.
        assert_eq!(c[9 + 4], 0, "zero endpoints");
    }

    #[test]
    fn strings_are_null_free_utf16() {
        let d = MockUsbDevice::new().descriptors();
        assert_eq!(d.strings[0], vec![0x04, DT_STRING, 0x09, 0x04]);
        // "limina" -> 2 header + 12 bytes.
        assert_eq!(d.strings[1].len(), 2 + "limina".len() * 2);
        assert_eq!(d.strings[1][1], DT_STRING);
    }
}
