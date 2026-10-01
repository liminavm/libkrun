// Copyright 2026, Red Hat Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Common display types for EDID generation.

use virtio_bindings::virtio_gpu::VIRTIO_GPU_MAX_SCANOUTS;

pub const MAX_DISPLAYS: usize = VIRTIO_GPU_MAX_SCANOUTS as usize;

use super::edid::EdidInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdidParams {
    pub refresh_rate: u32,
    pub physical_size: PhysicalSize,
    /// The *stable identity* half of the EDID: the fields a guest compositor uses to recognize
    /// this monitor across mode changes and reboots (and to key its remembered per-monitor
    /// configuration). `None` keeps the historical anonymous identity (`RHT`, product 1,
    /// serial 1, name `krun-display`), which is what a caller that never sets one gets.
    pub identity: Option<EdidIdentity>,
    /// Monitor range limits (descriptor tag `0xFD`). When set, the EDID also declares itself a
    /// continuous-frequency display, so the guest may infer modes within the range.
    pub range: Option<RefreshRange>,
    /// Replaces the built-in standard-timing list with caller-supplied modes. These are
    /// advertised *non-preferred* — only the detailed timing built from the display's current
    /// size is preferred (and the guest prunes any other preferred mode; see
    /// `docs/design/stable-edid-hotplug.md`).
    pub standard_timings: Option<StandardTimings>,
    /// An additional detailed timing (e.g. the same size at the panel's other refresh rate),
    /// emitted as the second detailed descriptor and therefore non-preferred.
    pub alt_mode: Option<DetailedMode>,
}

impl Default for EdidParams {
    fn default() -> Self {
        EdidParams {
            refresh_rate: 60,
            physical_size: PhysicalSize::Dpi(300),
            identity: None,
            range: None,
            standard_timings: None,
            alt_mode: None,
        }
    }
}

/// The identity fields of a generated EDID. Deliberately `Copy` and fixed-size so `EdidParams`
/// stays `Copy` and can be pushed at runtime without allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdidIdentity {
    /// Three upper-case letters, PNP-style (bytes 8-9, five bits per letter).
    pub manufacturer: [u8; 3],
    /// Manufacturer product code (bytes 10-11).
    pub product_id: u16,
    /// Serial number (bytes 12-15). Non-zero, unique per physical display.
    pub serial: u32,
    /// Display product name, ASCII, `0x0A`-terminated and space-padded to 13 bytes
    /// (descriptor tag `0xFC`).
    pub product_name: [u8; 13],
    /// Optional display product serial *string* (descriptor tag `0xFF`), same padding rule.
    pub serial_string: Option<[u8; 13]>,
}

/// Vertical/horizontal limits for the monitor-range-limits descriptor (tag `0xFD`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefreshRange {
    pub min_vertical_hz: u8,
    pub max_vertical_hz: u8,
    pub min_horizontal_khz: u8,
    pub max_horizontal_khz: u8,
    /// Maximum pixel clock in MHz; stored in the descriptor rounded up to a 10 MHz step.
    pub max_pixel_clock_mhz: u32,
}

/// One entry of the standard-timing list (bytes 38-53). The encoding only admits four aspect
/// ratios, widths that are a multiple of 8 in `256..=2288`, and refresh rates in `60..=123`;
/// entries that don't fit are dropped by the generator rather than mis-encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StandardTiming {
    pub width: u16,
    pub height: u16,
    pub refresh_hz: u16,
}

/// The eight standard-timing slots.
pub type StandardTimings = [Option<StandardTiming>; 8];

/// A full detailed timing: everything else about it (blanking, porches) uses the generator's
/// defaults, as the built-in preferred timing does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailedMode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PhysicalSize {
    Dpi(u32),
    DimensionsMillimeters(u16, u16),
}

/// User-configured display (monitor) properties.
/// Distinct from the scanout (guest framebuffer), which may be smaller.
#[derive(Clone, Debug)]
pub struct DisplayInfo {
    pub width: u32,
    pub height: u32,
    pub edid: DisplayInfoEdid,
}

#[derive(Debug, Clone)]
pub enum DisplayInfoEdid {
    Generated(EdidParams),
    Provided(Box<[u8]>),
}

impl DisplayInfo {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edid: DisplayInfoEdid::Generated(EdidParams::default()),
        }
    }

    pub fn edid_bytes(&self) -> Box<[u8]> {
        match &self.edid {
            DisplayInfoEdid::Provided(edid_bytes) => edid_bytes.clone(),
            DisplayInfoEdid::Generated(edid_params) => {
                let edid_info = EdidInfo::new(self.width, self.height, edid_params);
                edid_info.bytes()
            }
        }
    }
}
