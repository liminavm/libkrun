// Copyright 2022 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! Implementation of the EDID specification provided by software.
//! EDID spec: <https://glenwing.github.io/docs/VESA-EEDID-A2.pdf>

//! This module is used to create the Extended Display Identification Data (EDID), which will be
//! exposed to the guest system.
//!
//! We ignore most of the spec, the point here being for us to provide enough for graphics to work
//! and to allow us to configure the resolution and refresh rate (via the preferred timing mode
//! pixel clock).
//!
//! The EDID spec defines a number of methods to provide mode information, but in priority order the
//! "detailed" timing information is first, so we provide a single block of detailed timing
//! information and no other form of timing information.
//!
//! limina extension: [`EdidParams`] can additionally carry a stable *identity* (manufacturer,
//! product, serial, name), a caller-supplied standard-timing list, a second detailed timing and
//! monitor range limits — everything a guest compositor needs to recognize the display across
//! mode changes and to know its real refresh range. All of it is optional and defaults to the
//! historical output byte for byte. See `docs/design/stable-edid-hotplug.md`.
use super::types::{
    EdidIdentity, EdidParams, PhysicalSize, RefreshRange, StandardTiming, StandardTimings,
};

const EDID_DATA_LENGTH: usize = 128;
/// Offset of the first 18-byte descriptor block; there are four, back to back.
const DESCRIPTOR_BASE: usize = 54;
const DESCRIPTOR_LEN: usize = 18;
const DESCRIPTOR_COUNT: usize = 4;
/// Descriptor tags we emit (EDID 1.4 §3.10.3).
const TAG_SERIAL_STRING: u8 = 0xFF;
const TAG_RANGE_LIMITS: u8 = 0xFD;
const TAG_PRODUCT_NAME: u8 = 0xFC;
const TAG_DUMMY: u8 = 0x10;
/// Range-descriptor flag byte: "range limits only" — no GTF/CVT formula. This is the *only*
/// variant Linux accepts for the VRR/`monitor_range` info (`drm_edid.c` `get_monitor_range`),
/// and it keeps inferred modes to the standard DMT list filtered by the range.
const RANGE_LIMITS_ONLY: u8 = 0x01;
/// Feature-support bit 0 (byte 24): continuous-frequency display. Linux requires it *together*
/// with the range descriptor before it will read the monitor's refresh range at all.
const FEATURE_CONTINUOUS_FREQ: u8 = 1 << 0;
const DEFAULT_HORIZONTAL_BLANKING: u16 = 560;
const DEFAULT_VERTICAL_BLANKING: u16 = 50;
const DEFAULT_HORIZONTAL_FRONT_PORCH: u16 = 64;
const DEFAULT_VERTICAL_FRONT_PORCH: u16 = 1;
const DEFAULT_HORIZONTAL_SYNC_PULSE: u16 = 192;
const DEFAULT_VERTICAL_SYNC_PULSE: u16 = 3;
const MILLIMETERS_PER_INCH: f32 = 25.4;

#[derive(Copy, Clone)]
pub struct EdidInfo {
    width: u32,
    height: u32,
    refresh_rate: u32,
    horizontal_blanking: u16,
    vertical_blanking: u16,
    horizontal_front: u16,
    vertical_front: u16,
    horizontal_sync: u16,
    vertical_sync: u16,
    width_millimeters: u16,
    height_millimeters: u16,
    /// limina: the optional identity / mode-list / range extensions (see module docs).
    params: EdidParams,
}

impl EdidInfo {
    /// Only width, height and refresh rate are required for the graphics stack to work, so instead
    /// of pulling actual numbers from the system, we just use some typical values to populate other
    /// fields for now.
    pub fn new(width: u32, height: u32, params: &EdidParams) -> Self {
        let (width_millimeters, height_millimeters) = match params.physical_size {
            PhysicalSize::Dpi(dpi) => (
                ((width as f32 / dpi as f32) * MILLIMETERS_PER_INCH) as u16,
                ((height as f32 / dpi as f32) * MILLIMETERS_PER_INCH) as u16,
            ),
            PhysicalSize::DimensionsMillimeters(width, height) => (width, height),
        };

        Self {
            width,
            height,
            refresh_rate: params.refresh_rate,
            horizontal_blanking: DEFAULT_HORIZONTAL_BLANKING,
            vertical_blanking: DEFAULT_VERTICAL_BLANKING,
            horizontal_front: DEFAULT_HORIZONTAL_FRONT_PORCH,
            vertical_front: DEFAULT_VERTICAL_FRONT_PORCH,
            horizontal_sync: DEFAULT_HORIZONTAL_SYNC_PULSE,
            vertical_sync: DEFAULT_VERTICAL_SYNC_PULSE,
            width_millimeters,
            height_millimeters,
            params: *params,
        }
    }

    pub fn width_centimeters(&self) -> u8 {
        (self.width_millimeters / 10) as u8
    }

    pub fn height_centimeters(&self) -> u8 {
        (self.height_millimeters / 10) as u8
    }

    pub fn bytes(self) -> Box<[u8]> {
        let mut edid_box: Box<[u8]> = vec![0; EDID_DATA_LENGTH].into_boxed_slice();
        let edid = &mut edid_box[..];

        populate_header(edid, self.params.identity.as_ref());
        populate_edid_version(edid);
        populate_size(edid, &self);
        populate_features(edid, &self.params);
        populate_standard_timings(edid, self.params.standard_timings.as_ref());
        populate_descriptors(edid, &self);

        calculate_checksum(edid);

        edid_box
    }
}

/// Fill the four 18-byte descriptor blocks, in priority order:
///
/// 0. the **detailed timing for the display's current mode** — must come first: EDID 1.4 makes
///    the first detailed timing the preferred mode unconditionally (`drm_edid.c`
///    `add_detailed_modes`), and the guest prunes any *other* preferred mode that doesn't match
///    the size we simultaneously push through `GET_DISPLAY_INFO` (`virtgpu_display.c`
///    `virtio_gpu_conn_mode_valid`).
/// 1. the product name — the identity anchor, always present.
/// 2. the monitor range limits, when set.
/// 3. the alternate detailed timing, when set (non-preferred, being second).
/// 4. the serial string, when set.
///
/// Only the first four survive. With no limina extensions in play this reduces to exactly the
/// historical two blocks + two zeroed ones, so the default output is unchanged byte for byte.
fn populate_descriptors(edid: &mut [u8], info: &EdidInfo) {
    let identity = info.params.identity.as_ref();
    let mut blocks: Vec<[u8; DESCRIPTOR_LEN]> = Vec::with_capacity(DESCRIPTOR_COUNT);

    blocks.push(detailed_timing_descriptor(
        info,
        info.width,
        info.height,
        info.refresh_rate,
    ));
    blocks.push(string_descriptor(
        TAG_PRODUCT_NAME,
        identity.map_or(DEFAULT_PRODUCT_NAME, |i| i.product_name),
    ));
    if let Some(range) = info.params.range.as_ref() {
        blocks.push(range_limits_descriptor(range));
    }
    if let Some(alt) = info.params.alt_mode.as_ref() {
        blocks.push(detailed_timing_descriptor(
            info,
            alt.width,
            alt.height,
            alt.refresh_hz,
        ));
    }
    if let Some(serial) = identity.and_then(|i| i.serial_string) {
        blocks.push(string_descriptor(TAG_SERIAL_STRING, serial));
    }
    if blocks.len() > DESCRIPTOR_COUNT {
        // A base EDID block has room for four descriptors and no more. Everything load-bearing
        // (the preferred timing, the product name) is at the front of the priority order, so
        // what falls off here is an extra: say so rather than dropping it silently.
        debug!(
            "edid: {} descriptors requested, {DESCRIPTOR_COUNT} fit; dropping the lowest-priority \
             {} (a CTA extension block would carry them)",
            blocks.len(),
            blocks.len() - DESCRIPTOR_COUNT
        );
    }
    blocks.truncate(DESCRIPTOR_COUNT);

    // Spare blocks: a proper "unused" descriptor once we're emitting a limina identity, but
    // left as zeroes on the historical path so that output stays byte-identical. (Linux is
    // happy either way — it skips a descriptor whose pixel clock is 0 and whose tag it doesn't
    // know — but a zero tag is not a legal descriptor and stricter parsers do complain.)
    if identity.is_some() {
        while blocks.len() < DESCRIPTOR_COUNT {
            blocks.push(dummy_descriptor());
        }
    }

    for (index, block) in blocks.iter().enumerate() {
        let start = DESCRIPTOR_BASE + index * DESCRIPTOR_LEN;
        edid[start..start + DESCRIPTOR_LEN].copy_from_slice(block);
    }
}

/// The historical product name. Padded to 13 bytes with the `0x0A` terminator the spec asks for.
const DEFAULT_PRODUCT_NAME: [u8; 13] = *b"krun-display\n";

/// A string descriptor (product name `0xFC`, serial `0xFF`): 5-byte header then 13 bytes of
/// already-padded ASCII (§3.10.3.4).
fn string_descriptor(tag: u8, text: [u8; 13]) -> [u8; DESCRIPTOR_LEN] {
    let mut block = [0u8; DESCRIPTOR_LEN];
    block[0..5].copy_from_slice(&[0x00, 0x00, 0x00, tag, 0x00]);
    block[5..].copy_from_slice(&text);
    block
}

/// The monitor range-limits descriptor (`0xFD`). We always use the "range limits only" variant:
/// it is the one Linux requires before it will believe the refresh range (and hence before VRR
/// can ever be advertised), and it keeps guest-inferred modes to the DMT list clipped by the
/// range instead of a synthesized GTF/CVT continuum.
fn range_limits_descriptor(range: &RefreshRange) -> [u8; DESCRIPTOR_LEN] {
    let mut block = [0u8; DESCRIPTOR_LEN];
    block[0..4].copy_from_slice(&[0x00, 0x00, 0x00, TAG_RANGE_LIMITS]);
    // Offset flags: no 255 Hz/kHz offsets — our ranges fit in a byte.
    block[4] = 0x00;
    block[5] = range.min_vertical_hz;
    block[6] = range.max_vertical_hz;
    block[7] = range.min_horizontal_khz;
    block[8] = range.max_horizontal_khz;
    // Max pixel clock, in 10 MHz steps, rounded *up* so the advertised limit never sits below
    // a mode we also advertise.
    block[9] = range.max_pixel_clock_mhz.div_ceil(10).min(0xFF) as u8;
    block[10] = RANGE_LIMITS_ONLY;
    // Padding mandated for the range-limits-only variant.
    block[11] = 0x0A;
    block[12..].fill(0x20);
    block
}

/// The "unused descriptor" filler (§3.10.3.11).
fn dummy_descriptor() -> [u8; DESCRIPTOR_LEN] {
    let mut block = [0u8; DESCRIPTOR_LEN];
    block[3] = TAG_DUMMY;
    block[5] = 0x0A;
    block[6..].fill(0x20);
    block
}

/// Feature-support byte (24). Declaring continuous frequency is what lets the guest read the
/// range descriptor at all (`drm_edid.c` `drm_get_monitor_range` bails without it), so the two
/// are set together or not at all.
fn populate_features(edid: &mut [u8], params: &EdidParams) {
    if params.range.is_some() {
        edid[24] |= FEATURE_CONTINUOUS_FREQ;
    }
}

fn detailed_timing_descriptor(
    info: &EdidInfo,
    width: u32,
    height: u32,
    refresh_rate: u32,
) -> [u8; DESCRIPTOR_LEN] {
    let mut block = [0u8; DESCRIPTOR_LEN];
    populate_detailed_timing(&mut block, info, width, height, refresh_rate);
    block
}

fn populate_detailed_timing(
    edid_block: &mut [u8],
    info: &EdidInfo,
    width: u32,
    height: u32,
    refresh_rate: u32,
) {
    assert_eq!(edid_block.len(), 18);

    // Detailed timings
    //
    // 18 Byte Descriptors - 72 Bytes
    // The 72 bytes in this section are divided into four data fields. Each of the four data fields
    // are 18 bytes in length. These 18 byte data fields shall contain either detailed timing data
    // as described in Section 3.10.2 or other types of data as described in Section 3.10.3. The
    // addresses and the contents of the four 18 byte descriptors are shown in Table 3.20.
    //
    // We leave the bottom 6 bytes of this block purposefully empty.
    let horizontal_blanking_lsb: u8 = (info.horizontal_blanking & 0xFF) as u8;
    let horizontal_blanking_msb: u8 = ((info.horizontal_blanking >> 8) & 0x0F) as u8;

    let vertical_blanking_lsb: u8 = (info.vertical_blanking & 0xFF) as u8;
    let vertical_blanking_msb: u8 = ((info.vertical_blanking >> 8) & 0x0F) as u8;

    // The pixel clock is what controls the refresh timing information.
    //
    // The formula for getting refresh rate out of this value is:
    //   refresh_rate = clk * 10000 / (htotal * vtotal)
    // Solving for clk:
    //   clk = (refresh_rate * htotal * votal) / 10000
    //
    // where:
    //   clk - The setting here
    //   vtotal - Total lines
    //   htotal - Total pixels per line
    //
    // Value here is pixel clock + 10,000, in 10khz steps.
    //
    // Pseudocode of kernel logic for vrefresh:
    //    vtotal := mode->vtotal;
    //    calc_val := (clock * 1000) / htotal
    //    refresh := (calc_val + vtotal / 2) / vtotal
    //    if flags & INTERLACE: refresh *= 2
    //    if flags & DBLSCAN: refresh /= 2
    //    if vscan > 1: refresh /= vscan
    //
    let htotal = width + (info.horizontal_blanking as u32);
    let vtotal = height + (info.vertical_blanking as u32);
    // Compute in u32: the field is 16 bits in 10 kHz steps, so it tops out at 655.35 MHz, and a
    // high-resolution mode at a high refresh rate exceeds that (3024x1964 @ 120 Hz needs
    // ~866 MHz with these blanking values). Truncating to u16 *wraps*, which silently encodes a
    // completely different refresh rate — 3024x1964 @ 120 Hz decoded back as 30 Hz. Saturate
    // instead: the size still comes through exactly (that is what drives the scanout) and the
    // refresh is understated rather than fabricated. Expressing such modes honestly needs a
    // CTA-861 extension block, which `GET_EDID` has room for — see
    // `docs/design/stable-edid-hotplug.md`.
    let raw_clock = (refresh_rate * htotal * vtotal) / 10000;
    // Round to nearest 10khz.
    let rounded = ((raw_clock + 5) / 10) * 10;
    if rounded > u32::from(u16::MAX) {
        warn!(
            "edid: {width}x{height}@{refresh_rate} needs a {} MHz pixel clock, above the \
             655.35 MHz an EDID detailed timing can express; clamping the advertised refresh",
            raw_clock / 100
        );
    }
    let clock = rounded.min(u32::from(u16::MAX)) as u16;
    edid_block[0..2].copy_from_slice(&clock.to_le_bytes());

    let width_lsb: u8 = (width & 0xFF) as u8;
    let width_msb: u8 = ((width >> 8) & 0x0F) as u8;

    // Horizointal Addressable Video in pixels.
    edid_block[2] = width_lsb;
    // Horizontal blanking in pixels.
    edid_block[3] = horizontal_blanking_lsb;
    // Upper bits of the two above vals.
    edid_block[4] = horizontal_blanking_msb | (width_msb << 4);

    let vertical_active: u32 = height;
    let vertical_active_lsb: u8 = (vertical_active & 0xFF) as u8;
    let vertical_active_msb: u8 = ((vertical_active >> 8) & 0x0F) as u8;

    // Vertical addressable video in *lines*
    edid_block[5] = vertical_active_lsb;
    // Vertical blanking in lines
    edid_block[6] = vertical_blanking_lsb;
    // Sigbits of the above.
    edid_block[7] = vertical_blanking_msb | (vertical_active_msb << 4);

    let horizontal_front_lsb: u8 = (info.horizontal_front & 0xFF) as u8; // least sig 8 bits
    let horizontal_front_msb: u8 = ((info.horizontal_front >> 8) & 0x03) as u8; // most sig 2 bits
    let horizontal_sync_lsb: u8 = (info.horizontal_sync & 0xFF) as u8; // least sig 8 bits
    let horizontal_sync_msb: u8 = ((info.horizontal_sync >> 8) & 0x03) as u8; // most sig 2 bits

    let vertical_front_lsb: u8 = (info.vertical_front & 0x0F) as u8; // least sig 4 bits
    let vertical_front_msb: u8 = ((info.vertical_front >> 8) & 0x0F) as u8; // most sig 2 bits
    let vertical_sync_lsb: u8 = (info.vertical_sync & 0xFF) as u8; // least sig 4 bits
    let vertical_sync_msb: u8 = ((info.vertical_sync >> 8) & 0x0F) as u8; // most sig 2 bits

    // Horizontal front porch in pixels.
    edid_block[8] = horizontal_front_lsb;
    // Horizontal sync pulse width in pixels.
    edid_block[9] = horizontal_sync_lsb;
    // LSB of vertical front porch and sync pulse
    edid_block[10] = vertical_sync_lsb | (vertical_front_lsb << 4);
    // Upper 2 bits of these values.
    edid_block[11] = vertical_sync_msb
        | (vertical_front_msb << 2)
        | (horizontal_sync_msb << 4)
        | (horizontal_front_msb << 6);

    let width_millimeters_lsb: u8 = (info.width_millimeters & 0xFF) as u8; // least sig 8 bits
    let width_millimeters_msb: u8 = ((info.width_millimeters >> 8) & 0xF) as u8; // most sig 4 bits

    let height_millimeters_lsb: u8 = (info.height_millimeters & 0xFF) as u8; // least sig 8 bits
    let height_millimeters_msb: u8 = ((info.height_millimeters >> 8) & 0xF) as u8; // most sig 4 bits

    edid_block[12] = width_millimeters_lsb;
    edid_block[13] = height_millimeters_lsb;
    edid_block[14] = height_millimeters_msb | (width_millimeters_msb << 4);
}

// The EDID header. This is defined by the EDID spec.
fn populate_header(edid: &mut [u8], identity: Option<&EdidIdentity>) {
    edid[0] = 0x00;
    edid[1] = 0xFF;
    edid[2] = 0xFF;
    edid[3] = 0xFF;
    edid[4] = 0xFF;
    edid[5] = 0xFF;
    edid[6] = 0xFF;
    edid[7] = 0x00;

    // Red Hat 'RHT' is also used in QEMU, though it is not technically officially assigned.
    // A caller-supplied identity overrides it (limina names its own displays), and the whole
    // triple travels together: vendor + product + serial is what a guest compositor keys its
    // remembered per-monitor configuration on, so these must stay put across mode changes.
    let manufacturer_name = identity.map_or(*b"RHT", |i| i.manufacturer);
    // 00001 -> A, 00010 -> B, etc
    let manufacturer_id: u16 = manufacturer_name
        .iter()
        .map(|c| (c.to_ascii_uppercase().wrapping_sub(b'A').wrapping_add(1)) & 0x1F)
        .fold(0u16, |res, lsb| (res << 5) | (lsb as u16));
    edid[8..10].copy_from_slice(&manufacturer_id.to_be_bytes());

    let manufacture_product_id: u16 = identity.map_or(1, |i| i.product_id);
    edid[10..12].copy_from_slice(&manufacture_product_id.to_le_bytes());

    let serial_id: u32 = identity.map_or(1, |i| i.serial);
    edid[12..16].copy_from_slice(&serial_id.to_le_bytes());

    let manufacture_week: u8 = 30;
    edid[16] = manufacture_week;

    let manufacture_year: u32 = 2025;
    edid[17] = (manufacture_year - 1990u32) as u8;
}

// The standard timings are 8 timing modes with a lower priority (and different data format)
// than the 4 detailed timing modes. A caller may supply its own list (the host display's real
// modes); otherwise the built-in one below is used. These are never *preferred* modes, which is
// what makes it safe to advertise sizes the display isn't currently set to.
fn populate_standard_timings(edid: &mut [u8], supplied: Option<&StandardTimings>) {
    if let Some(timings) = supplied {
        for (index, slot) in timings.iter().enumerate() {
            let encoded = slot.and_then(encode_standard_timing);
            let (byte0, byte1) = encoded.unwrap_or(UNUSED_STANDARD_TIMING);
            edid[0x26 + (index * 2)] = byte0;
            edid[0x27 + (index * 2)] = byte1;
        }
        return;
    }

    const fn aspect_ratio(width: u32, height: u32) -> (u32, u32) {
        let divisor = gcd(width, height);
        (width / divisor, height / divisor)
    }

    const fn aspect_ratio_bits(width: u32, height: u32) -> u8 {
        match aspect_ratio(width, height) {
            (8, 5) => 0x0,
            (4, 3) => 0x1,
            (5, 4) => 0x2,
            (16, 9) => 0x3,
            _ => panic!("Not a standard aspect ratio"),
        }
    }

    const fn resolution(width: u32, height: u32) -> (u32, u32, u8) {
        (width, height, aspect_ratio_bits(width, height))
    }

    const RESOLUTIONS: [(u32, u32, u8); 8] = [
        resolution(1440, 900),
        resolution(1600, 900),
        resolution(800, 600),
        resolution(1680, 1050),
        resolution(1856, 1392),
        resolution(1280, 1024),
        resolution(1400, 1050),
        resolution(1920, 1200),
    ];

    // Index 0 is horizontal pixels / 8 - 31.
    // Index 1 is the aspect ratio in bits 7-6 and (refresh rate - 60) in bits 5-0 (EDID §3.9;
    // `EDID_TIMING_ASPECT_SHIFT` = 6 in Linux's `drm_edid.h`). The aspect bits MUST be shifted
    // into place — writing them unshifted lands them in the refresh field, which silently
    // turns e.g. 1600x900 (aspect code 3) into 1600x1000 @ 63 Hz. All eight entries here are
    // 60 Hz, so the refresh field is zero.
    for (index, (width, _height, aspect_ratio_bits)) in RESOLUTIONS.into_iter().enumerate() {
        edid[0x26 + (index * 2)] = (width / 8 - 31) as u8;
        edid[0x27 + (index * 2)] = aspect_ratio_bits << 6;
    }
}

/// The "unused" standard-timing slot (EDID §3.9: `0x01, 0x01`).
const UNUSED_STANDARD_TIMING: (u8, u8) = (0x01, 0x01);

/// Encode one caller-supplied standard timing, or `None` when the entry can't be represented:
/// the format only admits widths that are a multiple of 8 in `256..=2288`, refresh rates in
/// `60..=123`, and the four aspect ratios below. Dropping an unrepresentable mode is the right
/// failure — mis-encoding one advertises a resolution the caller never asked for.
fn encode_standard_timing(timing: StandardTiming) -> Option<(u8, u8)> {
    let width = u32::from(timing.width);
    let height = u32::from(timing.height);
    if width % 8 != 0 || !(256..=2288).contains(&width) || height == 0 {
        return None;
    }
    if !(60..=123).contains(&timing.refresh_hz) {
        return None;
    }
    let divisor = gcd(width, height);
    let aspect_bits: u8 = match (width / divisor, height / divisor) {
        (16, 10) | (8, 5) => 0x0,
        (4, 3) => 0x1,
        (5, 4) => 0x2,
        (16, 9) => 0x3,
        _ => return None,
    };
    let byte0 = (width / 8 - 31) as u8;
    let byte1 = (aspect_bits << 6) | ((timing.refresh_hz - 60) as u8 & 0x3F);
    Some((byte0, byte1))
}

// Per the EDID spec, needs to be 1 and 4.
fn populate_edid_version(edid: &mut [u8]) {
    edid[18] = 1;
    edid[19] = 4;
}

fn populate_size(edid: &mut [u8], info: &EdidInfo) {
    edid[21] = info.width_centimeters();
    edid[22] = info.height_centimeters();
}

fn calculate_checksum(edid: &mut [u8]) {
    let mut checksum: u8 = 0;
    for byte in edid.iter().take(EDID_DATA_LENGTH - 1) {
        checksum = checksum.wrapping_add(*byte);
    }

    if checksum != 0 {
        checksum = 255 - checksum + 1;
    }

    edid[127] = checksum;
}

const fn gcd(x: u32, y: u32) -> u32 {
    match y {
        0 => x,
        _ => gcd(y, x % y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::types::DetailedMode;

    /// Decode a 13-byte descriptor payload the way a parser would: up to the `0x0A` terminator.
    fn descriptor_text(block: &[u8]) -> String {
        let payload = &block[5..18];
        let end = payload.iter().position(|b| *b == 0x0A).unwrap_or(13);
        String::from_utf8_lossy(&payload[..end]).into_owned()
    }

    fn descriptor(edid: &[u8], index: usize) -> &[u8] {
        let start = DESCRIPTOR_BASE + index * DESCRIPTOR_LEN;
        &edid[start..start + DESCRIPTOR_LEN]
    }

    /// A descriptor is a *detailed timing* iff its pixel clock (first two bytes) is non-zero;
    /// otherwise byte 3 is the display-descriptor tag. Same rule Linux uses.
    fn is_detailed_timing(block: &[u8]) -> bool {
        block[0] != 0 || block[1] != 0
    }

    /// Decode a detailed timing back to (width, height, refresh Hz), inverting the generator via
    /// the kernel's own formula (`refresh = clock * 10000 / (htotal * vtotal)`).
    fn decode_detailed(block: &[u8]) -> (u32, u32, u32) {
        let clock = u16::from_le_bytes([block[0], block[1]]) as u32;
        let width = block[2] as u32 | (((block[4] >> 4) as u32) << 8);
        let h_blank = block[3] as u32 | ((block[4] & 0x0F) as u32) << 8;
        let height = block[5] as u32 | (((block[7] >> 4) as u32) << 8);
        let v_blank = block[6] as u32 | ((block[7] & 0x0F) as u32) << 8;
        let htotal = width + h_blank;
        let vtotal = height + v_blank;
        // Rounded, as `drm_mode_vrefresh` does — the pixel clock only has 10 kHz resolution, so
        // an exact division would report 121 Hz for a 120 Hz mode.
        let pixels = htotal * vtotal;
        let refresh = (clock * 10000 + pixels / 2) / pixels;
        (width, height, refresh)
    }

    /// Decode the standard-timing list the way `drm_mode_std` does: `hsize = byte0 * 8 + 248`,
    /// aspect ratio from bits 7-6, vertical frequency = bits 5-0 + 60.
    fn decode_standard_timings(edid: &[u8]) -> Vec<(u32, u32, u32)> {
        let mut out = Vec::new();
        for index in 0..8 {
            let byte0 = edid[0x26 + index * 2];
            let byte1 = edid[0x27 + index * 2];
            if (byte0, byte1) == UNUSED_STANDARD_TIMING {
                continue;
            }
            let hsize = byte0 as u32 * 8 + 248;
            let vsize = match byte1 >> 6 {
                0x0 => hsize * 10 / 16,
                0x1 => hsize * 3 / 4,
                0x2 => hsize * 4 / 5,
                _ => hsize * 9 / 16,
            };
            out.push((hsize, vsize, (byte1 & 0x3F) as u32 + 60));
        }
        out
    }

    fn checksum_is_valid(edid: &[u8]) -> bool {
        edid.iter().fold(0u8, |acc, b| acc.wrapping_add(*b)) == 0
    }

    fn name(text: &str) -> [u8; 13] {
        let mut out = [0x20u8; 13];
        let bytes = text.as_bytes();
        let len = bytes.len().min(12);
        out[..len].copy_from_slice(&bytes[..len]);
        out[len] = 0x0A;
        out
    }

    fn identity() -> EdidIdentity {
        EdidIdentity {
            manufacturer: *b"LMN",
            product_id: 0xBEEF,
            serial: 0x0123_4567,
            product_name: name("Built-in"),
            serial_string: Some(name("HOST-UUID-1")),
        }
    }

    fn build(width: u32, height: u32, params: &EdidParams) -> Box<[u8]> {
        EdidInfo::new(width, height, params).bytes()
    }

    #[test]
    fn default_edid_is_well_formed() {
        let edid = build(1920, 1080, &EdidParams::default());

        assert_eq!(
            &edid[0..8],
            &[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]
        );
        assert_eq!((edid[18], edid[19]), (1, 4), "EDID 1.4");
        assert!(checksum_is_valid(&edid));
        assert!(is_detailed_timing(descriptor(&edid, 0)));
        assert_eq!(descriptor(&edid, 1)[3], TAG_PRODUCT_NAME);
        assert_eq!(descriptor_text(descriptor(&edid, 1)), "krun-display");
        assert_eq!(decode_detailed(descriptor(&edid, 0)), (1920, 1080, 60));
        // No identity ⇒ the historical anonymous one, and no range ⇒ no continuous-frequency bit.
        assert_eq!(&edid[10..12], &1u16.to_le_bytes());
        assert_eq!(&edid[12..16], &1u32.to_le_bytes());
        assert_eq!(edid[24] & FEATURE_CONTINUOUS_FREQ, 0);
    }

    /// The built-in standard timings must decode back to the resolutions they name. This is the
    /// regression test for the aspect bits having been written unshifted, which silently turned
    /// five of the eight entries into different resolutions (1600x900 → 1600x1000 @ 63 Hz).
    #[test]
    fn builtin_standard_timings_decode_to_their_resolutions() {
        let edid = build(1920, 1080, &EdidParams::default());
        let decoded = decode_standard_timings(&edid);

        for expected in [
            (1440, 900, 60),
            (1600, 900, 60),
            (800, 600, 60),
            (1680, 1050, 60),
            (1856, 1392, 60),
            (1280, 1024, 60),
            (1400, 1050, 60),
            (1920, 1200, 60),
        ] {
            assert!(
                decoded.contains(&expected),
                "standard timing {expected:?} missing; decoded list was {decoded:?}"
            );
        }
    }

    /// The point of the whole exercise: a mode change must not disturb one identity byte.
    #[test]
    fn identity_survives_a_mode_change() {
        let params = EdidParams {
            identity: Some(identity()),
            ..EdidParams::default()
        };
        let small = build(1280, 800, &params);
        let large = build(2560, 1440, &params);

        // Vendor / product / serial / manufacture date: bytes 8..18.
        assert_eq!(&small[8..18], &large[8..18]);
        // And the string descriptors that carry the name and serial.
        assert_eq!(descriptor(&small, 1), descriptor(&large, 1));
        assert_eq!(descriptor_text(descriptor(&small, 1)), "Built-in");
        let serial_index = 2;
        assert_eq!(descriptor(&small, serial_index)[3], TAG_SERIAL_STRING);
        assert_eq!(
            descriptor(&small, serial_index),
            descriptor(&large, serial_index)
        );

        // The timing, meanwhile, must have moved.
        assert_eq!(decode_detailed(descriptor(&small, 0)), (1280, 800, 60));
        assert_eq!(decode_detailed(descriptor(&large, 0)), (2560, 1440, 60));
        assert!(checksum_is_valid(&small) && checksum_is_valid(&large));
    }

    #[test]
    fn identity_encodes_the_manufacturer_and_ids() {
        let params = EdidParams {
            identity: Some(identity()),
            ..EdidParams::default()
        };
        let edid = build(1920, 1080, &params);

        // 'LMN' → five bits per letter, big-endian: L=12, M=13, N=14.
        let expected = ((12u16) << 10) | ((13u16) << 5) | 14u16;
        assert_eq!(&edid[8..10], &expected.to_be_bytes());
        assert_eq!(&edid[10..12], &0xBEEFu16.to_le_bytes());
        assert_eq!(&edid[12..16], &0x0123_4567u32.to_le_bytes());
    }

    /// Two different host displays must never collide, and the same one must be reproducible.
    #[test]
    fn distinct_identities_produce_distinct_edids() {
        let a = EdidParams {
            identity: Some(identity()),
            ..EdidParams::default()
        };
        let mut other = identity();
        other.serial = 0x89AB_CDEF;
        other.product_name = name("External");
        let b = EdidParams {
            identity: Some(other),
            ..EdidParams::default()
        };

        assert_ne!(build(1920, 1080, &a), build(1920, 1080, &b));
        assert_eq!(build(1920, 1080, &a), build(1920, 1080, &a));
    }

    /// The range descriptor and the continuous-frequency feature bit must appear together —
    /// Linux ignores the range entirely without the bit (`drm_get_monitor_range`), and the
    /// "range limits only" flag is the only variant it accepts for the VRR range.
    #[test]
    fn range_limits_are_emitted_in_the_form_linux_accepts() {
        let params = EdidParams {
            identity: Some(identity()),
            range: Some(RefreshRange {
                min_vertical_hz: 48,
                max_vertical_hz: 120,
                min_horizontal_khz: 30,
                max_horizontal_khz: 200,
                max_pixel_clock_mhz: 675,
            }),
            ..EdidParams::default()
        };
        let edid = build(3024, 1964, &params);

        assert_eq!(edid[24] & FEATURE_CONTINUOUS_FREQ, FEATURE_CONTINUOUS_FREQ);
        let block = descriptor(&edid, 2);
        assert_eq!(block[3], TAG_RANGE_LIMITS);
        assert_eq!(block[4], 0x00, "no vfreq offsets");
        assert_eq!((block[5], block[6]), (48, 120));
        assert_eq!((block[7], block[8]), (30, 200));
        assert_eq!(block[9], 68, "max pixel clock rounds UP to a 10 MHz step");
        assert_eq!(block[10], RANGE_LIMITS_ONLY);
        assert!(checksum_is_valid(&edid));
    }

    /// The alternate mode must land in a *later* detailed-timing slot: EDID 1.4 makes the first
    /// detailed timing preferred, and the guest prunes any preferred mode that disagrees with
    /// the size pushed through GET_DISPLAY_INFO.
    #[test]
    fn alt_mode_is_a_non_preferred_second_detailed_timing() {
        // The match-host geometry of a ProMotion MacBook panel (screen *points*, which is what
        // limina drives the guest to), at both of its refresh rates.
        let params = EdidParams {
            identity: Some(identity()),
            refresh_rate: 120,
            alt_mode: Some(DetailedMode {
                width: 1512,
                height: 982,
                refresh_hz: 60,
            }),
            ..EdidParams::default()
        };
        let edid = build(1512, 982, &params);

        let first = descriptor(&edid, 0);
        assert!(is_detailed_timing(first));
        assert_eq!(decode_detailed(first), (1512, 982, 120));

        let alt_index = (0..DESCRIPTOR_COUNT)
            .skip(1)
            .find(|i| is_detailed_timing(descriptor(&edid, *i)))
            .expect("the alternate detailed timing should occupy a later slot");
        assert_eq!(
            decode_detailed(descriptor(&edid, alt_index)),
            (1512, 982, 60)
        );
    }

    /// A mode whose pixel clock overflows the 16-bit field must be *clamped*, never wrapped:
    /// wrapping silently advertised 3024x1964 @ 120 Hz as a 30 Hz mode. The size — which is
    /// what actually drives the scanout — must still be exact.
    #[test]
    fn an_overflowing_pixel_clock_saturates_instead_of_wrapping() {
        let params = EdidParams {
            identity: Some(identity()),
            refresh_rate: 120,
            ..EdidParams::default()
        };
        let edid = build(3024, 1964, &params);

        let block = descriptor(&edid, 0);
        assert_eq!(u16::from_le_bytes([block[0], block[1]]), u16::MAX);
        let (width, height, refresh) = decode_detailed(block);
        assert_eq!((width, height), (3024, 1964));
        assert!(
            refresh < 120,
            "an unencodable refresh must be understated, not fabricated (got {refresh})"
        );
    }

    #[test]
    fn supplied_standard_timings_replace_the_builtin_list() {
        let mut timings: StandardTimings = [None; 8];
        timings[0] = Some(StandardTiming {
            width: 1920,
            height: 1080,
            refresh_hz: 60,
        });
        timings[1] = Some(StandardTiming {
            width: 1280,
            height: 800,
            refresh_hz: 120,
        });
        let params = EdidParams {
            identity: Some(identity()),
            standard_timings: Some(timings),
            ..EdidParams::default()
        };
        let edid = build(1920, 1080, &params);

        let decoded = decode_standard_timings(&edid);
        assert_eq!(decoded, vec![(1920, 1080, 60), (1280, 800, 120)]);
    }

    /// Modes the format can't express are dropped, never mis-encoded into a different mode.
    #[test]
    fn unrepresentable_standard_timings_are_dropped() {
        for bad in [
            // Width not a multiple of 8.
            StandardTiming {
                width: 900,
                height: 600,
                refresh_hz: 60,
            },
            // Width out of the encodable range.
            StandardTiming {
                width: 3024,
                height: 1964,
                refresh_hz: 60,
            },
            // Refresh out of the 60..=123 window.
            StandardTiming {
                width: 1920,
                height: 1080,
                refresh_hz: 144,
            },
            // Not one of the four encodable aspect ratios.
            StandardTiming {
                width: 1024,
                height: 600,
                refresh_hz: 60,
            },
        ] {
            assert_eq!(
                encode_standard_timing(bad),
                None,
                "should not encode {bad:?}"
            );
        }
    }

    /// Spare descriptor slots get the proper "unused" tag on the limina path, and stay zeroed on
    /// the historical one (so that output is byte-identical to what callers already got).
    #[test]
    fn spare_descriptors_are_tagged_only_on_the_identity_path() {
        let plain = build(1920, 1080, &EdidParams::default());
        assert_eq!(descriptor(&plain, 2), [0u8; DESCRIPTOR_LEN]);
        assert_eq!(descriptor(&plain, 3), [0u8; DESCRIPTOR_LEN]);

        let mut ident = identity();
        ident.serial_string = None;
        let params = EdidParams {
            identity: Some(ident),
            ..EdidParams::default()
        };
        let edid = build(1920, 1080, &params);
        assert_eq!(descriptor(&edid, 2)[3], TAG_DUMMY);
        assert_eq!(descriptor(&edid, 3)[3], TAG_DUMMY);
        assert!(checksum_is_valid(&edid));
    }

    /// Physical size tracks the mode at constant DPI — deliberately, so a resize never changes
    /// the DPI the guest computes and therefore never flips its scale factor mid-drag.
    #[test]
    fn dpi_is_constant_across_a_resize() {
        let params = EdidParams {
            identity: Some(identity()),
            physical_size: PhysicalSize::Dpi(150),
            ..EdidParams::default()
        };
        let small = build(1500, 1000, &params);
        let large = build(3000, 2000, &params);

        // 1500 px @ 150 dpi = 10 in = 254 mm ⇒ 25 cm; double the pixels, double the size.
        assert_eq!((small[21], small[22]), (25, 16));
        assert_eq!((large[21], large[22]), (50, 33));
    }
}
