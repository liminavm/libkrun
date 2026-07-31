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
/// Video input definition (byte 20): digital input, 8 bits per colour, DisplayPort. Left zero,
/// this byte means *analog* — which is what every EDID parser then reported for a virtual
/// display that has never been anything but digital, and it denies the guest a colour depth.
const VIDEO_INPUT_DIGITAL_8BPC_DP: u8 = 0x80 | (0b010 << 4) | 0x05;
/// Byte 126: number of extension blocks following the base block.
const EXTENSION_COUNT_OFFSET: usize = 126;
/// EDID extension tag for a DisplayID structure (`DISPLAYID_EXT`).
const DISPLAYID_EXT_TAG: u8 = 0x70;
/// DisplayID structure version 2.0.
const DISPLAYID_REV_2_0: u8 = 0x20;
/// Primary use case "desktop productivity" — what a VM window is.
const DISPLAYID_PRIMARY_USE: u8 = 4;
/// `DATA_BLOCK_2_TYPE_7_DETAILED_TIMING`: the timing block whose pixel clock is 24-bit and in
/// **kHz**, so it can express modes the base block's 16-bit/10 kHz field cannot.
const DISPLAYID_BLOCK_TYPE_7: u8 = 0x22;
/// `struct displayid_header` — rev, payload bytes, primary use, extension count.
const DISPLAYID_HEADER_LEN: usize = 4;
/// `struct displayid_block` — tag, rev, payload bytes.
const DISPLAYID_BLOCK_HEADER_LEN: usize = 3;
/// `struct displayid_detailed_timings_1`, fixed size; the kernel rejects a block whose length
/// is not a multiple of it (`add_displayid_detailed_1_modes`).
const DISPLAYID_TIMING_LEN: usize = 20;
/// The DisplayID structure starts at extension byte 1 and must end before the EDID extension
/// checksum at byte 127 ("EDID extensions block checksum isn't for us" — `drm_displayid.c`),
/// with its own checksum byte in between.
const DISPLAYID_MAX_PAYLOAD: usize = EDID_DATA_LENGTH - 1 - DISPLAYID_HEADER_LEN - 1 - 1;
/// How many timings therefore fit in one extension block.
const DISPLAYID_MAX_TIMINGS: usize =
    (DISPLAYID_MAX_PAYLOAD - DISPLAYID_BLOCK_HEADER_LEN) / DISPLAYID_TIMING_LEN;
/// Type VII timing flags bit 7: this timing is the display's preferred mode.
const DISPLAYID_TIMING_PREFERRED: u8 = 0x80;
/// The highest pixel clock a base detailed timing can express: a 16-bit field in 10 kHz steps.
const BASE_DTD_MAX_CLOCK_KHZ: u32 = 655_350;

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
        // Modes the base block has to misreport get an honest copy in a DisplayID extension.
        // Deciding this first: whether one exists sets the extension count *inside* the base
        // block, which the base checksum then has to cover.
        let extended = self.overflowing_timings();

        let length = if extended.is_empty() {
            EDID_DATA_LENGTH
        } else {
            EDID_DATA_LENGTH * 2
        };
        let mut edid_box: Box<[u8]> = vec![0; length].into_boxed_slice();
        let edid = &mut edid_box[..EDID_DATA_LENGTH];

        populate_header(edid, self.params.identity.as_ref());
        populate_edid_version(edid);
        populate_size(edid, &self);
        populate_features(edid, &self.params);
        populate_standard_timings(edid, self.params.standard_timings.as_ref());
        populate_descriptors(edid, &self);
        if !extended.is_empty() {
            edid[EXTENSION_COUNT_OFFSET] = 1;
        }

        calculate_checksum(edid);

        if !extended.is_empty() {
            populate_displayid_extension(&mut edid_box[EDID_DATA_LENGTH..], &self, &extended);
        }

        edid_box
    }

    /// The advertised timings whose pixel clock does not fit a base detailed timing, in the
    /// order they should appear in the extension. The current mode comes first and is marked
    /// preferred: the base block is *forced* to carry it (that is the only way `GET_EDID` and
    /// `GET_DISPLAY_INFO` agree on the rect, without which `virtio_gpu_conn_mode_valid` prunes
    /// the preferred mode outright), so its clamped refresh is a lie we are obliged to tell.
    /// The DisplayID copy is the truth, and having the same active size it survives the same
    /// pruning check; `drm_mode_sort` orders equal-priority modes by descending clock, so the
    /// honest higher rate lands ahead of the clamped one.
    fn overflowing_timings(&self) -> Vec<ExtendedTiming> {
        let mut out = Vec::new();
        let mut consider = |width, height, refresh_hz, preferred| {
            if self.pixel_clock_khz(width, height, refresh_hz) > BASE_DTD_MAX_CLOCK_KHZ {
                out.push(ExtendedTiming {
                    width,
                    height,
                    refresh_hz,
                    preferred,
                });
            }
        };
        consider(self.width, self.height, self.refresh_rate, true);
        if let Some(alt) = self.params.alt_mode.as_ref() {
            consider(alt.width, alt.height, alt.refresh_hz, false);
        }
        if out.len() > DISPLAYID_MAX_TIMINGS {
            debug!(
                "edid: {} timings overflow the base block, {DISPLAYID_MAX_TIMINGS} fit one \
                 DisplayID extension; dropping {}",
                out.len(),
                out.len() - DISPLAYID_MAX_TIMINGS
            );
            out.truncate(DISPLAYID_MAX_TIMINGS);
        }
        out
    }

    /// Pixel clock a mode needs at our blanking, in kHz.
    fn pixel_clock_khz(&self, width: u32, height: u32, refresh_hz: u32) -> u32 {
        let htotal = width + u32::from(self.horizontal_blanking);
        let vtotal = height + u32::from(self.vertical_blanking);
        refresh_hz.saturating_mul(htotal).saturating_mul(vtotal) / 1000
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
             {} (a DisplayID extension block would carry them)",
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

/// A timing bound for the DisplayID extension block.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ExtendedTiming {
    width: u32,
    height: u32,
    refresh_hz: u32,
    preferred: bool,
}

/// Write a 128-byte DisplayID 2.0 extension block holding one type VII detailed-timing block.
///
/// Layout, per `drm_displayid.c` (which is the parser we must satisfy — the spec allows more
/// than the kernel reads):
///
/// ```text
///   [0]      0x70                     EDID extension tag
///   [1..5]   struct displayid_header  rev, payload bytes, primary use, extension count
///   [5..]    struct displayid_block   tag 0x22, rev, payload bytes
///            20 bytes per timing
///   [..]     DisplayID checksum       over the header, the blocks and itself
///   [127]    EDID extension checksum  "isn't for us" per the kernel, but written anyway
/// ```
fn populate_displayid_extension(ext: &mut [u8], info: &EdidInfo, timings: &[ExtendedTiming]) {
    debug_assert_eq!(ext.len(), EDID_DATA_LENGTH);
    debug_assert!(timings.len() <= DISPLAYID_MAX_TIMINGS);

    let payload = DISPLAYID_BLOCK_HEADER_LEN + timings.len() * DISPLAYID_TIMING_LEN;

    ext[0] = DISPLAYID_EXT_TAG;
    ext[1] = DISPLAYID_REV_2_0;
    ext[2] = payload as u8;
    ext[3] = DISPLAYID_PRIMARY_USE;
    // Extension count: DisplayID sections beyond this one, of which there are none.
    ext[4] = 0;

    let block = 1 + DISPLAYID_HEADER_LEN;
    ext[block] = DISPLAYID_BLOCK_TYPE_7;
    // Block revision 0; bits 4-3 hold the payload-bytes-per-timing selector, and 0 means the
    // 20-byte `displayid_detailed_timings_1` the kernel expects.
    ext[block + 1] = 0;
    ext[block + 2] = (timings.len() * DISPLAYID_TIMING_LEN) as u8;

    for (index, timing) in timings.iter().enumerate() {
        let at = block + DISPLAYID_BLOCK_HEADER_LEN + index * DISPLAYID_TIMING_LEN;
        write_displayid_timing(&mut ext[at..at + DISPLAYID_TIMING_LEN], info, timing);
    }

    // The DisplayID checksum covers the header, every block, and itself — it is *not* the EDID
    // extension checksum, and the kernel rejects the whole structure if it does not sum to zero.
    let end = 1 + DISPLAYID_HEADER_LEN + payload;
    let sum = ext[1..end]
        .iter()
        .fold(0u8, |acc, byte| acc.wrapping_add(*byte));
    ext[end] = sum.wrapping_neg();

    calculate_checksum(ext);
}

/// One `struct displayid_detailed_timings_1`. **Every field is stored as `value - 1`** and the
/// kernel adds it back (`drm_mode_displayid_detailed`); the sync fields additionally carry the
/// polarity in bit 15, left zero here to match the negative polarity the base timing implies.
fn write_displayid_timing(out: &mut [u8], info: &EdidInfo, timing: &ExtendedTiming) {
    debug_assert_eq!(out.len(), DISPLAYID_TIMING_LEN);

    // 24-bit, in kHz — the whole reason this block exists.
    let clock = info
        .pixel_clock_khz(timing.width, timing.height, timing.refresh_hz)
        .clamp(1, 0x00FF_FFFF)
        - 1;
    out[0] = (clock & 0xFF) as u8;
    out[1] = ((clock >> 8) & 0xFF) as u8;
    out[2] = ((clock >> 16) & 0xFF) as u8;
    out[3] = if timing.preferred {
        DISPLAYID_TIMING_PREFERRED
    } else {
        0
    };

    let mut put = |at: usize, value: u32| {
        let encoded = (value.max(1) - 1) as u16;
        out[at..at + 2].copy_from_slice(&encoded.to_le_bytes());
    };
    put(4, timing.width);
    put(6, u32::from(info.horizontal_blanking));
    put(8, u32::from(info.horizontal_front));
    put(10, u32::from(info.horizontal_sync));
    put(12, timing.height);
    put(14, u32::from(info.vertical_blanking));
    put(16, u32::from(info.vertical_front));
    put(18, u32::from(info.vertical_sync));
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
    // Byte 4 is the EDID 1.4 offset-flags byte: a set bit means "add 255 to this bound".
    // The horizontal rates need it — a 4K panel at 120 Hz is 265 kHz, past what the byte
    // holds — and understating the bound would have the guest prune a mode we advertise.
    let (min_h, min_h_offset) = split_offset(range.min_horizontal_khz);
    let (max_h, max_h_offset) = split_offset(range.max_horizontal_khz);
    block[4] = (u8::from(min_h_offset) << 2) | (u8::from(max_h_offset) << 3);
    block[5] = range.min_vertical_hz;
    block[6] = range.max_vertical_hz;
    block[7] = min_h;
    block[8] = max_h;
    // Max pixel clock, in 10 MHz steps, rounded *up* so the advertised limit never sits below
    // a mode we also advertise.
    block[9] = range.max_pixel_clock_mhz.div_ceil(10).min(0xFF) as u8;
    block[10] = RANGE_LIMITS_ONLY;
    // Padding mandated for the range-limits-only variant.
    block[11] = 0x0A;
    block[12..].fill(0x20);
    block
}

/// Split a range bound into the descriptor's byte plus its "+255" offset flag, saturating at
/// the 510 the encoding can express at all.
fn split_offset(value: u16) -> (u8, bool) {
    match value {
        0..=255 => (value as u8, false),
        _ => ((value - 255).min(255) as u8, true),
    }
}

/// The "unused descriptor" filler (§3.10.3.11).
fn dummy_descriptor() -> [u8; DESCRIPTOR_LEN] {
    let mut block = [0u8; DESCRIPTOR_LEN];
    block[3] = TAG_DUMMY;
    block[5] = 0x0A;
    block[6..].fill(0x20);
    block
}

/// Video input definition (20) and feature-support (24). Declaring continuous frequency is what
/// lets the guest read the range descriptor at all (`drm_edid.c` `drm_get_monitor_range` bails
/// without it), so the range and that bit are set together or not at all.
fn populate_features(edid: &mut [u8], params: &EdidParams) {
    edid[20] = VIDEO_INPUT_DIGITAL_8BPC_DP;
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
    // refresh is understated rather than fabricated. The honest timing is emitted alongside,
    // in a DisplayID type VII block (`overflowing_timings`), whose pixel clock is 24-bit and in
    // kHz. A *CTA-861* extension would not help: its detailed timings are the same 18-byte
    // descriptor with the same 16-bit field. See `docs/design/stable-edid-hotplug.md`.
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
    use crate::virtio::gpu::display::DetailedMode;

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

    /// Parse the DisplayID extension exactly as `drm_displayid.c` does: validate the structure
    /// checksum over `sizeof(header) + bytes + 1` starting at extension byte 1, walk the blocks
    /// from byte 5, and decode each type VII timing with `drm_mode_displayid_detailed`'s
    /// arithmetic. Returns (width, height, refresh Hz, preferred) per timing.
    ///
    /// Deliberately an independent reimplementation rather than a call back into the generator —
    /// a decoder that shares the encoder's mistakes proves nothing.
    fn decode_displayid(edid: &[u8]) -> Vec<(u32, u32, u32, bool)> {
        assert_eq!(edid.len(), 256, "no extension present");
        assert_eq!(edid[126], 1, "the base block must declare the extension");
        let ext = &edid[128..];
        assert_eq!(ext[0], DISPLAYID_EXT_TAG);

        // "EDID extensions block checksum isn't for us": the structure spans [1, 127).
        let bytes = ext[2] as usize;
        let dispid_length = DISPLAYID_HEADER_LEN + bytes + 1;
        assert!(
            dispid_length <= 127 - 1,
            "DisplayID structure overruns the block"
        );
        let csum = ext[1..1 + dispid_length]
            .iter()
            .fold(0u8, |acc, b| acc.wrapping_add(*b));
        assert_eq!(csum, 0, "DisplayID checksum invalid");

        let mut out = Vec::new();
        let mut idx = 1 + DISPLAYID_HEADER_LEN;
        let end = 1 + DISPLAYID_HEADER_LEN + bytes;
        while idx + DISPLAYID_BLOCK_HEADER_LEN <= end {
            let tag = ext[idx];
            let num_bytes = ext[idx + 2] as usize;
            assert!(idx + DISPLAYID_BLOCK_HEADER_LEN + num_bytes <= end);
            if tag == DISPLAYID_BLOCK_TYPE_7 {
                assert_eq!(
                    num_bytes % 20,
                    0,
                    "the kernel drops a block that isn't a multiple of 20"
                );
                for t in 0..num_bytes / 20 {
                    let b = &ext[idx + DISPLAYID_BLOCK_HEADER_LEN + t * 20..];
                    let clock_khz = (b[0] as u32 | (b[1] as u32) << 8 | (b[2] as u32) << 16) + 1;
                    let field = |at: usize| (b[at] as u32 | (b[at + 1] as u32) << 8) + 1;
                    let hactive = field(4);
                    let hblank = field(6);
                    let vactive = field(12);
                    let vblank = field(14);
                    let pixels = (hactive + hblank) * (vactive + vblank);
                    let refresh = (clock_khz * 1000 + pixels / 2) / pixels;
                    out.push((hactive, vactive, refresh, b[3] & 0x80 != 0));
                }
            }
            idx += DISPLAYID_BLOCK_HEADER_LEN + num_bytes;
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
        // The display is digital, not analog — byte 20 was left zero, which every parser reads
        // as an analog input.
        assert_eq!(edid[20] & 0x80, 0x80, "digital input bit");
        assert_eq!((edid[20] >> 4) & 0x07, 0b010, "8 bits per colour");
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

    /// A mode the base block cannot express gets an honest copy in a DisplayID type VII block:
    /// 3024x1964 @ 120 Hz needs an 866 MHz pixel clock, 1.3x the 655.35 MHz the base detailed
    /// timing tops out at. Without this the guest is told 90 Hz and can never select 120.
    #[test]
    fn a_mode_over_the_base_clock_ceiling_is_carried_in_a_displayid_block() {
        let params = EdidParams {
            refresh_rate: 120,
            identity: Some(identity()),
            ..EdidParams::default()
        };
        let edid = build(3024, 1964, &params);

        assert_eq!(edid.len(), 256, "an extension block must be appended");
        assert!(
            checksum_is_valid(&edid[..128]),
            "base checksum covers byte 126"
        );
        assert!(checksum_is_valid(&edid[128..]));

        // The base block still carries the mode at the *size* the guest is being driven to —
        // that is what `virtio_gpu_conn_mode_valid` matches against — with a clamped refresh.
        let base = decode_detailed(descriptor(&edid, 0));
        assert_eq!((base.0, base.1), (3024, 1964));
        assert!(
            base.2 < 120,
            "the base timing is necessarily clamped, got {}",
            base.2
        );

        // ...and the extension carries the truth, marked preferred so it sorts ahead of it.
        let timings = decode_displayid(&edid);
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0], (3024, 1964, 120, true));
    }

    /// A mode that fits the base block must not grow an extension: the historical single-block
    /// output stays byte for byte what it was.
    #[test]
    fn a_representable_mode_emits_no_extension() {
        let edid = build(1920, 1080, &EdidParams::default());
        assert_eq!(edid.len(), 128);
        assert_eq!(edid[126], 0, "extension count stays zero");
        assert!(checksum_is_valid(&edid));
    }

    /// Only the overflowing timings go to the extension. An alt mode that fits stays a base
    /// descriptor; one that doesn't joins the block, non-preferred (there is one preferred mode).
    #[test]
    fn only_overflowing_timings_move_to_the_extension() {
        // Current mode overflows, alt is a comfortable 60 Hz at the same size.
        let params = EdidParams {
            refresh_rate: 120,
            identity: Some(identity()),
            alt_mode: Some(DetailedMode {
                width: 3024,
                height: 1964,
                refresh_hz: 60,
            }),
            ..EdidParams::default()
        };
        let edid = build(3024, 1964, &params);
        let timings = decode_displayid(&edid);
        assert_eq!(
            timings,
            vec![(3024, 1964, 120, true)],
            "only the 120 Hz mode overflows"
        );
        // The 60 Hz alt is still an ordinary base descriptor (slot 2: no range is set here,
        // so the alt timing follows the product name directly).
        assert_eq!(decode_detailed(descriptor(&edid, 2)), (3024, 1964, 60));

        // Now make the alt overflow too: it joins the block and is NOT preferred.
        let params = EdidParams {
            alt_mode: Some(DetailedMode {
                width: 3024,
                height: 1964,
                refresh_hz: 100,
            }),
            ..params
        };
        let timings = decode_displayid(&build(3024, 1964, &params));
        assert_eq!(
            timings,
            vec![(3024, 1964, 120, true), (3024, 1964, 100, false)]
        );
    }

    /// The kernel drops the whole block if its length isn't a multiple of 20, and rejects the
    /// whole structure on a bad checksum — both are asserted inside `decode_displayid`, so this
    /// pins the block header fields it reads to get there.
    #[test]
    fn the_displayid_block_declares_what_the_kernel_expects() {
        let params = EdidParams {
            refresh_rate: 120,
            ..EdidParams::default()
        };
        let edid = build(3024, 1964, &params);
        let ext = &edid[128..];
        assert_eq!(ext[0], 0x70, "DISPLAYID_EXT");
        assert_eq!(ext[1], 0x20, "DisplayID 2.0");
        assert_eq!(ext[2] as usize, DISPLAYID_BLOCK_HEADER_LEN + 20);
        assert_eq!(ext[4], 0, "no further DisplayID sections");
        assert_eq!(ext[5], 0x22, "DATA_BLOCK_2_TYPE_7_DETAILED_TIMING");
        assert_eq!(ext[7], 20, "one 20-byte timing");
        // Everything past the structure and before the EDID extension checksum is padding.
        assert!(ext[1 + DISPLAYID_HEADER_LEN + ext[2] as usize + 1..127]
            .iter()
            .all(|b| *b == 0));
    }

    /// A horizontal rate past 255 kHz — a 4K panel at 120 Hz — must ride the EDID 1.4 "+255"
    /// offset flag rather than clamp, or the guest prunes the very mode we advertise.
    #[test]
    fn a_horizontal_rate_over_the_byte_uses_the_offset_flag() {
        assert_eq!(split_offset(242), (242, false));
        assert_eq!(split_offset(255), (255, false));
        assert_eq!(split_offset(265), (10, true));
        assert_eq!(split_offset(510), (255, true));
        // Past what the encoding can express at all: saturate rather than wrap.
        assert_eq!(split_offset(9000), (255, true));

        let range = RefreshRange {
            min_vertical_hz: 48,
            max_vertical_hz: 120,
            min_horizontal_khz: 106,
            max_horizontal_khz: 265,
            max_pixel_clock_mhz: 1200,
        };
        let params = EdidParams {
            identity: Some(identity()),
            range: Some(range),
            ..EdidParams::default()
        };
        let edid = build(3840, 2160, &params);
        let block = descriptor(&edid, 2);
        assert_eq!(block[3], TAG_RANGE_LIMITS);
        assert_eq!(block[4], 1 << 3, "only the max-horizontal offset is set");
        assert_eq!(block[7], 106, "the min fits the byte untouched");
        assert_eq!(block[8], 10, "265 - 255");
        assert_eq!(
            block[5..7],
            [48, 120],
            "vertical rates never need the offset"
        );
    }
}
