// Copyright 2020 The ChromiumOS Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! renderer_utils: Utility functions and structs used by virgl_renderer and gfxstream.

use crate::rutabaga_os::SafeDescriptor;
use crate::rutabaga_utils::RutabagaDebugHandler;
use crate::rutabaga_utils::RutabagaError;
use crate::rutabaga_utils::RutabagaFenceHandler;
use crate::rutabaga_utils::RutabagaResult;

// The remaining users of this module are all gfxstream's, and gfxstream is behind a feature:
// with it off, nothing here is reachable. virgl_renderer no longer uses any of it — it talks to
// virglrs's Rust API, where a box is a Box3 and a refusal is an error, not an errno.
#[allow(dead_code)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct VirglBox {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
}

#[allow(dead_code)]
pub fn ret_to_res(ret: i32) -> RutabagaResult<()> {
    match ret {
        0 => Ok(()),
        _ => Err(RutabagaError::ComponentError(ret)),
    }
}

#[allow(dead_code)]
pub struct RutabagaCookie {
    pub render_server_fd: Option<SafeDescriptor>,
    pub fence_handler: Option<RutabagaFenceHandler>,
    pub debug_handler: Option<RutabagaDebugHandler>,
}
