// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Portions Copyright 2017 The Chromium OS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the THIRD-PARTY file.

use std::cmp;
use std::convert::From;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
#[cfg(target_os = "linux")]
use std::os::linux::fs::MetadataExt;
#[cfg(target_os = "macos")]
use std::os::macos::fs::MetadataExt;
use std::path::PathBuf;
use std::result;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use imago::{
    DynStorage, FormatAccess, FormatDriverBuilder, PermissiveImplicitOpenGate, Storage,
    StorageOpenOptions, file::File as ImagoFile, qcow2::Qcow2, raw::Raw, vmdk::Vmdk,
};
use log::{error, warn};
use utils::eventfd::{EFD_NONBLOCK, EventFd};
use virtio_bindings::{
    virtio_blk::*, virtio_config::VIRTIO_F_VERSION_1, virtio_ring::VIRTIO_RING_F_EVENT_IDX,
};
use vm_memory::{ByteValued, GuestMemoryMmap};

#[cfg(target_os = "windows")]
use std::mem::MaybeUninit;
#[cfg(target_os = "windows")]
use std::os::windows::io::AsRawHandle;
#[cfg(target_os = "windows")]
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
};

use super::worker::BlockWorker;
use super::{
    super::{ActivateResult, DeviceQueue, DeviceState, QueueConfig, TYPE_BLOCK, VirtioDevice},
    Error, NUM_QUEUES, QUEUE_CONFIG, SECTOR_SHIFT, SECTOR_SIZE,
};

use crate::virtio::{
    ActivateError, DumpGate, InterruptTransport,
    block::{DiskFormat, SyncMode},
};

/// Configuration options for disk caching.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CacheType {
    /// Flushing mechanic will be advertised to the guest driver, but
    /// the operation will be a noop.
    #[default]
    Unsafe,
    /// Flushing mechanic will be advertised to the guest driver and
    /// flush requests coming from the guest will be performed using
    /// `fsync`.
    Writeback,
}

impl CacheType {
    /// Picks the appropriate cache type based on disk image or device path.
    /// Special files like `/dev/rdisk*` on macOS do not support flush/sync.
    pub fn auto(_path: &str) -> CacheType {
        #[cfg(target_os = "macos")]
        if _path.starts_with("/dev/rdisk") {
            return CacheType::Unsafe;
        }
        CacheType::Writeback
    }
}

/// Helper object for setting up all `Block` fields derived from its backing file.
pub(crate) struct DiskProperties {
    cache_type: CacheType,
    pub(crate) file: Arc<Mutex<FormatAccess<Box<dyn DynStorage>>>>,
    nsectors: u64,
    image_id: Vec<u8>,
}

impl DiskProperties {
    pub fn new(
        disk_image: Arc<Mutex<FormatAccess<Box<dyn DynStorage>>>>,
        disk_image_id: Vec<u8>,
        cache_type: CacheType,
    ) -> io::Result<Self> {
        let disk_size = disk_image.lock().unwrap().size();

        // We only support disk size, which uses the first two words of the configuration space.
        // If the image is not a multiple of the sector size, the tail bits are not exposed.
        if !disk_size.is_multiple_of(SECTOR_SIZE) {
            warn!(
                "Disk size {disk_size} is not a multiple of sector size {SECTOR_SIZE}; \
                 the remainder will not be visible to the guest."
            );
        }

        Ok(Self {
            cache_type,
            nsectors: disk_size >> SECTOR_SHIFT,
            image_id: disk_image_id,
            file: disk_image,
        })
    }

    pub fn nsectors(&self) -> u64 {
        self.nsectors
    }

    pub fn image_id(&self) -> &[u8] {
        &self.image_id
    }

    fn build_device_id(disk_file: &File) -> result::Result<String, Error> {
        // This is how kvmtool does it.
        #[cfg(unix)]
        let device_id = {
            let blk_metadata = disk_file.metadata().map_err(Error::GetFileMetadata)?;
            format!(
                "{}{}{}",
                blk_metadata.st_dev(),
                blk_metadata.st_rdev(),
                blk_metadata.st_ino()
            )
        };
        #[cfg(target_os = "windows")]
        let device_id = {
            let mut info = MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::zeroed();
            let ret =
                unsafe { GetFileInformationByHandle(disk_file.as_raw_handle(), info.as_mut_ptr()) };
            if ret != 0 {
                let info = unsafe { info.assume_init() };
                format!(
                    "{}{}{}",
                    info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow
                )
            } else {
                return Err(Error::GetFileMetadata(io::Error::last_os_error()));
            }
        };
        Ok(device_id)
    }

    fn build_disk_image_id(disk_file: &File, block_id: &str) -> Vec<u8> {
        let mut default_id = vec![0; VIRTIO_BLK_ID_BYTES as usize];

        // The public libkrun disk API accepts a caller-provided block_id. Make
        // that the virtio-blk GET_ID value so Linux can expose it as
        // /sys/block/<dev>/serial. Fall back to the historical backing-file
        // derived id only for callers that pass an empty block_id.
        let disk_id = if block_id.is_empty() {
            match Self::build_device_id(disk_file) {
                Err(_) => {
                    warn!("Could not generate device id. We'll use a default.");
                    return default_id;
                }
                Ok(m) => m,
            }
        } else {
            block_id.to_string()
        };

        // The kernel only knows to read a maximum of VIRTIO_BLK_ID_BYTES.
        // This will also zero out any leftover bytes.
        let disk_id = disk_id.as_bytes();
        let bytes_to_copy = cmp::min(disk_id.len(), VIRTIO_BLK_ID_BYTES as usize);
        default_id[..bytes_to_copy].clone_from_slice(&disk_id[..bytes_to_copy]);
        default_id
    }

    pub fn cache_type(&self) -> CacheType {
        self.cache_type
    }
}

impl Drop for DiskProperties {
    fn drop(&mut self) {
        match self.cache_type {
            CacheType::Writeback => {
                // flush() first to force any cached data out.
                if self.file.lock().unwrap().flush().is_err() {
                    error!("Failed to flush block data on drop.");
                }
                // Sync data out to physical media on host.
                if self.file.lock().unwrap().sync().is_err() {
                    error!("Failed to sync block data on drop.")
                }
            }
            CacheType::Unsafe => {
                // This is a noop.
            }
        };
    }
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
struct VirtioBlkGeometry {
    cylinders: u16,
    heads: u8,
    sectors: u8,
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
struct VirtioBlkTopology {
    physical_block_exp: u8,
    alignment_offset: u8,
    min_io_size: u16,
    opt_io_size: u32,
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
struct VirtioBlkConfig {
    capacity: u64,
    size_max: u32,
    seg_max: u32,
    geometry: VirtioBlkGeometry,
    blk_size: u32,
    topology: VirtioBlkTopology,
    writeback: u8,
    unused0: u8,
    num_queues: u16,
    max_discard_sectors: u32,
    max_discard_seg: u32,
    discard_sector_alignment: u32,
    max_write_zeroes_sectors: u32,
    max_write_zeroes_seg: u32,
    write_zeroes_may_unmap: u8,
}

// Safe because it only has data and has no implicit padding.
unsafe impl ByteValued for VirtioBlkConfig {}

/// Virtio device for exposing block level read/write operations on a host file.
pub struct Block {
    // Host file and properties.
    disk: Option<DiskProperties>,
    cache_type: CacheType,
    disk_image: Arc<Mutex<FormatAccess<Box<dyn DynStorage>>>>,
    disk_image_id: Vec<u8>,
    worker_thread: Option<JoinHandle<()>>,
    worker_stopfd: EventFd,
    /// Held closed by a snapshot while it copies guest RAM; every worker passes through it.
    dump_gate: Arc<DumpGate>,

    // Virtio fields.
    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,
    config: VirtioBlkConfig,

    // Transport related fields.
    pub(crate) device_state: DeviceState,

    // Implementation specific fields.
    pub(crate) id: String,
    pub(crate) partuuid: Option<String>,
}

impl Block {
    /// Create a new virtio block device that operates on the given file.
    ///
    /// The given file must be seekable and sizable.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        partuuid: Option<String>,
        cache_type: CacheType,
        disk_image_path: String,
        disk_image_format: DiskFormat,
        is_disk_read_only: bool,
        direct_io: bool,
        sync_mode: SyncMode,
    ) -> io::Result<Block> {
        let disk_image = OpenOptions::new()
            .read(true)
            .write(!is_disk_read_only)
            .open(PathBuf::from(&disk_image_path))?;

        let disk_image_id = DiskProperties::build_disk_image_id(&disk_image, &id);

        let file_opts = StorageOpenOptions::new()
            .write(!is_disk_read_only)
            .filename(disk_image_path)
            .direct(direct_io);

        #[cfg(target_os = "macos")]
        let file_opts = file_opts.relaxed_sync(sync_mode == SyncMode::Relaxed);
        let file = ImagoFile::open(file_opts)?;
        let discard_alignment = file.discard_align();

        let disk_image = match disk_image_format {
            DiskFormat::Qcow2 => {
                let mut qcow2 =
                    Qcow2::<Box<dyn DynStorage>, Arc<imago::FormatAccess<_>>>::open_image(
                        Box::new(file),
                        !is_disk_read_only,
                    )?;
                qcow2.open_implicit_dependencies()?;
                FormatAccess::new(qcow2)
            }
            DiskFormat::Raw => {
                let raw =
                    Raw::<Box<dyn DynStorage>>::open_image(Box::new(file), !is_disk_read_only)?;
                FormatAccess::new(raw)
            }
            DiskFormat::Vmdk => {
                let vmdk = Vmdk::<Box<dyn DynStorage>, Arc<imago::FormatAccess<_>>>::builder(
                    Box::new(file),
                )
                .open(PermissiveImplicitOpenGate::default())?;
                FormatAccess::new(vmdk)
            }
        };

        let disk_image = Arc::new(Mutex::new(disk_image));

        let disk_properties =
            DiskProperties::new(disk_image.clone(), disk_image_id.clone(), cache_type)?;

        let mut avail_features = (1u64 << VIRTIO_F_VERSION_1)
            | (1u64 << VIRTIO_BLK_F_SEG_MAX)
            | (1u64 << VIRTIO_BLK_F_DISCARD)
            | (1u64 << VIRTIO_BLK_F_WRITE_ZEROES)
            | (1u64 << VIRTIO_RING_F_EVENT_IDX);

        if sync_mode != SyncMode::None {
            avail_features |= 1u64 << VIRTIO_BLK_F_FLUSH;
        }

        if is_disk_read_only {
            avail_features |= 1u64 << VIRTIO_BLK_F_RO;
        };

        let config = VirtioBlkConfig {
            capacity: disk_properties.nsectors(),
            size_max: 0,
            // QUEUE_SIZE - 2
            seg_max: 254,
            max_discard_sectors: u32::MAX,
            max_discard_seg: 1,
            discard_sector_alignment: discard_alignment as u32 / 512,
            max_write_zeroes_sectors: u32::MAX,
            max_write_zeroes_seg: 1,
            write_zeroes_may_unmap: 1,
            ..Default::default()
        };

        Ok(Block {
            id,
            partuuid,
            config,
            disk: Some(disk_properties),
            cache_type,
            disk_image,
            disk_image_id,
            avail_features,
            acked_features: 0u64,
            device_state: DeviceState::Inactive,
            worker_thread: None,
            worker_stopfd: EventFd::new(EFD_NONBLOCK)?,
            dump_gate: Arc::default(),
        })
    }

    /// Provides the ID of this block device.
    pub fn id(&self) -> &String {
        &self.id
    }

    /// Provides the PARTUUID of this block device.
    pub fn partuuid(&self) -> Option<&String> {
        self.partuuid.as_ref()
    }

    /// Specifies if this block device is read only.
    pub fn is_read_only(&self) -> bool {
        self.avail_features & (1u64 << VIRTIO_BLK_F_RO) != 0
    }
}

impl VirtioDevice for Block {
    fn device_type(&self) -> u32 {
        TYPE_BLOCK
    }

    fn device_name(&self) -> &str {
        "block"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &QUEUE_CONFIG
    }

    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features;
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_slice = self.config.as_slice();
        let config_len = config_slice.len() as u64;
        if offset >= config_len {
            error!("Failed to read config space");
            return;
        }
        if let Some(end) = offset.checked_add(data.len() as u64) {
            // This write can't fail, offset and end are checked against config_len.
            data.write_all(&config_slice[offset as usize..cmp::min(end, config_len) as usize])
                .unwrap();
        }
    }

    fn write_config(&mut self, _offset: u64, _data: &[u8]) {
        error!("Guest attempted to write config");
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        if self.worker_thread.is_some() {
            panic!("virtio_blk: worker thread already exists");
        }

        let [blk_q]: [_; NUM_QUEUES] = queues.try_into().map_err(|_| {
            error!("Cannot perform activate. Expected {} queue(s)", NUM_QUEUES);
            ActivateError::BadActivate
        })?;

        let disk = match self.disk.take() {
            Some(d) => d,
            None => DiskProperties::new(
                Arc::clone(&self.disk_image),
                self.disk_image_id.clone(),
                self.cache_type,
            )
            .map_err(|_| ActivateError::BadActivate)?,
        };

        let worker = BlockWorker::new(
            blk_q,
            interrupt.clone(),
            mem.clone(),
            disk,
            self.worker_stopfd.try_clone().unwrap(),
            self.dump_gate.clone(),
        );
        self.worker_thread = Some(worker.run());

        self.device_state = DeviceState::Activated(mem, interrupt);
        Ok(())
    }

    fn reset(&mut self) -> bool {
        if let Some(worker) = self.worker_thread.take() {
            let _ = self.worker_stopfd.write(1);
            if let Err(e) = worker.join() {
                error!("error waiting for worker thread: {e:?}");
            }
        }
        self.device_state = DeviceState::Inactive;
        true
    }

    fn dump_gate(&self) -> Option<Arc<DumpGate>> {
        Some(self.dump_gate.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::{Duration, Instant};

    use utils::tempfile::TempFile;
    use vm_memory::{Bytes, GuestAddress};

    use crate::legacy::DummyIrqChip;
    use crate::virtio::queue::tests::VirtQueue as GuestQueue;
    use crate::virtio::queue::{VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE};

    /// While a snapshot holds the device's gate, a read the guest queued is not answered: the
    /// worker writes neither the data nor the used ring until the gate opens.
    #[test]
    fn a_held_worker_leaves_guest_ram_alone_until_released() {
        let backing = TempFile::new().expect("create backing file");
        backing
            .as_file()
            .set_len(16 * SECTOR_SIZE)
            .expect("size backing file");
        let mut block = Block::new(
            "held".to_string(),
            None,
            CacheType::Unsafe,
            backing.as_path().to_str().unwrap().to_string(),
            DiskFormat::Raw,
            false,
            false,
            SyncMode::None,
        )
        .unwrap();

        let mem = GuestMemoryMmap::from_ranges(&[(GuestAddress(0), 0x20000)]).unwrap();
        let vq = GuestQueue::new(GuestAddress(0x1000), &mem, 16);
        // A one-sector read: header, data buffer, status byte.
        mem.write_obj(VIRTIO_BLK_T_IN, GuestAddress(0x8000))
            .unwrap();
        mem.write_obj(0u64, GuestAddress(0x8008)).unwrap();
        mem.write_obj(0xffu8, GuestAddress(0xa000)).unwrap();
        vq.dtable[0].set(0x8000, 16, VIRTQ_DESC_F_NEXT, 1);
        vq.dtable[1].set(
            0x9000,
            SECTOR_SIZE as u32,
            VIRTQ_DESC_F_NEXT | VIRTQ_DESC_F_WRITE,
            2,
        );
        vq.dtable[2].set(0xa000, 1, VIRTQ_DESC_F_WRITE, 0);
        vq.avail.ring[0].set(0);
        vq.avail.idx.set(1);
        let kick = Arc::new(EventFd::new(EFD_NONBLOCK).unwrap());
        let queue = DeviceQueue::new(vq.create_queue(), kick.clone());
        let interrupt =
            InterruptTransport::new(DummyIrqChip::new().into(), "blk-test".to_string()).unwrap();

        let gate = block.dump_gate().unwrap();
        gate.close(Duration::ZERO).unwrap();
        block
            .activate(mem.clone(), interrupt.clone(), vec![queue])
            .unwrap();
        kick.write(1).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            vq.used.idx.get(),
            0,
            "the worker answered through a held gate"
        );
        assert_eq!(mem.read_obj::<u8>(GuestAddress(0xa000)).unwrap(), 0xff);

        gate.open();
        let deadline = Instant::now() + Duration::from_secs(5);
        while vq.used.idx.get() < 1 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            vq.used.idx.get(),
            1,
            "the released worker answered the read"
        );
        assert_eq!(
            mem.read_obj::<u8>(GuestAddress(0xa000)).unwrap(),
            VIRTIO_BLK_S_OK as u8
        );
        assert!(block.reset());
    }

    #[test]
    fn disk_image_id_prefers_supplied_block_id() {
        let backing = TempFile::new().expect("create backing file");
        backing
            .as_file()
            .set_len(SECTOR_SIZE)
            .expect("size backing file");

        let image_id = DiskProperties::build_disk_image_id(backing.as_file(), "workload-rootfs");

        assert_eq!(image_id.len(), VIRTIO_BLK_ID_BYTES as usize);
        assert_eq!(&image_id[..b"workload-rootfs".len()], b"workload-rootfs");
        assert!(
            image_id[b"workload-rootfs".len()..]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn disk_image_id_truncates_supplied_block_id_to_virtio_limit() {
        let backing = TempFile::new().expect("create backing file");
        backing
            .as_file()
            .set_len(SECTOR_SIZE)
            .expect("size backing file");
        let long_id = "volume-name-that-is-longer-than-virtio-limit";

        let image_id = DiskProperties::build_disk_image_id(backing.as_file(), long_id);

        assert_eq!(image_id.len(), VIRTIO_BLK_ID_BYTES as usize);
        assert_eq!(
            &image_id,
            &long_id.as_bytes()[..VIRTIO_BLK_ID_BYTES as usize]
        );
    }

    #[test]
    fn disk_image_id_accepts_empty_block_id() {
        let backing = TempFile::new().expect("create backing file");
        backing
            .as_file()
            .set_len(SECTOR_SIZE)
            .expect("size backing file");

        let image_id = DiskProperties::build_disk_image_id(backing.as_file(), "");

        assert_eq!(image_id.len(), VIRTIO_BLK_ID_BYTES as usize);
    }
}
