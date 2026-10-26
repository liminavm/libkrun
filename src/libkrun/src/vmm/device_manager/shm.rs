use std::collections::BTreeMap;

use arch::ArchMemoryInfo;
use vm_memory::GuestAddress;
use vmm_sys_util::align_upwards;

#[allow(unused)]
#[derive(Debug)]
pub enum Error {
    #[cfg(feature = "gpu")]
    DuplicatedGpuRegion,
    OutOfSpace,
}

#[derive(Clone)]
pub struct ShmRegion {
    pub guest_addr: GuestAddress,
    pub size: usize,
}

pub struct ShmManager {
    #[allow(unused)]
    next_guest_addr: u64,
    #[allow(unused)]
    page_size: usize,
    fs_regions: BTreeMap<usize, ShmRegion>,
    gpu_region: Option<ShmRegion>,
    /// limina: the GPU extension's doorbell and arena windows. Not guest memory: the doorbell
    /// window must trap, and arenas are mapped into theirs one by one.
    #[cfg(feature = "gpu")]
    gpu_ext: Option<(ShmRegion, ShmRegion)>,
    #[cfg(feature = "vhost-user")]
    vhost_user_regions: BTreeMap<usize, ShmRegion>,
}

impl ShmManager {
    pub fn new(info: &ArchMemoryInfo) -> ShmManager {
        Self {
            next_guest_addr: info.shm_start_addr,
            page_size: info.page_size,
            fs_regions: BTreeMap::new(),
            gpu_region: None,
            #[cfg(feature = "gpu")]
            gpu_ext: None,
            #[cfg(feature = "vhost-user")]
            vhost_user_regions: BTreeMap::new(),
        }
    }

    pub fn regions(&self) -> Vec<(GuestAddress, usize)> {
        let mut regions: Vec<(GuestAddress, usize)> = Vec::new();

        for region in self.fs_regions.iter() {
            regions.push((region.1.guest_addr, region.1.size));
        }

        if let Some(region) = &self.gpu_region {
            regions.push((region.guest_addr, region.size));
        }

        #[cfg(feature = "vhost-user")]
        for region in self.vhost_user_regions.values() {
            regions.push((region.guest_addr, region.size));
        }

        regions
    }

    #[cfg(not(any(feature = "tee", feature = "aws-nitro")))]
    pub fn fs_region(&self, index: usize) -> Option<&ShmRegion> {
        self.fs_regions.get(&index)
    }

    #[cfg(feature = "gpu")]
    pub fn gpu_region(&self) -> Option<&ShmRegion> {
        self.gpu_region.as_ref()
    }

    #[allow(unused)]
    fn create_region(&mut self, size: usize) -> Result<ShmRegion, Error> {
        let size = align_upwards!(size, self.page_size);

        let region = ShmRegion {
            guest_addr: GuestAddress(self.next_guest_addr),
            size,
        };

        if let Some(addr) = self.next_guest_addr.checked_add(size as u64) {
            self.next_guest_addr = addr;
            Ok(region)
        } else {
            Err(Error::OutOfSpace)
        }
    }

    #[cfg(feature = "gpu")]
    pub fn create_gpu_region(&mut self, size: usize) -> Result<(), Error> {
        if self.gpu_region.is_some() {
            Err(Error::DuplicatedGpuRegion)
        } else {
            self.gpu_region = Some(self.create_region(size)?);
            Ok(())
        }
    }

    /// limina: the GPU extension's windows, placed after everything allocated so far (so a VM
    /// without the extension keeps the layout it always had).
    #[cfg(feature = "gpu")]
    pub fn create_gpu_ext_regions(&mut self, doorbells: usize, arenas: usize) -> Result<(), Error> {
        if self.gpu_ext.is_some() {
            return Err(Error::DuplicatedGpuRegion);
        }
        let d = self.create_region(doorbells)?;
        let a = self.create_region(arenas)?;
        self.gpu_ext = Some((d, a));
        Ok(())
    }

    /// The GPU extension's (doorbell, arena) windows, if allocated.
    #[cfg(feature = "gpu")]
    pub fn gpu_ext_regions(&self) -> Option<&(ShmRegion, ShmRegion)> {
        self.gpu_ext.as_ref()
    }

    #[cfg(not(feature = "tee"))]
    pub fn create_fs_region(&mut self, index: usize, size: usize) -> Result<(), Error> {
        let region = self.create_region(size)?;
        self.fs_regions.insert(index, region);
        Ok(())
    }

    #[cfg(feature = "vhost-user")]
    pub fn create_vhost_user_region(&mut self, index: usize, size: usize) -> Result<(), Error> {
        let region = self.create_region(size)?;
        self.vhost_user_regions.insert(index, region);
        Ok(())
    }

    #[cfg(feature = "vhost-user")]
    pub fn vhost_user_region(&self, index: usize) -> Option<&ShmRegion> {
        self.vhost_user_regions.get(&index)
    }
}
