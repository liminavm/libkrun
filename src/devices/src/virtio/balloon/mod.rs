mod device;
mod event_handler;

pub use self::defs::uapi::VIRTIO_ID_BALLOON as TYPE_BALLOON;
pub use self::device::{Balloon, BalloonControlHandle, BalloonStats};

mod defs {
    use super::super::QueueConfig;

    pub const BALLOON_DEV_ID: &str = "virtio_balloon";
    pub const NUM_QUEUES: usize = 5;
    pub const QUEUE_SIZE: u16 = 256;
    pub static QUEUE_CONFIG: [QueueConfig; NUM_QUEUES] = [QueueConfig::new(QUEUE_SIZE); NUM_QUEUES];

    pub mod uapi {
        pub const VIRTIO_F_VERSION_1: u32 = 32;
        pub const VIRTIO_ID_BALLOON: u32 = 5;
        pub const VIRTIO_BALLOON_F_STATS_VQ: u32 = 1;
        // Deflate the balloon when the guest is under memory pressure (OOM safety net).
        pub const VIRTIO_BALLOON_F_DEFLATE_ON_OOM: u32 = 2;
        pub const VIRTIO_BALLOON_F_FREE_PAGE_HINT: u32 = 3;
        pub const VIRTIO_BALLOON_F_REPORTING: u32 = 5;
        // Inflate/deflate queues carry arrays of `__le32` page-frame numbers at this fixed shift,
        // independent of the guest's MMU page size (a 16 KiB guest balloon page is 4 PFNs).
        pub const VIRTIO_BALLOON_PFN_SHIFT: u32 = 12;
    }
}

#[derive(Debug)]
pub enum BalloonError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
}

type Result<T> = std::result::Result<T, BalloonError>;
