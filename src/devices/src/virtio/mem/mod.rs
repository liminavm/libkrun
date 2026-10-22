mod device;
mod event_handler;
mod plug;

pub use self::defs::uapi::VIRTIO_ID_MEM as TYPE_MEM;
pub use self::device::{Mem, MemControlHandle, MemState, MemStats};

/// The device's block size, and the size of the owned blocks its region is built from: an
/// unplugged block is returned to the host whole.
pub const HOTPLUG_BLOCK_SIZE: u64 = 2 << 20;

mod defs {
    use super::super::QueueConfig;

    pub const MEM_DEV_ID: &str = "virtio_mem";
    pub const NUM_QUEUES: usize = 1;
    pub const QUEUE_SIZE: u16 = 128;
    pub static QUEUE_CONFIG: [QueueConfig; NUM_QUEUES] = [QueueConfig::new(QUEUE_SIZE); NUM_QUEUES];

    pub mod uapi {
        pub const VIRTIO_F_VERSION_1: u32 = 32;
        pub const VIRTIO_ID_MEM: u32 = 24;
        /// The guest never touches unplugged memory, so the host may leave it unbacked.
        pub const VIRTIO_MEM_F_UNPLUGGED_INACCESSIBLE: u32 = 1;
        /// Plugged memory keeps its contents across a device reset on suspend.
        pub const VIRTIO_MEM_F_PERSISTENT_SUSPEND: u32 = 2;

        pub const VIRTIO_MEM_REQ_PLUG: u16 = 0;
        pub const VIRTIO_MEM_REQ_UNPLUG: u16 = 1;
        pub const VIRTIO_MEM_REQ_UNPLUG_ALL: u16 = 2;
        pub const VIRTIO_MEM_REQ_STATE: u16 = 3;

        pub const VIRTIO_MEM_RESP_ACK: u16 = 0;
        pub const VIRTIO_MEM_RESP_NACK: u16 = 1;
        pub const VIRTIO_MEM_RESP_ERROR: u16 = 3;

        pub const VIRTIO_MEM_STATE_PLUGGED: u16 = 0;
        pub const VIRTIO_MEM_STATE_UNPLUGGED: u16 = 1;
        pub const VIRTIO_MEM_STATE_MIXED: u16 = 2;

        /// `struct virtio_mem_req` and `struct virtio_mem_resp`.
        pub const REQ_SIZE: usize = 24;
        pub const RESP_SIZE: usize = 10;
    }
}

#[derive(Debug)]
pub enum MemError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
    /// The region is not a whole number of blocks, or a block is not a power of two.
    Geometry(String),
}

type Result<T> = std::result::Result<T, MemError>;
