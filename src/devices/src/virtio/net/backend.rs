use std::io;

#[cfg(unix)]
use std::os::fd::RawFd;
#[cfg(windows)]
use std::os::windows::io::RawSocket;

#[cfg(windows)]
use vm_memory::GuestMemoryMmap;

#[cfg(unix)]
pub type SysError = nix::Error;
#[cfg(windows)]
pub type SysError = io::Error;

#[allow(dead_code)]
#[derive(Debug)]
pub enum ConnectError {
    InvalidAddress(SysError),
    CreateSocket(SysError),
    Binding(SysError),
    #[cfg(windows)]
    Worker(SysError),
    #[cfg(not(target_os = "windows"))]
    SendingMagic(nix::Error),
    // Tap backend errors.
    #[cfg(not(target_os = "windows"))]
    OpenNetTun(nix::Error),
    #[cfg(not(target_os = "windows"))]
    TunSetIff(io::Error),
    #[cfg(not(target_os = "windows"))]
    TunSetVnetHdrSz(io::Error),
    #[cfg(not(target_os = "windows"))]
    TunSetOffload(io::Error),
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum ReadError {
    /// Nothing was written
    NothingRead,
    /// The guest queue ran out of available descriptors
    #[cfg(windows)]
    DescriptorStarvation,
    #[cfg(windows)]
    ProcessNotRunning,
    #[cfg(windows)]
    Queue(crate::virtio::queue::Error),
    /// Another internal error occurred
    Internal(SysError),
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum WriteError {
    /// Nothing was written, you can drop the frame or try to resend it later
    NothingWritten,
    /// Part of the buffer was written, the write has to be finished using try_finish_write
    PartialWrite,
    /// The proxy is gone (EPIPE, or a reset or unconnected datagram socket). The frame is lost;
    /// a backend that can be reconnected says so through [`NetBackend::reconnect`].
    ProcessNotRunning,
    /// Another internal error occurred
    Internal(SysError),
}

/// What came of an attempt to reconnect a backend whose proxy went away.
#[cfg(unix)]
#[derive(Debug)]
pub enum Reconnect {
    /// The backend talks to the proxy again, on the same descriptor or a new one.
    Done,
    /// The proxy is not back yet; try again later.
    NotYet(SysError),
    /// This backend cannot reconnect: it was handed over as a descriptor and has nothing to
    /// connect to.
    Unsupported,
}

#[cfg(unix)]
pub trait NetBackend {
    fn read_frame(&mut self, buf: &mut [u8]) -> Result<usize, ReadError>;
    fn write_frame(&mut self, hdr_len: usize, buf: &mut [u8]) -> Result<(), WriteError>;
    fn has_unfinished_write(&self) -> bool;
    fn try_finish_write(&mut self, hdr_len: usize, buf: &[u8]) -> Result<(), WriteError>;
    fn raw_socket_fd(&self) -> RawFd;

    /// Whether `read_frame` writes a blank virtio-net header in front of the frame. A backend
    /// that relays one (a tap, from the host kernel) returns false, so the header is left alone.
    fn synthesizes_vnet_hdr(&self) -> bool {
        true
    }

    /// Delay in microseconds before retrying after NothingWritten.
    /// Returns 0 if no delay-based retry is needed (e.g. on Linux where
    /// EAGAIN + EPOLLET handles retries via writable events).
    #[allow(dead_code)]
    fn write_retry_delay_us(&self) -> u64 {
        0
    }

    /// Connect to the proxy again after it went away. Called by the worker with a backoff
    /// between attempts, once a write reported [`WriteError::ProcessNotRunning`] or the socket
    /// hung up.
    fn reconnect(&mut self) -> Reconnect {
        Reconnect::Unsupported
    }
}

#[cfg(windows)]
#[derive(Debug, PartialEq, Eq)]
pub enum WriteStatus {
    Complete,
    Pending,
}

#[cfg(windows)]
pub trait NetBackend {
    fn prepare_tx_buffer(&mut self) -> &mut [u8];

    fn start_tx(&mut self, total_bytes: usize) -> Result<WriteStatus, WriteError>;

    fn resume_tx(&mut self) -> Result<WriteStatus, WriteError>;

    fn read_frames_to_guest(
        &mut self,
        mem: &GuestMemoryMmap,
        rx_queue: &mut crate::virtio::queue::Queue,
    ) -> Result<u32, ReadError>;

    fn raw_socket_fd(&self) -> RawSocket;
}
