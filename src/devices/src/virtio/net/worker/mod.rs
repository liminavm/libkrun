#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub use unix::NetWorker;
#[cfg(unix)]
pub(crate) use unix::connect_backend;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::NetWorker;
