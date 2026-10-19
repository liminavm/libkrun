//! A backend on Apple's vmnet framework: the guest's frames go to a vmnet interface in shared
//! (NAT), host-only or bridged mode, with the virtio-net header passed through in both directions
//! so segmentation and checksum offload reach vmnet instead of being undone on the way.

use std::ffi::{CString, c_char, c_int, c_void};
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixDatagram;
use std::sync::mpsc;
use std::time::Duration;
use std::{io, ptr};

use block2::{Block, RcBlock};
use virtio_bindings::virtio_net::{
    VIRTIO_NET_F_GUEST_TSO4, VIRTIO_NET_F_GUEST_TSO6, VIRTIO_NET_F_HOST_TSO4,
    VIRTIO_NET_F_HOST_TSO6,
};

use super::backend::{ConnectError, NetBackend, ReadError, WriteError};
use super::{MAX_BUFFER_SIZE, VNET_HDR_LEN};

/// The `vmnet_mode_t` values of `<vmnet/vmnet.h>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmnetMode {
    Host = 1000,
    Shared = 1001,
    Bridged = 1002,
}

impl TryFrom<u32> for VmnetMode {
    type Error = u32;

    fn try_from(mode: u32) -> Result<Self, u32> {
        match mode {
            1000 => Ok(Self::Host),
            1001 => Ok(Self::Shared),
            1002 => Ok(Self::Bridged),
            other => Err(other),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VmnetConfig {
    pub mode: VmnetMode,
    /// The host interface a bridged guest joins; ignored in the other modes.
    pub bridged_interface: Option<String>,
}

type XpcObject = *mut c_void;
type DispatchQueue = *mut c_void;
type InterfaceRef = *mut c_void;
type VmnetReturn = u32;

const VMNET_SUCCESS: VmnetReturn = 1000;
const VMNET_BUFFER_EXHAUSTED: VmnetReturn = 1007;
const VMNET_INTERFACE_PACKETS_AVAILABLE: u32 = 1 << 0;

#[repr(C)]
struct VmPktDesc {
    vm_pkt_size: usize,
    vm_pkt_iov: *mut libc::iovec,
    vm_pkt_iovcnt: u32,
    vm_flags: u32,
}

#[link(name = "vmnet", kind = "framework")]
unsafe extern "C" {
    static vmnet_operation_mode_key: *const c_char;
    static vmnet_shared_interface_name_key: *const c_char;
    static vmnet_allocate_mac_address_key: *const c_char;
    static vmnet_enable_virtio_header_key: *const c_char;
    static vmnet_enable_tso_key: *const c_char;
    static vmnet_mtu_key: *const c_char;
    static vmnet_max_packet_size_key: *const c_char;
    static vmnet_read_max_packets_key: *const c_char;
    static vmnet_write_max_packets_key: *const c_char;

    fn vmnet_start_interface(
        desc: XpcObject,
        queue: DispatchQueue,
        handler: &Block<dyn Fn(VmnetReturn, XpcObject)>,
    ) -> InterfaceRef;
    fn vmnet_interface_set_event_callback(
        interface: InterfaceRef,
        event_mask: u32,
        queue: DispatchQueue,
        callback: Option<&Block<dyn Fn(u32, XpcObject)>>,
    ) -> VmnetReturn;
    fn vmnet_stop_interface(
        interface: InterfaceRef,
        queue: DispatchQueue,
        handler: &Block<dyn Fn(VmnetReturn)>,
    ) -> VmnetReturn;
    fn vmnet_read(
        interface: InterfaceRef,
        packets: *mut VmPktDesc,
        pktcnt: *mut c_int,
    ) -> VmnetReturn;
    fn vmnet_write(
        interface: InterfaceRef,
        packets: *mut VmPktDesc,
        pktcnt: *mut c_int,
    ) -> VmnetReturn;
}

unsafe extern "C" {
    fn xpc_dictionary_create(
        keys: *const *const c_char,
        values: *const XpcObject,
        count: usize,
    ) -> XpcObject;
    fn xpc_dictionary_set_uint64(dict: XpcObject, key: *const c_char, value: u64);
    fn xpc_dictionary_set_bool(dict: XpcObject, key: *const c_char, value: bool);
    fn xpc_dictionary_set_string(dict: XpcObject, key: *const c_char, value: *const c_char);
    fn xpc_dictionary_get_uint64(dict: XpcObject, key: *const c_char) -> u64;
    fn xpc_release(object: XpcObject);
    fn dispatch_queue_create(label: *const c_char, attr: *mut c_void) -> DispatchQueue;
    fn dispatch_release(object: DispatchQueue);
}

/// How long vmnet gets to start or stop an interface; both normally answer within milliseconds.
const VMNET_TIMEOUT: Duration = Duration::from_secs(10);

/// The most frames one `vmnet_read` fetches.
const RX_BATCH: usize = 32;

/// How long to wait before retrying a write vmnet had no buffer for.
const WRITE_RETRY_DELAY_US: u64 = 50;

/// What vmnet reported about an interface it started.
#[derive(Debug)]
struct Started {
    mtu: u64,
    max_packet: usize,
    read_max: usize,
    write_max: usize,
}

pub struct Vmnet {
    interface: InterfaceRef,
    queue: DispatchQueue,
    /// The packets-available callback writes a byte here, which is what the worker's epoll
    /// waits on: vmnet itself has no descriptor to poll.
    wake_rx: UnixDatagram,
    _wake_tx: UnixDatagram,

    rx_bufs: Vec<Vec<u8>>,
    rx_lens: Vec<usize>,
    rx_next: usize,
    rx_count: usize,
    /// A read already failed and said so; the packets it left keep waking the worker.
    rx_failing: bool,
}

// SAFETY: a vmnet interface and a dispatch queue may be used from any thread; the worker owns the
// backend and is its only user.
unsafe impl Send for Vmnet {}

fn vmnet_error(what: &str, status: VmnetReturn) -> ConnectError {
    ConnectError::Vmnet(format!("{what} failed with vmnet status {status}"))
}

impl Vmnet {
    /// Start an interface for a guest whose virtio-net driver acked `vnet_features`, and whose
    /// MAC vmnet takes as given: the guest read it from config space before this runs.
    pub fn start(config: VmnetConfig, vnet_features: u64) -> Result<Self, ConnectError> {
        let acked = |bit: u32| vnet_features & (1 << bit) != 0;
        // vmnet's TSO is one switch for both directions, and the frames it hands up may be IPv6.
        let tso = acked(VIRTIO_NET_F_GUEST_TSO4) && acked(VIRTIO_NET_F_GUEST_TSO6);
        if !tso && (acked(VIRTIO_NET_F_HOST_TSO4) || acked(VIRTIO_NET_F_HOST_TSO6)) {
            warn!(
                "vmnet: the guest sends large segments but cannot take them, so vmnet runs \
                 without TSO and will refuse the guest's large frames"
            );
        }

        let bridged = match (&config.mode, &config.bridged_interface) {
            (VmnetMode::Bridged, Some(name)) => Some(
                CString::new(name.as_str())
                    .map_err(|_| ConnectError::Vmnet(format!("bad interface name {name:?}")))?,
            ),
            (VmnetMode::Bridged, None) => {
                return Err(ConnectError::Vmnet(
                    "bridged mode needs a host interface".into(),
                ));
            }
            _ => None,
        };

        let io_error = |e: io::Error| ConnectError::Vmnet(format!("wake socket: {e}"));
        let (wake_rx, wake_tx) = UnixDatagram::pair().map_err(io_error)?;
        wake_rx.set_nonblocking(true).map_err(io_error)?;
        wake_tx.set_nonblocking(true).map_err(io_error)?;

        // SAFETY: plain C calls on objects created here; the description is released once the
        // start call has copied what it needs.
        let (interface, queue, started) = unsafe {
            let queue = dispatch_queue_create(c"libkrun.vmnet".as_ptr(), ptr::null_mut());
            let desc = xpc_dictionary_create(ptr::null(), ptr::null(), 0);
            xpc_dictionary_set_uint64(desc, vmnet_operation_mode_key, config.mode as u64);
            if let Some(name) = &bridged {
                xpc_dictionary_set_string(desc, vmnet_shared_interface_name_key, name.as_ptr());
            }
            xpc_dictionary_set_bool(desc, vmnet_allocate_mac_address_key, false);
            xpc_dictionary_set_bool(desc, vmnet_enable_virtio_header_key, true);
            xpc_dictionary_set_bool(desc, vmnet_enable_tso_key, tso);

            let (tx, rx) = mpsc::channel();
            let handler = RcBlock::new(move |status: VmnetReturn, params: XpcObject| {
                // The parameters are only valid inside the handler.
                let started = (status == VMNET_SUCCESS && !params.is_null()).then(|| Started {
                    mtu: xpc_dictionary_get_uint64(params, vmnet_mtu_key),
                    max_packet: xpc_dictionary_get_uint64(params, vmnet_max_packet_size_key)
                        as usize,
                    read_max: xpc_dictionary_get_uint64(params, vmnet_read_max_packets_key)
                        as usize,
                    write_max: xpc_dictionary_get_uint64(params, vmnet_write_max_packets_key)
                        as usize,
                });
                let _ = tx.send((status, started));
            });
            let interface = vmnet_start_interface(desc, queue, &handler);
            xpc_release(desc);
            let reply = rx.recv_timeout(VMNET_TIMEOUT);
            match (interface.is_null(), reply) {
                (false, Ok((VMNET_SUCCESS, Some(started)))) => (interface, queue, started),
                (_, Ok((status, _))) => {
                    dispatch_release(queue);
                    return Err(vmnet_error("vmnet_start_interface", status));
                }
                (_, Err(_)) => {
                    // A start that never answered may still answer later, into this queue.
                    return Err(ConnectError::Vmnet(format!(
                        "vmnet_start_interface did not answer within {VMNET_TIMEOUT:?}"
                    )));
                }
            }
        };

        // The maximum vmnet reports leaves out the virtio-net header it puts in front, and
        // `vmnet_read` refuses (VMNET_INVALID_ARGUMENT) a buffer with no room for both.
        let rx_buf_len = started.max_packet + VNET_HDR_LEN;
        if rx_buf_len > MAX_BUFFER_SIZE {
            drop(Self::new(interface, queue, wake_rx, wake_tx, 0));
            return Err(ConnectError::Vmnet(format!(
                "vmnet's packets ({} bytes and the header) exceed the device's buffers \
                 ({MAX_BUFFER_SIZE})",
                started.max_packet
            )));
        }
        info!(
            "vmnet: {:?} interface up{}: mtu {}, packets up to {} bytes, TSO {}, up to {} \
             packets per read and {} per write",
            config.mode,
            bridged
                .as_ref()
                .map(|n| format!(" on {}", n.to_string_lossy()))
                .unwrap_or_default(),
            started.mtu,
            started.max_packet,
            if tso { "on" } else { "off" },
            started.read_max,
            started.write_max,
        );

        let batch = started.read_max.clamp(1, RX_BATCH);
        let mut backend = Self::new(interface, queue, wake_rx, wake_tx, batch);
        for buf in &mut backend.rx_bufs {
            buf.resize(rx_buf_len, 0);
        }
        backend.listen()?;
        Ok(backend)
    }

    fn new(
        interface: InterfaceRef,
        queue: DispatchQueue,
        wake_rx: UnixDatagram,
        wake_tx: UnixDatagram,
        batch: usize,
    ) -> Self {
        Self {
            interface,
            queue,
            wake_rx,
            _wake_tx: wake_tx,
            rx_bufs: vec![Vec::new(); batch],
            rx_lens: vec![0; batch],
            rx_next: 0,
            rx_count: 0,
            rx_failing: false,
        }
    }

    fn listen(&mut self) -> Result<(), ConnectError> {
        let wake = self._wake_tx.as_raw_fd();
        let callback = RcBlock::new(move |_events: u32, _params: XpcObject| {
            // A full socket already holds a wakeup the worker has not consumed.
            // SAFETY: the socket outlives the callback, which `Drop` turns off first.
            unsafe { libc::send(wake, [1u8].as_ptr().cast(), 1, libc::MSG_DONTWAIT) };
        });
        // SAFETY: the interface is live and vmnet copies the block.
        let status = unsafe {
            vmnet_interface_set_event_callback(
                self.interface,
                VMNET_INTERFACE_PACKETS_AVAILABLE,
                self.queue,
                Some(&callback),
            )
        };
        if status != VMNET_SUCCESS {
            return Err(vmnet_error("vmnet_interface_set_event_callback", status));
        }
        Ok(())
    }

    /// Fetch the next batch of frames from vmnet into `rx_bufs`.
    fn refill(&mut self) -> Result<(), ReadError> {
        // Drained before the read, so a packet arriving after it leaves a fresh wakeup behind.
        let mut sink = [0u8; 64];
        while self.wake_rx.recv(&mut sink).is_ok() {}

        let mut iovs: Vec<libc::iovec> = self
            .rx_bufs
            .iter_mut()
            .map(|buf| libc::iovec {
                iov_base: buf.as_mut_ptr().cast(),
                iov_len: buf.len(),
            })
            .collect();
        let mut descs: Vec<VmPktDesc> = iovs
            .iter_mut()
            .map(|iov| VmPktDesc {
                vm_pkt_size: iov.iov_len,
                vm_pkt_iov: iov,
                vm_pkt_iovcnt: 1,
                vm_flags: 0,
            })
            .collect();
        let mut count = descs.len() as c_int;
        // SAFETY: every descriptor points at a live buffer of the size it declares.
        let status = unsafe { vmnet_read(self.interface, descs.as_mut_ptr(), &mut count) };
        if status != VMNET_SUCCESS {
            if !self.rx_failing {
                error!("vmnet: vmnet_read failed with vmnet status {status}");
                self.rx_failing = true;
            }
            return Err(ReadError::NothingRead);
        }
        self.rx_failing = false;
        for (len, desc) in self.rx_lens.iter_mut().zip(&descs[..count as usize]) {
            *len = desc.vm_pkt_size;
        }
        self.rx_next = 0;
        self.rx_count = count as usize;
        Ok(())
    }
}

impl NetBackend for Vmnet {
    fn read_frame(&mut self, buf: &mut [u8]) -> Result<usize, ReadError> {
        loop {
            if self.rx_next == self.rx_count {
                self.refill()?;
                if self.rx_count == 0 {
                    return Err(ReadError::NothingRead);
                }
            }
            let len = self.rx_lens[self.rx_next];
            let frame = &self.rx_bufs[self.rx_next][..len];
            self.rx_next += 1;
            if len < VNET_HDR_LEN || len > buf.len() {
                warn!("vmnet: dropping a {len}-byte packet");
                continue;
            }
            buf[..len].copy_from_slice(frame);
            return Ok(len);
        }
    }

    fn synthesizes_vnet_hdr(&self) -> bool {
        false
    }

    fn write_frame(&mut self, _hdr_len: usize, buf: &mut [u8]) -> Result<(), WriteError> {
        let mut iov = libc::iovec {
            iov_base: buf.as_mut_ptr().cast(),
            iov_len: buf.len(),
        };
        let mut desc = VmPktDesc {
            vm_pkt_size: buf.len(),
            vm_pkt_iov: &mut iov,
            vm_pkt_iovcnt: 1,
            vm_flags: 0,
        };
        let mut count: c_int = 1;
        // SAFETY: the descriptor points at the guest's frame, header first, for its whole length.
        let status = unsafe { vmnet_write(self.interface, &mut desc, &mut count) };
        match (status, count) {
            (VMNET_SUCCESS, 1) => Ok(()),
            (VMNET_SUCCESS | VMNET_BUFFER_EXHAUSTED, _) => Err(WriteError::NothingWritten),
            (status, _) => {
                // The frame is lost, as one the wire dropped; the guest's stack retransmits.
                debug!(
                    "vmnet: vmnet_write of a {}-byte frame failed with vmnet status {status}",
                    buf.len()
                );
                Ok(())
            }
        }
    }

    fn has_unfinished_write(&self) -> bool {
        false
    }

    fn try_finish_write(&mut self, _hdr_len: usize, _buf: &[u8]) -> Result<(), WriteError> {
        Ok(())
    }

    fn raw_socket_fd(&self) -> RawFd {
        self.wake_rx.as_raw_fd()
    }

    fn write_retry_delay_us(&self) -> u64 {
        WRITE_RETRY_DELAY_US
    }
}

impl Drop for Vmnet {
    fn drop(&mut self) {
        let (tx, rx) = mpsc::channel();
        let handler = RcBlock::new(move |status: VmnetReturn| {
            let _ = tx.send(status);
        });
        // SAFETY: the interface is live until this stop completes. Its completion runs on the
        // same serial queue as the packets-available callback, so once it has run no callback is
        // left that could write to the wake socket after it is closed.
        unsafe {
            vmnet_interface_set_event_callback(
                self.interface,
                VMNET_INTERFACE_PACKETS_AVAILABLE,
                ptr::null_mut(),
                None,
            );
            let status = vmnet_stop_interface(self.interface, self.queue, &handler);
            if status != VMNET_SUCCESS {
                error!("vmnet: vmnet_stop_interface failed with vmnet status {status}");
                return;
            }
            match rx.recv_timeout(VMNET_TIMEOUT) {
                Ok(VMNET_SUCCESS) => dispatch_release(self.queue),
                Ok(status) => error!("vmnet: stopping the interface failed with status {status}"),
                Err(_) => error!("vmnet: the interface did not stop within {VMNET_TIMEOUT:?}"),
            }
        }
    }
}
