// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

//! The xHCI ring worker thread.
//!
//! A doorbell / USBCMD.RS / port-reset write on the vcpu thread only mutates
//! registers, queues work and kicks this thread's eventfd; the worker does all
//! the guest-memory work (walking rings, running the command set, posting
//! events). It locks the controller to collect work, then **releases the lock
//! before calling any gadget** so a gadget that blocks on Touch ID never stalls
//! MMIO. It holds only a `Weak` handle to the controller, so VMM teardown (the
//! bus dropping the device) breaks the loop with no reference cycle.

use std::sync::{Mutex, Weak};
use std::thread;

use utils::epoll::{ControlOperation, Epoll, EpollEvent, EventSet};
use utils::eventfd::EventFd;
use vm_memory::GuestMemoryMmap;

use super::device::XhciDevice;
use super::engine::DeferredCall;
use std::os::fd::AsRawFd;

/// Spawn the ring worker for a controller. `kick` is woken by register writes;
/// `stop` tears the thread down (best-effort — the controller `Weak` upgrade
/// failing also ends it).
pub fn spawn(
    dev: Weak<Mutex<XhciDevice>>,
    mem: GuestMemoryMmap,
    kick: EventFd,
    stop: EventFd,
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name("xhci worker".into())
        .spawn(move || work(dev, mem, kick, stop))
        .expect("spawn xhci worker")
}

fn work(dev: Weak<Mutex<XhciDevice>>, mem: GuestMemoryMmap, kick: EventFd, stop: EventFd) {
    let kick_fd = kick.as_raw_fd();
    let stop_fd = stop.as_raw_fd();

    let mut epoll = match Epoll::new() {
        Ok(e) => e,
        Err(e) => {
            error!("xhci worker: epoll create failed: {e}");
            return;
        }
    };
    let _ = epoll.ctl(
        ControlOperation::Add,
        kick_fd,
        &EpollEvent::new(EventSet::IN, kick_fd as u64),
    );
    let _ = epoll.ctl(
        ControlOperation::Add,
        stop_fd,
        &EpollEvent::new(EventSet::IN, stop_fd as u64),
    );

    let mut events = vec![EpollEvent::new(EventSet::empty(), 0); 4];
    loop {
        let n = match epoll.wait(events.len(), -1, events.as_mut_slice()) {
            Ok(n) => n,
            Err(e) => {
                debug!("xhci worker: epoll wait: {e}");
                continue;
            }
        };
        let mut do_work = false;
        for ev in &events[0..n] {
            let src = ev.fd();
            if src == stop_fd {
                let _ = stop.read();
                return;
            }
            if src == kick_fd {
                let _ = kick.read();
                do_work = true;
            }
        }
        if !do_work {
            continue;
        }

        // Upgrade the controller handle; a failed upgrade means the device was
        // dropped (VMM teardown) — end the thread.
        let Some(dev) = dev.upgrade() else {
            return;
        };
        let deferred = {
            let mut d = dev.lock().unwrap();
            d.run_worker_pass(&mem, &dev)
        };
        // The controller lock is now released — safe to call into gadgets, which
        // may block or defer completion to another thread.
        for call in deferred {
            match call {
                DeferredCall::Control(model, xfer) => model.handle_control(xfer),
                DeferredCall::Transfer(model, ep, xfer) => model.handle_transfer(ep, xfer),
                DeferredCall::Reset(model) => model.reset(),
                DeferredCall::EndpointStopped(model, ep) => model.endpoint_stopped(ep),
            }
        }
    }
}
