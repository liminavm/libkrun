#[cfg(unix)]
use std::os::unix::io::AsRawFd;
#[cfg(target_os = "windows")]
use utils::windows::AsRawFd;

use polly::event_manager::{EventManager, Subscriber};
use utils::epoll::{EpollEvent, EventSet};

use super::device::{Mem, REQ_INDEX};
use crate::virtio::device::VirtioDevice;

impl Mem {
    fn handle_req_event(&mut self, event: &EpollEvent) {
        if event.event_set() != EventSet::IN {
            warn!(
                "virtio-mem: unexpected request queue event {:?}",
                event.event_set()
            );
            return;
        }
        if let Err(e) = self.queue_event(REQ_INDEX).read() {
            error!("virtio-mem: failed to read the request queue event: {e:?}");
        } else if self.process_req() {
            self.device_state.signal_used_queue();
        }
    }

    fn handle_requested_event(&mut self) {
        if let Err(e) = self.requested_evt().read() {
            error!("virtio-mem: failed to read the requested-size event: {e:?}");
        }
        // Before activation the guest reads the size when it probes the device.
        if self.is_activated() {
            self.device_state.signal_config_change();
        }
    }

    fn handle_activate_event(&self, event_manager: &mut EventManager) {
        if let Err(e) = self.activate_evt.read() {
            error!("virtio-mem: failed to consume the activate event: {e:?}");
        }
        // The subscriber exists: activate_evt is in `interest_list()`.
        let self_subscriber = event_manager
            .subscriber(self.activate_evt.as_raw_fd())
            .unwrap();
        let req = self.queue_event(REQ_INDEX).as_raw_fd();
        event_manager
            .register(
                req,
                EpollEvent::new(EventSet::IN, req as u64),
                self_subscriber,
            )
            .unwrap_or_else(|e| error!("virtio-mem: failed to register the request queue: {e:?}"));
        event_manager
            .unregister(self.activate_evt.as_raw_fd())
            .unwrap_or_else(|e| error!("virtio-mem: failed to unregister activate_evt: {e:?}"));
    }
}

impl Subscriber for Mem {
    fn process(&mut self, event: &EpollEvent, event_manager: &mut EventManager) {
        let source = event.fd();
        if source == self.activate_evt.as_raw_fd() {
            self.handle_activate_event(event_manager);
        } else if source == self.requested_evt().as_raw_fd() {
            self.handle_requested_event();
        } else if self.is_activated() && source == self.queue_event(REQ_INDEX).as_raw_fd() {
            self.handle_req_event(event);
        } else {
            warn!("virtio-mem: unexpected event from fd {source}");
        }
    }

    fn interest_list(&self) -> Vec<EpollEvent> {
        vec![
            EpollEvent::new(EventSet::IN, self.activate_evt.as_raw_fd() as u64),
            EpollEvent::new(EventSet::IN, self.requested_evt().as_raw_fd() as u64),
        ]
    }
}
