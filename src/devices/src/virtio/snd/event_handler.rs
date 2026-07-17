// Copyright 2026 The limina Authors.
// SPDX-License-Identifier: Apache-2.0

use std::os::unix::io::AsRawFd;

use polly::event_manager::{EventManager, Subscriber};
use utils::epoll::{EpollEvent, EventSet};

use super::defs::{CONTROL_INDEX, TX_INDEX};
use super::device::Snd;
use crate::virtio::device::VirtioDevice;

impl Snd {
    fn handle_control_event(&mut self, event: &EpollEvent) {
        if event.event_set() != EventSet::IN {
            warn!(
                "snd: control queue unexpected event {:?}",
                event.event_set()
            );
            return;
        }
        if let Err(e) = self.queue_event(CONTROL_INDEX).read() {
            error!("snd: failed to read control queue event: {e:?}");
        } else if self.process_control() {
            self.device_state.signal_used_queue();
        }
    }

    fn handle_tx_event(&mut self, event: &EpollEvent) {
        if event.event_set() != EventSet::IN {
            warn!("snd: tx queue unexpected event {:?}", event.event_set());
            return;
        }
        if let Err(e) = self.queue_event(TX_INDEX).read() {
            error!("snd: failed to read tx queue event: {e:?}");
        } else if self.process_tx() {
            self.device_state.signal_used_queue();
        }
    }

    fn handle_activate_event(&self, event_manager: &mut EventManager) {
        debug!("snd: activate event");
        if let Err(e) = self.activate_evt.read() {
            error!("snd: failed to consume activate event: {e:?}");
        }

        // The subscriber must exist as we previously registered activate_evt via
        // `interest_list()`.
        let self_subscriber = event_manager
            .subscriber(self.activate_evt.as_raw_fd())
            .unwrap();

        for idx in [CONTROL_INDEX, TX_INDEX] {
            let fd = self.queue_event(idx).as_raw_fd();
            event_manager
                .register(
                    fd,
                    EpollEvent::new(EventSet::IN, fd as u64),
                    self_subscriber.clone(),
                )
                .unwrap_or_else(|e| {
                    error!("snd: failed to register queue {idx} with event manager: {e:?}");
                });
        }

        event_manager
            .unregister(self.activate_evt.as_raw_fd())
            .unwrap_or_else(|e| {
                error!("snd: failed to unregister activate evt: {e:?}");
            });
    }
}

impl Subscriber for Snd {
    fn process(&mut self, event: &EpollEvent, event_manager: &mut EventManager) {
        let source = event.fd();
        let activate_evt = self.activate_evt.as_raw_fd();
        if source == activate_evt {
            self.handle_activate_event(event_manager);
        } else if self.is_activated() {
            if source == self.queue_event(CONTROL_INDEX).as_raw_fd() {
                self.handle_control_event(event);
            } else if source == self.queue_event(TX_INDEX).as_raw_fd() {
                self.handle_tx_event(event);
            } else {
                warn!("snd: unexpected event on fd {source:?}");
            }
        } else {
            warn!("snd: not yet activated; spurious event on fd {source:?}");
        }
    }

    fn interest_list(&self) -> Vec<EpollEvent> {
        vec![EpollEvent::new(
            EventSet::IN,
            self.activate_evt.as_raw_fd() as u64,
        )]
    }
}
