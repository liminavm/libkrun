use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::{mem, thread};

use vm_memory::GuestMemoryMmap;

use crate::virtio::console::console_control::ConsoleControl;
use crate::virtio::console::port_io::{PortInput, PortOutput};
use crate::virtio::console::process_rx::process_rx;
use crate::virtio::console::process_tx::process_tx;
use crate::virtio::port_io::PortTerminalProperties;
use crate::virtio::{InterruptTransport, Queue};

pub struct PortDescription {
    pub name: Cow<'static, str>,
    pub input: Option<Box<dyn PortInput + Send>>,
    pub output: Option<Box<dyn PortOutput + Send>>,
    pub terminal: Option<Box<dyn PortTerminalProperties>>,
}

impl PortDescription {
    pub fn console(
        input: Option<Box<dyn PortInput + Send>>,
        output: Option<Box<dyn PortOutput + Send>>,
        terminal: Box<dyn PortTerminalProperties>,
    ) -> Self {
        Self {
            name: "".into(),
            input,
            output,
            terminal: Some(terminal),
        }
    }

    pub fn output_pipe(
        name: impl Into<Cow<'static, str>>,
        output: Box<dyn PortOutput + Send>,
    ) -> Self {
        Self {
            name: name.into(),
            input: None,
            output: Some(output),
            terminal: None,
        }
    }

    pub fn input_pipe(
        name: impl Into<Cow<'static, str>>,
        input: Box<dyn PortInput + Send>,
    ) -> Self {
        Self {
            name: name.into(),
            input: Some(input),
            output: None,
            terminal: None,
        }
    }
}

enum PortState {
    Inactive,
    Active {
        stopfd: utils::eventfd::EventFd,
        stop: Arc<AtomicBool>,
        /// The io threads own their queue while they run and hand it back when they join,
        /// so a stopped port can return both queues to the device (see [`Port::shutdown`]).
        rx_thread: Option<JoinHandle<Queue>>,
        tx_thread: Option<JoinHandle<Queue>>,
        /// A queue with no io thread to own it — the port has no input (or no output), but
        /// the device handed us both queues, so park them here rather than dropping them.
        idle_rx: Option<Queue>,
        idle_tx: Option<Queue>,
    },
}

pub(crate) struct Port {
    port_id: u32,
    /// Empty if no name given
    name: Cow<'static, str>,
    state: PortState,
    input: Option<Arc<Mutex<Box<dyn PortInput + Send>>>>,
    output: Option<Arc<Mutex<Box<dyn PortOutput + Send>>>>,
    terminal: Option<Box<dyn PortTerminalProperties>>,
}

impl Port {
    pub(crate) fn new(port_id: u32, description: PortDescription) -> Self {
        Self {
            port_id,
            name: description.name,
            state: PortState::Inactive,
            input: description.input.map(|input| Arc::new(Mutex::new(input))),
            output: description
                .output
                .map(|output| Arc::new(Mutex::new(output))),
            terminal: description.terminal,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn terminal(&self) -> Option<&dyn PortTerminalProperties> {
        self.terminal.as_deref()
    }

    pub fn notify_rx(&self) {
        if let PortState::Active {
            rx_thread: Some(handle),
            ..
        } = &self.state
        {
            handle.thread().unpark()
        }
    }

    pub fn notify_tx(&self) {
        if let PortState::Active {
            tx_thread: Some(handle),
            ..
        } = &self.state
        {
            handle.thread().unpark()
        }
    }

    pub fn start(
        &mut self,
        mem: GuestMemoryMmap,
        rx_queue: Queue,
        tx_queue: Queue,
        interrupt: InterruptTransport,
        control: Arc<ConsoleControl>,
    ) {
        if let PortState::Active { .. } = &mut self.state {
            // The caller is handing us fresh queues, so the ones the old run gives back are
            // redundant — drop them rather than leaking the threads.
            let _ = self.shutdown();
        };

        let input = self.input.as_ref().cloned();
        let output = self.output.as_ref().cloned();

        let stopfd = utils::eventfd::EventFd::new(utils::eventfd::EFD_NONBLOCK)
            .expect("Failed to create EventFd for interrupt_evt");
        let stop = Arc::new(AtomicBool::new(false));

        // Each io thread takes its queue by value and returns it when it stops, so joining
        // gives the queue back. Without an input (or output) there is no thread to hold the
        // queue, so it is parked in the state instead — either way the device gets both
        // queues back on shutdown and can start the port again.
        let (rx_thread, idle_rx) = match input {
            Some(input) => {
                let mem = mem.clone();
                let interrupt = interrupt.clone();
                let port_id = self.port_id;
                let stopfd = stopfd.try_clone().unwrap();
                let stop = stop.clone();
                let handle = thread::Builder::new()
                    .name("console port".into())
                    .spawn(move || {
                        let mut queue = rx_queue;
                        process_rx(
                            mem, &mut queue, interrupt, input, control, port_id, stopfd, stop,
                        );
                        queue
                    })
                    .unwrap();
                (Some(handle), None)
            }
            None => (None, Some(rx_queue)),
        };

        let (tx_thread, idle_tx) = match output {
            Some(output) => {
                let stop = stop.clone();
                let handle = thread::spawn(move || {
                    let mut queue = tx_queue;
                    process_tx(mem, &mut queue, interrupt, output, stop);
                    queue
                });
                (Some(handle), None)
            }
            None => (None, Some(tx_queue)),
        };

        self.state = PortState::Active {
            stopfd,
            stop,
            rx_thread,
            tx_thread,
            idle_rx,
            idle_tx,
        }
    }

    /// Stop the port's io threads and give the device its queues back as `(rx, tx)`.
    ///
    /// A side is `None` only if its io thread panicked (the queue died with it) or the port
    /// was already inactive. The caller must handle that without assuming a queue is there:
    /// the guest drives this path by closing a port, so an `unwrap` here would hand the
    /// guest a way to kill the VM.
    pub fn shutdown(&mut self) -> (Option<Queue>, Option<Queue>) {
        let (mut rx_queue, mut tx_queue) = (None, None);
        if let PortState::Active {
            stopfd,
            stop,
            tx_thread,
            rx_thread,
            idle_rx,
            idle_tx,
        } = &mut self.state
        {
            rx_queue = idle_rx.take();
            tx_queue = idle_tx.take();
            stop.store(true, Ordering::Release);
            if let Some(tx_thread) = mem::take(tx_thread) {
                tx_thread.thread().unpark();
                match tx_thread.join() {
                    Ok(queue) => tx_queue = Some(queue),
                    Err(e) => log::error!(
                        "Failed to flush tx for port {port_id}, thread panicked: {e:?}",
                        port_id = self.port_id
                    ),
                }
            }
            stopfd.write(1).unwrap();
            if let Some(rx_thread) = mem::take(rx_thread) {
                rx_thread.thread().unpark();
                match rx_thread.join() {
                    Ok(queue) => rx_queue = Some(queue),
                    Err(e) => log::error!(
                        "Failed to flush rx for port {port_id}, thread panicked: {e:?}",
                        port_id = self.port_id
                    ),
                }
            }
            self.state = PortState::Inactive;
        };
        (rx_queue, tx_queue)
    }
}
