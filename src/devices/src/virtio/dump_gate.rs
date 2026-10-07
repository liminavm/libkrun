//! Holds a device's guest-RAM writers off while the VMM copies guest RAM into a snapshot.
//!
//! Parking the vCPUs stops new requests, not the device threads already serving old ones: a
//! block read landing, a frame arriving from a network proxy or a GPU fence retiring all write
//! guest RAM from threads of their own. One of those writes during the copy tears the snapshot
//! (a used index advanced over a payload half copied, or the reverse).
//!
//! Each such thread does its guest-RAM work inside a [`DumpSection`], one wake's worth at a
//! time. [`DumpGate::close`] waits for the sections already running to end and holds every new
//! one at its start until [`DumpGate::open`]. A thread idle between wakes holds no section, so
//! it costs the close nothing and simply waits at its next wake.

use std::cell::RefCell;
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

#[derive(Default)]
struct State {
    closed: bool,
    /// Sections running, counting each thread once however deeply it nested.
    active: usize,
}

#[derive(Default)]
pub struct DumpGate {
    state: Mutex<State>,
    changed: Condvar,
}

thread_local! {
    /// The gates this thread is inside a section of. A section nested in one of them passes
    /// straight through: the thread is already counted, and holding it at the nested start
    /// would leave the outer section waiting on itself.
    static HELD: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// A stretch of guest-RAM work. Ends when dropped.
#[must_use = "a section ends when it is dropped"]
pub struct DumpSection<'a> {
    /// `None` for a section nested in one this thread already holds.
    gate: Option<&'a DumpGate>,
}

impl DumpGate {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn key(&self) -> usize {
        self as *const Self as usize
    }

    /// Start a stretch of guest-RAM work, waiting first while the gate is closed.
    pub fn enter(&self) -> DumpSection<'_> {
        let key = self.key();
        if HELD.with(|held| held.borrow().contains(&key)) {
            return DumpSection { gate: None };
        }
        let mut state = self.lock();
        while state.closed {
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.active += 1;
        drop(state);
        HELD.with(|held| held.borrow_mut().push(key));
        DumpSection { gate: Some(self) }
    }

    /// Hold new sections back and wait up to `timeout` for the running ones to end. On a
    /// timeout the gate opens again and the number of sections still running comes back, so a
    /// caller that cannot get the device quiet goes on without holding anything.
    pub fn close(&self, timeout: Duration) -> Result<(), usize> {
        let deadline = Instant::now() + timeout;
        let mut state = self.lock();
        state.closed = true;
        while state.active > 0 {
            let now = Instant::now();
            if now >= deadline {
                let active = state.active;
                state.closed = false;
                drop(state);
                self.changed.notify_all();
                return Err(active);
            }
            state = self
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        Ok(())
    }

    /// Let the sections held at their start run. Opening an open gate does nothing.
    pub fn open(&self) {
        self.lock().closed = false;
        self.changed.notify_all();
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }
}

impl Drop for DumpSection<'_> {
    fn drop(&mut self) {
        let Some(gate) = self.gate else {
            return;
        };
        let key = gate.key();
        HELD.with(|held| {
            let mut held = held.borrow_mut();
            if let Some(i) = held.iter().rposition(|k| *k == key) {
                held.swap_remove(i);
            }
        });
        gate.lock().active -= 1;
        gate.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;

    const SETTLE: Duration = Duration::from_millis(100);

    #[test]
    fn closing_a_gate_nobody_is_inside_returns_at_once() {
        let gate = DumpGate::new();
        gate.close(Duration::ZERO).unwrap();
        assert!(gate.is_closed());
        gate.open();
        assert!(!gate.is_closed());
    }

    #[test]
    fn a_section_started_while_closed_waits_for_the_open() {
        let gate = Arc::new(DumpGate::new());
        gate.close(Duration::ZERO).unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let worker = {
            let (gate, ran) = (gate.clone(), ran.clone());
            thread::spawn(move || {
                let _section = gate.enter();
                ran.store(true, Ordering::SeqCst);
            })
        };
        thread::sleep(SETTLE);
        assert!(
            !ran.load(Ordering::SeqCst),
            "the section ran through a closed gate"
        );

        gate.open();
        worker.join().unwrap();
        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn closing_waits_for_the_running_section_to_end() {
        let gate = Arc::new(DumpGate::new());
        let (entered_tx, entered) = mpsc::channel();
        let (finish, finish_rx) = mpsc::channel::<()>();
        let ended = Arc::new(AtomicBool::new(false));
        let worker = {
            let (gate, ended) = (gate.clone(), ended.clone());
            thread::spawn(move || {
                let section = gate.enter();
                entered_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
                ended.store(true, Ordering::SeqCst);
                drop(section);
            })
        };
        entered.recv().unwrap();
        let releaser = thread::spawn(move || {
            thread::sleep(SETTLE);
            finish.send(()).unwrap();
        });

        gate.close(Duration::from_secs(10)).unwrap();
        assert!(
            ended.load(Ordering::SeqCst),
            "the close returned with the section still running"
        );
        worker.join().unwrap();
        releaser.join().unwrap();
        gate.open();
    }

    /// A writer that never finishes must not hold the snapshot forever: the close gives up,
    /// and gives up open, so nothing stays held behind it.
    #[test]
    fn a_close_that_times_out_leaves_the_gate_open() {
        let gate = Arc::new(DumpGate::new());
        let (entered_tx, entered) = mpsc::channel();
        let (finish, finish_rx) = mpsc::channel::<()>();
        let stuck = {
            let gate = gate.clone();
            thread::spawn(move || {
                let _section = gate.enter();
                entered_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
            })
        };
        entered.recv().unwrap();

        assert_eq!(gate.close(SETTLE), Err(1));
        assert!(!gate.is_closed());
        let other = {
            let gate = gate.clone();
            thread::spawn(move || drop(gate.enter()))
        };
        other.join().unwrap();
        finish.send(()).unwrap();
        stuck.join().unwrap();
    }

    /// A section nested in one the thread already holds passes through even once the gate is
    /// closing, or the thread would wait on its own outer section.
    #[test]
    fn a_nested_section_passes_a_closing_gate() {
        let gate = Arc::new(DumpGate::new());
        let (entered_tx, entered) = mpsc::channel();
        let (nest, nest_rx) = mpsc::channel::<()>();
        let worker = {
            let gate = gate.clone();
            thread::spawn(move || {
                let _outer = gate.enter();
                entered_tx.send(()).unwrap();
                nest_rx.recv().unwrap();
                let _inner = gate.enter();
            })
        };
        entered.recv().unwrap();
        let closer = {
            let gate = gate.clone();
            thread::spawn(move || gate.close(Duration::from_secs(10)))
        };
        while !gate.is_closed() {
            thread::yield_now();
        }
        nest.send(()).unwrap();
        worker.join().unwrap();
        assert_eq!(closer.join().unwrap(), Ok(()));
        gate.open();
    }
}
