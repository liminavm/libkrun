// limina: a change counter the VMM can block on for guest power-state transitions.

// loom's in this crate's own tests under `--cfg loom`, so `loom_model` can race a transition
// against a waiter.
#[cfg(all(test, loom))]
use loom::sync::{Condvar, Mutex};
#[cfg(not(all(test, loom)))]
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

/// Counts guest power-state transitions — a virtio device changing status (a driver resetting it
/// on the way into suspend, or bringing it back to `DRIVER_OK`), and a vCPU parking in or
/// resuming from PSCI `SYSTEM_SUSPEND` — so the VMM can wait for the next one instead of polling.
///
/// It carries no state of its own: after every wakeup a waiter re-reads whatever it cares about
/// (the VMM's device statuses, whether the guest is system-suspended). Mechanism only; what a
/// transition means is the VMM's policy.
#[derive(Default)]
pub struct GuestPowerWatch {
    generation: Mutex<u64>,
    changed: Condvar,
}

impl GuestPowerWatch {
    /// Record a transition and wake every waiter. Returns the new generation, which orders this
    /// transition against every other one the counter has seen.
    pub fn notify(&self) -> u64 {
        let mut generation = self.generation.lock().unwrap();
        *generation = generation.wrapping_add(1);
        self.changed.notify_all();
        *generation
    }

    /// The current generation. Read it BEFORE inspecting the state it guards, then pass it to
    /// [`Self::wait_past`]: a transition that lands in between is then never missed.
    pub fn generation(&self) -> u64 {
        *self.generation.lock().unwrap()
    }

    /// Block until the generation moves past `seen`, or `timeout` elapses. Returns the current
    /// generation either way.
    pub fn wait_past(&self, seen: u64, timeout: Duration) -> u64 {
        let deadline = Instant::now() + timeout;
        let mut generation = self.generation.lock().unwrap();
        while *generation == seen {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            generation = self
                .changed
                .wait_timeout(generation, deadline - now)
                .unwrap()
                .0;
        }
        *generation
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;

    #[test]
    fn a_transition_already_recorded_is_not_waited_for() {
        let w = GuestPowerWatch::default();
        let seen = w.generation();
        w.notify();
        let start = Instant::now();
        assert_eq!(w.wait_past(seen, Duration::from_secs(10)), seen + 1);
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_quiet_guest_times_out() {
        let w = GuestPowerWatch::default();
        let seen = w.generation();
        assert_eq!(w.wait_past(seen, Duration::from_millis(20)), seen);
    }

    #[test]
    fn a_transition_from_another_thread_wakes_the_waiter() {
        let w = Arc::new(GuestPowerWatch::default());
        let seen = w.generation();
        let notifier = w.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            notifier.notify();
        });
        assert_eq!(w.wait_past(seen, Duration::from_secs(10)), seen + 1);
        t.join().unwrap();
    }
}

/// loom model of the protocol [`GuestPowerWatch::generation`] documents: a waiter reads the
/// generation, then the state it guards, and waits only if the state is not yet what it wants —
/// racing the transition that changes the state and records it.
///
/// Run with `cargo xtask check loom third_party/libkrun/src/devices`. loom's timed wait never times
/// out, so a transition the waiter slept through is not a late wakeup here but a deadlock, which
/// loom reports.
#[cfg(all(test, loom))]
mod loom_model {
    use super::GuestPowerWatch;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, Ordering};
    use loom::thread;
    use std::time::Duration;

    #[test]
    fn a_transition_is_never_slept_through() {
        loom::model(|| {
            let watch = Arc::new(GuestPowerWatch::default());
            let suspended = Arc::new(AtomicBool::new(false));
            let guest = {
                let (watch, suspended) = (watch.clone(), suspended.clone());
                thread::spawn(move || {
                    suspended.store(true, Ordering::Relaxed);
                    watch.notify();
                })
            };
            loop {
                let seen = watch.generation();
                if suspended.load(Ordering::Relaxed) {
                    break;
                }
                watch.wait_past(seen, Duration::from_secs(3600));
            }
            guest.join().unwrap();
        });
    }
}
