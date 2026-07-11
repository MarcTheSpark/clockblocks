"""
Real-time tests that a tempo change reschedules a parked wait, exercised from each of the three thread
origins the redesign has to handle:

  * the clock's own thread (trivial — the running clock changes its own tempo and its next wait reflects it),
  * a sibling clock's thread (Step 3 — reschedule a foreign clock's pending wakeup),
  * a non-clock thread (Step 8 — an external mutator takes held() + the tree lock),
  * another family's clock thread (a clock, but of a different scheduler, so it too must take held()).

The common shape for the foreign-origin cases: a child parks in a long wait at tempo 60; a mutator on the
other thread speeds the child's tempo up dramatically; the child must wake far sooner than its original
(un-rescheduled) wake time.
"""
import threading
import unittest

from clockblocks.clock import Clock
from clockblocks.utilities import current_clock
from tests import timing


class TempoChangeOriginTestCase(unittest.TestCase):
    def setUp(self):
        self.master = Clock(name="master", initial_tempo=60)

    def tearDown(self):
        self.master.kill()

    # ---- owning thread ----

    def test_tempo_change_from_owning_thread(self):
        """A clock changing its own tempo mid-run: the immediately following wait honors the new tempo."""
        t0 = timing.stopwatch()
        woke = {}

        def child():
            c = current_clock()
            c.tempo = 240                 # 1 beat == 0.25 sec
            c.wait(2.0)                   # 2 beats == 0.5 sec wall
            woke["wall"] = timing.elapsed(t0)

        self.master.fork(child)
        self.master.wait(1.5)
        self.assertIn("wall", woke)
        self.assertAlmostEqual(woke["wall"], 0.5, delta=0.1,
                               msg=f"woke at {woke['wall']:.3f}s, expected ~0.5s under self-set tempo 240")

    # ---- sibling clock thread ----

    def test_tempo_change_from_sibling_clock(self):
        """A sibling clock speeds up a parked child; the child's long wait is rescheduled and wakes early."""
        t0 = timing.stopwatch()
        woke = {}
        child_ready = threading.Event()

        def sleeper():
            child_ready.set()
            current_clock().wait(8.0)     # 8 beats == 8 sec at tempo 60 if never rescheduled
            woke["wall"] = timing.elapsed(t0)

        child = self.master.fork(sleeper)

        def speeder():
            child_ready.wait(timeout=1)
            current_clock().wait(0.2)     # let the sleeper actually park in its wait
            child.tempo = 2400            # 1 beat == 0.025 sec; ~8 remaining beats now ~0.2 sec
            current_clock().wait(2.0)     # stay alive to observe

        self.master.fork(speeder)
        self.master.wait(2.0)

        self.assertIn("wall", woke)
        self.assertLess(woke["wall"], 2.0,
                        f"sleeper woke at {woke['wall']:.3f}s — sibling tempo change didn't reschedule it.")

    # ---- non-clock thread ----

    def test_tempo_change_from_non_clock_thread(self):
        """A bare thread (no clock bound) speeds up a parked child; Step-8 path must reschedule it."""
        t0 = timing.stopwatch()
        woke = {}
        child_ready = threading.Event()

        def sleeper():
            child_ready.set()
            current_clock().wait(8.0)
            woke["wall"] = timing.elapsed(t0)

        child = self.master.fork(sleeper)

        def mutate_from_plain_thread():
            child_ready.wait(timeout=1)
            timing.sleep(0.2)            # let the sleeper park (scheduler-seconds)
            self.assertIsNone(current_clock())   # genuinely off any clock
            child.tempo = 2400
        mutator = threading.Thread(target=mutate_from_plain_thread)
        mutator.start()

        self.master.wait(2.0)
        mutator.join(timeout=1)

        self.assertIn("wall", woke)
        self.assertLess(woke["wall"], 2.0,
                        f"sleeper woke at {woke['wall']:.3f}s — non-clock-thread tempo change didn't reschedule it.")

    # ---- another family's clock thread ----

    def test_tempo_change_from_other_family_takes_held(self):
        """A mutator that is a clock, but of a *different* family, must still take held() on the target's
        scheduler — it can't skip it the way an own-family clock legitimately does. Proven by holding the
        target family in a long action and checking the cross-family mutation does not return until that
        action finishes. Fails if the hold_scheduler() branch mistreats another family as same-family
        (the pre-fix `current_clock() is None` predicate did exactly that, taking only the tree lock)."""
        in_action = threading.Event()      # set while family A is mid-action (its exec lock is held)
        action_done = threading.Event()
        mutation_done = threading.Event()
        observed = {}

        # Family A: `worker`'s turn holds A's execution lock for a real 0.3s; `sleeper` is a parked target.
        def worker():
            in_action.set()
            timing.sleep(0.3)              # A's scheduler is parked in this turn, holding _execution_lock
            in_action.clear()
            action_done.set()
        sleeper = self.master.fork(lambda: current_clock().wait(5.0))
        self.master.fork(worker)

        def other_family():
            other_master = Clock(name="other", initial_tempo=60)
            try:
                def mutator():
                    in_action.wait(timeout=1)                 # mutate WHILE A is mid-action
                    sleeper.tempo = 120                       # cross-family: held() must block until A is idle
                    observed["action_still_running"] = in_action.is_set()
                    mutation_done.set()
                other_master.fork(mutator)
                other_master.wait(1.0)
            finally:
                other_master.kill()

        threading.Thread(target=other_family, daemon=True).start()
        self.master.wait(1.0)
        self.assertTrue(mutation_done.wait(timeout=2), "cross-family mutation never completed")
        self.assertTrue(action_done.is_set())
        self.assertFalse(observed.get("action_still_running", True),
                         "cross-family tempo change returned while the target scheduler was mid-action — "
                         "held() was skipped.")


if __name__ == "__main__":
    unittest.main()
