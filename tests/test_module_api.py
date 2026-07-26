import threading
import unittest
import warnings

from tests import timing
from clockblocks.clock import Clock, ClockState
from clockblocks.exceptions import ClockblocksError, NoActiveClockError, NotMasterClockError
from clockblocks.moment import Moment
from clockblocks.metric_phase import MetricPhaseTarget
from clockblocks.tempo_envelope import TempoEnvelope
from clockblocks import utilities
from clockblocks.utilities import current_clock


class ModuleApiTestCase(unittest.TestCase):
    """
    Tests for the Step-5 surface: Clock.wait_forever / wait_for_children_to_finish / run_as_server,
    and the module-level wrappers in clockblocks.utilities that delegate to current_clock().

    Same isolation pattern as test_fork.py / test_kill.py: fresh master + fresh scheduler per test.
    The master's __init__ binds it as current_clock() on this (the test) thread.
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- wait_for_children_to_finish ----

    def test_wait_for_children_to_finish(self):
        done = []

        def child():
            current_clock().wait(0.5)
            done.append("done")

        self.master.fork(child)
        self.master.wait_for_children_to_finish()
        self.assertEqual(done, ["done"])
        self.assertEqual(self.master.children(), ())

    def test_module_wait_for_children_to_finish(self):
        done = []

        def child():
            current_clock().wait(0.5)
            done.append("done")

        # module-level fork + wait, both resolving to self.master via current_clock()
        utilities.fork(child)
        utilities.wait_for_children_to_finish()
        self.assertEqual(done, ["done"])
        self.assertEqual(self.master.children(), ())

    def test_wait_for_children_returns_promptly_after_last_child(self):
        # The child finishes ~0.3s in; wait_for_children_to_finish must return right then, not poll to
        # the next whole beat (~1.0s under the old implementation).
        def child():
            current_clock().wait(0.3)

        self.master.fork(child)
        t0 = timing.stopwatch()
        self.master.wait_for_children_to_finish()
        elapsed = timing.elapsed(t0)
        self.assertLess(elapsed, 0.7,
                        f"wait_for_children_to_finish over-waited ({elapsed:.3f}s) past the last child's end")
        self.assertGreater(elapsed, 0.2, "returned before the child could possibly have finished")
        self.assertEqual(self.master.children(), ())

    def test_kill_of_last_child_releases_wait_for_children(self):
        # Killing the last child from another thread must release a parent blocked in
        # wait_for_children_to_finish() (kill()'s detach runs through the same _detach_child path).
        child = self.master.fork(lambda: current_clock().wait(100))   # would otherwise block ~forever

        def killer():
            timing.sleep(0.2)
            child.kill()

        threading.Thread(target=killer).start()
        t0 = timing.stopwatch()
        self.master.wait_for_children_to_finish()
        elapsed = timing.elapsed(t0)
        self.assertLess(elapsed, 1.5,
                        f"kill of last child didn't release wait_for_children_to_finish ({elapsed:.3f}s)")
        self.assertEqual(self.master.children(), ())

    # ---- wait_forever ----

    def test_wait_forever_raises_when_clock_killed(self):
        # wait_forever() unblocks by *raising* ClockKilledError, not returning. For a forked clock the
        # fork wrapper catches it, so the line after wait_forever() is never reached and the child ends DEAD.
        reached_after = threading.Event()

        def proc():
            current_clock().wait_forever()   # blocks until the clock is killed, then raises
            reached_after.set()              # must NOT be reached

        child = self.master.fork(proc)
        self.master.wait(0.05)               # let the child enter wait_forever
        child.kill()
        self.master.wait(0.1)                # give the child thread time to unwind
        self.assertFalse(reached_after.is_set(),
                         "code after wait_forever() ran; it should have raised ClockKilledError instead")
        self.assertIs(child._state, ClockState.DEAD)

    # ---- module-level fork ----

    def test_module_fork_uses_current_clock(self):
        seen = []

        def proc():
            seen.append(current_clock())

        child = utilities.fork(proc)
        self.assertIs(child.parent, self.master)
        self.master.wait(0.05)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], child)

    def test_module_fork_without_clock_raises(self):
        captured = {}

        def runner():
            try:
                utilities.fork(lambda: None)
            except ClockblocksError as e:
                captured["err"] = e

        t = threading.Thread(target=runner)   # fresh thread, no clock bound
        t.start()
        t.join(timeout=2)
        self.assertIn("err", captured)

    # ---- run_as_server ----

    def test_run_as_server_releases_calling_thread_and_keeps_running(self):
        self.assertIs(current_clock(), self.master)
        self.master.run_as_server()
        # the calling (test) thread no longer owns the clock
        self.assertIsNone(current_clock())

        ran = threading.Event()
        # fork directly on the server clock (module-level fork can't be used: this thread has no clock)
        self.master.fork(lambda: ran.set())
        self.assertTrue(ran.wait(timeout=3),
                        "forked work did not run on the backgrounded server clock")

        # clean up the background wait_forever thread so it doesn't linger
        self.master.kill()

    def test_run_as_server_on_child_raises(self):
        child = self.master.fork(lambda: None)
        with self.assertRaises(NotMasterClockError):
            child.run_as_server()

    # ---- strictness: no-clock threads raise instead of silently sleeping ----

    @staticmethod
    def _capture_on_fresh_thread(call):
        """Run `call` on a fresh thread (no clock bound) and return any exception it raised."""
        captured = {}

        def runner():
            try:
                call()
            except BaseException as e:  # noqa: BLE001 - we want whatever it raised
                captured["err"] = e

        t = threading.Thread(target=runner)
        t.start()
        t.join(timeout=2)
        return captured.get("err")

    def test_module_wait_without_clock_raises(self):
        err = self._capture_on_fresh_thread(lambda: utilities.wait(0.01))
        self.assertIsInstance(err, NoActiveClockError)

    def test_module_wait_forever_without_clock_raises(self):
        err = self._capture_on_fresh_thread(utilities.wait_forever)
        self.assertIsInstance(err, NoActiveClockError)

    def test_module_wait_for_children_without_clock_raises(self):
        err = self._capture_on_fresh_thread(utilities.wait_for_children_to_finish)
        self.assertIsInstance(err, NoActiveClockError)

    # ---- tempo helpers (Step 13): act on the CURRENT clock, raise off-thread ----

    def test_set_get_tempo_rate_beat_length_on_current_clock(self):
        # On the test thread, current_clock() is the master (bound in its __init__).
        utilities.set_tempo(90)
        self.assertEqual(self.master.tempo, 90)
        self.assertEqual(utilities.get_tempo(), 90)
        utilities.set_rate(3)
        self.assertEqual(self.master.rate, 3)
        self.assertEqual(utilities.get_rate(), 3)
        utilities.set_beat_length(0.5)
        self.assertAlmostEqual(self.master.beat_length, 0.5)
        self.assertAlmostEqual(utilities.get_beat_length(), 0.5)

    # ---- position readers: act on the CURRENT clock, raise off-thread ----

    def test_get_beat_and_get_time_track_current_clock(self):
        self.assertEqual(utilities.get_beat(), 0)
        self.assertEqual(utilities.get_time(), 0)
        self.master.wait(3)
        self.assertAlmostEqual(utilities.get_beat(), 3)
        self.assertAlmostEqual(utilities.get_time(), 3)

    def test_get_beat_and_get_time_return_plain_floats(self):
        # Not the _CallableFloat shim that Clock.beat/Clock.time still return for backwards compatibility:
        # a new function has no old method spelling to support.
        self.assertIs(type(utilities.get_beat()), float)
        self.assertIs(type(utilities.get_time()), float)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            utilities.get_beat()
            utilities.get_time()

    def test_position_readers_act_on_fork_not_master(self):
        # Inside a fork at double tempo, get_beat() reports the fork's own beat, while get_time() reports
        # its position in the parent's beats — neither is the master's reading.
        seen = {}

        def child():
            utilities.wait(4)
            seen["child_beat"] = utilities.get_beat()
            seen["child_time"] = utilities.get_time()
            seen["master_beat"] = self.master.beat

        self.master.fork(child, initial_rate=2)
        self.master.wait_for_children_to_finish()
        self.assertAlmostEqual(seen["child_beat"], 4)
        self.assertAlmostEqual(seen["child_time"], 2)
        self.assertAlmostEqual(seen["master_beat"], 2)

    def test_tempo_helpers_act_on_fork_not_master(self):
        # set_tempo() inside a fork changes *that fork's* tempo, leaving the master untouched.
        self.master.tempo = 60
        seen = {}

        def child():
            utilities.set_tempo(180)
            utilities.set_tempo_target(200, Moment.after_beats(1))  # exercise a curve forwarder too
            seen["child_tempo"] = utilities.get_tempo()
            seen["master_tempo"] = self.master.tempo

        self.master.fork(child)
        self.master.wait_for_children_to_finish()
        self.assertEqual(seen["child_tempo"], 180)
        self.assertEqual(seen["master_tempo"], 60)

    def test_tempo_target_when_is_moment_not_bare_number(self):
        # `when` is a ResolvableMoment; a bare number is rejected (no back-compat duration path).
        self.master.tempo = 60
        # reaching tempo 120 after 2 beats: the curve is now projected to land at beat 2.
        self.master.set_tempo_target(120, Moment.after_beats(2))
        self.assertAlmostEqual(self.master.tempo_history.length(), 2, delta=1e-9)
        self.assertAlmostEqual(self.master.tempo_history.tempo_at(2), 120, delta=1e-9)
        with self.assertRaises(TypeError):
            self.master.set_tempo_target(120, 2)   # bare number no longer accepted

    def test_set_tempo_targets_mixes_beats_and_time_axes(self):
        # One call mixing beats- and time-axis whens: the curve is built left-to-right, with the
        # time-axis middle segment converted to a beat endpoint using the curve built so far.
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [90, 120, 80],
            [Moment.after_beats(2), Moment.after_time(5), Moment.after_beats(10)],
        )
        th = self.master.tempo_history
        self.assertAlmostEqual(th.length(), 10, delta=1e-9)          # last (beats) endpoint
        self.assertAlmostEqual(th.tempo_at(0), 60, delta=1e-9)
        self.assertAlmostEqual(th.tempo_at(2), 90, delta=1e-9)
        self.assertAlmostEqual(th.tempo_at(10), 80, delta=1e-9)
        # the time-axis (middle) target really landed where exactly 5 seconds elapse from now:
        # its endpoint is the boundary between segment 2 and 3, reaching tempo 120.
        mid_beat = th.segments[1].end_time
        self.assertAlmostEqual(th.tempo_at(mid_beat), 120, delta=1e-9)
        self.assertAlmostEqual(th.integrate_interval(0, mid_beat), 5, delta=1e-9)

    def test_set_tempo_targets_backwards_when_raises_with_index(self):
        self.master.tempo = 60
        with self.assertRaises(ValueError) as cm:
            self.master.set_tempo_targets([90, 120], [Moment.after_beats(5), Moment.after_beats(3)])
        self.assertIn("#1", str(cm.exception))   # the offending index is named

    def test_set_tempo_targets_with_metric_phase_and_min_duration(self):
        # A MetricPhaseTarget resolves against *now*; min_duration keeps it past the prior segment.
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [100, 140],
            [Moment.after_beats(3), MetricPhaseTarget(0, divisor=4, min_duration=3)],
        )
        self.assertAlmostEqual(self.master.tempo_history.length(), 4, delta=1e-9)

    def test_set_tempo_targets_per_element_none_curve_shape_is_linear(self):
        # A per-element None in curve_shapes means linear (0), like the singular setters' curve_shape=None
        # default. Must work on the time axis too (an un-normalized None used to raise abs(None) there).
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [90, 120], [Moment.after_time(2), Moment.after_time(4)], curve_shapes=[None, None])
        self.assertEqual([s.curve_shape for s in self.master.tempo_history.segments], [0, 0])

    def test_set_tempo_target_align_to_phase_lands_on_downbeat(self):
        # "accelerate to 130 over 20 seconds, curvature solved so it lands on a downbeat (divisor 4)"
        self.master.tempo = 60
        self.master.set_tempo_target(130, Moment.after_time(20), align_to=MetricPhaseTarget(0, divisor=4))
        th = self.master.tempo_history
        end_beat = th.length()
        self.assertAlmostEqual(end_beat % 4, 0, delta=1e-9, msg="did not land on a downbeat")
        self.assertAlmostEqual(th.integrate_interval(0, end_beat), 20, delta=1e-6)   # time axis still pinned
        self.assertAlmostEqual(th.tempo_at(end_beat), 130, delta=1e-9)

    def test_set_tempo_target_align_to_fixed_coordinate_and_warns_on_curve_shape(self):
        self.master.tempo = 60
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # when pins time (20s); align_to pins the free (beat) axis to exactly 26; curvature solved
            self.master.set_tempo_target(130, Moment.after_time(20), curve_shape=2, align_to=Moment.at_beat(26))
            self.assertTrue(any("curve_shape" in str(w.message) for w in caught),
                            "expected a warning that curve_shape is discarded for a fixed align_to")
        th = self.master.tempo_history
        self.assertAlmostEqual(th.length(), 26, delta=1e-9)
        self.assertAlmostEqual(th.integrate_interval(0, 26), 20, delta=1e-6)

    def test_set_tempo_target_align_to_same_axis_raises(self):
        self.master.tempo = 60
        with self.assertRaises(ValueError):
            # `when` pins time; align_to on the time axis too -> not the free axis
            self.master.set_tempo_target(130, Moment.after_time(20), align_to=Moment.at_time(25))

    def test_set_tempo_target_align_to_unreachable_raises_and_rolls_back(self):
        self.master.tempo = 60
        with self.assertRaises(ValueError):
            self.master.set_tempo_target(130, Moment.after_time(20), align_to=Moment.at_beat(1000))
        self.assertEqual(self.master.tempo_history.length(), 0)   # curve untouched

    def test_metric_phase_target_align_infers_free_axis_ignoring_its_units(self):
        # A MetricPhaseTarget with units=None infers the free axis; an explicit conflicting axis raises.
        self.master.tempo = 60
        self.master.set_tempo_target(130, Moment.after_time(20),
                                     align_to=MetricPhaseTarget(0, divisor=4, units="beats"))  # beats == free
        self.assertAlmostEqual(self.master.tempo_history.length() % 4, 0, delta=1e-9)
        m2 = Clock(name="m2"); m2.tempo = 60
        try:
            with self.assertRaises(ValueError):
                m2.set_tempo_target(130, Moment.after_time(20),
                                    align_to=MetricPhaseTarget(0, divisor=4, units="time"))  # time == pinned
        finally:
            m2.kill()

    def test_set_tempo_targets_group_align_to_phase_time_free(self):
        # Whole-run align: a beats-pinned 3-segment run is bent collectively so the run's *time* (the free
        # axis) lands on a multiple of 3, while every segment's end *beat* is left untouched.
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [90, 120, 80],
            [Moment.after_beats(2), Moment.after_beats(5), Moment.after_beats(9)],
            align_to=MetricPhaseTarget(0, divisor=3),   # free axis (time); units inferred
        )
        th = self.master.tempo_history
        self.assertAlmostEqual(th.length(), 9, delta=1e-9)                       # end beat unchanged
        self.assertEqual([s.end_time for s in th.segments], [2, 5, 9])           # per-segment beats unchanged
        self.assertAlmostEqual(th.integrate_interval(0, 9) % 3, 0, delta=1e-6)   # run lands on a time phase

    def test_set_tempo_targets_group_align_to_phase_beats_free(self):
        # A time-pinned 2-segment run aligned to a beat phase (free axis = beats): the run's end *beat* lands
        # on a multiple of 2, while the run's end *time* (pinned) is preserved.
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [90, 120],
            [Moment.after_time(3), Moment.after_time(6)],
            align_to=MetricPhaseTarget(0, divisor=2),
        )
        th = self.master.tempo_history
        end_beat = th.length()
        self.assertAlmostEqual(end_beat - round(end_beat / 2) * 2, 0, delta=1e-6)   # end beat on a phase of 2
        self.assertAlmostEqual(th.integrate_interval(0, end_beat), 6, delta=1e-6)   # end time pinned

    def test_set_tempo_targets_per_segment_list_two_runs(self):
        # A per-segment align_to list defines two independent aligned runs: segments 0-1 (phase target) and
        # segments 2-3 (fixed coordinate). Each lands its free (time) axis; the final end beat is unchanged.
        self.master.tempo = 60
        self.master.set_tempo_targets(
            [90, 120, 100, 80],
            [Moment.after_beats(2), Moment.after_beats(5), Moment.after_beats(8), Moment.after_beats(12)],
            align_to=[None, MetricPhaseTarget(0, divisor=3), None, Moment.at_time(7)],
        )
        th = self.master.tempo_history
        self.assertAlmostEqual(th.length(), 12, delta=1e-9)
        self.assertAlmostEqual(th.integrate_interval(0, 5), 3, delta=1e-6)    # first run on its phase (3)
        self.assertAlmostEqual(th.integrate_interval(0, 12), 7, delta=1e-6)   # second run on its fixed time

    def test_set_tempo_targets_single_align_over_mixed_axes_raises(self):
        # A single align_to value forms one run over the whole (mixed-axis) call -> must be single-axis.
        self.master.tempo = 60
        with self.assertRaises(ValueError):
            self.master.set_tempo_targets(
                [90, 120],
                [Moment.after_beats(2), Moment.after_time(5)],
                align_to=MetricPhaseTarget(0, divisor=4),
            )
        self.assertEqual(self.master.tempo_history.length(), 0)   # curve untouched

    def test_set_tempo_targets_group_align_unreachable_rolls_back(self):
        # An unreachable group align rolls the *whole* call back, leaving the curve untouched.
        self.master.tempo = 60
        with self.assertRaises(ValueError):
            self.master.set_tempo_targets(
                [90, 120],
                [Moment.after_beats(2), Moment.after_beats(5)],
                align_to=Moment.at_time(1000),   # free (time) axis, but far out of curvature range
            )
        self.assertEqual(self.master.tempo_history.length(), 0)

    def test_apply_tempo_envelope_loops_until_stopped(self):
        self.master.tempo = 60
        env = TempoEnvelope([60, 120, 60], [1, 1])
        self.master.apply_tempo_envelope(env, loop=True)
        self.assertIsNotNone(self.master.tempo_history.follow_func_or_envelope_loop)
        self.master.stop_tempo_loop_or_function()
        self.assertIsNone(self.master.tempo_history.follow_func_or_envelope_loop)

    def test_tempo_helpers_without_clock_raise(self):
        for call in (
            lambda: utilities.set_tempo(120),
            lambda: utilities.set_rate(2),
            lambda: utilities.set_beat_length(0.5),
            utilities.get_tempo,
            utilities.get_rate,
            utilities.get_beat_length,
            utilities.get_beat,
            utilities.get_time,
            lambda: utilities.set_tempo_target(120, Moment.after_beats(1)),
            lambda: utilities.set_rate_target(2, Moment.after_beats(1)),
            lambda: utilities.set_beat_length_target(0.5, Moment.after_beats(1)),
            lambda: utilities.set_tempo_targets([120], [Moment.after_beats(1)]),
            lambda: utilities.set_rate_targets([2], [Moment.after_beats(1)]),
            lambda: utilities.set_beat_length_targets([0.5], [Moment.after_beats(1)]),
            lambda: utilities.apply_tempo_function(lambda b: 60),
            lambda: utilities.apply_rate_function(lambda b: 1),
            lambda: utilities.apply_beat_length_function(lambda b: 1),
            lambda: utilities.apply_tempo_envelope(TempoEnvelope([60, 120], [1])),
            utilities.stop_tempo_loop_or_function,
        ):
            err = self._capture_on_fresh_thread(call)
            self.assertIsInstance(err, NoActiveClockError)


if __name__ == "__main__":
    unittest.main()
