"""
Regression tests for extract_absolute_tempo_envelope on clocks following an open-ended
tempo function or looping tempo envelope.

TempoHistory auto-extends an open-ended follow inside advance() (via time_at_beat), so the
extraction loop's `th.beat < th.length()` condition would chase a receding horizon forever
unless the deepcopied histories are frozen first. Before the fix, a function follower hung
outright; the envelope-loop case terminated but is covered here too, since it shares the
frozen-copy code path.
"""
import threading
import unittest
from math import sin

from clockblocks.clock import Clock
from clockblocks.tempo_envelope import TempoEnvelope
from tests import timing


def extract_with_timeout(clock, timeout=20):
    """Run extract_absolute_tempo_envelope in a daemon thread so a regression hangs the
    thread, not the suite; returns the envelope or None on timeout."""
    result = {}
    thread = threading.Thread(target=lambda: result.update(env=clock.extract_absolute_tempo_envelope()),
                              daemon=True)
    thread.start()
    thread.join(timeout)
    return result.get("env")


class ExtractTempoEnvelopeTestCase(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "master"):
            self.master.kill()

    def _run_child_and_extract(self, apply_tempo_shape):
        self.master = Clock("master")

        def child_process():
            from clockblocks import current_clock, wait
            apply_tempo_shape(current_clock())
            while True:
                wait(0.25)

        child = self.master.fork(child_process, name="child")
        self.master.fast_forward()
        self.master.wait(8)
        env = extract_with_timeout(child)
        self.assertIsNotNone(env, "extract_absolute_tempo_envelope did not terminate")
        self.assertGreater(env.length(), 0)

    def test_extract_from_function_follower(self):
        self._run_child_and_extract(
            lambda c: c.apply_tempo_function(lambda t: 60 + 30 * sin(t), duration_units="time"))

    def test_extract_from_envelope_loop_incommensurable_length(self):
        # loop length 2.97 never lines up with the extraction's sampling step, so this
        # exercises the receding-horizon case even for envelope loops
        self._run_child_and_extract(
            lambda c: c.apply_tempo_envelope(
                TempoEnvelope.from_levels_and_durations((160, 100, 70), (1.5, 1.47)), loop=True))

    def test_stepwise_tempo_keeps_jumps_sharp(self):
        # A stepwise tempo (flat beats joined by instantaneous jumps) must extract as flat segments
        # plus zero-duration jumps — never as short accelerandi/ritardandi in the flat stretches.
        self.master = Clock("master")

        def child_process():
            from clockblocks import current_clock, wait
            current_clock().apply_tempo_envelope(
                TempoEnvelope.from_levels_and_durations((160, 160, 100, 100, 70, 70), (1, 0, 1, 0, 1)),
                loop=True)
            while True:
                wait(0.25)

        child = self.master.fork(child_process, name="child")
        self.master.fast_forward()
        self.master.wait(8)
        env = extract_with_timeout(child)
        self.assertIsNotNone(env)
        for seg in env.segments:
            if abs(seg.end_level - seg.start_level) < 1e-6:
                self.assertAlmostEqual(seg.curve_shape, 0, places=6,
                                       msg="flat segment rendered with spurious curvature")
        # the jumps themselves survive as (near) zero-duration segments with a real level change
        self.assertTrue(any(seg.duration < 1e-6 and abs(seg.end_level - seg.start_level) > 0.05
                            for seg in env.segments), "instantaneous tempo jumps were smeared away")


if __name__ == '__main__':
    unittest.main()
