import unittest

from emgimu.phases import PhaseTracker
from emgimu.state import Direction, Phase


class PhaseTests(unittest.TestCase):
    def test_onset_requires_two_frames_then_holds(self):
        tracker = PhaseTracker(Direction.NONE, Direction.UNKNOWN, confirm_frames=2, hold_after_frames=2)
        first = tracker.update(Direction.RIGHT, 0.9)
        self.assertEqual(first.stable_label, Direction.NONE)
        self.assertEqual(first.phase, Phase.ONSET)
        second = tracker.update(Direction.RIGHT, 0.9)
        self.assertEqual(second.stable_label, Direction.RIGHT)
        self.assertEqual(second.phase, Phase.ACTIVE)
        tracker.update(Direction.RIGHT, 0.9)
        held = tracker.update(Direction.RIGHT, 0.9)
        self.assertEqual(held.phase, Phase.HOLD)

    def test_release_and_unknown_are_distinct(self):
        tracker = PhaseTracker(Direction.NONE, Direction.UNKNOWN)
        tracker.update(Direction.RIGHT, 1); tracker.update(Direction.RIGHT, 1)
        release = tracker.update(Direction.NONE, 1)
        self.assertEqual(release.phase, Phase.RELEASE)
        unknown = tracker.update(Direction.UNKNOWN, 0, valid=False)
        self.assertEqual(unknown.phase, Phase.UNKNOWN)
        self.assertEqual(unknown.stable_label, Direction.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

