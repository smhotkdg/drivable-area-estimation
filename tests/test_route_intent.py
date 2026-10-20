import unittest
import numpy as np
from drivable_area.tracking.route_intent import route_evidence, IntentTracker


def evidence(label, angle):
    return {"direction": label, "far_bearing_deg": angle}


class RouteIntentTests(unittest.TestCase):
    def test_route_geometry_distinguishes_directions(self):
        for label, x in [("straight", 180), ("left", 30), ("right", 330)]:
            points = np.column_stack([np.linspace(180, x, 28), np.linspace(600, 280, 28)])
            self.assertEqual(route_evidence(points, 360, 640)["direction"], label)

    def test_short_or_unconfirmed_path_is_not_straight(self):
        short = np.column_stack([np.full(10, 180), np.linspace(600, 540, 10)])
        self.assertEqual(route_evidence(short, 360, 640)["direction"], "unknown")
        self.assertEqual(route_evidence(short, 360, 640, confirmed=False)["direction"], "unknown")

    def test_single_spike_does_not_switch_label(self):
        tracker = IntentTracker(15)
        for _ in range(3):
            result = tracker.update(evidence("straight", 0))
        self.assertEqual(result["direction"], "straight")
        result = tracker.update(evidence("left", -20))
        self.assertEqual(result["direction"], "straight")
        self.assertFalse(result["fresh"])

    def test_hold_is_not_scored_as_fresh_and_expires(self):
        tracker = IntentTracker(15)
        for _ in range(3):
            tracker.update(evidence("left", -20))
        result = tracker.update(evidence("unknown", None))
        self.assertEqual(result["state"], "held")
        self.assertFalse(result["fresh"])
        for _ in range(10):
            result = tracker.update(evidence("unknown", None))
        self.assertEqual(result["direction"], "unknown")

    def test_hysteresis_avoids_turn_threshold_flicker(self):
        tracker = IntentTracker(15)
        for _ in range(3):
            tracker.update(evidence("left", -13))
        result = tracker.update(evidence("straight", -8))
        self.assertEqual(result["direction"], "left")
        self.assertTrue(result["fresh"])


if __name__ == "__main__":
    unittest.main()
