import unittest
import numpy as np
from drivable_area.tracking.steering_indicator import ImageSteeringIndicator
from drivable_area.visualization.steering_wheel import draw_steering_wheel


def direction(label="right", pending=None):
    return {"direction": label, "stale": False, "pending_direction": pending}


class SteeringIndicatorTests(unittest.TestCase):
    def test_positive_and_negative_image_bearings_turn_the_wheel_correspondingly(self):
        for label, angle in [("left", -20), ("right", 20)]:
            tracker = ImageSteeringIndicator(60)
            for _ in range(90):
                value = tracker.update(
                    {"direction": label, "far_bearing_deg": angle}, direction(label), True
                )
            self.assertAlmostEqual(value["image_bearing_deg"], angle, delta=0.2)
            self.assertIn("NOT physical", value["note"])

    def test_spurious_opposite_candidate_does_not_reverse_the_wheel(self):
        tracker = ImageSteeringIndicator(60)
        for _ in range(30):
            tracker.update({"direction": "left", "far_bearing_deg": -20}, direction("left"), True)
        value = tracker.update(
            {"direction": "right", "far_bearing_deg": 25}, direction("left", "right"), True
        )
        self.assertLess(value["wheel_rotation_deg"], 0)
        self.assertFalse(value["new_angle_observation"])

    def test_angle_changes_are_rate_limited_even_after_a_confirmed_reversal(self):
        tracker = ImageSteeringIndicator(60)
        for _ in range(60):
            before = tracker.update(
                {"direction": "left", "far_bearing_deg": -60}, direction("left"), True
            )
        after = tracker.update(
            {"direction": "right", "far_bearing_deg": 60}, direction("right"), True
        )
        self.assertLessEqual(abs(after["wheel_rotation_deg"] - before["wheel_rotation_deg"]), 1.501)

    def test_predictions_do_not_add_new_angle_observations(self):
        tracker = ImageSteeringIndicator(30)
        tracker.update({"direction": "right", "far_bearing_deg": 10}, direction(), True)
        value = tracker.update({"direction": "right", "far_bearing_deg": 55}, direction(), False)
        self.assertFalse(value["new_angle_observation"])
        self.assertLess(value["image_bearing_deg"], 10)

    def test_expired_or_missing_angles_are_not_displayed_as_known(self):
        tracker = ImageSteeringIndicator(10)
        tracker.update({"direction": "right", "far_bearing_deg": 20}, direction(), True)
        for _ in range(8):
            value = tracker.update({"direction": "unknown"}, direction("unknown"), False)
        self.assertFalse(value["available"])
        self.assertIsNone(value["image_bearing_deg"])

    def test_wheel_glyph_rotation_changes_pixels_but_keeps_dimensions(self):
        blank = np.zeros((80, 80, 3), np.uint8)
        a = draw_steering_wheel(
            blank.copy(), (40, 40), 18, {"wheel_rotation_deg": 0, "available": True}
        )
        b = draw_steering_wheel(
            blank.copy(), (40, 40), 18, {"wheel_rotation_deg": 25, "available": True}
        )
        self.assertEqual(a.shape, b.shape)
        self.assertGreater(np.count_nonzero(a != b), 20)
