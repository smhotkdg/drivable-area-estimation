import unittest
from unittest.mock import patch
import cv2
import numpy as np

from drivable_area.tracking.path_tracker import PathTracker, motion


class VideoTrackingTests(unittest.TestCase):
    def test_continuation_start_stays_camera_relative_under_flow(self):
        tracker = PathTracker(360, 640, 15, planner="continuation")
        tracker.update(self.mask(), True)
        anchor = tracker.state[0, :2].copy()
        for i in range(12):
            tracker.update(self.mask(), i % 5 == 0, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 4.0]]))
            np.testing.assert_allclose(tracker.state[0, :2], anchor, atol=0.01)

    def mask(self, center=170):
        mask = np.zeros((640, 360), bool)
        mask[250:, max(0, center - 60) : min(360, center + 61)] = True
        return mask

    def test_intermediate_frames_do_not_recalculate_centerline(self):
        tracker = PathTracker(360, 640, 15)
        with patch.object(tracker, "measurement", wraps=tracker.measurement) as measure:
            for i in range(15):
                tracker.update(self.mask(), detected=(i % 5 == 0))
        self.assertEqual(measure.call_count, 3)

    def test_large_detection_jump_is_rejected_and_direction_hidden(self):
        tracker = PathTracker(360, 640, 15)
        tracker.update(self.mask(), True)
        before = tracker.state[:, 0].copy()
        _, info = tracker.update(self.mask(280), True)
        self.assertLessEqual(
            float(np.max(np.abs(tracker.state[:, 0] - before))), 360 * 0.018 + 1e-6
        )
        self.assertIsNone(info["image_angle_deg"])
        self.assertGreater(tracker.rejected, 0)

    def test_tracking_expires_without_detection_updates(self):
        tracker = PathTracker(360, 640, 15, max_missing_seconds=0.5)
        tracker.update(self.mask(), True)
        for _ in range(10):
            _, info = tracker.update(np.zeros((640, 360), bool), False)
        self.assertIsNone(tracker.state)
        self.assertIsNone(info["image_angle_deg"])

    def test_normal_measurement_jitter_is_smoothed(self):
        tracker = PathTracker(360, 640, 15)
        raw, filtered = [], []
        for i in range(40):
            center = 170 + (6 if i % 2 else -6)
            tracker.update(self.mask(center), True)
            raw.append(center)
            filtered.append(float(tracker.state[len(tracker.ys) // 2, 0]))
        self.assertLess(np.std(filtered[5:]), np.std(raw[5:]))

    def test_optical_flow_recovers_small_motion(self):
        rng = np.random.default_rng(10)
        old = rng.integers(0, 255, (240, 320), np.uint8)
        new = cv2.warpAffine(
            old, np.float32([[1, 0, 3], [0, 1, 2]]), (320, 240), borderMode=cv2.BORDER_REFLECT
        )
        matrix, inliers = motion(old, new, np.ones(old.shape, bool))
        self.assertIsNotNone(matrix)
        self.assertGreaterEqual(inliers, 8)
        self.assertAlmostEqual(float(matrix[0, 2]), 3, delta=0.5)
        self.assertAlmostEqual(float(matrix[1, 2]), 2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
