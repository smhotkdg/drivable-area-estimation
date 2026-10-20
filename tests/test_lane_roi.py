import unittest
import cv2
import numpy as np

from drivable_area.geometry.lane_roi import EgoLaneTracker, _curve


class LaneTrackerTests(unittest.TestCase):
    def setUp(self):
        self.h, self.w = 240, 320
        self.road = np.ones((self.h, self.w), bool)
        self.lanes = np.zeros((self.h, self.w), np.uint8)
        for y in range(80, self.h):
            offset = int(25 * ((y - 160) / 100) ** 2)
            cv2.circle(self.lanes, (100 + offset, y), 2, 1, -1)
            cv2.circle(self.lanes, (220 + offset, y), 2, 1, -1)
        rng = np.random.default_rng(3)
        self.image = rng.integers(0, 255, (self.h, self.w, 3), dtype=np.uint8)

    def test_curved_lane_excludes_adjacent_road(self):
        result, info = EgoLaneTracker().update(self.road, self.lanes)
        self.assertTrue(result[200, 160])
        self.assertFalse(result[200, 20])
        self.assertFalse(result[200, 310])
        self.assertEqual(info["basis"], "lane_boundaries")

    def test_missing_lanes_are_explicit_road_edge_estimates(self):
        result, info = EgoLaneTracker().update(self.road, np.zeros_like(self.lanes))
        self.assertTrue(result[200, 160])
        self.assertEqual(info["basis"], "road_edge_assisted")

    def test_disconnected_road_and_obstacle_remain_excluded(self):
        road = self.road.copy()
        road[90:105] = False
        road[180:200, 140:180] = False
        result, _ = EgoLaneTracker().update(road, np.zeros_like(self.lanes))
        self.assertFalse(result[:90].any())
        self.assertFalse(result[180:200, 140:180].any())
        self.assertFalse(np.any(result & ~road))

    def test_history_tracks_small_motion_and_expires_on_seek(self):
        tracker = EgoLaneTracker()
        tracker.update(self.road, self.lanes, self.image, 10)
        moved = cv2.warpAffine(
            self.image,
            np.float32([[1, 0, 2], [0, 1, 1]]),
            (self.w, self.h),
            borderMode=cv2.BORDER_REFLECT,
        )
        _, info = tracker.update(self.road, np.zeros_like(self.lanes), moved, 11)
        self.assertTrue(info["temporal_used"])
        _, info = tracker.update(self.road, np.zeros_like(self.lanes), moved, 40)
        self.assertFalse(info["temporal_used"])
        self.assertEqual(info["basis"], "road_edge_assisted")

    def test_history_expires_after_eight_source_frames(self):
        tracker = EgoLaneTracker(max_history_frames=8)
        tracker.update(self.road, self.lanes, self.image, 1)
        for frame in range(2, 11):
            _, info = tracker.update(self.road, np.zeros_like(self.lanes), self.image, frame)
        self.assertFalse(info["temporal_used"])

    def test_no_road_resets_history(self):
        tracker = EgoLaneTracker()
        tracker.update(self.road, self.lanes, self.image, 1)
        result, info = tracker.update(np.zeros_like(self.road), self.lanes, self.image, 2)
        self.assertFalse(result.any())
        self.assertEqual(info["status"], "road_unavailable")
        self.assertIsNone(tracker.previous)

    def test_switch_between_markings_does_not_reverse_extrapolation(self):
        outer = [(y, 240 + 1.4 * (y - 260)) for y in range(260, 301)]
        nearer = [(y, 240 + 0.8 * (y - 305)) for y in range(305, 351)]
        curve = _curve(outer + nearer, 640, 360)
        self.assertIsNotNone(curve)
        self.assertGreater(curve[500], curve[350])
        self.assertGreater(curve[500], 360)


if __name__ == "__main__":
    unittest.main()
