import unittest
import cv2
import numpy as np
from drivable_area.geometry.single_lane_center import (
    SingleLaneTracker,
    lane_center_path,
    center_metrics,
)
from drivable_area.geometry.corridor_path import path_inside


class SingleLaneCenterTests(unittest.TestCase):
    def pair(self, h=320, left=50, right=110):
        return {
            "left": np.full(h, float(left)),
            "right": np.full(h, float(right)),
            "marking_left": np.ones(h, bool),
            "marking_right": np.ones(h, bool),
        }

    def test_suddenly_widened_pair_does_not_change_selected_lane(self):
        tracker = SingleLaneTracker(fps=10)
        mask = np.ones((320, 180), bool)
        sources = {"left": "marking", "right": "marking"}
        tracker.update(mask, self.pair(), sources)
        selected, meta = tracker.update(mask, self.pair(left=20, right=150), sources)
        self.assertGreater(meta["rejected_widening_rows"], 250)
        self.assertFalse(selected[:, 120:].any())
        self.assertTrue(selected[200, 80])

    def test_unknown_wide_road_is_not_silently_split_into_lanes(self):
        selected, meta = SingleLaneTracker().update(np.ones((320, 180), bool))
        self.assertTrue(selected.all())
        self.assertEqual(meta["mode"], "road_center_estimate")
        self.assertFalse(meta["true_lane_identity_verified"])

    def test_identity_expires_instead_of_preserving_an_old_lane_forever(self):
        tracker = SingleLaneTracker(fps=10, expiry_seconds=0.3)
        mask = np.ones((320, 180), bool)
        tracker.update(mask, self.pair(), {"left": "marking", "right": "marking"})
        for _ in range(5):
            selected, meta = tracker.update(mask, detected=False)
        self.assertEqual(meta["mode"], "road_center_estimate")
        self.assertTrue(selected.all())

    def test_curve_center_does_not_follow_the_distant_outer_edge(self):
        mask = np.zeros((640, 360), np.uint8)
        cv2.fillPoly(
            mask, [np.array([[0, 639], [359, 639], [359, 180], [305, 180], [0, 400]], np.int32)], 1
        )
        path, info = lane_center_path(mask > 0)
        self.assertIsNotNone(path, info)
        self.assertTrue(path_inside(path, mask > 0))
        self.assertLess(center_metrics(path, mask > 0)["mean_center_offset_width_fraction"], 0.04)
        near = path[(path[:, 1] > 370) & (path[:, 1] < 430)]
        self.assertLess(near[:, 0].max(), 210)

    def test_straight_selected_lane_center_stays_centered(self):
        mask = np.zeros((640, 360), bool)
        mask[200:, 100:260] = True
        path, info = lane_center_path(mask)
        self.assertIsNotNone(path, info)
        self.assertLess(np.ptp(path[:, 0]), 2)
        self.assertTrue(path_inside(path, mask))

    def test_center_path_never_bridges_a_blocked_road(self):
        mask = np.zeros((640, 360), bool)
        mask[200:, 100:260] = True
        mask[410:435] = False
        path, _ = lane_center_path(mask)
        if path is not None:
            self.assertTrue(path_inside(path, mask))
            self.assertGreaterEqual(path[:, 1].min(), 435)

    def test_direction_lookahead_includes_a_bend_beyond_the_near_route(self):
        from drivable_area.tracking.route_intent import curve_evidence

        road = np.zeros((640, 360), bool)
        for y in range(180, 640):
            center = 180 + max(0, 300 - y) * 0.75
            road[y, max(0, int(center - 45)) : min(360, int(center + 45))] = True
        short, _ = lane_center_path(road, max_distance_fraction=0.4)
        full, _ = lane_center_path(road, 48, max_distance_fraction=None)
        self.assertEqual(curve_evidence(short, 360, 640)["direction"], "straight")
        self.assertEqual(curve_evidence(full, 360, 640)["direction"], "right")

    def test_invalid_far_tail_does_not_erase_supported_direction(self):
        from drivable_area.geometry.lane_boundary_tracker import BoundaryDirectionObserver

        t = np.linspace(0, 1, 48)
        path = np.column_stack([180 + 90 * t * t, 600 - 400 * t])
        mask = np.ones((640, 360), bool)
        mask[:250] = False
        result = BoundaryDirectionObserver(trim_route=True).update(mask, path, detected=True)
        self.assertEqual(result["direction"], "right")
        self.assertLess(result["supported_points"], 48)

    def test_flow_cannot_keep_a_filtered_route_at_the_corridor_edge(self):
        from drivable_area.tracking.path_tracker import PathTracker

        mask = np.zeros((640, 360), bool)
        mask[180:, 100:260] = True
        tracker = PathTracker(360, 640, 15, planner="lane-center")
        tracker.update(mask, True)
        for i in range(10):
            route, info = tracker.update(
                mask, i % 5 == 0, np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]])
            )
            self.assertIsNotNone(route, info)
            far = route[0][route[0][:, 1] < 640 * 0.84]
            self.assertLessEqual(np.max(abs(far[:, 0] - 179.5)), 7)
