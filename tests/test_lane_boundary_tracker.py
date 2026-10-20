import unittest
from unittest.mock import patch
import cv2
import numpy as np
from drivable_area.geometry.lane_boundary_tracker import (
    LaneBoundaryTracker,
    marking_boundaries,
    boundary_center_path,
    BoundaryDirectionObserver,
)
from drivable_area.geometry.corridor_path import path_inside
from drivable_area.tracking.route_intent import curve_evidence


class BoundaryTests(unittest.TestCase):
    def scene(self):
        road = np.zeros((320, 180), bool)
        road[90:, 10:170] = True
        marks = np.zeros(road.shape, np.uint8)
        cv2.line(marks, (115, 310), (95, 120), 1, 3)
        return road, marks

    def test_observed_divider_is_a_hard_constraint(self):
        road, marks = self.scene()
        tracker = LaneBoundaryTracker()
        mask, info = tracker.update(road, marks)
        self.assertEqual(info["sources"]["right"], "marking")
        for y in range(125, 300):
            xs = np.flatnonzero(mask[y])
            if len(xs):
                self.assertLess(xs.max(), (115 - 95) * (y - 120) / 190 + 95 + 3)
        path, meta = boundary_center_path(mask)
        self.assertIsNotNone(path, meta)
        self.assertTrue(path_inside(path, mask))

    def test_crosswalk_is_not_a_side_boundary(self):
        road, marks = self.scene()
        marks[:] = 0
        for y in (160, 185, 210):
            cv2.line(marks, (20, y), (160, y), 1, 5)
        boundaries = marking_boundaries(road, marks)
        self.assertFalse(any(np.isfinite(b).any() for b in boundaries.values()))

    def test_missing_marking_expires_even_with_other_rows_observed(self):
        road, marks = self.scene()
        tracker = LaneBoundaryTracker(fps=10, expiry_seconds=0.3)
        tracker.update(road, marks)
        for _ in range(5):
            mask, info = tracker.update(road, np.zeros_like(marks))
        self.assertEqual(info["sources"]["right"], "road_edge_estimate")
        self.assertFalse(tracker.last_bounds["marking_right"].any())

    def test_boundary_motion_and_path_are_separate(self):
        road, marks = self.scene()
        tracker = LaneBoundaryTracker()
        tracker.update(road, marks)
        before = tracker.last_bounds["right"].copy()
        _, info = tracker.update(
            road, transform=np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 0.0]]), detected=False
        )
        self.assertEqual(info["sources"]["right"], "tracked_marking")
        self.assertAlmostEqual(tracker.last_bounds["right"][180] - before[180], 3, delta=0.2)

    def test_road_edges_do_not_claim_verified_ego_lane(self):
        road, marks = self.scene()
        _, info = LaneBoundaryTracker().update(road, np.zeros_like(marks))
        self.assertFalse(info["ego_lane_verified"])
        self.assertEqual(info["sources"]["left"], "road_edge_estimate")

    def test_curve_with_small_bearing_is_not_forced_straight(self):
        t = np.linspace(0, 1, 40)
        for sign, label in [(-1, "left"), (1, "right")]:
            points = np.column_stack([180 + sign * 45 * t**3, 600 - 320 * t])
            evidence = curve_evidence(points, 360, 640)
            self.assertLess(abs(evidence["far_bearing_deg"]), 10)
            self.assertEqual(evidence["direction"], label)
        points = np.column_stack([180 + 10 * t, 600 - 320 * t])
        self.assertEqual(curve_evidence(points, 360, 640)["direction"], "straight")

    def test_direction_observation_survives_path_filter_independently(self):
        observer = BoundaryDirectionObserver(fps=10)
        mask = np.ones((640, 360), bool)
        t = np.linspace(0, 1, 28)
        path = np.column_stack([180 + 60 * t * t, 600 - 320 * t])
        result = observer.update(mask, path, detected=True)
        self.assertEqual(result["direction"], "right")
        for _ in range(4):
            result = observer.update(mask, transform=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 3.0]]))
            self.assertEqual(result["direction"], "right")
            np.testing.assert_allclose(observer.path[0], path[0], atol=0.01)
        for _ in range(4):
            result = observer.update(mask)
        self.assertEqual(result["direction"], "unknown")

    def test_partial_refresh_does_not_keep_unobserved_far_rows_forever(self):
        road, marks = self.scene()
        tracker = LaneBoundaryTracker(fps=10, expiry_seconds=0.3)
        left = np.full(320, np.nan)
        right = np.full(320, np.nan)
        left[120:300] = 30
        right[120:300] = 120
        with patch(
            "drivable_area.geometry.lane_boundary_tracker.marking_boundaries",
            return_value={"left": left, "right": right},
        ):
            tracker.update(road, marks)
        partial_left = left.copy()
        partial_right = right.copy()
        partial_left[:230] = np.nan
        partial_right[:230] = np.nan
        for _ in range(5):
            with patch(
                "drivable_area.geometry.lane_boundary_tracker.marking_boundaries",
                return_value={"left": partial_left, "right": partial_right},
            ):
                tracker.update(road, marks)
        self.assertFalse(tracker.last_bounds["marking_right"][150])
        self.assertTrue(tracker.last_bounds["marking_right"][250])

    def test_filtered_path_stays_inside_a_new_close_boundary(self):
        from drivable_area.tracking.path_tracker import PathTracker

        road = np.zeros((640, 360), bool)
        road[240:, 110:240] = True
        tracker = PathTracker(360, 640, 15, planner="boundaries")
        tracker.update(road, True)
        narrowed = road.copy()
        narrowed[:, 178:] = False
        path, info = tracker.update(narrowed, False)
        self.assertIsNotNone(path, info)
        self.assertTrue(path_inside(path[0], narrowed))
