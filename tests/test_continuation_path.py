import unittest
import numpy as np
from drivable_area.geometry.continuation_path import continuation_center_path
from drivable_area.geometry.corridor_path import path_inside


class ContinuationTests(unittest.TestCase):
    def test_wide_side_branch_does_not_replace_forward_exit(self):
        road = np.zeros((320, 180), bool)
        road[80:, 65:115] = True
        road[150:230, :] = True
        cue = road.copy()
        cue[:160] = False  # An upstream fit wrongly erased the forward exit.
        path, info = continuation_center_path(road, cue)
        self.assertIsNotNone(path, info)
        self.assertTrue(path_inside(path, road))
        self.assertLess(abs(path[-1, 0] - 90), 12)
        self.assertLess(path[-1, 1], 160)

    def test_curves_remain_on_the_connected_road(self):
        for sign in (-1, 1):
            road = np.zeros((320, 180), bool)
            for y in range(105, 320):
                center = 90 + sign * min(65, max(0, 240 - y) * 0.6)
                road[y, max(0, int(center - 25)) : min(180, int(center + 25))] = True
            path, info = continuation_center_path(road)
            self.assertIsNotNone(path, info)
            self.assertTrue(path_inside(path, road))
            self.assertGreater(sign * (path[-1, 0] - path[0, 0]), 30)

    def test_blocked_road_does_not_reach_distant_component(self):
        road = np.zeros((320, 180), bool)
        road[80:, 65:115] = True
        road[170:190] = False
        path, _ = continuation_center_path(road)
        if path is not None:
            self.assertTrue(path_inside(path, road))
            self.assertGreaterEqual(path[:, 1].min(), 190)

    def test_empty_road_abstains(self):
        path, _ = continuation_center_path(np.zeros((320, 180), bool))
        self.assertIsNone(path)
