import unittest
import numpy as np
from drivable_area.geometry.corridor_path import connected_center_path, path_inside


class ConnectedPathTests(unittest.TestCase):
    def mask(self, turn):
        mask = np.zeros((320, 180), bool)
        mask[140:, 65:115] = True
        if turn == "left":
            mask[105:165, 5:115] = True
        elif turn == "right":
            mask[105:165, 65:175] = True
        else:
            mask[50:, 65:115] = True
            mask[130:155, 115:175] = True
        return mask

    def test_left_and_right_turns_have_correct_path_direction(self):
        for side, sign in [("left", -1), ("right", 1)]:
            mask = self.mask(side)
            path, info = connected_center_path(mask)
            self.assertIsNotNone(path, info)
            self.assertTrue(path_inside(path, mask))
            self.assertGreater(sign * (path[-1, 0] - path[0, 0]), 20)

    def test_intersection_side_bulge_does_not_create_zigzag(self):
        mask = self.mask("straight")
        path, info = connected_center_path(mask)
        self.assertIsNotNone(path, info)
        self.assertLess(np.ptp(path[:, 0]), 15)
        self.assertTrue(path_inside(path, mask))

    def test_no_route_across_disconnected_road(self):
        mask = self.mask("straight")
        mask[170:190] = False
        path, _ = connected_center_path(mask)
        if path is not None:
            self.assertTrue(path_inside(path, mask))
            self.assertTrue(np.all(path[:, 1] >= 190))

    def test_empty_mask_gives_no_path(self):
        path, _ = connected_center_path(np.zeros((320, 180), bool))
        self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
