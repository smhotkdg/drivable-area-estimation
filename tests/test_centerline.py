import unittest
import cv2
import numpy as np

from drivable_area.geometry.draw_centerline import estimate_centerline, draw_centerline


class CenterlineTests(unittest.TestCase):
    def test_straight_center_is_halfway_between_boundaries(self):
        mask = np.zeros((240, 320), bool)
        mask[80:, 70:250] = True
        result = estimate_centerline(mask)
        self.assertEqual(result["status"], "estimated")
        self.assertTrue(all(abs(x - 160) <= 1 for line in result["segments"] for x, y in line))

    def test_empty_mask_has_no_centerline(self):
        info = estimate_centerline(np.zeros((240, 320), bool))
        self.assertEqual(info["status"], "unavailable")
        self.assertIsNone(info["representative_point"])

    def test_obstacle_is_not_bridged_by_line_or_rendering(self):
        mask = np.zeros((240, 320), bool)
        mask[70:, 50:270] = True
        mask[120:170, 140:180] = False
        image = np.zeros((240, 320, 3), np.uint8)
        vis, info = draw_centerline(image, mask)
        for line in info["segments"]:
            raster = np.zeros_like(mask, np.uint8)
            cv2.polylines(raster, [np.array(line, np.int32).reshape(-1, 1, 2)], False, 1, 1)
            self.assertFalse(np.any((raster > 0) & ~mask))
        self.assertFalse(np.any(vis[~mask]))

    def test_center_follows_curving_corridor(self):
        mask = np.zeros((240, 320), bool)
        for y in range(70, 240):
            center = int(160 - 60 * ((240 - y) / 170) ** 2)
            mask[y, center - 40 : center + 41] = True
        info = estimate_centerline(mask)
        points = [p for line in info["segments"] for p in line]
        self.assertLess(points[-1][0], points[0][0] - 30)
        self.assertTrue(all(mask[y, x] for x, y in points))


if __name__ == "__main__":
    unittest.main()
