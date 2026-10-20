"""Single connected image-space route favoring clearance from mask boundaries.

This describes mask geometry, not vehicle feasibility or intersection intent.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.graph import route_through_array
from skimage.morphology import skeletonize


def resample_path(points, count=28, max_distance=None):
    points = np.asarray(points, dtype=float)
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-6]
    lengths, points = lengths[keep], points[keep]
    if len(points) < 2 or lengths[-1] <= 0:
        return None
    at = np.linspace(
        0, min(lengths[-1], max_distance) if max_distance is not None else lengths[-1], count
    )
    return np.column_stack([np.interp(at, lengths, points[:, dim]) for dim in range(2)])


def path_inside(points, mask):
    raster = np.zeros(mask.shape, np.uint8)
    pts = np.rint(points).astype(np.int32)
    h, w = mask.shape
    if (
        np.any(pts[:, 0] < 0)
        or np.any(pts[:, 0] >= w)
        or np.any(pts[:, 1] < 0)
        or np.any(pts[:, 1] >= h)
    ):
        return False
    if not np.all(mask[pts[:, 1], pts[:, 0]]):
        return False
    cv2.polylines(raster, [pts.reshape(-1, 1, 2)], False, 1, 1)
    return not np.any((raster > 0) & ~mask)


def connected_center_path(mask, count=28):
    height, width = mask.shape
    scale = min(1.0, 180.0 / width)
    w, h = max(16, round(width * scale)), max(32, round(height * scale))
    small = cv2.resize(np.asarray(mask, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    start_y = min(h - 1, round(h * 0.94))
    start = None
    for y in range(start_y, int(h * 0.75), -1):
        xs = np.flatnonzero(small[y])
        if len(xs):
            x = int(xs[np.argmin(abs(xs - (w - 1) * 0.5))])
            if abs(x - w * 0.5) <= w * 0.2:
                start = (y, x)
                break
    if start is None:
        return None, {"status": "no_near_road"}
    _, labels = cv2.connectedComponents(small)
    component = labels == labels[start]
    component[: int(h * 0.25)] = False
    # Padding treats the image border as a boundary instead of infinite road.
    clearance = cv2.distanceTransform(np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 5)[
        1:-1, 1:-1
    ]
    skeleton = skeletonize(component).astype(np.uint8)
    neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    goal_candidates = (skeleton > 0) & (neighbors == 1) & (clearance >= 2)
    goal_candidates[int(start[0] - h * 0.12) :] = False
    ys, xs = np.nonzero(goal_candidates)
    if not len(ys):
        goal_candidates = (skeleton > 0) & (clearance >= 2)
        goal_candidates[int(start[0] - h * 0.12) :] = False
        ys, xs = np.nonzero(goal_candidates)
    if not len(ys):
        return None, {"status": "insufficient_forward_road"}
    # Follow a medial-axis continuation. This supports a road turning sideways
    # in either direction; a horizontal slice cannot represent such a turn.
    scores = start[0] - ys + 0.35 * np.abs(xs - start[1])
    best = int(np.argmax(scores))
    far_y, goal_x = int(ys[best]), int(xs[best])
    costs = np.full((h, w), np.inf)
    costs[component] = 1.0 + (w * 0.12 / (clearance[component] + 1.0)) ** 2
    try:
        route, _ = route_through_array(
            costs, start, (far_y, goal_x), fully_connected=True, geometric=True
        )
    except ValueError:
        return None, {"status": "disconnected_goal"}
    path = np.asarray(route, dtype=float)[:, ::-1]
    if len(path) < 8:
        return None, {"status": "short_path"}
    smoothed = gaussian_filter1d(path, sigma=5.0, axis=0, mode="nearest")
    result = resample_path(smoothed / scale, count, max_distance=height * 0.55)
    if result is None or not path_inside(result, np.asarray(mask, bool)):
        result = resample_path(path / scale, count, max_distance=height * 0.55)
    if result is None or not path_inside(result, np.asarray(mask, bool)):
        return None, {"status": "unsafe_chord"}
    return result, {
        "status": "estimated",
        "method": "connected clearance-weighted route",
        "goal_pixel": result[-1].round(1).tolist(),
        "note": "Visible-corridor geometry only; intersection route intent is unknown.",
    }
