"""Forward connected-road continuation, without medial-axis endpoint selection.

This is an image-space default continuation, not a navigation route. A lane
region is a soft goal cue: it cannot erase forward road when its fit fails.
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from skimage.graph import route_through_array
from drivable_area.geometry.corridor_path import path_inside, resample_path


def _spans(row):
    delta = np.diff(np.r_[0, row, 0].astype(np.int8))
    return list(zip(np.flatnonzero(delta == 1), np.flatnonzero(delta == -1) - 1))


def continuation_center_path(road, lane_region=None, count=28):
    height, width = road.shape
    scale = min(1.0, 180.0 / width)
    w, h = round(width * scale), round(height * scale)
    small = cv2.resize(np.asarray(road, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    start = None
    for y in range(round(h * 0.94), round(h * 0.75), -1):
        xs = np.flatnonzero(small[y])
        if len(xs):
            x = int(xs[np.argmin(abs(xs - (w - 1) * 0.5))])
            if abs(x - w * 0.5) < w * 0.2:
                start = (y, x)
                break
    if start is None:
        return None, {"status": "no_near_road"}
    _, labels = cv2.connectedComponents(small)
    component = labels == labels[start]
    component[: round(h * 0.25)] = False
    # Trace overlapping road spans. Crossing an occlusion or a disconnected
    # island is forbidden, rather than extrapolating through it.
    previous = (start[1], start[1])
    observed = []
    for y in range(start[0], round(h * 0.25), -1):
        spans = [s for s in _spans(component[y]) if s[1] - s[0] >= max(3, w * 0.035)]
        spans = [s for s in spans if min(s[1], previous[1]) - max(s[0], previous[0]) >= -2]
        if not spans:
            break
        previous = max(
            spans,
            key=lambda s: (
                min(s[1], previous[1]) - max(s[0], previous[0]),
                -abs(sum(s) - sum(previous)),
            ),
        )
        observed.append((y, *previous))
    candidates = [s for s in observed if s[2] - s[1] >= w * 0.08 and start[0] - s[0] >= h * 0.12]
    if not candidates:
        return None, {"status": "insufficient_forward_road"}
    # Stop slightly below the segmentation tip, where a span has actual width.
    top = candidates[-1][0] + round(h * 0.018)
    far_y, a, b = min(candidates, key=lambda s: abs(s[0] - top))
    goal_x = (a + b) * 0.5
    cue_used = False
    if lane_region is not None:
        cue = cv2.resize(np.asarray(lane_region, np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        spans = [
            (max(a, span_left), min(b, span_right)) for span_left, span_right in _spans(cue[far_y])
        ]
        spans = [s for s in spans if s[1] - s[0] >= w * 0.06]
        if spans:
            span_left, span_right = min(spans, key=lambda s: abs((s[0] + s[1]) * 0.5 - start[1]))
            goal_x, cue_used = (span_left + span_right) * 0.5, True
    goal = (far_y, int(round(goal_x)))
    clearance = cv2.distanceTransform(np.pad(component.astype(np.uint8), 1), cv2.DIST_L2, 5)[
        1:-1, 1:-1
    ]
    yy, xx = np.indices((h, w))
    progress = np.clip((start[0] - yy) / max(1, start[0] - far_y), 0, 1)
    # Zero lateral tangent at the near seed prevents a far side goal from
    # immediately pulling the start diagonally across the current corridor.
    guide = start[1] + progress**2 * (3.0 - 2.0 * progress) * (goal[1] - start[1])
    costs = np.full((h, w), np.inf)
    # Clearance alone pulls the route into wide side branches. A forward guide
    # counteracts that attraction; mask boundaries still constrain every pixel.
    values = 1.0 + 0.35 * (w * 0.10 / (clearance + 1)) ** 2 + 2.0 * ((xx - guide) / (w * 0.16)) ** 2
    costs[component] = values[component]
    costs[:far_y] = np.inf
    try:
        route, _ = route_through_array(costs, start, goal, fully_connected=True, geometric=True)
    except ValueError:
        return None, {"status": "disconnected_goal"}
    path = np.asarray(route, dtype=float)[:, ::-1]
    if len(path) < 8:
        return None, {"status": "short_path"}
    smooth = gaussian_filter1d(path, 5.0, axis=0, mode="nearest")
    result = resample_path(smooth / scale, count, max_distance=height * 0.55)
    if result is None or not path_inside(result, np.asarray(road, bool)):
        result = resample_path(path / scale, count, max_distance=height * 0.55)
    if result is None or not path_inside(result, np.asarray(road, bool)):
        return None, {"status": "unsafe_chord"}
    return result, {
        "status": "estimated",
        "method": "forward road continuation",
        "lane_goal_cue": cue_used,
        "goal_pixel": [round(goal[1] / scale, 1), round(goal[0] / scale, 1)],
        "note": "Default visible-road continuation; intersection navigation intent is unknown.",
    }
