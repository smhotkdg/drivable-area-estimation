"""Maintain a selected lane hypothesis; center it without a distant side goal.

No lane count is inferred from road-mask width. With no observed/history pair,
the fallback is explicitly a ROAD center estimate, not an identified ego lane.
"""

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from drivable_area.geometry.lane_boundary_tracker import _warp_boundary
from drivable_area.geometry.lane_roi import _road_path
from drivable_area.geometry.corridor_path import path_inside, resample_path


class SingleLaneTracker:
    def __init__(self, fps=15, expiry_seconds=1.0):
        self.expiry = max(1, round(fps * expiry_seconds))
        self.left = None
        self.right = None
        self.age = None
        self.last_bounds = None

    def update(self, mask, bounds=None, sources=None, transform=None, detected=True):
        h, w = mask.shape
        sources = sources or {}
        if self.left is None or len(self.left) != h:
            self.left = np.full(h, np.nan)
            self.right = np.full(h, np.nan)
            self.age = np.full(h, self.expiry + 1.0)
        else:
            age_values = np.where(np.isfinite(self.left), self.age, np.nan)
            self.left = _warp_boundary(self.left, transform)
            self.right = _warp_boundary(self.right, transform)
            self.age = _warp_boundary(age_values, transform) + 1
        available = np.isfinite(self.left) & np.isfinite(self.right) & (self.age <= self.expiry)
        rejected = 0
        observed = np.zeros(h, bool)
        if (
            bounds is not None
            and detected
            and sources.get("left") == "marking"
            and sources.get("right") == "marking"
        ):
            a, b = bounds["left"], bounds["right"]
            width = b - a
            observed = (
                bounds["marking_left"]
                & bounds["marking_right"]
                & np.isfinite(a)
                & np.isfinite(b)
                & (width > w * 0.06)
                & (width < w * 0.9)
            )
            old_width = self.right - self.left
            compatible = (~available) | (
                (width >= old_width * 0.7)
                & (width <= old_width * 1.3)
                & (abs((a + b - self.left - self.right) * 0.5) < w * 0.10)
            )
            rejected = int(np.count_nonzero(observed & ~compatible))
            observed &= compatible
            self.left[observed] = a[observed]
            self.right[observed] = b[observed]
            self.age[observed] = 0
        available = np.isfinite(self.left) & np.isfinite(self.right) & (self.age <= self.expiry)
        selected = mask.copy()
        columns = np.arange(w)[None, :]
        selected &= (~available[:, None]) | (
            (columns >= self.left[:, None]) & (columns <= self.right[:, None])
        )
        self.left[~available] = np.nan
        self.right[~available] = np.nan
        # Only active lane rows are constrained. Unknown rows retain the road
        # estimate and are labelled as such instead of inventing a half-road.
        self.last_bounds = None if bounds is None else {k: v.copy() for k, v in bounds.items()}
        if self.last_bounds is not None:
            self.last_bounds["left"][available] = self.left[available]
            self.last_bounds["right"][available] = self.right[available]
        mode = (
            "observed_pair"
            if observed.sum() >= h * 0.025
            else "tracked_pair"
            if available.sum() >= h * 0.025
            else "road_center_estimate"
        )
        return selected, {
            "mode": mode,
            "locked_rows": int(available.sum()),
            "new_pair_rows": int(observed.sum()),
            "rejected_widening_rows": rejected,
            "true_lane_identity_verified": False,
            "note": "A wide road mask does not establish lane count. Unobserved rows remain explicitly road-center estimates.",
        }


def center_profile(mask):
    data = _road_path(np.asarray(mask, bool))
    if data is None:
        return None
    component, left, right = data
    valid = np.isfinite(left) & np.isfinite(right) & (right - left >= 5)
    rows = np.flatnonzero(valid)
    if len(rows) < 12:
        return None
    # Reject brief row-wise spikes from segmentation/side branches, then
    # smooth only inside contiguous support. No global polynomial or side goal.
    center = (left + right) * 0.5
    for part in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
        if len(part) < 3:
            continue
        values = median_filter(center[part], size=min(15, len(part) // 2 * 2 + 1), mode="nearest")
        center[part] = gaussian_filter1d(values, sigma=max(3, len(mask) * 0.025), mode="nearest")
    center[valid] = np.clip(center[valid], left[valid] + 1, right[valid] - 1)
    return component, left, right, center


def lane_center_path(mask, count=28, max_distance_fraction=0.70):
    h, w = mask.shape
    profile = center_profile(mask)
    if profile is None:
        return None, {"status": "no_center_support"}
    component, left, right, center = profile
    start = None
    for y in range(round(h * 0.94), round(h * 0.75), -1):
        if np.isfinite(center[y]) and component[y, round(center[y])]:
            start = y
            break
    if start is None:
        return None, {"status": "no_near_center"}
    rows = []
    for y in range(start, round(h * 0.25), -1):
        if not np.isfinite(center[y]):
            break
        # A vanishing sliver is not an observed lane center.
        if right[y] - left[y] < max(6, w * 0.035):
            break
        rows.append(y)
    if len(rows) < h * 0.12:
        return None, {"status": "short_center_support"}
    rows = np.asarray(rows)
    x = center[rows].copy()
    # Connect the camera-relative origin gradually to the observed center.
    # This is a near-field attachment, not a far endpoint pulling the curve.
    near_weight = np.clip(1 - (start - rows) / (h * 0.16), 0, 1) ** 2
    anchor = np.clip(w * 0.5, left[start] + 1, right[start] - 1)
    x += near_weight * (anchor - x[0])
    x = np.clip(x, left[rows] + 1, right[rows] - 1)
    raw = np.column_stack([x, rows])
    distance = None if max_distance_fraction is None else h * max_distance_fraction
    result = resample_path(raw, count, max_distance=distance)
    if result is None or not path_inside(result, np.asarray(mask, bool)):
        # Retain a connected supported prefix, never draw chords across holes.
        stop = len(raw)
        for i in range(1, len(raw)):
            if not path_inside(raw[i - 1 : i + 1], np.asarray(mask, bool)):
                stop = i
                break
        result = resample_path(raw[:stop], count, max_distance=distance)
    if result is None or not path_inside(result, np.asarray(mask, bool)):
        # Coarse display segments can cut a sharp supported bend. Shorten the
        # visible horizon until every connecting segment is supported.
        length = float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum())
        initial = min(length, distance) if distance is not None else length
        for factor in (0.9, 0.8, 0.7, 0.6, 0.5):
            candidate = resample_path(raw, count, max_distance=initial * factor)
            if candidate is not None and path_inside(candidate, np.asarray(mask, bool)):
                result = candidate
                break
        else:
            return None, {"status": "center_path_unavailable"}
    return result, {
        "status": "estimated",
        "method": "selected-corridor center, no far side goal",
        "note": "Road center fallback is not proof of single-lane identity.",
    }


def center_metrics(points, mask):
    profile = center_profile(mask)
    if points is None or profile is None:
        return None
    _, left, right, center = profile
    pts = np.asarray(points)
    ys = np.clip(np.rint(pts[:, 1]).astype(int), 0, len(center) - 1)
    valid = np.isfinite(center[ys]) & (right[ys] - left[ys] > 5)
    if not valid.any():
        return None
    fraction = np.abs(pts[valid, 0] - center[ys[valid]]) / (right[ys[valid]] - left[ys[valid]])
    return {
        "mean_center_offset_width_fraction": round(float(np.mean(fraction)), 4),
        "max_center_offset_width_fraction": round(float(np.max(fraction)), 4),
        "scope": "Relative to estimated visible corridor, not annotated true-lane center.",
    }


def supported_prefix(points, mask):
    if points is None:
        return None
    points = np.asarray(points)
    if len(points) < 2 or not path_inside(points[:1], mask):
        return None
    stop = 1
    for i in range(1, len(points)):
        if not path_inside(points[i - 1 : i + 1], mask):
            break
        stop = i + 1
    return points[:stop] if stop >= 5 else None
