"""Curved lane/road-edge corridor with short-lived, motion-aligned history.

A road-edge-assisted result is an estimate of the visible road corridor, not
proof that the ego lane has been identified. Original road/obstacle labels
always bound the result; disconnected roads are excluded from the seed region.
"""

import cv2
import numpy as np
from drivable_area.geometry.lane_roi_linear import restrict_to_ego_lane as linear_lane_region


def _runs(row):
    edges = np.diff(np.r_[False, row, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def _road_path(road):
    h, w = road.shape
    cleaned = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    seeds = labels[int(h * 0.82) : int(h * 0.98), int(w * 0.35) : int(w * 0.65)]
    ids, counts = np.unique(seeds[seeds > 0], return_counts=True)
    if not len(ids):
        return None
    label = int(ids[np.argmax(counts)])
    if stats[label, cv2.CC_STAT_AREA] < h * w * 0.015:
        return None
    component = labels == label
    left, right = np.full(h, np.nan), np.full(h, np.nan)
    previous = (w * 0.4, w * 0.6)
    for y in range(h - 1, int(h * 0.25), -1):
        spans = [(a, b) for a, b in _runs(component[y]) if b - a >= max(3, w * 0.015)]
        if not spans:
            continue
        # Follow the same branch up the image rather than the outer envelope of
        # all roads. This leaves other roads across a median disconnected.
        a, b = max(
            spans,
            key=lambda p: (
                max(0, min(p[1], previous[1]) - max(p[0], previous[0])),
                -abs((p[0] + p[1] - previous[0] - previous[1]) / 2),
            ),
        )
        overlap = min(b, previous[1]) - max(a, previous[0])
        if overlap < -w * 0.06:
            continue
        left[y], right[y] = a, b
        previous = a, b
    return component, left, right


def _curve(points, h, w):
    if len(points) < max(10, int(h * 0.025)):
        return None
    y, x = np.asarray(points, dtype=float).T
    # A row-wise nearest point can jump from the outer edge to a center line.
    # Split those identities before fitting; prefer the coherent track nearest
    # the vehicle, where ego-lane identity is least ambiguous.
    breaks = np.flatnonzero((np.abs(np.diff(x)) > max(6, w * 0.04)) | (np.diff(y) > h * 0.04)) + 1
    tracks = np.split(np.column_stack([y, x]), breaks)
    tracks = [
        track
        for track in tracks
        if len(track) >= max(10, int(h * 0.025)) and np.ptp(track[:, 0]) >= h * 0.055
    ]
    if not tracks:
        return None
    y, x = max(tracks, key=lambda t: (t[-1, 0], len(t))).T
    if np.ptp(y) < h * 0.055:
        return None
    keep = np.ones(len(y), bool)
    degree = 2 if np.ptp(y) > h * 0.12 else 1
    straight = np.polyfit(y / h, x / w, 1)
    linear_error = np.abs(x / w - np.polyval(straight, y / h))
    reliable_linear = np.quantile(linear_error, 0.90) < 0.025 and np.median(linear_error) < 0.01
    if reliable_linear:
        degree = 1
    for _ in range(3):
        if keep.sum() < 8:
            return None
        coef = np.polyfit(y[keep] / h, x[keep] / w, degree)
        residual = np.abs(x / w - np.polyval(coef, y / h))
        keep = residual < max(0.015, 3 * np.median(residual[keep]))
    if keep.sum() < max(8, len(y) * 0.5):
        return None
    lo, hi = float(y[keep].min()), float(y[keep].max())
    rows = np.arange(h, dtype=float)
    clipped = np.clip(rows, lo, hi)
    prediction = np.polyval(coef, clipped / h) * w
    # Quadratic within observed rows; bounded tangent outside them prevents
    # runaway curvature when the marking leaves the image near the vehicle.
    derivative = np.clip(np.polyval(np.polyder(coef), clipped / h) * w / h, -2.5, 2.5)
    prediction += (rows - clipped) * derivative
    prediction[rows < lo - h * 0.05] = np.nan
    if not reliable_linear:
        prediction[rows > hi + h * 0.10] = np.nan
    return prediction


class EgoLaneTracker:
    def __init__(self, max_history_frames=8):
        self.max_history_frames = max_history_frames
        self.previous = None

    def reset(self):
        self.previous = None

    def update(self, road, lanes, image=None, frame_number=None):
        height, width = road.shape
        h = min(640, height)
        w = max(32, round(width * h / height))
        small_road = cv2.resize(
            road.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        small_lanes = cv2.resize(lanes.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        gray = (
            None if image is None else cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2GRAY)
        )
        meta = {
            "status": "road_unavailable",
            "method": "curved lanes, temporal tracking and road edges",
        }
        empty = np.zeros_like(road, dtype=bool)
        path = _road_path(small_road)
        if path is None:
            self.reset()
            return empty, meta
        component, road_left, road_right = path
        # Retain the earlier straight-lane solution when it is available and
        # reaches the near-vehicle seed. Curved fallback must not replace a
        # working straight corridor with an unstable polynomial extrapolation.
        linear, linear_info = linear_lane_region(road, lanes)
        if linear_info["status"] == "estimated":
            linear_small = cv2.resize(
                linear.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            linear_small &= component
            near = linear_small[int(h * 0.82) : int(h * 0.98), int(w * 0.35) : int(w * 0.65)]
            if near.mean() > 0.15 and linear_small.sum() > component.sum() * 0.15:
                left, right = np.full(h, np.nan), np.full(h, np.nan)
                for y in range(h):
                    xs = np.flatnonzero(linear_small[y])
                    if len(xs):
                        left[y], right[y] = xs[0], xs[-1]
                self.previous = {
                    "left": left,
                    "right": right,
                    "gray": gray,
                    "frame": frame_number,
                    "age": {"left": 0, "right": 0},
                    "shape": (h, w),
                }
                full = cv2.resize(
                    linear_small.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                meta.update(
                    status="estimated",
                    method="validated straight-lane estimate",
                    boundary_sources={"left": "linear_lane", "right": "linear_lane"},
                    basis="lane_boundaries",
                    temporal_used=False,
                )
                return full & road, meta
        valid = np.isfinite(road_left) & np.isfinite(road_right)
        center = (road_left + road_right) / 2
        # Remove top/bottom artifacts and long horizontal stop-line interiors.
        small_lanes[: int(h * 0.25)] = 0
        small_lanes[int(h * 0.98) :] = 0
        horizontal = cv2.morphologyEx(
            small_lanes, cv2.MORPH_OPEN, np.ones((1, max(15, int(w * 0.20))), np.uint8)
        )
        small_lanes[horizontal > 0] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(small_lanes)
        accepted = np.zeros_like(small_lanes)
        for i in range(1, count):
            if (
                stats[i, cv2.CC_STAT_HEIGHT] >= max(8, h * 0.035)
                and stats[i, cv2.CC_STAT_AREA] >= 12
            ):
                accepted[labels == i] = 1
        points = {"left": [], "right": []}
        for y in np.flatnonzero(valid):
            xs = np.flatnonzero(accepted[y])
            xs = xs[(xs >= road_left[y] - w * 0.03) & (xs <= road_right[y] + w * 0.03)]
            left_x = xs[xs < center[y] - w * 0.03]
            right_x = xs[xs > center[y] + w * 0.03]
            if len(left_x):
                points["left"].append((y, float(left_x.max())))
            if len(right_x):
                points["right"].append((y, float(right_x.min())))
        curves = {side: _curve(samples, h, w) for side, samples in points.items()}
        sources, ages = {}, {}
        history = self._history(gray, frame_number, h, w)
        boundaries = {}
        assisted_rows = {}
        for side, edge in [("left", road_left), ("right", road_right)]:
            curve = curves[side]
            if curve is not None:
                sources[side], ages[side] = "lane_curve", 0
            elif history is not None and history["age"][side] <= self.max_history_frames:
                curve = history[side]
                sources[side], ages[side] = "tracked_lane", history["age"][side]
            else:
                curve = edge.copy()
                sources[side], ages[side] = "road_edge", self.max_history_frames + 1
            # Reject a candidate on rows where it crosses the other side of
            # the road or leaves too little road. Fall back to an observed edge
            # instead of allowing a bad fit to erase the complete corridor.
            row_width = road_right - road_left
            bad = (
                (curve > road_right - row_width * 0.15)
                if side == "left"
                else (curve < road_left + row_width * 0.15)
            )
            unavailable = ~np.isfinite(curve) | bad
            assisted_rows[side] = int(np.count_nonzero(unavailable & valid))
            if np.count_nonzero(bad & valid) > np.count_nonzero(valid) * 0.10:
                ages[side] = self.max_history_frames + 1
            curve = np.where(unavailable, edge, curve)
            # The observed road edge bounds extrapolation. Missing curve rows
            # use the road edge explicitly, rather than an unconstrained fill.
            curve = np.where(np.isfinite(curve), curve, edge)
            boundary = (
                np.maximum(curve, edge + 1) if side == "left" else np.minimum(curve, edge - 1)
            )
            boundaries[side] = boundary
        left, right = boundaries["left"], boundaries["right"]
        usable = (
            valid & np.isfinite(left) & np.isfinite(right) & (right - left >= max(3, w * 0.025))
        )
        columns = np.arange(w)[None, :]
        corridor = (
            component & usable[:, None] & (columns >= left[:, None]) & (columns <= right[:, None])
        )
        # Keep only a component reachable from the near-vehicle seed after
        # clipping, so crossing lines cannot expose a separate road island.
        n, cc, cc_stats, _ = cv2.connectedComponentsWithStats(corridor.astype(np.uint8))
        seed_labels = cc[int(h * 0.82) : int(h * 0.98), int(w * 0.35) : int(w * 0.65)]
        ids, counts = np.unique(seed_labels[seed_labels > 0], return_counts=True)
        if not len(ids):
            self.reset()
            meta["status"] = "ambiguous_boundaries"
            return empty, meta
        corridor = cc == int(ids[np.argmax(counts)])
        full = cv2.resize(
            corridor.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        result = full & road
        self.previous = {
            "left": left,
            "right": right,
            "gray": gray,
            "frame": frame_number,
            "age": ages,
            "shape": (h, w),
        }
        assisted = any(source == "road_edge" for source in sources.values()) or any(
            n > np.count_nonzero(valid) * 0.10 for n in assisted_rows.values()
        )
        meta.update(
            status="estimated",
            boundary_sources=sources,
            road_edge_assisted_rows=assisted_rows,
            basis="road_edge_assisted" if assisted else "lane_boundaries",
            temporal_used=any(s == "tracked_lane" for s in sources.values()),
            note="Road-edge fallback estimates the visible road corridor; lane identity may be ambiguous.",
        )
        return result, meta

    def _history(self, gray, frame_number, h, w):
        old = self.previous
        if old is None or old["shape"] != (h, w) or gray is None or old["gray"] is None:
            return None
        if frame_number is None or old["frame"] is None:
            return None
        gap = frame_number - old["frame"]
        if gap < 1 or gap > self.max_history_frames:
            return None
        # Compare coarse appearance so ordinary pixel motion is not mistaken
        # for a cut; optical-flow consistency below rejects unrelated frames.
        coarse = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        old_coarse = cv2.resize(old["gray"], (32, 32), interpolation=cv2.INTER_AREA)
        if np.mean(np.abs(coarse.astype(float) - old_coarse.astype(float))) > 40:
            return None
        features = cv2.goodFeaturesToTrack(
            old["gray"], maxCorners=100, qualityLevel=0.02, minDistance=10
        )
        if features is None or len(features) < 8:
            return None
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(old["gray"], gray, features, None)
        if moved is None or ok is None or ok.sum() < 8:
            return None
        displacement = (moved - features)[ok.ravel() > 0, 0]
        dx, dy = np.median(displacement, axis=0)
        if abs(dx) > w * 0.08 or abs(dy) > h * 0.08:
            return None
        if np.median(np.linalg.norm(displacement - [dx, dy], axis=1)) > 6:
            return None
        rows = np.arange(h, dtype=float)
        return {
            side: np.interp(rows - dy, rows, old[side], left=np.nan, right=np.nan) + dx
            for side in ("left", "right")
        } | {"age": {s: old["age"][s] + gap for s in ("left", "right")}}


def restrict_to_ego_lane(road, lanes):
    """Stateless compatibility wrapper; use EgoLaneTracker for video history."""
    return EgoLaneTracker().update(road, lanes)
