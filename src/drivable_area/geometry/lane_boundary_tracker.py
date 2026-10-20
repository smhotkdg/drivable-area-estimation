"""Track longitudinal marking boundaries separately from the road/path.

Road-mask edges are labelled as road estimates, never as observed curbs or
verified ego-lane boundaries. Observed/tracked markings are hard constraints.
"""

import cv2
import numpy as np
from drivable_area.geometry.lane_roi import _road_path


def marking_boundaries(road, lanes):
    h, w = road.shape
    marks = np.asarray(lanes, np.uint8).copy()
    marks &= cv2.dilate(road.astype(np.uint8), np.ones((7, 7), np.uint8))
    marks[: round(h * 0.30)] = 0
    marks[round(h * 0.98) :] = 0
    horizontal = cv2.morphologyEx(
        marks, cv2.MORPH_OPEN, np.ones((1, max(15, round(w * 0.12))), np.uint8)
    )
    marks[horizontal > 0] = 0
    segments = cv2.HoughLinesP(
        marks * 255,
        1,
        np.pi / 180,
        threshold=max(10, round(h * 0.025)),
        minLineLength=max(12, round(h * 0.045)),
        maxLineGap=round(h * 0.04),
    )
    boundaries = {s: np.full(h, np.nan) for s in ("left", "right")}
    if segments is None:
        return boundaries
    anchor = h * 0.72
    groups = {"left": [], "right": []}
    for x1, y1, x2, y2 in segments[:, 0].astype(float):
        if abs(y2 - y1) < h * 0.045 or abs(y2 - y1) < 0.7 * abs(x2 - x1):
            continue
        slope = (x2 - x1) / (y2 - y1)
        intercept = x1 - slope * y1
        at = slope * anchor + intercept
        if abs(at - w * 0.5) < w * 0.035 or not -w < at < 2 * w:
            continue
        side = "left" if at < w * 0.5 else "right"
        groups[side].append((at, slope, intercept, min(y1, y2), max(y1, y2)))
    rows = np.arange(h)
    for side, candidates in groups.items():
        if not candidates:
            continue
        # Require support before selecting the closest identity. A short spur
        # at a crosswalk must not beat a supported longitudinal marking.
        clusters = []
        for seed in candidates:
            selected = [
                c
                for c in candidates
                if abs(c[0] - seed[0]) < w * 0.06 and abs(c[1] - seed[1]) < 0.35
            ]
            lo = min(c[3] for c in selected)
            hi = max(c[4] for c in selected)
            if hi - lo < h * 0.08:
                continue
            clusters.append((seed, selected, lo, hi))
        if not clusters:
            continue
        seed, selected, lo, hi = min(clusters, key=lambda c: abs(c[0][0] - w * 0.5))
        slope = float(np.median([c[1] for c in selected]))
        intercept = float(np.median([c[2] for c in selected]))
        prediction = slope * rows + intercept
        supported = (rows >= lo - h * 0.025) & (rows <= hi + h * 0.16)
        boundaries[side][supported] = prediction[supported]
    return boundaries


def _warp_boundary(values, transform):
    if transform is None:
        return values.copy()
    ys = np.flatnonzero(np.isfinite(values))
    result = np.full_like(values, np.nan)
    if len(ys) < 2:
        return result
    xy = np.column_stack([values[ys], ys, np.ones(len(ys))]) @ transform.T
    order = np.argsort(xy[:, 1])
    xy = xy[order]
    rows = np.arange(len(values))
    valid = (rows >= xy[0, 1]) & (rows <= xy[-1, 1])
    result[valid] = np.interp(rows[valid], xy[:, 1], xy[:, 0])
    return result


class LaneBoundaryTracker:
    def __init__(self, fps=15, expiry_seconds=0.6):
        self.expiry = max(1, round(fps * expiry_seconds))
        self.history = None
        self.ages = None
        self.last_bounds = None
        self.width_history = None
        self.width_age = None

    def update(self, road, lanes=None, transform=None, detected=True):
        road = np.asarray(road, bool)
        h, w = road.shape
        road_data = _road_path(road)
        if road_data is None:
            self.history = None
            self.last_bounds = None
            self.width_history = None
            return np.zeros_like(road), {"status": "no_road", "sources": {}, "marking_rows": 0}
        component, road_left, road_right = road_data
        if self.ages is None or len(self.ages["left"]) != h:
            self.ages = {s: np.full(h, self.expiry + 1, dtype=float) for s in ("left", "right")}
        observed = (
            marking_boundaries(component, lanes)
            if detected and lanes is not None
            else {s: np.full(h, np.nan) for s in ("left", "right")}
        )
        bounds = {}
        sources = {}
        mark_rows = {}
        fresh_rows = {}
        used_history = False
        for side, edge in [("left", road_left), ("right", road_right)]:
            measured = observed[side]
            valid = np.isfinite(measured) & np.isfinite(edge)
            # A marking must be inside the current connected road and leave
            # room on its opposite side. No invalid fit silently replaces it.
            valid &= (measured >= road_left - w * 0.02) & (measured <= road_right + w * 0.02)
            valid &= (
                measured < road_right - w * 0.05
                if side == "left"
                else measured > road_left + w * 0.05
            )
            measured = np.where(valid, measured, np.nan)
            raw = measured.copy()
            fresh_rows[side] = np.isfinite(raw)
            tracked = np.full(h, np.nan)
            tracked_age = np.full(h, self.expiry + 1, dtype=float)
            if self.history is not None and len(self.history[side]) == h:
                tracked = _warp_boundary(self.history[side], transform)
                age_values = np.where(np.isfinite(self.history[side]), self.ages[side], np.nan)
                tracked_age = _warp_boundary(age_values, transform) + 1
                tracked[~np.isfinite(tracked_age) | (tracked_age > self.expiry)] = np.nan
            if np.isfinite(measured).sum() >= h * 0.04:
                # Fuse only compatible identities, retaining the current
                # observation as a hard bound even when smoothing history.
                compatible = (
                    np.isfinite(measured)
                    & np.isfinite(tracked)
                    & (abs(measured - tracked) < w * 0.06)
                )
                measured[compatible] = 0.7 * measured[compatible] + 0.3 * tracked[compatible]
                curve = np.where(np.isfinite(measured), measured, tracked)
                sources[side] = "marking"
            elif np.isfinite(tracked).sum() >= h * 0.04:
                curve = tracked
                sources[side] = "tracked_marking"
                used_history = True
            else:
                curve = np.full(h, np.nan)
                sources[side] = "road_edge_estimate"
            # Preserve current observed boundary as an additional constraint:
            # smoothing cannot move the permitted region across a marking.
            curve = np.where(
                np.isfinite(raw),
                np.fmax(curve, raw) if side == "left" else np.fmin(curve, raw),
                curve,
            )
            self.ages[side] = np.where(np.isfinite(raw), 0, tracked_age)
            mark_rows[side] = np.isfinite(curve)
            bounds[side] = (
                np.fmax(edge + 2, curve + 2) if side == "left" else np.fmin(edge - 2, curve - 2)
            )
            observed[side] = curve
        # Infer a temporarily missing side only from a recently observed pair.
        # The inferred side is labelled separately and never called observed.
        both = np.isfinite(observed["left"]) & np.isfinite(observed["right"])
        if self.width_history is None or len(self.width_history) != h:
            self.width_history = np.full(h, np.nan)
            self.width_age = np.full(h, self.expiry + 1, dtype=float)
        else:
            if transform is not None and abs(transform[1, 1]) > 0.5:
                rows = np.arange(h)
                source = (rows - transform[1, 2]) / transform[1, 1]
                self.width_history = np.interp(
                    source, rows, self.width_history, left=np.nan, right=np.nan
                ) * abs(transform[0, 0])
                self.width_age = np.interp(
                    source, rows, self.width_age, left=self.expiry + 1, right=self.expiry + 1
                )
            self.width_age += 1
        widths = observed["right"] - observed["left"]
        usable = (
            both & fresh_rows["left"] & fresh_rows["right"] & (widths > w * 0.06) & (widths < w)
        )
        self.width_history[usable] = widths[usable]
        self.width_age[usable] = 0
        for side, other in [("left", "right"), ("right", "left")]:
            missing = (
                ~np.isfinite(observed[side])
                & np.isfinite(observed[other])
                & np.isfinite(self.width_history)
                & (self.width_age <= self.expiry)
            )
            inferred = observed[other] + (-1 if side == "left" else 1) * self.width_history
            if side == "left":
                bounds[side][missing] = np.fmax(bounds[side][missing], inferred[missing] + 2)
            else:
                bounds[side][missing] = np.fmin(bounds[side][missing], inferred[missing] - 2)
            if missing.sum() >= h * 0.04 and sources[side] == "road_edge_estimate":
                sources[side] = "inferred_width"
        left, right = bounds["left"], bounds["right"]
        valid = np.isfinite(left) & np.isfinite(right) & (right - left >= max(5, w * 0.035))
        columns = np.arange(w)[None, :]
        corridor = (
            component & valid[:, None] & (columns >= left[:, None]) & (columns <= right[:, None])
        )
        # Keep the component containing the near-camera origin.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(corridor.astype(np.uint8))
        seeds = labels[round(h * 0.82) : round(h * 0.98), round(w * 0.4) : round(w * 0.6)]
        ids, counts = np.unique(seeds[seeds > 0], return_counts=True)
        corridor &= labels == int(ids[np.argmax(counts)]) if len(ids) else False
        self.history = observed
        self.last_bounds = {
            "left": left,
            "right": right,
            "marking_left": mark_rows["left"],
            "marking_right": mark_rows["right"],
        }
        return corridor, {
            "status": "bounded" if corridor.any() else "boundary_conflict",
            "sources": sources,
            "marking_rows": {s: int(mark_rows[s].sum()) for s in mark_rows},
            "temporal_used": used_history,
            "ego_lane_verified": False,
            "note": "Observed/tracked markings constrain the route. Road edges are estimates; ego-lane identity remains unverified.",
        }


def boundary_center_path(mask, count=28):
    """One contiguous center track within the bounded corridor, with no cuts."""
    from drivable_area.geometry.continuation_path import continuation_center_path

    return continuation_center_path(mask, count=count)


class BoundaryDirectionObserver:
    """A bounded route observation independent of the smoothed path tracker."""

    def __init__(self, fps=15, trim_route=False):
        self.expiry = max(1, round(fps * 0.5))
        self.path = None
        self.anchor = None
        self.age = 0
        self.trim_route = trim_route

    def update(self, mask, measurement=None, detected=False, transform=None):
        from drivable_area.geometry.corridor_path import path_inside
        from drivable_area.tracking.route_intent import curve_evidence

        h, w = mask.shape
        if detected:
            self.path = None if measurement is None else measurement.copy()
            self.anchor = None if self.path is None else self.path[0].copy()
            self.age = 0
        else:
            self.age += 1
            if self.path is not None and transform is not None:
                self.path = np.column_stack([self.path, np.ones(len(self.path))]) @ transform.T
                weights = (1 - np.linspace(0, 1, len(self.path))) ** 3
                self.path += weights[:, None] * (self.anchor - self.path[0])
        observed = self.path
        if self.trim_route:
            from drivable_area.geometry.single_lane_center import supported_prefix

            observed = supported_prefix(self.path, mask)
        valid = observed is not None and self.age <= self.expiry and path_inside(observed, mask)
        result = curve_evidence(observed, w, h, confirmed=valid)
        result["supported_points"] = 0 if observed is None else len(observed)
        result.update(
            source="boundary_observation" if detected else "boundary_prediction",
            age_frames=self.age,
        )
        return result
