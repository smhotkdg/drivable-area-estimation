"""Estimate an ego-lane corridor from detected lane markings.

This is a geometric postprocess, not a trained ego-lane model. Missing or
ambiguous boundaries return an empty region rather than unconstrained road.
"""

import cv2
import numpy as np


def restrict_to_ego_lane(road, lanes):
    height, width = road.shape
    small_h = min(640, height)
    small_w = max(32, round(width * small_h / height))
    markings = cv2.resize(
        lanes.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST
    )
    road_small = cv2.resize(
        road.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST
    )
    # Suppress image-border artifacts, sky, and lane detections far from road.
    margin = max(3, round(small_w * 0.025))
    near_road = cv2.dilate(road_small, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    markings &= near_road
    markings[: round(small_h * 0.30)] = 0
    markings[round(small_h * 0.98) :] = 0
    segments = cv2.HoughLinesP(
        markings * 255,
        1,
        np.pi / 180,
        threshold=max(12, round(small_h * 0.035)),
        minLineLength=max(12, round(small_h * 0.065)),
        maxLineGap=max(5, round(small_h * 0.025)),
    )
    metadata = {"status": "boundaries_unavailable", "method": "lane-segment linear fit"}
    empty = np.zeros_like(road, dtype=bool)
    if segments is None:
        return empty, metadata
    # Select the pair surrounding the image center where the road first becomes
    # broad enough to establish a lane. The camera is assumed to face forward.
    road_rows = np.flatnonzero(road_small.sum(axis=1) > small_w * 0.25)
    if not len(road_rows):
        metadata["status"] = "road_unavailable"
        return empty, metadata
    anchor_y = float(np.clip(road_rows[0] + small_h * 0.08, small_h * 0.45, small_h * 0.70))
    groups = {"left": [], "right": []}
    center = small_w * 0.5
    for line in segments[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        dx, dy = x2 - x1, y2 - y1
        # Horizontal stop lines and crosswalks are not longitudinal boundaries.
        if abs(dy) < small_h * 0.055 or abs(dy) < 0.7 * abs(dx):
            continue
        slope = dx / dy
        intercept = x1 - slope * y1
        x_anchor = slope * anchor_y + intercept
        if not -0.15 * small_w <= x_anchor <= 1.15 * small_w:
            continue
        if abs(x_anchor - center) < small_w * 0.04:
            continue
        side = "left" if x_anchor < center else "right"
        groups[side].append(
            {
                "x": x_anchor,
                "slope": slope,
                "intercept": intercept,
                "ymin": min(y1, y2),
                "ymax": max(y1, y2),
                "length": abs(dy),
            }
        )
    boundaries = {}
    min_y = []
    for side, candidates in groups.items():
        if not candidates:
            metadata["missing_side"] = side
            return empty, metadata
        # Use the nearest marking on each side; group duplicates from the same
        # thick marking rather than averaging neighboring lanes together.
        nearest = (
            max(candidates, key=lambda c: c["x"])
            if side == "left"
            else min(candidates, key=lambda c: c["x"])
        )
        selected = [
            c
            for c in candidates
            if abs(c["x"] - nearest["x"]) < small_w * 0.045
            and abs(c["slope"] - nearest["slope"]) < 0.35
        ]
        span = max(c["ymax"] for c in selected) - min(c["ymin"] for c in selected)
        if span < small_h * 0.08:
            metadata["missing_side"] = side
            return empty, metadata
        slope = float(np.median([c["slope"] for c in selected]))
        intercept = float(np.median([c["intercept"] for c in selected]))
        boundaries[side] = (slope, intercept)
        min_y.append(min(c["ymin"] for c in selected))
    rows = np.arange(height, dtype=np.float64)
    left = (
        (boundaries["left"][0] * rows * small_h / height + boundaries["left"][1]) * width / small_w
    )
    right = (
        (boundaries["right"][0] * rows * small_h / height + boundaries["right"][1])
        * width
        / small_w
    )
    # Do not expand above observed markings or through crossing boundaries.
    y_start = int(np.ceil(max(min_y) * height / small_h))
    valid_rows = (rows >= y_start) & (right - left > width * 0.08)
    bottom = height - 1
    if not (left[bottom] < width * 0.5 < right[bottom]) or not valid_rows[bottom]:
        metadata["status"] = "ambiguous_boundaries"
        return empty, metadata
    columns = np.arange(width)[None, :]
    corridor = (columns > left[:, None]) & (columns < right[:, None]) & valid_rows[:, None]
    restricted = road & corridor
    metadata.update(
        status="estimated",
        anchor_y=round(anchor_y * height / small_h),
        left_x_at_bottom=round(float(left[-1]), 1),
        right_x_at_bottom=round(float(right[-1]), 1),
        first_valid_row=y_start,
        note="Straight-line extrapolation; centered forward-facing camera assumed.",
    )
    return restricted, metadata
