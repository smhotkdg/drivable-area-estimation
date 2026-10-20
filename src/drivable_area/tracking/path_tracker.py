import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from drivable_area.geometry.corridor_path import connected_center_path, path_inside
from drivable_area.geometry.continuation_path import continuation_center_path
from drivable_area.geometry.lane_boundary_tracker import boundary_center_path
from drivable_area.geometry.single_lane_center import (
    lane_center_path,
    center_profile,
    supported_prefix,
)


class PathTracker:
    def __init__(self, width, height, fps, max_missing_seconds=1.0, planner="medial"):
        self.w, self.h, self.dt = width, height, 1.0 / fps
        self.count = 28
        self.ys = np.linspace(height * 0.94, height * 0.44, self.count)
        self.state = None
        self.covariance = None
        self.valid = np.zeros(self.count, bool)
        self.missing = 0
        self.max_missing = round(fps * max_missing_seconds)
        self.angle = None
        self.rejected = 0
        self.conflict = False
        self.pending = None
        self.planner = planner
        self.near_anchor = None

    def measurement(self, mask, lane_region=None):
        if self.planner == "lane-center":
            path, self.path_info = lane_center_path(mask, self.count)
            self.direction_measurement, _ = lane_center_path(mask, 48, max_distance_fraction=None)
        elif self.planner == "boundaries":
            path, self.path_info = boundary_center_path(mask, self.count)
        elif self.planner == "continuation":
            path, self.path_info = continuation_center_path(mask, lane_region, self.count)
        else:
            path, self.path_info = connected_center_path(mask, self.count)
        return path

    def _initialize(self, measured):
        self.state = np.column_stack([measured, np.zeros_like(measured)])
        self.covariance = np.tile(np.diag([25.0, 25.0, 100.0, 100.0]), (self.count, 1, 1))
        self.valid[:] = True
        self.missing = 0
        self.angle = None
        self.conflict = False
        self.pending = None
        self.near_anchor = measured[0].copy()

    def _anchor_near_route(self, mask):
        # A corridor path is redefined relative to the camera each frame. Its
        # start is not a fixed patch of asphalt that should flow off the image.
        if (
            self.planner not in ("continuation", "boundaries", "lane-center")
            or self.near_anchor is None
        ):
            return
        x, y = np.rint(self.near_anchor).astype(int)
        if not (0 <= x < self.w and 0 <= y < self.h and mask[y, x]):
            return
        weight = (1.0 - np.linspace(0.0, 1.0, self.count)) ** 3
        self.state[:, :2] += weight[:, None] * (self.near_anchor - self.state[0, :2])
        self.state[:, 2:] *= 1.0 - weight[:, None]

    def _correct(self, measured, variance, gate):
        innovations = measured - self.state[:, :2]
        good = np.isfinite(measured).all(axis=1) & (np.linalg.norm(innovations, axis=1) <= gate)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        for i in np.flatnonzero(good):
            p = self.covariance[i]
            gain = (
                p
                @ observation.T
                @ np.linalg.inv(observation @ p @ observation.T + np.eye(2) * variance)
            )
            self.state[i] += gain @ innovations[i]
            self.covariance[i] = (np.eye(4) - gain @ observation) @ p
        self.rejected += int((~good).sum())
        return good

    def update(self, mask, detected=False, transform=None, lane_region=None):
        measured = (
            (
                self.measurement(mask, lane_region)
                if self.planner == "continuation"
                else self.measurement(mask)
            )
            if detected
            else None
        )
        if detected:
            self.last_measurement = measured
        if self.state is None:
            if measured is None:
                return None, {"status": "lost", "image_angle_deg": None}
            self._initialize(measured)
            mode = "detected"
        else:
            old_position = self.state[:, :2].copy()
            transition = np.eye(4)
            transition[0, 2] = transition[1, 3] = self.dt
            self.state = self.state @ transition.T
            self.covariance = transition @ self.covariance @ transition.T + np.diag(
                [0.5, 0.5, 8.0, 8.0]
            )
            if transform is not None:
                flow = np.column_stack([old_position, np.ones(self.count)]) @ transform.T
                self._correct(flow, variance=9.0, gate=self.w * 0.08)
            self._anchor_near_route(mask)
            self.missing += 1
            mode = "optical-flow" if transform is not None else "prediction"
            if detected:
                error = (
                    np.linalg.norm(measured - self.state[:, :2], axis=1)
                    if measured is not None
                    else None
                )
                compatible = (
                    error is not None
                    and np.median(error) <= self.w * 0.12
                    and np.mean(error > self.w * 0.12) <= 0.4
                )
                if compatible:
                    self._correct(measured, variance=16.0, gate=self.w * 0.12)
                    self.missing, self.conflict, self.pending = 0, False, None
                    mode = "corrected"
                else:
                    self.rejected += self.count
                    self.conflict = True
                    mode = "path-conflict"
                    if (
                        measured is not None
                        and self.pending is not None
                        and np.median(np.linalg.norm(measured - self.pending, axis=1))
                        < self.w * 0.08
                    ):
                        self._initialize(measured)
                        mode = "reacquired"
                    else:
                        self.pending = None if measured is None else measured.copy()
            if mode != "reacquired":
                self.state[:, :2] = old_position + np.clip(
                    self.state[:, :2] - old_position, -self.w * 0.018, self.w * 0.018
                )
            self.state[:, 2:] = np.clip(self.state[:, 2:], -self.w * 0.25, self.w * 0.25)
            self._anchor_near_route(mask)
            if self.missing > self.max_missing:
                self.state = None
                self.valid[:] = False
                self.angle = None
                return None, {"status": "lost", "image_angle_deg": None}
        if self.conflict:
            self.angle = None
            if self.pending is not None:
                candidate = self.pending.copy()
                if not detected and transform is not None:
                    candidate = np.column_stack([candidate, np.ones(self.count)]) @ transform.T
                    if (
                        self.planner in ("boundaries", "lane-center")
                        and self.near_anchor is not None
                    ):
                        weights = (1 - np.linspace(0, 1, self.count)) ** 3
                        candidate += weights[:, None] * (self.near_anchor - candidate[0])
                    self.pending = candidate
                if path_inside(candidate, mask):
                    points = np.rint(candidate).astype(np.int32)
                    target = points[round((len(points) - 1) * 0.65)]
                    return (points, np.ones(len(points), bool), target), {
                        "status": "unconfirmed-route",
                        "image_angle_deg": None,
                    }
                if self.planner == "lane-center":
                    prefix = supported_prefix(candidate, mask)
                    if (
                        prefix is not None
                        and len(prefix) >= 7
                        and np.linalg.norm(prefix[-1] - prefix[0]) >= self.h * 0.12
                    ):
                        points = np.rint(prefix).astype(np.int32)
                        if path_inside(points, mask):
                            target = points[round((len(points) - 1) * 0.65)]
                            return (points, np.ones(len(points), bool), target), {
                                "status": "unconfirmed-prefix",
                                "image_angle_deg": None,
                            }
            return None, {"status": "path-conflict", "image_angle_deg": None}
        # One arc-length-ordered route is tracked in 2D. No independent
        # horizontal rows can jump to different branches of an intersection.
        smooth = gaussian_filter1d(self.state[:, :2], sigma=1.0, axis=0, mode="nearest")
        points = np.rint(smooth).astype(np.int32)
        if self.planner in ("boundaries", "lane-center"):
            # Constrained correction for small filter drift: a smoothed path
            # may approach an observed marking but cannot cross it. Large
            # changes still require reacquisition instead of being projected.
            projected = points.copy()
            for i, (x, y) in enumerate(points):
                if 0 <= y < self.h:
                    xs = np.flatnonzero(mask[y])
                    if len(xs):
                        nearest = int(xs[np.argmin(abs(xs - x))])
                        if abs(nearest - x) <= self.w * 0.06:
                            projected[i, 0] = nearest
            if path_inside(projected, mask):
                points = projected
                self.state[:, :2] = points
        if self.planner == "lane-center":
            profile = center_profile(mask)
            if profile is not None:
                _, left, right, center = profile
                for i, (x, y) in enumerate(points):
                    if 0 <= y < self.h * 0.84 and np.isfinite(center[y]):
                        margin = (right[y] - left[y]) * 0.04
                        points[i, 0] = round(np.clip(x, center[y] - margin, center[y] + margin))
                if path_inside(points, mask):
                    self.state[:, :2] = points
        # An uncertain far endpoint should not erase a verified near route.
        # Keep only the continuously connected prefix, never disconnected dots.
        first = 0
        while first < min(7, len(points)) and not path_inside(points[first : first + 1], mask):
            first += 1
        prefix = first
        for i in range(first, len(points)):
            if not path_inside(points[max(first, i - 1) : i + 1], mask):
                break
            prefix = i + 1
        if prefix - first < 7 or np.linalg.norm(points[prefix - 1] - points[first]) < self.h * 0.12:
            self.angle = None
            if detected and measured is not None and path_inside(measured, mask):
                points = np.rint(measured).astype(np.int32)
                target = points[round((len(points) - 1) * 0.65)]
                return (points, np.ones(len(points), bool), target), {
                    "status": "unconfirmed-route",
                    "image_angle_deg": None,
                }
            return None, {"status": "path-outside-mask", "image_angle_deg": None}
        points = points[first:prefix]
        target = points[round((len(points) - 1) * 0.65)]
        raw_angle = math.degrees(math.atan2(target[0] - self.w * 0.5, self.h * 0.96 - target[1]))
        if self.angle is None:
            self.angle = raw_angle
        else:
            self.angle += float(np.clip(0.25 * (raw_angle - self.angle), -2.0, 2.0))
        return (points, np.ones(len(points), bool), target), {
            "status": mode,
            "image_angle_deg": round(self.angle, 2),
            "target_pixel": target.tolist(),
            "visible_points": len(points),
            "path_method": self.planner + " route, 2D tracking by arc length",
            "rejected_measurements_total": self.rejected,
        }


def motion(previous, current, mask):
    if previous is None or mask is None:
        return None, 0
    # Only road texture contributes; do not track the sky or nearby buildings.
    features = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=140,
        qualityLevel=0.015,
        minDistance=6,
        mask=(mask.astype(np.uint8) * 255),
    )
    if features is None or len(features) < 8:
        return None, 0
    moved, ok, errors = cv2.calcOpticalFlowPyrLK(
        previous, current, features, None, winSize=(21, 21), maxLevel=3
    )
    if moved is None:
        return None, 0
    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, moved, None, winSize=(21, 21), maxLevel=3
    )
    if back is None:
        return None, 0
    good = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (errors.ravel() < 25)
    good &= np.linalg.norm(back[:, 0] - features[:, 0], axis=1) < 1.5
    if good.sum() < 8:
        return None, int(good.sum())
    transform, inliers = cv2.estimateAffinePartial2D(
        features[good, 0],
        moved[good, 0],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=500,
    )
    if transform is None or inliers.sum() < 8 or inliers.mean() < 0.5:
        return None, 0
    scale = math.hypot(transform[0, 0], transform[1, 0])
    rotation = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    h, w = current.shape
    if (
        not 0.92 < scale < 1.08
        or abs(rotation) > 8
        or abs(transform[0, 2]) > w * 0.12
        or abs(transform[1, 2]) > h * 0.12
    ):
        return None, 0
    return transform, int(inliers.sum())
