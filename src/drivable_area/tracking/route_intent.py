"""Direction-only readout from a visible route; not navigation intent inference.

No reference labels or reference curves are accepted by this module. Thresholds
are image-space heuristics for this experiment, not physical steering limits.
"""

import math
import numpy as np


def route_evidence(points, width, height, confirmed=True):
    if not confirmed or points is None or len(points) < 5:
        return {"direction": "unknown", "reason": "unconfirmed_path", "far_bearing_deg": None}
    points = np.asarray(points, dtype=float)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if length < height * 0.30 or points[0, 1] - points[-1, 1] < height * 0.15:
        return {
            "direction": "unknown",
            "reason": "insufficient_lookahead",
            "far_bearing_deg": None,
            "length_px": round(length, 1),
        }
    # The far visible route describes a coming bend better than a single
    # near-field center point. This still describes route appearance only.
    end = np.median(points[-3:], axis=0)
    bearing = math.degrees(math.atan2(end[0] - width * 0.5, height * 0.96 - end[1]))
    direction = "left" if bearing < -10 else "right" if bearing > 10 else "straight"
    return {
        "direction": direction,
        "reason": "route_geometry",
        "far_bearing_deg": round(bearing, 2),
        "length_px": round(length, 1),
    }


class IntentTracker:
    def __init__(self, fps, confirmations=3, expiry_seconds=0.5):
        self.fps = fps
        self.confirmations = confirmations
        self.expiry_frames = max(1, round(expiry_seconds * fps))
        self.label = "unknown"
        self.pending = None
        self.count = 0
        self.missing = 0

    def update(self, evidence):
        candidate = evidence["direction"]
        if candidate == "unknown":
            self.missing += 1
            self.pending, self.count = None, 0
            if self.missing > self.expiry_frames:
                self.label = "unknown"
            # A held label is explicitly NOT a fresh prediction for scoring.
            return {
                "direction": self.label,
                "fresh": False,
                "state": "held" if self.label != "unknown" else "abstain",
            }
        self.missing = 0
        angle = evidence.get("decision_angle_deg", evidence["far_bearing_deg"])
        if self.label == "left" and angle < -6:
            candidate = "left"
        elif self.label == "right" and angle > 6:
            candidate = "right"
        if candidate == self.label:
            self.pending, self.count = None, 0
            return {"direction": self.label, "fresh": True, "state": "stable"}
        if self.pending == candidate:
            self.count += 1
        else:
            self.pending, self.count = candidate, 1
        if self.count >= self.confirmations:
            self.label = candidate
            self.pending, self.count = None, 0
            return {"direction": self.label, "fresh": True, "state": "confirmed"}
        return {"direction": self.label, "fresh": False, "state": "pending"}


def curve_evidence(points, width, height, confirmed=True):
    """Combine displacement with robust near/far tangent change (image only)."""
    base = route_evidence(points, width, height, confirmed)
    if base["direction"] == "unknown":
        return base
    points = np.asarray(points, dtype=float)
    # Average chords over a quarter of the arc, rather than differentiating
    # adjacent noisy dots. Positive heading means a rightward image direction.
    k = max(2, len(points) // 4)
    near = points[k] - points[0]
    far = points[-1] - points[-1 - k]
    near_angle = math.degrees(math.atan2(near[0], -near[1]))
    far_angle = math.degrees(math.atan2(far[0], -far[1]))
    change = (far_angle - near_angle + 180) % 360 - 180
    displacement = points[-1, 0] - points[0, 0]
    bearing = base["far_bearing_deg"]
    curved = (
        abs(change) > 12
        and abs(displacement) > width * 0.04
        and change * displacement > 0
        and change * bearing >= 0
    )
    decision = bearing
    if curved and abs(change) > abs(bearing):
        decision = change
    base.update(
        near_tangent_deg=round(near_angle, 2),
        far_tangent_deg=round(far_angle, 2),
        tangent_change_deg=round(change, 2),
        decision_angle_deg=round(decision, 2),
        direction="left" if decision < -10 else "right" if decision > 10 else "straight",
        reason="curve_and_displacement" if curved else "route_geometry",
    )
    return base
