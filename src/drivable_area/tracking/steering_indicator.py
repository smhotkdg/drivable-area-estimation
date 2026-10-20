"""Smooth a visual steering-wheel indicator from IMAGE bearing only.

No wheelbase, speed or camera calibration is available. This module never
produces physical steering angles or vehicle commands.
"""

import math
import numpy as np


class ImageSteeringIndicator:
    def __init__(
        self,
        fps,
        smoothing_seconds=0.18,
        max_rate_deg_per_second=90.0,
        max_angle_deg=60.0,
        expiry_seconds=0.5,
    ):
        if (
            fps <= 0
            or min(smoothing_seconds, max_rate_deg_per_second, max_angle_deg, expiry_seconds) <= 0
        ):
            raise ValueError("FPS and steering display limits must be positive")
        self.fps = fps
        self.smoothing = smoothing_seconds
        self.max_rate = max_rate_deg_per_second
        self.max_angle = max_angle_deg
        self.expiry = expiry_seconds
        self.frame = -1
        self.last_observation = None
        self.target = 0.0
        self.angle = 0.0

    def update(self, evidence, direction, observed=False):
        self.frame += 1
        now = self.frame / self.fps
        label = direction.get("direction", "unknown")
        bearing = evidence.get("far_bearing_deg")
        accepted = (
            observed
            and label != "unknown"
            and not direction.get("stale", False)
            and evidence.get("direction") == label
            and bearing is not None
            and math.isfinite(bearing)
        )
        if accepted:
            self.target = float(np.clip(bearing, -self.max_angle, self.max_angle))
            self.last_observation = now
        available = (
            self.last_observation is not None
            and now - self.last_observation <= self.expiry
            and label != "unknown"
            and not direction.get("stale", False)
        )
        if not available:
            self.target = 0.0
        alpha = 1 - math.exp(-1 / (self.fps * self.smoothing))
        limit = self.max_rate / self.fps
        self.angle += float(np.clip((self.target - self.angle) * alpha, -limit, limit))
        pending = bool(direction.get("pending_direction"))
        return {
            "wheel_rotation_deg": round(self.angle, 3),
            "image_bearing_deg": round(self.angle, 1) if available else None,
            "available": available,
            "status": "unavailable"
            if not available
            else "held"
            if pending
            else "observed"
            if accepted
            else "tracking",
            "new_angle_observation": bool(accepted),
            "note": "Image-space bearing visualization only; NOT physical steering angle or a vehicle command.",
        }
