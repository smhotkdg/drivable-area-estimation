"""Temporal observation votes and a directional arrow display.

Weights are empirical relative support, NOT calibrated correctness
probabilities. Optical-flow predictions never add new detection votes.
Adjacent real detections may still be temporally correlated.
"""

import math
from collections import deque
import numpy as np

DIRECTIONS = ("left", "straight", "right")
KOREAN = {"left": "좌회전", "straight": "직진", "right": "우회전", "unknown": "판단 대기"}


class DirectionVoteTracker:
    def __init__(
        self,
        fps,
        window_seconds=1.0,
        half_life_seconds=0.35,
        min_consecutive=3,
        enter_weight=0.60,
        switch_margin=0.20,
        expiry_seconds=0.5,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = fps
        self.window = window_seconds
        self.half_life = half_life_seconds
        self.min_consecutive = min_consecutive
        self.enter = enter_weight
        self.margin = switch_margin
        self.expiry = expiry_seconds
        self.history = deque()
        self.frame = -1
        self.label = "unknown"
        self.last_valid = None
        self.raw = "unknown"
        self.run = 0
        self.total_observations = 0
        self.display_weights = np.zeros(3, float)
        self.last_time = None

    def update(self, evidence, observed=False):
        self.frame += 1
        now = self.frame / self.fps
        raw = evidence.get("direction", "unknown")
        if raw not in DIRECTIONS:
            raw = "unknown"
        if observed:
            self.total_observations += 1
            self.history.append((now, raw))
            self.run = (
                self.run + 1
                if raw == self.raw and raw != "unknown"
                else 1
                if raw != "unknown"
                else 0
            )
            self.raw = raw
            if raw != "unknown":
                self.last_valid = now
        while self.history and now - self.history[0][0] > self.window:
            self.history.popleft()
        scores = np.zeros(3, float)
        for at, label in self.history:
            if label in DIRECTIONS:
                scores[DIRECTIONS.index(label)] += 2 ** (-(now - at) / self.half_life)
        total = float(scores.sum())
        weights = scores / total if total else scores
        winner = DIRECTIONS[int(np.argmax(weights))] if total else "unknown"
        changed = False
        if observed and winner != "unknown" and raw == winner and self.run >= self.min_consecutive:
            candidate_weight = weights[DIRECTIONS.index(winner)]
            runner_up = float(np.partition(weights, -2)[-2])
            if candidate_weight >= self.enter and candidate_weight - runner_up >= self.margin:
                changed = self.label != winner
                self.label = winner
        stale = self.last_valid is None or now - self.last_valid > self.expiry
        if stale:
            self.label = "unknown"
        dt = 1 / self.fps if self.last_time is None else now - self.last_time
        self.last_time = now
        # Only the visual lengths are eased. Decisions use actual vote weights.
        target = weights if not stale else np.zeros(3)
        alpha = 1 - math.exp(-dt / 0.12)
        self.display_weights += (target - self.display_weights) * alpha
        agreement = (
            sum(label == self.label for _, label in self.history) if self.label in DIRECTIONS else 0
        )
        held = self.label != "unknown" and (
            self.raw == "unknown"
            or not observed
            and self.last_valid is not None
            and now - self.last_valid > 0.15
        )
        pending = self.raw if self.raw in DIRECTIONS and self.raw != self.label else None
        state = (
            "waiting"
            if self.label == "unknown"
            else "held"
            if held
            else "checking"
            if pending
            else "confirmed"
            if changed
            else "stable"
        )
        return {
            "direction": self.label,
            "state": state,
            "weights": dict(zip(DIRECTIONS, map(float, weights))),
            "arrow_weights": dict(zip(DIRECTIONS, map(float, self.display_weights))),
            "raw_direction": self.raw,
            "consecutive_observations": self.run,
            "agreement_count": agreement,
            "recent_observations": len(self.history),
            "agreement_fraction": agreement / len(self.history) if self.history else 0.0,
            "new_observation": bool(observed),
            "total_observations": self.total_observations,
            "pending_direction": pending,
            "changed": changed,
            "stale": stale,
            "note": "Empirical observation support; not calibrated correctness probability. Predicted frames add no votes.",
        }
