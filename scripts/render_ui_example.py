"""Render the README's synthetic layout example (no user video or real probabilities)."""

from pathlib import Path
import cv2
import numpy as np
from drivable_area.visualization.dashboard import draw_direction_dashboard


def main():
    root = Path(__file__).resolve().parents[1]
    weights = {"left": 0.12, "straight": 0.25, "right": 0.63}
    state = {
        "direction": "right",
        "state": "stable",
        "stale": False,
        "arrow_weights": weights,
        "recent_observations": 12,
        "agreement_count": 8,
        "agreement_fraction": 8 / 12,
        "raw_direction": "right",
        "consecutive_observations": 4,
        "steering": {
            "wheel_rotation_deg": 18.0,
            "image_bearing_deg": 18.0,
            "available": True,
            "status": "tracking",
        },
    }
    image = draw_direction_dashboard(
        np.zeros((640, 360, 3), np.uint8),
        state,
        {"lane_selection": {"mode": "road_center_estimate"}},
    )
    path = root / "docs/images/steering-ui.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image[:133]):
        raise RuntimeError("Failed to save UI example")
    print(path)


if __name__ == "__main__":
    main()
