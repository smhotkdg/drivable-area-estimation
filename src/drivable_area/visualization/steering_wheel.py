"""Vector steering-wheel glyph, with clockwise-positive image rotation."""

import math
import cv2


def draw_steering_wheel(image, center, radius, state):
    cx, cy = map(int, center)
    r = max(9, int(radius))
    angle = state.get("wheel_rotation_deg", 0.0)
    available = state.get("available", False)
    color = (245, 215, 100) if available else (110, 120, 125)
    if state.get("status") == "held":
        color = (80, 185, 235)
    cv2.circle(image, (cx, cy), r, (8, 12, 15), 6, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), r, color, 3, cv2.LINE_AA)
    for base in (0, 90, 180):
        theta = math.radians(base + angle)
        end = (round(cx + math.cos(theta) * r * 0.80), round(cy + math.sin(theta) * r * 0.80))
        cv2.line(image, (cx, cy), end, color, 2, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), max(3, r // 5), color, -1, cv2.LINE_AA)
    theta = math.radians(-90 + angle)
    marker = (round(cx + math.cos(theta) * r), round(cy + math.sin(theta) * r))
    cv2.circle(image, marker, 3, (80, 200, 255) if available else (110, 120, 125), -1, cv2.LINE_AA)
    cv2.line(image, (cx, cy - r - 6), (cx, cy - r - 3), (205, 210, 215), 1, cv2.LINE_AA)
    return image
