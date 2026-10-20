import cv2
import numpy as np


def render(image, mask, track, info, source_time, detect_every, fps, assisted, boundaries=None):
    vis = image.copy()
    vis[mask] = (vis[mask].astype(np.float32) * 0.7 + np.array([0, 76.5, 0])).astype(np.uint8)
    h, w = mask.shape
    if track is not None:
        pts, valid, target = track
        canvas = vis.copy()
        for i in range(len(pts) - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            raster = np.zeros_like(mask, np.uint8)
            cv2.line(raster, tuple(pts[i]), tuple(pts[i + 1]), 1, 1)
            if np.any((raster > 0) & ~mask):
                continue
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (20, 20, 20), 5, cv2.LINE_AA)
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 255), 2, cv2.LINE_AA)
        for point in pts[valid][::2]:
            cv2.circle(canvas, tuple(point), 3, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(target), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(target), 4, (0, 0, 255), -1, cv2.LINE_AA)
        vis[mask] = canvas[mask]
    if boundaries is not None:
        for side, color in [("left", (255, 220, 0)), ("right", (0, 150, 255))]:
            values = boundaries[side]
            for y in range(round(h * 0.30), h - 1, 3):
                if not np.isfinite(values[y : y + 4]).all():
                    continue
                if not mask[y : y + 4].any(axis=1).all():
                    continue
                if abs(values[min(y + 3, h - 1)] - values[y]) > w * 0.08:
                    continue
                a = (int(round(values[y])), y)
                b = (int(round(values[min(y + 3, h - 1)])), min(y + 3, h - 1))
                observed = boundaries["marking_" + side][y]
                if observed or y % 12 < 3:
                    cv2.line(vis, a, b, color if observed else (170, 170, 170), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (w - 1, 105), (20, 20, 20), -1)
    angle = info["image_angle_deg"]
    if angle is None:
        heading, color = "PATH UNCERTAIN - no direction", (0, 150, 255)
    else:
        direction = "RIGHT" if angle > 1.5 else "LEFT" if angle < -1.5 else "STRAIGHT"
        heading, color = f"{direction}  {angle:+.1f} deg (image)", (0, 255, 255)
    lines = [
        (heading, color),
        ("Image bearing only - NOT steering", (210, 210, 210)),
        (f"{info['status']} | detect: every {detect_every} frames", (230, 230, 230)),
        (
            f"t={source_time:.2f}s  {w}x{h}  {fps:g}fps" + (" | edge aided" if assisted else ""),
            (200, 200, 200),
        ),
    ]
    if "intent" in info:
        intent = info["intent"]
        prefix = "LAST" if intent["state"] == "held" else "INTENT"
        lines[0] = (f"{prefix}: {intent['direction'].upper()} [{intent['state']}]", (0, 255, 255))
        angle_text = (
            "--"
            if info["intent_evidence"]["far_bearing_deg"] is None
            else str(info["intent_evidence"]["far_bearing_deg"])
        )
        lines[1] = (f"Far route bearing: {angle_text} deg (image)", (210, 210, 210))
        lines[2] = (
            f"Old: {info['intent_previous_rule'].upper()} | {info['status']}",
            (230, 230, 230),
        )
        if boundaries is not None:
            sources = info["boundary"]["sources"]
            left = (
                "lane"
                if sources.get("left") == "marking"
                else "tracked"
                if sources.get("left") == "tracked_marking"
                else "road"
            )
            right = (
                "lane"
                if sources.get("right") == "marking"
                else "tracked"
                if sources.get("right") == "tracked_marking"
                else "road"
            )
            curve = info["intent_evidence"].get("tangent_change_deg")
            lines[1] = (
                f"Image bend: {curve if curve is not None else '--'} deg | not steering",
                (210, 210, 210),
            )
            lines[2] = (f"L:{left} R:{right} | path:{info['status']}", (230, 230, 230))
        if "lane_selection" in info:
            mode = info["lane_selection"]["mode"]
            prefix = "ROAD EST" if mode == "road_center_estimate" else "LANE EST"
            lines[0] = (
                f"{prefix}: {intent['direction'].upper()} [{intent['state']}]",
                (0, 180, 255) if prefix == "ROAD EST" else (0, 255, 255),
            )
            lines[3] = (f"t={source_time:.2f}s | {mode.replace('_', ' ')}", (200, 200, 200))
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            vis, text, (8, 23 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA
        )
    return vis
