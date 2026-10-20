"""Draw image-space corridor centers; this is not a validated vehicle trajectory."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _runs(row):
    changes = np.diff(np.r_[False, row, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1))


def estimate_centerline(mask):
    """Return near-to-far segments, with every line segment inside the mask.

    Follow contiguous row intervals rather than averaging across two roads or
    an obstacle hole. Discontinue the line where an interval is lost.
    Coordinates use the original image's top-left origin, in pixels.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    minimum_width = max(3, round(w * 0.025))
    step = max(1, h // 240)
    previous = (w * 0.35, w * 0.65)
    records = []
    started = False
    for y in range(min(h - 1, round(h * 0.96)), max(0, round(h * 0.25)), -step):
        spans = [(a, b) for a, b in _runs(mask[y]) if b - a >= minimum_width]
        if not spans:
            if started:
                break
            continue
        a, b = max(
            spans,
            key=lambda p: (
                max(0, min(p[1], previous[1]) - max(p[0], previous[0])),
                -abs(p[0] + p[1] - previous[0] - previous[1]),
            ),
        )
        if min(b, previous[1]) - max(a, previous[0]) <= 0:
            if started:
                break
            continue
        records.append((int(round((a + b) / 2)), y, a, b))
        previous, started = (a, b), True
    if len(records) < 3:
        return {"status": "unavailable", "segments": [], "points": [], "representative_point": None}
    xs = np.array([p[0] for p in records], dtype=float)
    smooth = np.convolve(np.pad(xs, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    centers = [
        (int(np.clip(round(x), a + 1, b - 1)), y) for x, (_, y, a, b) in zip(smooth, records)
    ]
    segments, segment = [], [centers[0]]
    for point in centers[1:]:
        x0, y0 = segment[-1]
        x1, y1 = point
        # Check the complete rasterized segment, so a connecting line cannot
        # cross a non-road pixel between two otherwise valid center points.
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        raster = np.zeros((y_max - y_min + 1, x_max - x_min + 1), np.uint8)
        cv2.line(raster, (x0 - x_min, y0 - y_min), (x1 - x_min, y1 - y_min), 1, 1)
        valid = np.all(mask[y_min : y_max + 1, x_min : x_max + 1][raster > 0])
        if valid:
            segment.append(point)
        else:
            if len(segment) >= 2:
                segments.append(segment)
            segment = [point]
    if len(segment) >= 2:
        segments.append(segment)
    if not segments:
        return {"status": "unavailable", "segments": [], "points": [], "representative_point": None}
    spacing = max(1, round(h * 0.05 / step))
    dots = [p for line in segments for p in line[::spacing]]
    # Representative image-space center at ~65% image height, not a metric
    # look-ahead target or a steering instruction.
    representative = min((p for line in segments for p in line), key=lambda p: abs(p[1] - h * 0.65))
    return {
        "status": "estimated",
        "segments": segments,
        "points": dots,
        "representative_point": representative,
        "coordinate_system": "image pixels (x, y)",
        "note": "Geometric center of the estimated mask; not a validated driving path.",
    }


def draw_centerline(image, mask):
    result = image.copy()
    info = estimate_centerline(mask)
    if info["status"] != "estimated":
        return result, info
    h, w = mask.shape
    thickness = max(2, round(w / 220))
    canvas = result.copy()
    for segment in info["segments"]:
        pts = np.asarray(segment, np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (30, 30, 30), thickness + 4, cv2.LINE_AA)
        cv2.polylines(canvas, [pts], False, (0, 255, 255), thickness, cv2.LINE_AA)
    for point in info["points"]:
        cv2.circle(canvas, tuple(point), max(4, thickness + 2), (255, 100, 0), -1, cv2.LINE_AA)
    point = tuple(info["representative_point"])
    cv2.circle(canvas, point, max(8, thickness * 2), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, point, max(5, thickness * 2 - 3), (0, 0, 255), -1, cv2.LINE_AA)
    result[np.asarray(mask, bool)] = canvas[np.asarray(mask, bool)]
    return result, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", required=True, help="folder containing ego_lane and ego_lane_masks"
    )
    parser.add_argument("--save-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.result_dir), Path(args.save_dir)
    images = sorted((source / "ego_lane").glob("frame_*.jpg"))
    if not images:
        parser.error("No ego_lane images found")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        mask = cv2.imread(str(source / "ego_lane_masks" / (path.stem + ".png")), 0)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise RuntimeError(f"Missing or invalid input for {path.name}")
        vis, info = draw_centerline(image, mask > 0)
        if not cv2.imwrite(str(output / path.name), vis):
            raise RuntimeError(f"Cannot save {path.name}")
        rows.append({"frame": int(path.stem.split("_")[-1]), **info})
    (output / "centerlines.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(
        f"Saved {len(rows)} images; {sum(r['status'] == 'estimated' for r in rows)} centerlines: {output}"
    )


if __name__ == "__main__":
    main()
