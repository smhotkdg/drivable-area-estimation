"""Add observation-weighted arrows to an existing rendered video.

Reuses stored evidence; does not re-run or modify segmentation/path tracking.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
import cv2
from dataclasses import asdict
from drivable_area.config import load_settings
from drivable_area.video_io import require_ffmpeg, encode_video
from drivable_area.tracking.direction_votes import DirectionVoteTracker
from drivable_area.tracking.steering_indicator import ImageSteeringIndicator
from drivable_area.visualization.dashboard import draw_direction_dashboard


def switches(labels):
    known = [x for x in labels if x != "unknown"]
    return sum(a != b for a, b in zip(known, known[1:]))


def main(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    preliminary, _ = bootstrap.parse_known_args(argv)
    settings = load_settings(preliminary.config)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--font")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--input", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audio-source")
    p.add_argument("--fps", type=float)
    p.add_argument("--work-dir", default=".work/arrows")
    args = p.parse_args(argv)
    cv2.setNumThreads(1)
    if args.font:
        settings.display.font_path = args.font
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(args.input).resolve():
        p.error("Output must differ from input")
    if output.exists() and not args.overwrite:
        p.error("Output exists; choose another path or use --overwrite")
    require_ffmpeg()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf8"))
    rows = metadata["frames"]
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid FPS")
    w, h = int(cap.get(3)), int(cap.get(4))
    raw = work / (output.stem + "_render.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Cannot open temporary video writer")
    tracker = DirectionVoteTracker(fps, **asdict(settings.votes))
    steering_values = asdict(settings.steering)
    steering_values.pop("enabled")
    steering_indicator = (
        ImageSteeringIndicator(fps, **steering_values) if settings.steering.enabled else None
    )
    records = []
    started = time.perf_counter()
    mixed_qa = False
    try:
        for i, info in enumerate(rows):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Input ended at frame {i}; expected {len(rows)}")
            evidence = info["intent_evidence"]
            observed = (
                info["detected"]
                and evidence.get("source", "boundary_observation") == "boundary_observation"
            )
            state = tracker.update(evidence, observed)
            if steering_indicator is not None:
                state["steering"] = steering_indicator.update(evidence, state, observed)
            result = draw_direction_dashboard(image, state, info, settings=settings.display)
            writer.write(result)
            records.append(
                {"frame": i + 1, "source_frame": info.get("source_frame", i + 1), **state}
            )
            weights = list(state["arrow_weights"].values())
            if i in (200, 638, 3679, 5211) or not mixed_qa and sorted(weights)[-2] > 0.25:
                cv2.imwrite(str(work / f"qa_{i + 1:06d}.jpg"), result)
                if sorted(weights)[-2] > 0.25:
                    mixed_qa = True
            if (i + 1) % 300 == 0:
                print(f"{i + 1}/{len(rows)} arrow frames", flush=True)
        ok, _ = cap.read()
        if ok:
            raise RuntimeError("Input has more frames than metadata")
    finally:
        cap.release()
        writer.release()
    audio = args.audio_source or args.input
    encode_video(raw, output, audio, fps, len(rows))
    check = cv2.VideoCapture(str(output))
    count = 0
    actual_fps = check.get(cv2.CAP_PROP_FPS)
    while True:
        ok, image = check.read()
        if not ok:
            break
        if image.shape[:2] != (h, w):
            raise RuntimeError("Unexpected output resolution")
        count += 1
    check.release()
    if count != len(rows) or abs(actual_fps - fps) > 0.001:
        raise RuntimeError(f"Output validation failed: {count} frames / {actual_fps} FPS")
    before = [row["intent"]["direction"] for row in rows]
    after = [r["direction"] for r in records]
    summary = {
        "input": args.input,
        "evidence_metadata": args.metadata,
        "output": str(output),
        "decoded_frames": count,
        "resolution": [w, h],
        "fps": actual_fps,
        "duration_seconds": count / actual_fps,
        "configuration": {
            "votes": asdict(settings.votes),
            "display": asdict(settings.display),
            "steering": asdict(settings.steering),
            "arrow_easing_seconds": 0.12,
        },
        "detection_observations": records[-1]["total_observations"] if records else 0,
        "display_switches_skipping_unknown": {
            "previous": switches(before),
            "weighted": switches(after),
        },
        "display_states": dict(Counter(r["state"] for r in records)),
        "display_direction_counts": dict(Counter(after)),
        "note": "Weights are relative recent detection support, not calibrated probabilities or accuracy. Adjacent detections are temporally correlated. Predictions add no votes. Lower switching is not proof of better accuracy; transitions intentionally have a confirmation delay.",
        "path_and_road_evidence_unchanged": True,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    output.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "frames": records}, ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    output.with_suffix(".validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
