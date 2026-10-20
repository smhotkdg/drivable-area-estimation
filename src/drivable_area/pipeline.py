"""Fast cached-mask video preview with optical flow and Kalman path tracking.

Displayed angles are IMAGE-SPACE bearings, not calibrated vehicle steering.
No vehicle commands are generated. Cached model masks are read only on
re-detection frames; the intervening frames use tracked motion/prediction.
"""

import argparse
import json
import math
from pathlib import Path
import time
import uuid
from dataclasses import asdict
from drivable_area.config import load_settings, VideoSettings
from drivable_area.video_io import require_ffmpeg, encode_video

import cv2
import numpy as np

from drivable_area.geometry.corridor_path import path_inside
from drivable_area.geometry.lane_roi import EgoLaneTracker
from drivable_area.tracking.route_intent import route_evidence, IntentTracker
from drivable_area.geometry.lane_boundary_tracker import (
    LaneBoundaryTracker,
    BoundaryDirectionObserver,
)
from drivable_area.geometry.single_lane_center import SingleLaneTracker, center_metrics
from drivable_area.tracking.direction_votes import DirectionVoteTracker
from drivable_area.visualization.dashboard import draw_direction_dashboard


from drivable_area.tracking.path_tracker import PathTracker, motion
from drivable_area.visualization.overlay import render


def main(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    preliminary, _ = bootstrap.parse_known_args(argv)
    settings = load_settings(preliminary.config)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="TOML settings; CLI arguments take precedence")
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--cache-dir",
        help="existing segformer_masks and lane_masks folder; omit for live inference",
    )
    parser.add_argument("--output", required=True, help="output .mp4")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--start-seconds", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument(
        "--intent-only",
        action=argparse.BooleanOptionalAction,
        help="show image-space direction, not steering",
    )
    parser.add_argument(
        "--direction-arrows",
        action=argparse.BooleanOptionalAction,
        help="directional votes and compact arrows",
    )
    parser.add_argument(
        "--planner",
        choices=["medial", "continuation", "boundaries", "lane-center"],
        default="medial",
    )
    parser.add_argument("--weight", help="TwinLiteNet+ checkpoint path (live inference)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--font", help="Korean TrueType/OpenType font path")
    parser.add_argument("--work-dir", default=".work")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(**asdict(settings.video))
    args = parser.parse_args(argv)
    settings.video = VideoSettings(**{key: getattr(args, key) for key in asdict(settings.video)})
    if args.weight:
        settings.models.twin_weight = args.weight
    if args.device:
        settings.models.device = args.device
    if args.font:
        settings.display.font_path = args.font
    try:
        settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    if not Path(args.source).is_file():
        parser.error("Source video does not exist")
    if Path(args.output).resolve() == Path(args.source).resolve():
        parser.error("Output must differ from source")
    if Path(args.output).suffix.lower() != ".mp4":
        parser.error("Output must be .mp4")
    if Path(args.output).exists() and not args.overwrite:
        parser.error("Output exists; choose another path or use --overwrite")
    require_ffmpeg()
    cv2.setNumThreads(1)
    output = Path(args.output)
    cache = Path(args.cache_dir) if args.cache_dir else None
    models = None
    if cache is None:
        from drivable_area.models.segmentation import SegmentationModels

        models = SegmentationModels(settings.models)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    raw = work / (uuid.uuid4().hex + ".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    writer = None
    rows = []
    started = time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open source video")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or args.fps > source_fps:
            raise ValueError("Output FPS must not exceed valid source FPS")
        height = args.height
        width = 2 * round(cap.get(3) / cap.get(4) * height / 2)
        fps = args.fps or source_fps
        first = round(args.start_seconds * source_fps)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        remaining = total - first
        frame_count = (
            remaining if fps == source_fps else math.ceil(remaining * fps / source_fps - 1e-9)
        )
        if args.duration:
            frame_count = min(round(args.duration * fps), frame_count)
        if frame_count <= 0:
            raise ValueError("Requested clip is outside source duration")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, first):
            raise RuntimeError("Could not seek to clip start")
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open output video writer")
        tracker, roi = PathTracker(width, height, fps, planner=args.planner), EgoLaneTracker()
        intent_tracker = IntentTracker(fps)
        vote_tracker = (
            DirectionVoteTracker(fps, **asdict(settings.votes)) if args.direction_arrows else None
        )
        boundary_tracker = LaneBoundaryTracker(fps)
        lane_identity = SingleLaneTracker(fps)
        lane_identity_info = None
        boundary_direction = BoundaryDirectionObserver(
            fps, trim_route=args.planner == "lane-center"
        )
        road_mask = None
        boundary_info = None
        previous_gray, previous_mask = None, None
        current_source = first - 1
        assisted = False
        for i in range(frame_count):
            target_source = first + round(i * source_fps / fps)
            while current_source < target_source - 1:
                if not cap.grab():
                    raise RuntimeError("Unexpected end of source while skipping frames")
                current_source += 1
            ok, full = cap.read()
            current_source += 1
            if not ok:
                raise RuntimeError(f"Cannot decode source frame {current_source + 1}")
            image = cv2.resize(full, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            transform, inliers = motion(previous_gray, gray, previous_mask)
            detected = i % args.detect_every == 0 or tracker.state is None
            lane_region = None
            if detected:
                name = f"frame_{current_source + 1:06d}.png"
                if models is not None:
                    road, lanes = models.predict(full, (width, height))
                else:
                    road = cv2.imread(str(cache / "segformer_masks" / name), 0)
                    lanes = cv2.imread(str(cache / "lane_masks" / name), 0)
                    if road is None or lanes is None:
                        raise RuntimeError(f"Missing cached model masks: {name}")
                    road = cv2.resize(road, (width, height), interpolation=cv2.INTER_NEAREST) > 0
                    lanes = cv2.resize(lanes, (width, height), interpolation=cv2.INTER_NEAREST) > 0
                mask, roi_info = roi.update(road, lanes, image, current_source + 1)
                if args.planner == "continuation":
                    lane_region, mask = mask, road
                assisted = roi_info.get("basis") == "road_edge_assisted"
                if args.planner in ("boundaries", "lane-center"):
                    road_mask = road
            elif transform is not None:
                mask = cv2.warpAffine(
                    previous_mask.astype(np.uint8),
                    transform,
                    (width, height),
                    flags=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                mask = previous_mask.copy()
            if args.planner in ("boundaries", "lane-center"):
                if not detected and transform is not None:
                    road_mask = cv2.warpAffine(
                        road_mask.astype(np.uint8),
                        transform,
                        (width, height),
                        flags=cv2.INTER_NEAREST,
                    ).astype(bool)
                mask, boundary_info = boundary_tracker.update(
                    road_mask, lanes if detected else None, transform, detected
                )
                if args.planner == "lane-center":
                    mask, lane_identity_info = lane_identity.update(
                        mask,
                        boundary_tracker.last_bounds,
                        boundary_info.get("sources"),
                        transform,
                        detected,
                    )
            tracked, info = tracker.update(mask, detected, transform, lane_region)
            if args.intent_only:
                old_angle = info["image_angle_deg"]
                info["intent_previous_rule"] = (
                    "unknown"
                    if old_angle is None
                    else "left"
                    if old_angle < -1.5
                    else "right"
                    if old_angle > 1.5
                    else "straight"
                )
                evidence = route_evidence(
                    None if tracked is None else tracked[0],
                    width,
                    height,
                    confirmed=info["image_angle_deg"] is not None,
                )
                if args.planner in ("boundaries", "lane-center"):
                    direction_measurement = (
                        getattr(tracker, "direction_measurement", None)
                        if args.planner == "lane-center"
                        else getattr(tracker, "last_measurement", None)
                    )
                    evidence = boundary_direction.update(
                        mask, direction_measurement, detected, transform
                    )
                    if args.planner == "lane-center":
                        evidence["horizon"] = "full observed selected corridor"
                info["intent_evidence"] = evidence
                info["intent"] = intent_tracker.update(evidence)
            info.update(
                frame=i + 1,
                source_frame=current_source + 1,
                detected=detected,
                flow_inliers=inliers,
                road_edge_assisted=assisted,
                planner=args.planner,
            )
            if args.planner in ("boundaries", "lane-center"):
                info["boundary"] = boundary_info
                info["path_within_estimated_corridor"] = tracked is not None and path_inside(
                    tracked[0], mask
                )
                info["path_points"] = None if tracked is None else tracked[0].tolist()
                if args.planner == "lane-center":
                    info["lane_selection"] = lane_identity_info
                    info["center_metrics"] = center_metrics(
                        None if tracked is None else tracked[0], mask
                    )
            rendered = render(
                image,
                mask,
                tracked,
                info,
                current_source / source_fps,
                args.detect_every,
                fps,
                assisted,
                lane_identity.last_bounds
                if args.planner == "lane-center"
                else boundary_tracker.last_bounds
                if args.planner == "boundaries"
                else None,
            )
            if vote_tracker is not None:
                observed = (
                    detected
                    and info["intent_evidence"].get("source", "boundary_observation")
                    == "boundary_observation"
                )
                info["direction_dashboard"] = vote_tracker.update(info["intent_evidence"], observed)
                rendered = draw_direction_dashboard(
                    rendered, info["direction_dashboard"], info, settings=settings.display
                )
            writer.write(rendered)
            previous_gray, previous_mask = gray, mask
            rows.append(info)
            if (i + 1) % 75 == 0:
                print(f"{i + 1}/{frame_count} video frames", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        report = {
            "source": args.source,
            "cached_masks": args.cache_dir,
            "settings": asdict(settings),
            "complete": len(rows) == locals().get("frame_count", -1),
            "planner": args.planner,
            "note": "Angles are image-space bearings, not physical steering or vehicle commands.",
            "detection_interval_output_frames": args.detect_every,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "frames": rows,
        }
        output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    encode_video(raw, output, args.source, fps, len(rows), args.start_seconds, not args.no_audio)
    raw.unlink()
    print(f"Saved {len(rows)} frames to {output} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
