import json
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from drivable_area.pipeline import main as process_video
from drivable_area.replay import main as replay
from drivable_area.visualization.dashboard import _font


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for video integration")
class PipelineIntegrationTests(unittest.TestCase):
    def test_fresh_cached_pipeline_and_ui_replay_preserve_every_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            output = root / "processed.mp4"
            cache = root / "cache"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (180, 320))
            self.assertTrue(writer.isOpened())
            for index in range(9):
                image = np.zeros((320, 180, 3), np.uint8)
                image[100:, 20:160] = (80 + index, 80 + index, 80 + index)
                writer.write(image)
                for name, mask in [
                    (
                        "segformer_masks",
                        np.pad(np.ones((220, 140), np.uint8) * 255, ((100, 0), (20, 20))),
                    ),
                    ("lane_masks", np.zeros((320, 180), np.uint8)),
                ]:
                    path = cache / name / f"frame_{index + 1:06d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    self.assertTrue(cv2.imwrite(str(path), mask))
            writer.release()
            process_video(
                [
                    "--source",
                    str(source),
                    "--cache-dir",
                    str(cache),
                    "--output",
                    str(output),
                    "--height",
                    "240",
                    "--no-direction-arrows",
                    "--no-audio",
                    "--work-dir",
                    str(root / "work"),
                ]
            )
            report = json.loads(output.with_suffix(".json").read_text())
            self.assertTrue(report["complete"])
            self.assertEqual([r["source_frame"] for r in report["frames"]], list(range(1, 10)))
            self.check_video(output, 9)
            try:
                _font(12)
            except RuntimeError:
                self.skipTest("A Korean font is required for replay")
            arrows = root / "arrows.mp4"
            replay(
                [
                    "--input",
                    str(output),
                    "--metadata",
                    str(output.with_suffix(".json")),
                    "--output",
                    str(arrows),
                    "--work-dir",
                    str(root / "arrows-work"),
                ]
            )
            self.check_video(arrows, 9)
            stats = json.loads(arrows.with_suffix(".validation.json").read_text())
            self.assertEqual(stats["decoded_frames"], 9)

    def check_video(self, path, expected):
        cap = cv2.VideoCapture(str(path))
        count = 0
        self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30.0, places=2)
        while True:
            ok, image = cap.read()
            if not ok:
                break
            self.assertEqual(image.shape[:2], (240, 136))
            count += 1
        cap.release()
        self.assertEqual(count, expected)
