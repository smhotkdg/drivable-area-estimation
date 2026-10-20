import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from drivable_area.config import load_settings
from drivable_area.cli import main


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_the_accepted_full_video_and_top_dashboard_mode(self):
        settings = load_settings()
        self.assertEqual(settings.video.planner, "lane-center")
        self.assertEqual(settings.video.fps, 0)
        self.assertEqual(settings.video.duration, 0)
        self.assertTrue(settings.video.direction_arrows)
        self.assertEqual(settings.display.panel_height, 132)

    def test_toml_rejects_typo_instead_of_silently_ignoring_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.toml"
            path.write_text("[votes]\nmin_consecutve=7\n", encoding="utf8")
            with self.assertRaisesRegex(ValueError, "min_consecutve"):
                load_settings(path)

    def test_toml_rejects_invalid_vote_ranges_and_types(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "settings.toml"
            for value in ("enter_weight=1.2", 'min_consecutive="three"'):
                path.write_text("[votes]\n" + value + "\n", encoding="utf8")
                with self.assertRaises(ValueError):
                    load_settings(path)

    def test_video_help_is_dispatched_to_video_options(self):
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as result:
            main(["video", "--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("--source", stream.getvalue())
        self.assertIn("--cache-dir", stream.getvalue())
