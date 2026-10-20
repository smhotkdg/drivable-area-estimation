"""Final H.264 export shared by video processing and UI replay."""

import shutil
import subprocess
from pathlib import Path


def require_ffmpeg():
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("FFmpeg를 설치하고 PATH에 추가하세요. docs/SETUP.md 참고")
    return executable


def encode_video(raw, output, source, fps, frames, start_seconds=0.0, keep_audio=True):
    command = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw)]
    if keep_audio:
        command += ["-ss", str(start_seconds), "-i", str(source)]
    command += ["-map", "0:v:0"]
    if keep_audio:
        command += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-fps_mode",
        "cfr",
        "-t",
        str(frames / fps),
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    if not Path(output).is_file():
        raise RuntimeError("최종 영상이 생성되지 않았습니다")
