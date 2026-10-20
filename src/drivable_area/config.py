"""Validated, shared TOML settings. CLI arguments override video settings."""

from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib


@dataclass
class VideoSettings:
    height: int = 640
    fps: float = 0.0
    start_seconds: float = 0.0
    duration: float = 0.0
    detect_every: int = 5
    planner: str = "lane-center"
    intent_only: bool = True
    direction_arrows: bool = True


@dataclass
class VoteSettings:
    window_seconds: float = 1.0
    half_life_seconds: float = 0.35
    min_consecutive: int = 3
    enter_weight: float = 0.60
    switch_margin: float = 0.20
    expiry_seconds: float = 0.5


@dataclass
class DisplaySettings:
    font_path: str = ""
    panel_height: int = 132
    arrow_length: int = 40


@dataclass
class ModelSettings:
    twin_weight: str = "weights/small.pth"
    twin_config: str = "small"
    model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    cache_dir: str = ".cache/huggingface"
    img_size: int = 640
    seg_size: int = 1024
    device: str = "auto"
    cpu_threads: int = 1
    local_files_only: bool = False


@dataclass
class SteeringSettings:
    enabled: bool = True
    smoothing_seconds: float = 0.18
    max_rate_deg_per_second: float = 90.0
    max_angle_deg: float = 60.0
    expiry_seconds: float = 0.5


@dataclass
class Settings:
    video: VideoSettings = field(default_factory=VideoSettings)
    votes: VoteSettings = field(default_factory=VoteSettings)
    display: DisplaySettings = field(default_factory=DisplaySettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    steering: SteeringSettings = field(default_factory=SteeringSettings)

    def validate(self):
        v, vote, display, m = self.video, self.votes, self.display, self.models
        steering = self.steering
        if (
            min(
                steering.smoothing_seconds,
                steering.max_rate_deg_per_second,
                steering.max_angle_deg,
                steering.expiry_seconds,
            )
            <= 0
        ):
            raise ValueError("스티어링 표시의 시간/각도 제한은 양수여야 합니다")
        if steering.enabled and display.panel_height < 120:
            raise ValueError("스티어링 표시에는 panel_height >=120이 필요합니다")
        if (
            v.height < 240
            or v.height % 2
            or v.fps < 0
            or v.duration < 0
            or v.start_seconds < 0
            or v.detect_every < 1
        ):
            raise ValueError("height: 짝수 >=240; fps/duration/start_seconds >=0; detect_every >=1")
        if v.planner not in {"medial", "continuation", "boundaries", "lane-center"}:
            raise ValueError("알 수 없는 planner")
        if v.direction_arrows and not v.intent_only:
            raise ValueError("direction_arrows에는 intent_only=true가 필요합니다")
        if (
            vote.window_seconds <= 0
            or vote.half_life_seconds <= 0
            or vote.expiry_seconds <= 0
            or vote.min_consecutive < 1
        ):
            raise ValueError("투표 시간과 연속 검출 횟수는 양수여야 합니다")
        if not 0 < vote.enter_weight <= 1 or not 0 <= vote.switch_margin <= 1:
            raise ValueError("투표 가중치/차이 기준은 0~1 범위여야 합니다")
        if display.panel_height < 100 or display.arrow_length <= 0:
            raise ValueError("panel_height >=100, arrow_length >0이어야 합니다")
        if m.img_size < 32 or m.img_size % 32 or m.seg_size < 32 or m.cpu_threads < 1:
            raise ValueError("모델 입력 크기/CPU 스레드 설정을 확인하세요")
        if m.device not in {"auto", "cpu", "cuda"} or m.twin_config not in {
            "nano",
            "small",
            "medium",
            "large",
        }:
            raise ValueError("모델 장치/구성 설정을 확인하세요")
        return self


def load_settings(path=None):
    result = Settings()
    if path:
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        for section, values in data.items():
            if section not in {f.name for f in fields(Settings)} or not isinstance(values, dict):
                raise ValueError(f"알 수 없는 설정 구역: {section}")
            target = getattr(result, section)
            known = {f.name for f in fields(target)}
            for key, value in values.items():
                if key not in known:
                    raise ValueError(f"알 수 없는 설정: {section}.{key}")
                if type(value) is not type(getattr(target, key)):
                    # TOML integer values are valid for floating-point fields.
                    if isinstance(getattr(target, key), float) and type(value) is int:
                        value = float(value)
                    else:
                        raise ValueError(f"설정 형식 오류: {section}.{key}")
                setattr(target, key, value)
    return result.validate()
