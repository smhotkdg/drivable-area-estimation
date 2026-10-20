from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from drivable_area.tracking.direction_votes import DIRECTIONS, KOREAN
from drivable_area.config import DisplaySettings


@lru_cache(maxsize=12)
def _font(size, configured=""):
    candidates = (
        [Path(configured)]
        if configured
        else [
            Path("C:/Windows/Fonts/malgun.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        ]
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise RuntimeError(
            "한글 글꼴을 찾을 수 없습니다. --font 또는 display.font_path를 지정하세요"
        )
    return ImageFont.truetype(str(path), size)


def draw_direction_dashboard(image, state, info=None, replace_header=True, settings=None):
    settings = settings or DisplaySettings()
    image = image.copy()
    h, w = image.shape[:2]
    scale = min(w / 360.0, h / 640.0)
    x0 = 0
    x1 = w - 1
    top = 0 if replace_header else round(8 * scale)
    ui_scale = settings.panel_height / 132
    bottom = min(h - 4, top + round(settings.panel_height * scale))
    cv2.rectangle(image, (x0, top), (x1, bottom), (20, 25, 30), -1)
    cv2.line(image, (x0, bottom), (x1, bottom), (85, 100, 110), 1, cv2.LINE_AA)
    label = state["direction"]
    fresh = label != "unknown" and not state["stale"]
    colors = {"left": (255, 210, 80), "straight": (110, 245, 150), "right": (80, 195, 255)}
    base_y = top + round(94 * scale * ui_scale)
    for index, direction in enumerate(DIRECTIONS):
        weight = state["arrow_weights"][direction]
        origin = np.array([w * (0.20 + 0.30 * index), base_y])
        length = min(settings.arrow_length, settings.panel_height * 0.32) * scale * max(0.0, weight)
        unit = (
            np.array([[0.0, 0.0], [0.0, -1.0]])
            if direction == "straight"
            else np.array([[0, 0], [0, -0.35], [-0.10, -0.70], [-0.40, -0.85], [-0.82, -0.85]])
        )
        if direction == "right":
            unit[:, 0] *= -1
        unit /= np.linalg.norm(np.diff(unit, axis=0), axis=1).sum()
        points = np.rint(origin + unit * length).astype(np.int32)
        color = colors[direction] if weight > 0.02 else (90, 95, 100)
        if length >= 3:
            thickness = max(2, round((4 if direction == label and fresh else 2) * scale))
            cv2.polylines(
                image, [points.reshape(-1, 1, 2)], False, (8, 12, 15), thickness + 3, cv2.LINE_AA
            )
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
            tip = points[-1].astype(float)
            vector = points[-1] - points[-2]
            norm = np.linalg.norm(vector)
            if norm > 0:
                u = vector / norm
                n = np.array([-u[1], u[0]])
                head = min(7 * scale, max(3 * scale, length * 0.20))
                triangle = np.rint(
                    [tip, tip - u * head + n * head * 0.62, tip - u * head - n * head * 0.62]
                ).astype(np.int32)
                cv2.fillConvexPoly(image, triangle, color, cv2.LINE_AA)
        else:
            cv2.circle(
                image,
                tuple(np.rint(origin).astype(int)),
                max(2, round(2 * scale)),
                color,
                -1,
                cv2.LINE_AA,
            )
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    def text(x, y, value, size=12, color=(235, 240, 245), anchor=None):
        draw.text(
            (round(x), round(top + (y - top) * ui_scale)),
            value,
            font=_font(max(9, round(size * scale)), settings.font_path),
            fill=color,
            anchor=anchor,
        )

    status = {
        "waiting": "확인 중",
        "held": "이전 판단 유지",
        "checking": "전환 확인 중",
        "confirmed": "확정",
        "stable": "유지",
    }[state["state"]]
    mode = (info or {}).get("lane_selection", {}).get("mode")
    prefix = "도로 중심 추정" if mode == "road_center_estimate" else "차로 추정"
    text(10 * scale, top + 4 * scale, f"{prefix} | {KOREAN[label]} · {status}", 13)
    n = state["recent_observations"]
    agree = state["agreement_count"]
    pct = round(state["agreement_fraction"] * 100)
    text(
        10 * scale,
        top + 27 * scale,
        f"현재 {KOREAN[state['raw_direction']]} 연속 {state['consecutive_observations']}회 · 일치 {agree}/{n} ({pct}%)",
        10,
        (200, 210, 220),
    )
    for index, direction in enumerate(DIRECTIONS):
        cx = w * (0.20 + 0.30 * index)
        text(
            cx,
            top + 101 * scale,
            f"{KOREAN[direction]} {round(state['arrow_weights'][direction] * 100)}%",
            11,
            anchor="mt",
        )
    text(10 * scale, top + 119 * scale, "최근 1초 검출 가중치 · 정답 확률 아님", 9, (160, 177, 190))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
