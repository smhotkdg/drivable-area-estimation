# 수정과 회귀 검사

## 작은 변경부터

1. `configs/default.toml`을 `configs/local.toml`로 복사해 로컬 경로를 지정합니다.
2. 변경 대상 모듈을 README 표에서 찾습니다.
3. 관련 단위 테스트를 추가/수정하고 짧은 저해상도 영상으로 확인합니다.
4. 차로 폭, 중심 편향, 경계 침범, 경로 끊김, 방향 전환 횟수와 지연을 따로 확인합니다.
5. 전체 테스트와 코드 형식 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
```

코드 형식을 정리하려면 `ruff format src tests`를 사용합니다. 모델 가중치가 없는 환경에서도 핵심 테스트를 실행할 수 있습니다.

## 자주 수정하는 설정

```toml
[votes]
min_consecutive = 3    # 늘리면 짧은 흔들림에 둔감해지고 전환이 늦어짐
enter_weight = 0.60
switch_margin = 0.20

[display]
panel_height = 132
arrow_length = 40
```

상단 UI의 글자/색상은 `visualization/dashboard.py`, 기본값은 `config.py`에서 관리합니다. 새 필드를 만들면 TOML, 검증, 설명 문서를 함께 수정하세요.

경로 생성 수치까지 무작정 설정으로 노출하지는 않았습니다. 기하학적 변경은 `geometry/single_lane_center.py`와 관련 테스트에서 검토합니다. 추정 영역 중심에 대한 편차가 줄었다고 실제 차로 정답률이 오른 것으로 보고하지 않습니다.

## 기존 파일에서 옮겨온 위치

| 이전 파일 | 새 위치 |
|---|---|
| `track_road_video.py` | `pipeline.py`, `tracking/path_tracker.py`, `visualization/overlay.py` |
| `direction_dashboard.py` | `tracking/direction_votes.py`, `visualization/dashboard.py` |
| `add_direction_arrows.py` | `replay.py` |
| `compare_road.py`의 모델 추론 | `models/segmentation.py` |
| 차로·중심선 실험 파일 | `geometry/` |

원본 학습 스크립트, 개인 영상, 모델 캐시, 과거 모든 실험 결과를 복사한 저장소는 아닙니다. 현재 채택한 추론·추적·표시 흐름과 필요한 비교 알고리즘에 집중했습니다.

## 변경 시 주의할 회귀

- 교차로에서 도로가 넓어져도 선택 차로가 함께 두 배로 넓어지지 않아야 합니다.
- 먼 한 점을 향해 경로가 연석으로 붙지 않아야 합니다.
- 횡단보도/정지선이 좌우 차선 경계가 되지 않아야 합니다.
- 관측이 끊겼을 때 유지한 값과 새 판단을 구분해야 합니다.
- `arrows`는 원본 처리 JSON과 정확히 같은 프레임 순서의 영상을 받아야 합니다.
- 프레임 수만 같다고 같은 원본임이 보장되지는 않습니다. 다른 영상의 캐시를 재사용하지 마세요.
