# 설치와 실행

## 실행 환경

- Python 3.11 이상. 최초 검증은 Windows/Python 3.11.9에서 수행했습니다.
- FFmpeg 실행 파일이 PATH에 있어야 H.264 저장과 오디오 처리가 가능합니다.
- 한글 글꼴: Windows 맑은 고딕, Linux Noto Sans CJK, macOS Apple SD Gothic Neo를 자동 탐색합니다.
- 자동 탐색이 실패하면 `--font "글꼴파일경로"` 또는 `[display].font_path`를 지정합니다.

Linux 예시:

```sh
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-noto-cjk
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

FFmpeg는 [공식 다운로드 안내](https://ffmpeg.org/download.html)를 참고하세요.

## 두 가지 설치 모드

### 기존 마스크 사용

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

PyTorch와 모델 다운로드가 필요 없습니다. `--cache-dir` 아래에 `segformer_masks`, `lane_masks`가 필요합니다.

### 새 영상 추론

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[inference]"
```

GPU를 쓰려면 해당 환경에 맞는 PyTorch를 먼저 설치하세요. 기본 `models.device="auto"`는 CUDA를 사용할 수 있으면 CUDA, 아니면 CPU를 선택합니다. 체크포인트도 선택한 장치로 읽습니다. `--device cpu`로 CPU를 강제할 수 있습니다.

## 가중치와 캐시

- TwinLiteNet+ small 가중치: `weights/small.pth` 또는 `--weight`로 지정합니다.
- SegFormer 첫 실행은 다운로드가 필요합니다. 캐시 위치는 `[models].cache_dir`입니다.
- 기존 Hugging Face 캐시를 쓸 때는 해당 경로를 `configs/local.toml`에 지정할 수 있습니다.
- 다운로드가 끝난 환경에서 오프라인으로 쓰려면 `local_files_only=true`를 사용합니다.
- 가중치는 저장소에 포함하지 않습니다. 출처는 [weights/README.md](../weights/README.md)를 확인하세요.

## 주요 옵션

| 옵션 | 의미 |
|---|---|
| `--height 640` | 종횡비를 유지한 출력 높이. 짝수 240 이상 |
| `--fps 0` | 원본 평균 FPS 사용. 모든 원본 프레임 처리 |
| `--duration 0` | 지정 시작점에서 영상 끝까지 |
| `--start-seconds 80 --duration 8` | 짧은 구간 검사 |
| `--detect-every 5` | 5개 출력 프레임마다 새 마스크 검출/읽기 |
| `--cache-dir PATH` | 모델 대신 기존 마스크 사용 |
| `--no-direction-arrows` | 화살표 UI 제외. UI 재생성용 바탕 영상에 권장 |
| `--no-audio` | 오디오 제외 |
| `--overwrite` | 기존 결과 파일 덮어쓰기 명시 |

원본 FPS를 낮추면 처리 프레임도 줄어듭니다. **해상도만 줄이려면 FPS를 0으로 유지**하세요. 가변 FPS 원본은 평균 FPS의 일정 프레임 간격 영상으로 저장하므로 개별 원본 타임스탬프를 그대로 유지하는 기능은 아닙니다.

## 결과 파일

- `result.mp4`: H.264 영상. 오디오가 있으면 AAC로 저장합니다.
- `result.json`: 입력, 적용 설정, 프레임 번호, 경로·경계·방향 정보. 중단된 처리의 `complete`는 false입니다.
- `arrows.validation.json`: UI 재생성 시 전체 프레임 디코딩 검증과 표시 통계.
- `.work/`: 중간 영상과 내부 QA 자료. Git에서 제외됩니다.

원본·출력 경로는 다르게 지정해야 합니다. 결과가 이미 있으면 `--overwrite` 없이는 덮어쓰지 않습니다.
