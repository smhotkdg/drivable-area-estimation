# 가중치 준비

1. [TwinLiteNet+ 원본 저장소](https://github.com/chequanghuy/TwinLiteNetPlus)의 Pretrained 안내를 확인합니다.
2. 원본이 제공하는 [다운로드 폴더](https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz?usp=sharing)에서 small 가중치를 준비합니다.
3. `weights/small.pth`에 놓거나 `--weight` / `[models].twin_weight`로 다른 경로를 지정합니다.

다른 크기의 가중치를 사용하면 `[models].twin_config`도 `nano`, `small`, `medium`, `large` 중 해당 값으로 바꿔야 합니다.

SegFormer는 첫 실행 시 Hugging Face에서 다운로드하며 `[models].cache_dir`에 캐시합니다. 가중치를 Git에 커밋하지 마세요. 별도 모델 라이선스는 [외부 모델 안내](../docs/THIRD_PARTY.md)를 확인하세요.
