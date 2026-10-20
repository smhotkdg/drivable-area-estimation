# 외부 코드와 모델

## TwinLiteNet+

- 원본: https://github.com/chequanghuy/TwinLiteNetPlus
- 로컬 원본 기준 커밋: `90f1b8695ae311d5123b05f8534b2e11e42499d2`
- 포함: `_vendor/twinlitenet/model.py`, `config.py`, `preprocess.py`
- 라이선스: MIT. 원본 저작권 고지는 저장소 `LICENSE`와 vendor 폴더의 `LICENSE`에 유지합니다.
- 변경: 패키지 상대 import, 사용하지 않는 import 제거, 코드 형식 정리, 전처리 함수만 분리. 네트워크 구성과 체크포인트 키는 유지합니다.
- 원본의 학습/평가 코드와 데이터셋은 포함하지 않습니다.

## SegFormer

- 기본 모델 카드: https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024
- 모델 카드가 연결한 라이선스: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
- 해당 라이선스는 연구·평가를 위한 비상업적 사용 제한을 명시합니다. 본 저장소 코드의 MIT 라이선스가 모델의 별도 조건을 대체하지 않습니다.
- 이 저장소는 Transformers의 모델 API를 호출하며 SegFormer 가중치나 원본 SegFormer 저장소 코드를 재배포하지 않습니다.

## 기타

PyTorch, Transformers, OpenCV, NumPy, SciPy, scikit-image, Pillow, FFmpeg는 각 프로젝트의 라이선스를 따릅니다. 한글 글꼴은 실행 환경에서 찾아 사용하며 글꼴 파일을 저장소에 포함하지 않습니다.
