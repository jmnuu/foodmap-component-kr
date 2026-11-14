# 목적 지향 정보 추출 Fine-Tuning

## Goal

제한된 메모리에서 문맥·부정 표현·날짜·생략된 주체를 해석해 한국어 식사 사건을 정확하게 추출한다. 현재 출력은 음식, 날짜, 시각, 주체, 섭취 상태 및 명시 근거다. 식사량/일반 구성요소 추정과 웨어러블 예측은 이번 비교의 평가 대상이 아니다.

## Insight

정답 접두사를 제공한 token accuracy와 자유 생성의 사건 전체 정확도는 다르다. 반복 JSON 토큰을 잘 맞춰도 ‘엄마가 만든 김밥을 내가 먹었다’의 역할을 틀릴 수 있다. 형식 오류와 의미 오류를 함께 보되 별도로 분석한다.

## 구현 내용

Qwen3-0.6B + NF4 double quantization + LoRA + AdamW. Batch는 micro1 × accumulation4, epoch4다. 고정 예산 비교는 `q/v rank8`과 `q/k/v/o rank4`를3개 seed에서 새로 학습한다. 두 조건의 학습 파라미터는1,146,880개다. 위치와 rank가 동시에 달라지는 배치 전략 비교이며 위치 하나만의 효과는 아니다.

`src/`는 실제 실험에 사용한37개 Python 모듈의 고정 snapshot이다. 일부 기존 parser 지원 모듈도 의존성과 코드 해시 재현을 위해 보존했다. GPU 환경에 접근하기 위한 SSH/Tailscale 스크립트는 필요하지 않다. 로컬 경로를 CLI로 전달한다.

## CPU 검증

저장소 root에서:

```bash
cd fine_tuning
PYTHONPATH=src python -m foodmap_parser.meal_context_experiment validate --input data/data_n224.jsonl
PYTHONPATH=src python checks.py
PYTHONPATH=src python verify_results.py
```

## GPU 학습과 평가

실행 환경: Linux/WSL, Python3.12, NVIDIA CUDA. 실제 사용 버전은 `requirements-cuda.txt`에 적었다. 장치/driver 호환성은 별도 확인한다. 모델 checkpoint는 `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`를 로컬에 준비한다. 스크립트는 모델을 자동 다운로드하지 않는다.

```bash
PYTHONPATH=src python -m foodmap_parser.meal_context_training \
  --input data/data_n224.jsonl --config configs/qv_r8_s42.json \
  --variant uniform --model-dir /path/to/Qwen3-0.6B \
  --output-dir runs/train_qv_r8_s42
```

`run_allocation.py`에 `--dry-run`을 붙이면 모델 호출 없이18개 명령을 확인할 수 있다.

6개 설정 전체를 동일한 순서로 학습하고 기존37개, 확인63개를 평가하려면:

```bash
PYTHONPATH=src python run_allocation.py --model-dir /path/to/Qwen3-0.6B --output-dir runs/allocation_001
```

모든 학습/기존37개 평가 후에 모든 확인63개 평가를 수행한다. 결과를 본 뒤 조건을 추가하거나 checkpoint를 고르지 않는다. `STOP_AFTER_STAGE` 파일을 output 폴더에 만들면 현재 단계가 끝난 후 멈춘다. 중단된 output을 자동 재사용하지 않는다.

## 결과

[검증된 결과 요약](results/summary_ko.md)을 본다. 자료는 검수된 합성 train224개, 반복 사용한 development37개, 이 비교에서 처음 평가한 development63개다.63개는 이 실행 이후 관찰된 자료이므로 다음 연구에서 미사용 자료라고 부르지 않는다. 총600개 예측은100개 문장×6개 모델이며600개 독립 문장이 아니다.

이 배포에는 기존 adapter의 학습 적합도 진단을 다시 실행하는 모델 파일은 포함하지 않는다. 새로 학습하는 비교군과 데이터·소스·설정 해시를 제공한다. CPU 검사는 코드/데이터 무결성을 확인하며 GPU 성능 재현을 대신하지 않는다.

## 논문과 다음 가설

- [QLoRA](https://arxiv.org/abs/2305.14314): 양자화된 base와 adapter 학습 및 대상 모듈 분석.
- [LoRA+](https://arxiv.org/abs/2402.12354): adapter 두 행렬의 학습률을 달리하는 선행 방법. FoodMap에서 자동 우월성을 가정하지 않는다.
- [Critical Tokens Matter](https://proceedings.mlr.press/v267/lin25j.html): 중요한 결정 토큰 학습의 후속 가설 참고. 수학 추론 결과를 음식 추출 성능으로 옮겨 주장하지 않는다.

후속 질문은 검수된 역할·부정 범위의 대조 사례와 선택적 학습이 정확도를 개선하는지다. 현재 개발 문장을 학습에 섞지 않는다. 성능 개선 여부는 결과가 나온 뒤 판단한다.
