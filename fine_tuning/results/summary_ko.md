# 고정 예산 LoRA 배치 비교 — 검증 결과

2026-11-14. 사람 승인 합성 train224개, microbatch1×accumulation4,epoch4,AdamW,LR0.0002. Qwen3-0.6B NF4. 학습 파라미터 예산1,146,880개를 유지하고 q/v rank8과 q/k/v/o rank4를 비교했다.

| 조건 | 기존37 정답 seed42/43/44 | 평균 전체 일치 | 확인63 정답 seed42/43/44 | 평균 전체 일치 | 확인 joint F1 |
|---|---|---:|---|---:|---:|
| qv_r8 | [23, 16, 16] | 49.5% | [29, 24, 26] | 41.8% | 0.478 |
| qkvo_r4 | [22, 15, 17] | 48.6% | [21, 18, 18] | 30.2% | 0.335 |

사전 가설 판정: **이번 파일럿에서 지지하지 않음**. 확인63개 정답 수 증가 seed: 0/3.

전체 일치는 음식·날짜·시각·주체·섭취 상태를 연결한 사건 점수다. 근거 문자열·주체 basis·명시 재료까지 비교한 strict annotation 점수는 각 evaluation JSON에 별도 기록했다.

## 검증

- 기준선도 같은 환경에서 새로 학습한6개 모델과12개 평가다. 원래 실험은 모든 학습/기존37 평가 뒤 확인63개 평가를 수행했다.
- raw600건을 Mac에서 다시 파싱·평가해 저장된 결과와 일치함을 확인했다.100문장×6조건이며600개 독립 문장이 아니다.
- 이 배포의 `verify_results.py`로 원시 출력→validator→지표를 모델 없이 다시 계산할 수 있다.
- 원래 protocol의 SHA256은 metrics.json 및 reproduction_manifest.json에 보존했다. 배포 protocol은 실행 장치 경로를 생략한 파생본이다.

## 학습 적합도와 후속 질문

기존 adapter의 별도 학습 적합도 진단은 학습46개 자유 생성에서40개 전체 일치였다. 이미 본 문장에서도 주체·날짜 등의 오류가 있었다. 정답 접두사를 제공한 높은 token accuracy를 사건 연결 정확도로 해석하지 않는다. 이46개는 층화 진단 표본이며 독립 일반화 평가가 아니다. 해당 과거 adapter 파일은 이 코드 배포에 포함하지 않았다.

다음 질문은 검수된 역할·부정 범위 대조 자료와 선택적 학습이 오류를 줄이는지다. 이 결과를 보고 기존 개발 문장을 학습으로 옮기지 않는다.

## 한계

- Three seeds; synthetic small development data is not real-world ASR performance.
- Placement and per-matrix rank jointly change: this tests allocation at equal parameter budget, not a pure placement-only effect.
- Matched alpha/r and LR do not imply equal update norms or effective optimization geometry.
- Equal parameters do not guarantee equal training activations, physical GPU residency, or inference speed.
- The old adapter used for the fit diagnostic came from a different WSL patch; contemporary baselines are retrained for the controlled comparison.
- Reserved63 becomes observed after this experiment and cannot subsequently be advertised as untouched.
- The stratified training-generation probe is descriptive and is not representative of natural prevalence.

- 합성 개발 자료의 작은 파일럿이다. 실제 사용자/ASR 성능과 통계적 우월성을 입증하지 않는다.
- 위치와 rank가 함께 바뀐 예산 배치 비교다. 동일 파라미터 수가 동일 optimizer 효과나 메모리를 보장하지 않는다.
- GPU allocated/reserved와 CPU RSS는 다른 측정값이다. reserved의 물리 VRAM 대비 이상치는 원인 미확인으로 보존했다. FLOPs는 측정하지 않았다.
- 확인63개는 이 실험 이후 관찰된 자료다. 이후 미사용 확인 자료라고 주장하지 않는다.
