# FoodMap — Task-Specific Information Extraction with Fine-Tuning

**한국어 자연어 기록에서 목적에 맞는 정보를 추출하는 저메모리 fine-tuning 연구 프로젝트**입니다. 첫 응용은 음식·날짜·시각·섭취 주체·섭취 상태를 하나의 식사 사건으로 정확히 연결하는 것입니다.

Qwen3-0.6B의 QLoRA를 기반으로 학습 데이터, 목적함수, adapter 설정과 메모리 효율을 통제 실험합니다. 음식 단어 하나를 찾는 정확도와 사건 전체의 일치율을 구분합니다.

본 프로젝트의 코드 구현, 합성 데이터 초안 작성, 실험 실행·분석, 문서 정리에 OpenAI Codex를 활용했습니다.

## 바로 시작하기

- [Fine-tuning 코드·실험 안내](fine_tuning/README.md)
- [검수된 합성 데이터와 분할 설명](fine_tuning/DATA_CARD.md)
- [재현 가능한 설정·출처 해시](fine_tuning/reproduction_manifest.json)
- [학습/추론 코드](fine_tuning/src/foodmap_parser/meal_context_training.py)
- [고정 예산 LoRA 비교 결과](fine_tuning/results/summary_ko.md)

현재 실험의 음식·시간·주체·섭취 상태 추출과 향후 식사량/구성요소 확장을 구분합니다. 합성 개발 자료의 점수를 실제 ASR 또는 새로운 사용자 성능으로 주장하지 않습니다. 메모리는 학습/추론과 GPU/CPU를 구분해 기록합니다.

**연구 분리:** 본 저장소는 정보 추출 fine-tuning에 집중합니다. 식사와 웨어러블 이력으로 미래 반응 분포를 생성하는 diffusion 연구는 별도 프로젝트로 진행하며, 이 저장소의 parser 성능으로 그 효과를 주장하지 않습니다.

기존 사전·component 연구는 원래 파일과 Git 이력에 보존되어 있습니다. 최신 학습 실험은 `fine_tuning/` 묶음에서 시작하세요. Qwen 가중치·adapter 파일·개인 연결 설정은 이 Git 배포에 포함하지 않습니다.


[이전 component 데이터셋 소개](https://github.com/jmnuu/foodmap-component-kr/blob/72ac3be2c0dc231bb87f8c56efd4006ed38117bf/README.md)
