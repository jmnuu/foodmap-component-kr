# 데이터 카드

- 원천: 사람이 승인한 assistant-synthetic 한국어 식사 기록. 실제 환자·웨어러블 자료가 아니다.
- `data_n224.jsonl`: train224개 + 기존 development37개가 함께 들어 있는261행 파일. 파일명224는 학습 수다.
- `evaluation_original37.jsonl`: 위 development37개만 담은 평가 입력. 이미 반복 관찰한 자료다.
- `reserved_development63.jsonl`: 추가 development63개.2026-11-14 비교에서 처음 평가했고, 이제 관찰된 자료다. 최종 사용자 테스트셋이 아니다.
- 사람의 승인 기록은 선언된 메타데이터이며 독립 평가자 간 일치도를 측정한 것은 아니다.
- source group·화자·정규화 문장 중복 검사를 적용했다. 모든 의미상 유사 문장이 분리됨을 보장하지 않는다.
- 원문 근거 quote·날짜 anchor·검수 상태·split을 보존한다. 해시는 `reproduction_manifest.json`에 있다.
- 공식 평가기는 train split이나 미승인 정답의 일반 평가를 거부한다.
- 원래의 recipe/component 자료는 별도 역사 자료이며 이 fine-tuning의 사건 정확도 정답과 혼합하지 않는다.

확장 시 실제/합성·학습/개발/최종 평가·출처·검수 버전을 구분한다. 가상의 문장 수가 많아지는 것을 독립적인 실제 사용자 수가 증가한 것으로 해석하지 않는다.
