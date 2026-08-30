# General FoodMap 핵심 논문 정리

## 읽는 방법

이 문서는 General FoodMap을 만들 때 우선 읽을 논문 8편을 같은 기준으로
정리한 연구 노트다. 여기서 "연관"은 논문이 보고한 연관성이지, 음식이 어떤
개인에게 결과를 반드시 일으킨다는 뜻이 아니다. 특히 umbrella review와
prospective cohort 기반 결과는 FoodMap의 변수 설계와 근거 강도 기록에 쓰고,
개인별 임상 예측값으로 바로 변환하지 않는다.

## 한눈에 보는 역할

| 논문 | General FoodMap에서의 역할 | 가장 중요한 설계 결정 |
| --- | --- | --- |
| 당 umbrella review | 당 노출의 기본 분해 | SSB, 첨가당, 과당을 분리한다. |
| SSB umbrella review | 음료 패턴의 별도 분석 | 당 g와 음료 섭취를 동시에 저장한다. |
| 단백질 meta-analysis | 영양소 원천 보존 | 식물성·동물성·총단백질을 분리한다. |
| 유제품 umbrella review | 식품군 내부 이질성 | 제품 유형과 지방 함량을 분리한다. |
| 나트륨 umbrella review | 용량-생리 결과 연결 | `sodium_mg`와 혈압 측정을 연결한다. |
| 식이섬유 meta-analysis | 영양소 유형·원천 보존 | 총량, soluble/insoluble, 원천을 분리한다. |
| 초가공식품 umbrella review | 가공도 노출 설계 | NOVA 등 분류 체계를 기록한다. |
| GBD 식이 위험 연구 | 온톨로지 우선순위 | 우선 포함할 식이 위험 개념을 정한다. |

## 1. Dietary sugar consumption and health: umbrella review

- 서지: Huang Y, et al. *BMJ*. 2023;381:e071609. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37019448/)
- 연구 유형: 기존 메타분석을 종합한 umbrella review.
- 규모: 73개 메타분석, 83개 건강 결과를 포함했다.
- 무엇을 봤는가: 식이 당 섭취와 건강 결과의 관계를 검토했고, SSB, 첨가당,
  과당처럼 노출 정의가 다른 분석을 함께 다뤘다.
- FoodMap 적용: `added_sugar_g`, `fructose_g`, `sugar_sweetened_beverage`,
  `serving_ml`, `food_source`를 별도 필드로 둔다. `sugar=true` 하나로 합치지
  않는다.
- 주의: 연구 설계와 비교 기준이 다양한 umbrella review다. 서로 다른 당 정의의
  효과크기를 하나의 평균 점수로 합산하면 안 된다.

## 2. Sugar-Sweetened Beverages and Adverse Human Health Outcomes

- 서지: Lane MM, et al. *Annual Review of Nutrition*. 2024;44:383-404.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/39207876/)
- 연구 유형: 관찰연구 메타분석을 종합한 umbrella review.
- 범위: 사용자가 확인한 원문 정리 기준 47개 메타분석과 약 2,205만 명 자료를
  검토했다. 제2형 당뇨병, 심혈관 질환, 우울, 신장결석, 요산 등의 결과를
  분리해 평가한다.
- FoodMap 적용: SSB는 단순한 당 함량이 아니라 음료 섭취 패턴이므로
  `beverage_type`, `serving_ml`, `frequency`, `added_sugar_g`를 함께 저장한다.
- 주의: 관찰연구 중심 근거다. SSB 섭취가 건강 상태의 원인이라고 개인 수준에서
  단정하는 규칙으로 바꾸지 않는다.

## 3. Total, animal, and plant proteins and mortality

- 서지: Naghshi S, et al. *BMJ*. 2020;370:m2412.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/32699048/)
- 연구 유형: 성인 prospective cohort의 체계적 문헌고찰 및 용량-반응 메타분석.
- 규모: 32개 코호트 연구, 715,128명, 추적 기간 3.5-32년을 포함했다.
- 핵심 결과: 총단백질, 동물성 단백질, 식물성 단백질을 구분했다. 초록에서는
  식물성 단백질 섭취가 전체 사망 및 심혈관 사망과 다른 연관 패턴을 보인다고
  보고했다.
- FoodMap 적용: `protein_g` 외에 `protein_source`를 `plant`, `animal`,
  `mixed`, `unknown`으로 저장하고, 식품 원천을 보존한다.
- 주의: 단백질 자체뿐 아니라 무엇을 대체해 먹었는지, 전체 식사 패턴과 생활
  습관이 관계에 영향을 줄 수 있다.

## 4. Dairy foods and health: an umbrella review of observational studies

- 서지: Godos J, et al. *International Journal of Food Sciences and Nutrition*.
  2020;71(2):138-151. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31199182/)
- 연구 유형: 관찰연구 기반 umbrella review.
- 핵심 메시지: 전체 유제품이라는 큰 범주만으로 건강 결과를 설명하기 어렵고,
  제품 종류, 지방 함량, 지역과 식사 패턴 같은 맥락이 중요하다.
- FoodMap 적용: `dairy_product_type`을 `milk`, `yogurt`, `cheese`, `butter`,
  `other`로 나누고, 가능한 경우 `fat_content`, `portion_g`, `region`도 저장한다.
- 주의: `dairy=true`를 건강 점수에 바로 연결하지 않는다. 식품군 내부의 차이와
  교란요인을 보존하는 사례로 사용한다.

## 5. Impact of sodium intake on blood pressure, mortality and major cardiovascular events

- 서지: Yang Q, et al. *Critical Reviews in Food Science and Nutrition*.
  2025;65(28):5903-5913. [PubMed](https://pubmed.ncbi.nlm.nih.gov/39624982/)
- 연구 유형: systematic review와 메타분석을 종합한 umbrella review.
- 규모: 사용자가 확인한 원문 정리 기준 56개 systematic review/meta-analysis를
  검토했다.
- 핵심 메시지: 나트륨 섭취 감소와 수축기 혈압 감소 방향의 근거를 종합한다.
  사용자가 정리한 수치 범위는 약 2-8.7 mmHg이며, 최종 효과크기 채택 전에는
  비교군, 기간, 측정 방식, certainty를 원문 표에서 다시 확인해야 한다.
- FoodMap 적용: `high_sodium` 태그 대신 `sodium_mg`, `portion_g`,
  `measurement_window`를 저장한다. 결과에는 검증된 혈압 측정값과 기기 정보를
  연결한다.
- 주의: 혈압은 운동, 수면, 약물, 스트레스, 측정 자세와 시간의 영향을 받는다.
  해당 변수를 기록하지 않으면 개인 연관성 분석의 신뢰도가 낮아진다.

## 6. Dietary fiber intake and mortality

- 서지: Ramezani F, et al. *Clinical Nutrition*. 2024;43(1):65-83.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/38011755/)
- 연구 유형: prospective cohort의 업데이트된 체계적 문헌고찰 및 메타분석.
- 범위: 사용자가 확인한 원문 정리 기준 64개 prospective study, 약 351만 명을
  포함했으며, 총 식이섬유와 유형·식품 원천을 구분해 분석한다.
- FoodMap 적용: 공통 기준은 `fiber_g`로 두되, 알 수 있으면 `fiber_type`
  (`soluble`, `insoluble`, `total`)과 `food_source`를 추가로 기록한다.
- 주의: 음식 기록만으로 용해성/불용성 섬유를 항상 정확히 알 수 있는 것은
  아니다. 근거 없는 추정 대신 `unknown`을 허용하고 영양성분 DB 연결로 보완한다.

## 7. Ultra-processed food exposure and adverse health outcomes

- 서지: Lane MM, et al. *BMJ*. 2024;384:e077310.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/38418082/)
- 연구 유형: 역학 메타분석을 종합한 systematic umbrella review.
- 범위: 사용자가 확인한 원문 정리 기준 약 990만 명, 45개 pooled analysis를
  정리했다. 논문은 NOVA 식품 분류 체계로 정의된 초가공식품 노출을 사용한다.
- FoodMap 적용: `processing_classification`, `classification_system`,
  `portion_g`, `frequency`를 저장한다. 처리 정도는 영양소 하나가 아닌 식품
  분류 특징으로 둔다.
- 주의: 논문도 근거의 신뢰도를 등급화하며, 다수 결과는 관찰연구에서 왔다.
  따라서 `ultra_processed`를 인과적 임상 효과로 저장하지 않는다.

## 8. Health effects of dietary risks in 195 countries, 1990-2017

- 서지: GBD 2017 Diet Collaborators. *The Lancet*. 2019;393:1958-1972.
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/30954305/)
- 연구 유형: 195개국을 대상으로 한 Global Burden of Disease 식이 위험 분석.
- 핵심 메시지: 높은 나트륨, 낮은 통곡물, 낮은 과일 섭취 등을 포함한 식이 위험
  요인의 인구 수준 질병 부담을 추정했다.
- FoodMap 적용: 어떤 성분과 식품군을 온톨로지에 먼저 포함할지 정하는
  우선순위 근거로 사용한다. 예: `sodium_mg`, 통곡물 식별, 과일 식품군.
- 주의: 국가·인구 수준의 질병 부담 추정치를 개인의 식사 한 끼 효과나 개인화
  계수로 옮기면 안 된다.

## 이 8편에서 바로 도출되는 데이터 수집 원칙

1. 음식명만 추출하지 말고, 음식별 양과 단위를 기록한다.
2. 영양소 양과 식품 원천을 함께 보존한다.
3. SSB·유제품·초가공식품처럼 식품군 또는 분류 체계인 노출은 영양소 g와
   별도로 저장한다.
4. 건강 결과를 연결할 때는 연구 설계, 대상 집단, 교란요인, 효과크기 단위,
   근거 수준, 출처 위치를 함께 기록한다.
5. 값이 없을 때 LLM이 채우지 않도록 `unknown`과 검토 상태를 허용한다.

## 다음 읽기 순서

1. 당 umbrella review: General FoodMap의 노출 분해 원칙을 잡는다.
2. 나트륨 논문: 용량과 웨어러블 또는 측정 결과를 연결하는 방식을 설계한다.
3. 단백질·식이섬유·유제품 논문: 식품 원천과 변이 정보가 왜 필요한지 확인한다.
4. 초가공식품·GBD 논문: 식품 분류와 온톨로지 우선순위를 정한다.
