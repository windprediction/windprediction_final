# 풍력 발전량 예측 — 제출용

## 개요
- `main.py` 실행 시 **단일 `result.csv`**를 생성합니다.
- 대상: **양양풍력 → 영덕풍력 → 경주2풍력** (이 순서로 병합·저장)
- 규정 준수: **실측/재분석/NWP 미사용**, 전날 15시 이후 생성 예보 미사용, 데이터 외부 반출 금지.

---

## 동작 원리(알고리즘)
1. **데이터 로드**
   - 타깃 발전량(`train_y_*.parquet`)과 SCADA(`scada_*.parquet` / 경주는 연도별 Excel)를 읽습니다.
2. **시간 정규화**
   - 모든 시각을 **KST tz-naive(ns)**로 통일.
3. **전처리 & 피처 엔지니어링**
   - 풍속 이상치 **1–99% 클리핑**.
   - 시간 파생: `month/hour/dayofweek/is_weekend`, `hour_sin/cos`.
   - 풍향 파생: `sin/cos(wind_direction_degree)`.
   - 풍속 **Lag(1h/24h)**, **24h rolling(mean/std/max/min)**, 상호작용(`wind_x_hour_sin`).
   - 터빈 ID가 있으면 **원-핫 인코딩**(없으면 `turbine_WTG_UNKNOWN=1`).
4. **타깃–SCADA 결합**
   - `merge_asof(backward, tolerance≈65min)`로 타깃 시각에 가장 가까운 SCADA를 매칭.
5. **학습/예측 분할(누설 방지)**
   - **양양:** 짝수월 학습 → 홀수월 예측  
   - **영덕:** 홀수월 학습 → 짝수월 예측  
   - **경주2:** 홀수월 학습 → 짝수월 예측
6. **미래 입력 구성(클라이마톨로지)**
   - 과거 SCADA의 **(월×시) 그룹 평균**으로 예측 구간의 입력 베이스를 만듭니다.
   - Lag/rolling은 해당 평균값으로 **자기지시 없이** 단순 대치(미래 누수 차단).
7. **모델 학습/예측**
   - 회귀기: **XGBoost**(하이퍼파라미터는 코드 내 고정값).
   - 터빈별 샘플을 만들어 **각 터빈 예측 → 평균**하여 농장 출력 근사.
8. **출력 생성**
   - 세 발전소 결과를 **양양→영덕→경주2** 순으로 정렬·결합해 `result.csv` 저장.  
   - 스키마: `plant_name,end_datetime,yield_kwh`

---

## 실행 방법 (uv)
```bash
uv sync --python 3.10
uv run python main.py
# 결과: ./result.csv 생성
