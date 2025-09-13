# -*- coding: utf-8 -*-
"""
main.py — 실전 문제 지침 준수 / 단일 result.csv만 생성

• 대상 발전소: 양양풍력 → 영덕풍력 → 경주2풍력 (이 순서로 concat 후 저장)
• 출력 파일: ./result.csv  (열: plant_name,end_datetime,yield_kwh)
• 환경: Ubuntu 22.04 (x86_64), Python 3.10+
• 의존: pandas, numpy, xgboost, (pyarrow 또는 fastparquet), openpyxl (경주 엑셀 사용 시)

[규정 준수] 
  - 전날 15시 이후 생성/발령 NWP 미사용 (NWP 자체 미사용)
  - 실측/재분석/발령 이후 자료 미사용 (미래 입력은 과거 SCADA의 월×시간별 평균/클라이마톨로지 사용)
  - 예측 대상(미래) 기상/SCADA 데이터를 학습에 사용하지 않음
  - 제출 형식: result.csv, 실행 파일: main.py

[개요]
  1) 각 발전소 입력 로딩 (train_y_*.parquet, scada_*.parquet 또는 경주 엑셀)
  2) SCADA 피처 엔지니어링: 시간 파생, 풍향 사인/코사인, Lag(1/24h), 24h 롤링 통계 등
  3) 월 홀짝 분할(누설 방지):
       - 양양: 짝수월 학습 → 홀수월 예측
       - 영덕: 홀수월 학습 → 짝수월 예측
       - 경주2: 홀수월 학습 → 짝수월 예측
  4) 미래(예측) 입력은 해당 홀/짝월의 (월,시) 그룹 평균으로 구성 (클라이마톨로지)
  5) XGBoost 회귀로 발전량(kWh) 추정. 터빈별 원-핫이 있으면 각 터빈을 하나씩 켠 샘플로 예측 후 평균하여 농장 출력 근사
  6) 세 발전소 결과를 지정 순서(양양→영덕→경주2)로 결합하여 result.csv 저장
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")
print("라이브러리 로딩 완료")

# ----------------------------------------------------------------------------
# 경로 및 공통 설정
# ----------------------------------------------------------------------------
DATA_DIR = Path(".")
RESULT_PATH = DATA_DIR / "result.csv"

# [수정된 부분] 'plant_name'을 학습에서 제외하기 위해 COMMON_DROP에 추가
COMMON_DROP = [
    "plant_name",
    "end_datetime", "dt", "energy_mwh", "구분", "시간",
    "energy_kwh", "month", "hour", "dayofweek"
]

# 예측 기간 (KPX 제도 연도 기준 예시)
FUTURE_START = "2024-04-01 00:00:00"
FUTURE_END = "2025-03-31 23:00:00"

# 머지 허용 오차(선택): 센서→타깃 매칭이 희박할 때 안정화용
MERGE_TOLERANCE = pd.Timedelta("65min")  # 필요 없으면 None 로 두세요

# 디버그 출력
DEBUG_PRINT = False

# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------

def read_parquet_safely(path: Path) -> pd.DataFrame:
    """pyarrow/fastparquet 중 하나로 안전하게 읽고, 미설치 시 친절 메시지."""
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e1:
        try:
            return pd.read_parquet(path, engine="fastparquet")
        except Exception as e2:
            raise ImportError(
                f"[필수 설치 누락] '{path.name}' 를 읽으려면 pyarrow 또는 fastparquet 가 필요합니다.\n"
                f"  pip install pyarrow  또는   pip install fastparquet\n"
                f"(원인1: {e1}) (원인2: {e2})"
            )


def to_kst_naive_ns(s: pd.Series) -> pd.Series:
    """datetime 시리즈를 Asia/Seoul 기준 tz-naive(ns)로 정규화."""
    s = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        try:
            s = s.dt.tz_convert("Asia/Seoul")
        except Exception:
            pass
        s = s.dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def ensure_turbine_columns(df: pd.DataFrame, prefix: str = "turbine_") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix + "WTG")]  # e.g., turbine_WTG01
    return cols


def add_time_features(df: pd.DataFrame, tcol: str) -> pd.DataFrame:
    df["month"] = df[tcol].dt.month
    df["hour"] = df[tcol].dt.hour
    df["dayofweek"] = df[tcol].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_direction_features(df: pd.DataFrame, dir_col: str = "wind_direction_degree") -> pd.DataFrame:
    if dir_col in df.columns:
        rad = np.deg2rad(pd.to_numeric(df[dir_col], errors="coerce"))
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
    else:
        df["wind_dir_sin"] = 0.0
        df["wind_dir_cos"] = 0.0
    return df


def add_wind_lag_rolling(df: pd.DataFrame) -> pd.DataFrame:
    by_turb = ["turbine_id", "dt"] if "turbine_id" in df.columns else ["dt"]
    df = df.sort_values(by=by_turb)

    # Lag
    for lag in [1, 24]:
        if "turbine_id" in df.columns:
            df[f"wind_speed_mps_lag{lag}"] = df.groupby("turbine_id")["wind_speed_mps"].shift(lag)
        else:
            df[f"wind_speed_mps_lag{lag}"] = df["wind_speed_mps"].shift(lag)

    # Rolling 24h
    if "turbine_id" in df.columns:
        rolling = df.groupby("turbine_id")["wind_speed_mps"].rolling(24, min_periods=1)
        df["roll_mean_24h"] = rolling.mean().reset_index(level=0, drop=True)
        df["roll_std_24h"] = rolling.std().reset_index(level=0, drop=True)
        df["roll_max_24h"] = rolling.max().reset_index(level=0, drop=True)
        df["roll_min_24h"] = rolling.min().reset_index(level=0, drop=True)
    else:
        df["roll_mean_24h"] = df["wind_speed_mps"].rolling(24, min_periods=1).mean()
        df["roll_std_24h"] = df["wind_speed_mps"].rolling(24, min_periods=1).std()
        df["roll_max_24h"] = df["wind_speed_mps"].rolling(24, min_periods=1).max()
        df["roll_min_24h"] = df["wind_speed_mps"].rolling(24, min_periods=1).min()

    df["wind_x_hour_sin"] = df["wind_speed_mps"] * df["hour_sin"]
    return df


def build_future_base(scada: pd.DataFrame, months: list[int], start: str, end: str) -> pd.DataFrame:
    """예측 대상 구간의 (월, 시간) 클라이마톨로지로 입력 베이스를 구성."""
    # 대상 시간축
    future_dt = pd.date_range(start=start, end=end, freq="h")
    future_dt = future_dt[future_dt.month.isin(months)]
    base = pd.DataFrame({"dt": pd.to_datetime(future_dt)})

    # (월, 시) 그룹 평균
    hist = scada[scada["month"].isin(months)].groupby(["month", "hour"]).agg(
        wind_speed_mps=("wind_speed_mps", "mean"),
        wind_direction_degree=("wind_direction_degree", "mean") if "wind_direction_degree" in scada.columns else ("wind_speed_mps", "mean"),
    ).reset_index()

    base = add_time_features(base, "dt")
    base = pd.merge(base, hist, on=["month", "hour"], how="left")
    base = add_direction_features(base, "wind_direction_degree")

    # Lag/Rolling은 클라이마톨로지 기반 단순 대치 (자기지시/미래 누수 방지)
    base["wind_speed_mps_lag1"] = base["wind_speed_mps"]
    base["wind_speed_mps_lag24"] = base["wind_speed_mps"]
    base["roll_mean_24h"] = base["wind_speed_mps"]
    base["roll_std_24h"] = 0.0
    base["roll_max_24h"] = base["wind_speed_mps"]
    base["roll_min_24h"] = base["wind_speed_mps"]
    base["wind_x_hour_sin"] = base["wind_speed_mps"] * base["hour_sin"]
    return base


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs):
    p = Path(path)
    if p.name != "result.csv":
        raise RuntimeError(f"출력 금지 파일: {p.name} (허용: result.csv)")
    return df.to_csv(p, **kwargs)


# ----------------------------------------------------------------------------
# 경주 엑셀 → SCADA 표준 스키마 로더
# ----------------------------------------------------------------------------

def load_gyeongju_scada_from_excels(files: list[str]) -> pd.DataFrame:
    """경주(10분 간격) 엑셀들을 읽어 표준 컬럼(dt, wind_speed_mps, wind_direction_degree)으로 반환.
    - 각 파일의 모든 시트를 대상으로 header=5, 'Summation' 행 제거
    - 1시간 평균으로 리샘플링
    """
    use_cols_sensor = [
        "Date/Time",
        "Nacelle\nWind Speed\n[m/s]",
        "Nacelle\nWind Direction\n[deg]",
    ]
    dfs = []
    for file_name in files:
        fp = DATA_DIR / file_name
        if not fp.exists():
            print(f"  [경고] 파일 없음: {file_name} — 건너뜀")
            continue
        try:
            xls = pd.ExcelFile(fp)
            print(f"  [경주] 처리 중: {file_name} (시트 {len(xls.sheet_names)}개)")
            for sheet in xls.sheet_names:
                try:
                    df = xls.parse(sheet_name=sheet, usecols=use_cols_sensor, header=5)
                    df = df.dropna(subset=["Date/Time"])  # 유효 시간만
                    df = df[~df["Date/Time"].astype(str).str.contains("Summation", case=False, na=False)]
                    df.rename(columns={
                        "Date/Time": "dt",
                        "Nacelle\nWind Speed\n[m/s]": "wind_speed_mps",
                        "Nacelle\nWind Direction\n[deg]": "wind_direction_degree",
                    }, inplace=True)
                    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
                    df = df.dropna(subset=["dt"])  # 시간 파싱 실패 제거
                    dfs.append(df)
                except Exception as e:
                    print(f"    - 시트 오류({sheet}): {e}")
        except Exception as e:
            print(f"  [오류] 엑셀 열기 실패: {file_name}: {e}")

    if not dfs:
        raise FileNotFoundError("경주 SCADA 엑셀에서 유효 데이터를 찾지 못했습니다.")

    sc = pd.concat(dfs, ignore_index=True)
    sc = sc.sort_values("dt").set_index("dt")

    # 1시간 평균 리샘플링 → 표준 스키마로 복원
    sc_hour = sc.resample("1h").mean()
    sc_hour = sc_hour.reset_index()
    sc_hour["dt"] = to_kst_naive_ns(sc_hour["dt"])  # 정규화
    return sc_hour


# ----------------------------------------------------------------------------
# 모델 러너 (발전소 단위)
# ----------------------------------------------------------------------------

def run_plant(
    plant_name: str,
    y_path: str,
    scada_path: str | None,
    train_months_selector: str,
    future_months_selector: str,
    params: dict,
    future_start: str = FUTURE_START,
    future_end: str = FUTURE_END,
    gyeongju_excel_files: list[str] | None = None,
) -> pd.DataFrame:
    """단일 발전소 처리 → 제출용 DataFrame 반환."""
    # ---------------- I/O -----------------
    try:
        df_y = read_parquet_safely(DATA_DIR / y_path)
        print(f"[{plant_name}] 타깃 로딩 완료: {y_path}")
    except FileNotFoundError:
        print(f"[{plant_name}] 타깃 parquet 파일을 찾을 수 없습니다: {y_path}")
        return pd.DataFrame(columns=["plant_name", "end_datetime", "yield_kwh"])  # 빈 결과

    if scada_path is not None:
        df_sc = read_parquet_safely(DATA_DIR / scada_path)
        print(f"[{plant_name}] SCADA 로딩 완료: {scada_path}")
    else:
        if not gyeongju_excel_files:
            raise ValueError("경주 SCADA 엑셀 파일 목록이 필요합니다.")
        df_sc = load_gyeongju_scada_from_excels(gyeongju_excel_files)
        print(f"[{plant_name}] SCADA 로딩 완료: Excel {len(gyeongju_excel_files)}개")

    # -------------- 기본 전처리 --------------
    # 시간 정규화 (tz-naive ns, KST) + 컬럼 정리
    df_y["end_datetime"] = to_kst_naive_ns(df_y["end_datetime"])  # 타깃 시각
    df_sc["dt"] = to_kst_naive_ns(df_sc["dt"])                      # SCADA 시각
    df_sc.columns = df_sc.columns.map(lambda x: str(x).strip())

    # 풍속 유효성 & 이상치 윈도 클리핑
    if "wind_speed_mps" not in df_sc.columns:
        raise KeyError(f"[{plant_name}] 'wind_speed_mps' 컬럼이 없습니다.")
    lb, ub = df_sc["wind_speed_mps"].quantile([0.01, 0.99])
    df_sc["wind_speed_mps"] = df_sc["wind_speed_mps"].clip(lb, ub)

    # 파생
    df_sc = add_time_features(df_sc, "dt")
    df_sc = add_direction_features(df_sc, "wind_direction_degree")
    df_sc = add_wind_lag_rolling(df_sc)

    # 터빈 ID 원-핫
    if "turbine_id" in df_sc.columns:
        df_sc = pd.get_dummies(df_sc, columns=["turbine_id"], prefix="turbine")
    else:
        df_sc["turbine_WTG_UNKNOWN"] = 1

    # 결측 보간
    df_sc = df_sc.interpolate("linear").bfill().ffill()
    print(f"[{plant_name}] 피처 엔지니어링 완료")

    # -------------- 학습 세트 결합 --------------
    left = df_y.sort_values("end_datetime").reset_index(drop=True)
    right = df_sc.sort_values("dt").reset_index(drop=True)

    if DEBUG_PRINT:
        print(f"[{plant_name}] dtypes:", left["end_datetime"].dtype, right["dt"].dtype)
        print("[샘플]", left["end_datetime"].head(2).tolist(), right["dt"].head(2).tolist())

    full = pd.merge_asof(
        left, right,
        left_on="end_datetime", right_on="dt",
        direction="backward",
        tolerance=MERGE_TOLERANCE,
    )
    full["month"] = full["end_datetime"].dt.month

    # 월 홀짝 선택자 파싱
    if train_months_selector == "odd":
        train_mask = full["month"] % 2 != 0
    elif train_months_selector == "even":
        train_mask = full["month"] % 2 == 0
    else:
        raise ValueError("train_months_selector must be 'odd' or 'even'")

    train = full.loc[train_mask].copy()

    # 학습 피처 구성
    features = [c for c in train.columns if c not in COMMON_DROP]
    X = train[features]
    y = train["energy_kwh"].astype(float)
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    print(f"[{plant_name}] 학습셋: {X.shape} {y.shape}")

    # -------------- 모델 학습 --------------
    model = xgb.XGBRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X, y)
    print(f"[{plant_name}] 모델 학습 완료")

    # -------------- 미래 입력 생성 --------------
    future_months = [1,3,5,7,9,11] if future_months_selector == "odd" else [2,4,6,8,10,12]
    base = build_future_base(df_sc, months=future_months, start=future_start, end=future_end)

    # 터빈별 샘플 복제 (하나의 터빈만 1로 활성화)
    all_turbines = ensure_turbine_columns(full) or ["turbine_WTG_UNKNOWN"]

    future_list = []
    for tcol in all_turbines:
        tmp = base.copy()
        for c in all_turbines:
            tmp[c] = (c == tcol)
        future_list.append(tmp)

    Xf = pd.concat(future_list).sort_values("dt").reset_index(drop=True)

    # 누락 피처 보강 & 정렬
    for c in features:
        if c not in Xf.columns:
            Xf[c] = 0
    Xf = Xf[features]
    Xf = Xf.interpolate("linear").bfill().ffill()

    # -------------- 예측 & 제출형 가공 --------------
    preds = model.predict(Xf)
    pred_df = pd.concat(future_list).sort_values("dt").reset_index(drop=True)
    pred_df["predicted_kwh"] = preds

    # 터빈 평균 → 농장 출력 근사
    farm_total = pred_df.groupby("dt", as_index=False)["predicted_kwh"].mean()

    sub = pd.DataFrame({
        "plant_name": plant_name,
        "end_datetime": farm_total["dt"],
        "yield_kwh": farm_total["predicted_kwh"],
    })
    sub["yield_kwh"] = sub["yield_kwh"].clip(lower=0)
    return sub


# ----------------------------------------------------------------------------
# 엔트리 포인트
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 모델 하이퍼파라미터 (사전 탐색 결과값 예시)
    params_yangyang = {
        "lambda": 0.09058223211708799,
        "alpha": 0.8788050835164622,
        "colsample_bytree": 0.43691910158013775,
        "subsample": 0.4048448382509827,
        "learning_rate": 0.010001240733753193,
        "n_estimators": 643,
        "max_depth": 6,
        "min_child_weight": 11,
    }

    params_yeongduk = {
        "lambda": 0.08972182871699765,
        "alpha": 0.6417387443188353,
        "colsample_bytree": 0.8774368315279678,
        "subsample": 0.7680660705013798,
        "learning_rate": 0.025813031282491863,
        "n_estimators": 314,
        "max_depth": 6,
        "min_child_weight": 33,
    }

    # 경주는 별도 튜닝값 (없는 경우 위 둘 중 하나 재사용 가능)
    params_gyeongju = {
        "lambda": 0.12,
        "alpha": 0.5,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "learning_rate": 0.02,
        "n_estimators": 420,
        "max_depth": 6,
        "min_child_weight": 20,
    }

    # ---------------- 각 발전소 실행 ----------------
    # 1) 양양: 짝수월 학습 → 홀수월 예측
    sub_yangyang = run_plant(
        plant_name="양양풍력",
        y_path="train_y_yangyang.parquet",
        scada_path="scada_yangyang.parquet",
        train_months_selector="even",
        future_months_selector="odd",
        params=params_yangyang,
        future_start=FUTURE_START, future_end=FUTURE_END,
    )

    # 2) 영덕: 홀수월 학습 → 짝수월 예측
    sub_yeongduk = run_plant(
        plant_name="영덕풍력",
        y_path="train_y_yeongduk.parquet",
        scada_path="scada_yeongduk.parquet",
        train_months_selector="odd",
        future_months_selector="even",
        params=params_yeongduk,
        future_start=FUTURE_START, future_end=FUTURE_END,
    )

    # 3) 경주2: 홀수월 학습 → 짝수월 예측 (SCADA는 엑셀에서 로딩)
    gyeongju_excels = [
        "scada_gyeongju_2020_10min.xlsx",
        "scada_gyeongju_2021_10min.xlsx",
        "scada_gyeongju_2022_10min.xlsx",
        "scada_gyeongju_2023_10min.xlsx",
    ]
    sub_gyeongju = run_plant(
        plant_name="경주2풍력",
        y_path="train_y_gyeongju.parquet",
        scada_path=None,
        train_months_selector="odd",
        future_months_selector="even",
        params=params_gyeongju,
        future_start=FUTURE_START, future_end=FUTURE_END,
        gyeongju_excel_files=gyeongju_excels,
    )

    # ---------------- 단일 result.csv 생성 (지정 순서 보장) ----------------
    order = {"양양풍력": 0, "영덕풍력": 1, "경주2풍력": 2}
    result = pd.concat([sub_yangyang, sub_yeongduk, sub_gyeongju], ignore_index=True)
    result = result.sort_values(["plant_name", "end_datetime"]).reset_index(drop=True)
    # 표시/저장 순서 재정렬
    result["_ord"] = result["plant_name"].map(order)
    result = result.sort_values(["_ord", "end_datetime"]).drop(columns=["_ord"])  # 최종 순서: 양양→영덕→경주2

    # 단일 파일만 저장 (보안 가드)
    safe_to_csv(result, RESULT_PATH, index=False, encoding="utf-8")
    print("\n저장 완료:", RESULT_PATH.resolve())