# src/preprocess.py
# -*- coding: utf-8 -*-
"""
전처리 파이프라인 (raw BMS CSV -> merged_clean.csv)
- Time 파싱(XdYhZmWs) + 같은 초 중복 균등 분할
- 고정 dt 리샘플(선형 보간)
- Total Pressure -> Voltage 리네임, Power = Voltage * Current 재계산
- Temp 집계(High/Avg/Low), dT/dt(평활 후 과거차분, 클리핑)
- 최근 에너지(Wh_recent_2s/5s)
- 이상치 처리(EWMA 잔차 robust z-score -> 선형보간)
- 변환(Yeo–Johnson: Current/Power), 차분(1차: Current/Power)
- 스케일링(Robust/Standard/MinMax) — train 세션만으로 fit, 전체 transform
"""

import os
import re
import glob
import pickle
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.preprocessing import PowerTransformer

from src.config import load_config

# -------------------------------
# 유틸: 시간/세션 파싱
# -------------------------------
TIME_RE = re.compile(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.I)
_SES_RE = re.compile(r"(\d{4})")

def parse_time_to_sec(s: str) -> int:
    """'10d2h46m17s' -> 초(int). 숫자면 그대로 캐스팅."""
    if not isinstance(s, str):
        try:
            return int(s)
        except Exception:
            return 0
    m = TIME_RE.search(s.strip())
    if not m:
        return 0
    d, h, m_, s_ = [(int(x) if x else 0) for x in m.groups()]
    return d * 86400 + h * 3600 + m_ * 60 + s_

def _extract_session_name(path: str) -> str:
    """파일명에서 4자리 숫자 세션ID 추출. 없으면 앞 4글자 fallback."""
    base = os.path.basename(path)
    m = _SES_RE.search(base)
    return m.group(1) if m else base[:4]

# -------------------------------
# 유틸: 숫자 추출(보강)
# -------------------------------
def _to_num(x):
    if isinstance(x, str):
        s = x.strip().upper()
        for tok in ["MV", "V", "MA", "A", "AH", "%"]:
            s = s.replace(tok, "")
        s = s.replace(",", "")
        try:
            return float(s)
        except Exception:
            return np.nan
    return x

# -------------------------------
# 유틸: 이상치 처리 (pandas 2.x 호환)
# -------------------------------
def ewma_outlier_clip(series: pd.Series, alpha=0.2, z_thresh=3.5) -> pd.Series:
    """
    EWMA 기반 잔차의 robust z-score로 이상치 감지 -> NaN -> 선형보간.
    pandas 2.x에는 Series.mad() 없음 -> MAD 직접 계산.
    """
    x = pd.to_numeric(series, errors="coerce").astype(float).copy()
    ema = x.ewm(alpha=alpha, adjust=False).mean()
    resid = x - ema

    med = np.nanmedian(resid)
    mad = np.nanmedian(np.abs(resid - med))
    if not np.isfinite(mad) or mad <= 0:
        mad = 1e-6

    z = (resid - med) / (1.4826 * mad)
    z = np.clip(z, -10, 10)

    mask = np.abs(z) > z_thresh
    x[mask] = np.nan
    return x.interpolate(limit_direction="both")

def apply_outlier_block(df: pd.DataFrame, cols: List[str], alpha: float, z_thresh: float) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = ewma_outlier_clip(df[c], alpha=alpha, z_thresh=z_thresh)
    return df

# -------------------------------
# 유틸: 변환/차분
# -------------------------------
def yeo_johnson_safe(col: pd.Series) -> Tuple[np.ndarray, PowerTransformer]:
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    vals = pd.to_numeric(col, errors="coerce").fillna(0.0).values.reshape(-1, 1)
    tr = pt.fit_transform(vals).reshape(-1)
    return tr, pt

# -------------------------------
# 1파일 전처리 (시간, 리샘플, 파생)
# -------------------------------
def load_one_bms(path: str, dt: float) -> pd.DataFrame:
    # CSV 로드 (인코딩 유연)
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949", engine="python")

    # 컬럼 정리
    df.columns = [c.strip() for c in df.columns]
    if "Total Pressure" in df.columns and "Voltage" not in df.columns:
        df = df.rename(columns={"Total Pressure": "Voltage"})

    # 숫자화
    for c in ["Voltage", "Current", "Power", "Remaining Capacity", "Temp1", "Temp2", "Temp3", "Temp4", "SOC"]:
        if c in df.columns:
            df[c] = df[c].apply(_to_num)

    # 시간 파싱
    if "Time" not in df.columns:
        raise KeyError(f"{os.path.basename(path)}에 'Time' 컬럼이 없습니다.")
    df["t_sec"] = df["Time"].apply(parse_time_to_sec)

    # 같은 초 내 중복 -> 균등 분할(offset = i/k)
    df = df.sort_values(["t_sec"]).reset_index(drop=True)
    counts = df.groupby("t_sec")["t_sec"].transform("count").astype(int)
    order = df.groupby("t_sec").cumcount().astype(int)
    offset = order / counts.replace(0, 1)
    df["t_float"] = df["t_sec"] + offset

    # 고정 간격 그리드 생성
    t0, t1 = float(df["t_float"].iloc[0]), float(df["t_float"].iloc[-1])
    grid = np.arange(t0, t1 + 1e-12, dt)

    # 보간할 숫자 채널
    num_cols = ["Voltage", "Current", "SOC", "Temp1", "Temp2", "Temp3", "Temp4"]
    base = df[num_cols + ["t_float"]].dropna(subset=["t_float"]).set_index("t_float").sort_index()
    interp = base.reindex(base.index.union(grid)).interpolate(method="index").reindex(grid)
    interp.index.name = "t_float"
    out = interp.reset_index()

    # Power 재계산 (원본 Power 무시)
    out["Power"] = out["Voltage"].fillna(0.0) * out["Current"].fillna(0.0)

    # Temp 집계
    out["Temp_high"] = out[["Temp1", "Temp2", "Temp3", "Temp4"]].max(axis=1)
    out["Temp_avg"]  = out[["Temp1", "Temp2", "Temp3", "Temp4"]].mean(axis=1)
    out["Temp_low"]  = out[["Temp1", "Temp2", "Temp3", "Temp4"]].min(axis=1)

    # dT/dt: 0.5~1.0s 평활 후 과거차분(k ≈ 1초)
    try:
        sm = savgol_filter(out["Temp_avg"].interpolate().bfill().values, 9, 2)
    except Exception:
        sm = out["Temp_avg"].interpolate().bfill().values
    k = max(1, int(round(1.0 / dt)))  # 약 1초 차분
    dtdt = np.zeros_like(sm)
    dtdt[k:] = (sm[k:] - sm[:-k]) / (k * dt)
    out["dT_dt"] = np.clip(dtdt, -1.0, 1.0)

    # 최근 에너지(Wh) 이동합
    dWh = (out["Power"].fillna(0.0) * dt) / 3600.0
    n2, n5 = int(round(2.0 / dt)), int(round(5.0 / dt))
    out["Wh_recent_2s"] = dWh.rolling(n2, min_periods=1).sum()
    out["Wh_recent_5s"] = dWh.rolling(n5, min_periods=1).sum()

    # 세션 id
    out["session"] = _extract_session_name(path)
    return out

# -------------------------------
# 스케일러 도우미
# -------------------------------
def build_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    if kind == "standard":
        return StandardScaler()
    return MinMaxScaler()

def fit_scaler(train_df: pd.DataFrame, use_cols: List[str], kind: str):
    scaler = build_scaler(kind)
    scaler.fit(train_df[use_cols].values)
    return scaler

# -------------------------------
# 메인
# -------------------------------
def main():
    cfg = load_config()

    raw_dir = cfg["data"]["raw_dir"]
    interim_dir = cfg["data"]["interim_dir"]
    os.makedirs(interim_dir, exist_ok=True)

    dt = float(cfg["window"]["dt"])
    past_sec = cfg["window"].get("past_seconds", cfg["window"].get("past", 15.0))
    horizon_sec = cfg["window"].get("horizon_seconds", cfg["window"].get("future", 5.0))

    # preprocess 설정 (기본값 보정)
    pp_cfg = cfg.get("preprocess", {})
    out_cfg = pp_cfg.get("outlier", {"enabled": True, "alpha": 0.2, "z_thresh": 3.5})
    tf_cfg = pp_cfg.get("transform", {"yeo_johnson_cols": ["Current", "Power"], "diff_cols": ["Current", "Power"]})
    sc_cfg = pp_cfg.get("scaling", {"type": "robust", "save_path": "artifacts/scalers/input_robust.pkl"})

    # 파일 수집 (유연한 glob)  ➜  r0_soc_file 제외 + Time 컬럼 검증
    r0_path = os.path.abspath(cfg["data"].get("r0_soc_file", ""))

    patts = [
        os.path.join(raw_dir, "*bms데이터.csv"),
        os.path.join(raw_dir, "*BMS*.csv"),
        os.path.join(raw_dir, "*.csv"),
    ]
    cands = []
    for p in patts:
        cands.extend(glob.glob(p))
    # 중복 제거
    cands = sorted(set(cands))

    # 1) r0_soc_file 제외
    cands = [f for f in cands if os.path.abspath(f) != r0_path]

    # 2) 헤더에 Time 컬럼이 있는 CSV만 채택(인코딩 자동 판별)
    files = []
    for f in cands:
        try:
            hdr = pd.read_csv(f, nrows=0, encoding="utf-8")
        except UnicodeDecodeError:
            hdr = pd.read_csv(f, nrows=0, encoding="cp949")
        cols = [c.strip() for c in hdr.columns.tolist()]
        if "Time" in cols:
            files.append(f)

    if not files:
        raise FileNotFoundError(
            f"raw_dir({raw_dir})에서 전처리 대상 CSV를 찾지 못했습니다. "
            f"(r0_soc_file은 자동 제외, Time 컬럼 있는 파일만 선택)"
        )

    # 각 파일 처리 후 병합  ➜  각 파일마다 session 컬럼 보장
    dfs = []
    for f in files:
        df_i = load_one_bms(f, dt)
        if "session" not in df_i.columns:
            df_i["session"] = _extract_session_name(f)
        if df_i["session"].isna().any() or (df_i["session"] == "").any():
            df_i["session"] = df_i["session"].replace("", _extract_session_name(f)).fillna(_extract_session_name(f))
        dfs.append(df_i)

    if not dfs:
        raise RuntimeError("전처리 대상 파일 로드 결과가 비어 있습니다.")

    all_df = pd.concat(dfs, ignore_index=True)

    # 세션 백업(가드)
    session_backup = all_df["session"].astype(str).copy()

    # 방어: 세션 요약 로그
    try:
        ses_counts = all_df["session"].value_counts().sort_index()
        print(f"sessions detected: {list(ses_counts.index)}")
        print(ses_counts.to_string())
    except Exception:
        print("WARNING: 'session' 요약에 실패했습니다. 컬럼 상태를 확인하세요.")

    # SOC 안정화
    if "SOC" in all_df.columns:
        all_df["SOC"] = all_df["SOC"].clip(lower=0, upper=100)

    # ====== 이상치 처리 (세션별로 적용) ======
    if out_cfg.get("enabled", True):
        alpha = float(out_cfg.get("alpha", 0.2))
        zt = float(out_cfg.get("z_thresh", 3.5))
        outlier_cols = [
            "Voltage", "Current", "Power",
            "Temp1", "Temp2", "Temp3", "Temp4",
            "Temp_high", "Temp_avg", "Temp_low",
            "SOC"
        ]
        gb = all_df.groupby("session", group_keys=False)
        try:
            all_df = gb.apply(lambda g: apply_outlier_block(g, outlier_cols, alpha, zt),
                              include_groups=False).reset_index(drop=True)
        except TypeError:
            all_df = gb.apply(lambda g: apply_outlier_block(g, outlier_cols, alpha, zt)).reset_index(drop=True)

        # 세션 복원 가드
        if "session" not in all_df.columns:
            all_df.insert(0, "session", session_backup)
        else:
            all_df["session"] = all_df["session"].astype(str).fillna(session_backup)

    # ====== 변환(YJ) & 차분(1차) ======
    yj_cols = tf_cfg.get("yeo_johnson_cols", ["Current", "Power"])
    diff_cols = tf_cfg.get("diff_cols", ["Current", "Power"])

    yj_transformers: Dict[str, PowerTransformer] = {}
    for c in yj_cols:
        if c in all_df.columns:
            tr, pt = yeo_johnson_safe(all_df[c])
            all_df[c + "_yj"] = tr
            yj_transformers[c] = pt

    for c in diff_cols:
        if c in all_df.columns:
            all_df[c + "_diff1"] = all_df[c].diff().fillna(0.0)

    # ====== 스케일링 (Train 세션으로 fit -> 전체 transform) ======
    base_feat = [
        "Voltage", "Current", "Power", "SOC",
        "Temp_high", "Temp_avg", "Temp_low", "dT_dt",
        "Wh_recent_2s", "Wh_recent_5s",
        "Current_yj", "Power_yj", "Current_diff1", "Power_diff1"
    ]
    use_cols = [c for c in base_feat if c in all_df.columns]

    # 세션 복원 가드 (스케일러 fit 전에 보장)
    if "session" not in all_df.columns:
        all_df.insert(0, "session", session_backup)
    else:
        all_df["session"] = all_df["session"].astype(str).fillna(session_backup)

    train_sessions = set(cfg["data"]["sessions_train"])
    mask_tr = all_df["session"].isin(train_sessions)
    if not mask_tr.any():
        raise RuntimeError("train 세션이 비어 있습니다. config.yaml의 data.sessions_train을 확인하세요.")

    scaler_kind = sc_cfg.get("type", "robust")
    scaler = fit_scaler(all_df[mask_tr], use_cols, scaler_kind)

    # 변환은 피처 컬럼에만 적용 (다른 메타 컬럼 보존)
    all_df[use_cols] = scaler.transform(all_df[use_cols].values)

    save_path = sc_cfg.get("save_path", "artifacts/scalers/input_scaler.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"scaler": scaler, "cols": use_cols, "kind": scaler_kind}, f)

    save_csv = os.path.join(interim_dir, "merged_clean.csv")
    all_df.to_csv(save_csv, index=False, encoding="utf-8")

    print("saved:", save_csv)
    print(f"rows={len(all_df)} | dt={dt}s | past={past_sec}s | horizon={horizon_sec}s")
    print("scaled columns:", use_cols)

if __name__ == "__main__":
    main()
