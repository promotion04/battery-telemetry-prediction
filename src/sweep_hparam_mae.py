# src/sweep_hparam_mae.py
# -*- coding: utf-8 -*-
"""
하이퍼파라미터 스윕 + QCD 판단/표시(보강판)
- config 변형 → (선택) 전처리/라벨 → 학습 → 평가
- SOC/PCELL/TH 3개 메트릭 동시 수집 + (선택) 집계(mean/median/min/max)
- 결과:
  - artifacts/sweeps/<model>_<key_sanitized>/metrics.csv
  - <metric>_vs_<key>.png (metric=단일/집계 시)
  - 각 단일지표별 PNG (metric=all 시)
  - best.json (선택 목적지표 기준 최적값)
"""
import os, sys, json, argparse, shutil, subprocess, yaml
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------
# util: dict dotted get/set
# ----------------------------
def dotted_get(d: Dict[str, Any], key: str, default=None):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def dotted_set(d: Dict[str, Any], key: str, value: Any):
    parts = key.split("."); cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

# ----------------------------
# parse values: "3,5,7" or json "[3,5,7]"
# ----------------------------
def parse_values(raw: str) -> List[Any]:
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    out = []
    for tok in raw.split(","):
        s = tok.strip()
        if s == "true":  out.append(True);  continue
        if s == "false": out.append(False); continue
        try:
            out.append(float(s) if ("." in s or "e" in s.lower()) else int(s))
            continue
        except Exception:
            pass
        out.append(s)
    return out

def run(cmd: List[str]):
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError(f"Command failed (code={r.returncode}): {' '.join(cmd)}")
    tail = "\n".join(r.stdout.strip().splitlines()[-15:])
    if tail: print(tail)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def _norm_key(k: str) -> str:
    return k.strip().lower().replace("-", "_")

# ----------------------------
# metric alias
# ----------------------------
_ALIAS_MAP = {
    "soc_seq_mae": ["soc_mae", "mae_soc", "soc_l1", "soc_abs_err"],
    "pcell_seq_mae": ["pcell_mae", "p_cell_mae", "pcell_l1", "p_kw_mae", "p_avail_mae", "pcell_seq_mae"],
    "th_seq_mae": ["temp_seq_mae", "temp_h_mae", "th_mae", "th_seq_mae"],
}

def _alias_lookup(blob: Dict[str, Any], want_key: str):
    want = _norm_key(want_key)
    # 정확 일치
    for k, v in blob.items():
        if _norm_key(k) == want:
            return v
    # 별칭
    for std, cands in _ALIAS_MAP.items():
        if want == std or want in cands:
            for k, v in blob.items():
                if _norm_key(k) in ([std] + cands):
                    return v
    # 뒤집힌 키(예: th_seq_mae <-> mae_seq_th) 방지용 간단 처리
    parts = want.split("_")
    rev = "_".join(reversed(parts))
    for k, v in blob.items():
        if _norm_key(k) == rev:
            return v
    return None

def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

# ----------------------------
# read metrics from artifacts
# ----------------------------
def read_metric_value(art_dir: str, split: str, metric_key: str) -> Optional[float]:
    candidates = [
        os.path.join(art_dir, "metrics_ext.json"),
        os.path.join(art_dir, "qcd_report.json"),
        os.path.join(art_dir, f"mae_summary_{split}.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                v = _alias_lookup(blob, metric_key)
                if v is not None:
                    return _safe_float(v)
            except Exception:
                pass
    return None

def read_all_three(art_dir: str, split: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    soc = read_metric_value(art_dir, split, "soc_seq_MAE")
    pcell = read_metric_value(art_dir, split, "pcell_seq_MAE")
    th = read_metric_value(art_dir, split, "th_seq_MAE")
    return soc, pcell, th

# ----------------------------
# aggregation helpers
# ----------------------------
def aggregate_values(vals: List[Optional[float]], how: str) -> Optional[float]:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return None
    how = how.lower()
    if how == "mean":   return float(np.mean(arr))
    if how == "median": return float(np.median(arr))
    if how == "min":    return float(np.min(arr))
    if how == "max":    return float(np.max(arr))
    return None

# ----------------------------
# plotting
# ----------------------------
def plot_xy(xs: List[Any], ys: List[Optional[float]], xlabels: List[str],
            key_label: str, metric_label: str, out_path: str):
    vals = np.array([np.nan if y is None else y for y in ys], dtype=float)
    try:
        plt.figure(figsize=(7, 4.5))
        idx = np.arange(len(xs))
        plt.plot(idx, vals, marker="o")
        plt.xticks(idx, xlabels, rotation=15)
        plt.xlabel(key_label); plt.ylabel(metric_label)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out_path, dpi=600, bbox_inches="tight"); plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--model_name", type=str, default="tcn",
                    choices=["tcn", "lstm", "transformer"])
    ap.add_argument("--key", type=str, required=True,
                    help="스윕할 점표기 키 (예: model.tcn.kernel_size)")
    ap.add_argument("--values", type=str, required=True,
                    help="값 목록 (CSV: 3,5,7 또는 JSON: [3,5,7])")
    ap.add_argument("--metric", type=str, default="soc_seq_MAE",
                    help="단일 지표(soc_seq_MAE/pcell_seq_MAE/th_seq_MAE) "
                         "또는 all/mean/median/min/max")
    ap.add_argument("--split", type=str, default="val",
                    choices=["train", "val", "test"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rerun_preprocess", action="store_true")
    ap.add_argument("--limit_rows", type=int, default=0,
                    help="전처리 단계에서 데이터 행 수 제한(해당 스크립트가 지원할 때만 전달)")
    args = ap.parse_args()

    all_metrics = ["soc_seq_MAE", "pcell_seq_MAE", "th_seq_MAE"]
    agg_metrics = ["mean", "median", "min", "max"]

    metric_mode = "single"
    target_metric = args.metric
    if args.metric.lower() == "all":
        metric_mode = "all"
        target_metric = None
    elif args.metric.lower() in agg_metrics:
        metric_mode = "agg"
        target_metric = args.metric.lower()  # mean/median/min/max

    base_cfg = load_yaml(args.config)
    values = parse_values(args.values)

    key_sanitized = args.key.replace(".", "_")
    sweep_root = os.path.join("artifacts", "sweeps", f"{args.model_name}_{key_sanitized}")
    os.makedirs(sweep_root, exist_ok=True)
    shutil.copy2(args.config, os.path.join(sweep_root, "base_config.yaml"))

    rows = []
    xlabels = [str(v) for v in values]

    for v in values:
        # 1) cfg 생성
        cfg = load_yaml(args.config)
        dotted_set(cfg, args.key, v)
        if "train" not in cfg:
            cfg["train"] = {}
        cfg["train"]["epochs"] = int(args.epochs)

        tmp_cfg_path = os.path.join(sweep_root, f"cfg_{key_sanitized}_{str(v).replace(' ','')}.yaml")
        save_yaml(cfg, tmp_cfg_path)

        # 2) (선택) 전처리/라벨
        if args.rerun_preprocess:
            cmd_pre = [sys.executable, "-m", "src.preprocess", "--config", tmp_cfg_path]
            if args.limit_rows and args.limit_rows > 0:
                cmd_pre += ["--limit_rows", str(args.limit_rows)]
            run(cmd_pre)
            run([sys.executable, "-m", "src.labels", "--config", tmp_cfg_path])

        # 3) 학습/평가
        run([sys.executable, "-m", "src.train", "--config", tmp_cfg_path, "--model_name", args.model_name])
        run([sys.executable, "-m", "src.evaluate", "--config", tmp_cfg_path, "--model_name", args.model_name, "--split", args.split])

        # 4) 메트릭 수집
        art_dir = os.path.join("artifacts", args.model_name, args.split)
        soc, pcell, th = read_all_three(art_dir, args.split)
        row = dict(value=v, soc_seq_MAE=soc, pcell_seq_MAE=pcell, th_seq_MAE=th)

        # 5) 집계값(참고용) 모두 추가
        row["mean"]   = aggregate_values([soc, pcell, th], "mean")
        row["median"] = aggregate_values([soc, pcell, th], "median")
        row["min"]    = aggregate_values([soc, pcell, th], "min")
        row["max"]    = aggregate_values([soc, pcell, th], "max")

        rows.append(row)

    # 6) CSV 저장
    df = pd.DataFrame(rows)
    out_csv = os.path.join(sweep_root, "metrics.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8", float_format="%.6f")

    # 7) 플롯 저장
    def _series_safe(col: str) -> List[Optional[float]]:
        return [None if pd.isna(x) else float(x) for x in df[col].tolist()]

    if metric_mode == "single":
        y = _series_safe(target_metric)
        out_png = os.path.join(sweep_root, f"{target_metric}_vs_{key_sanitized}.png")
        plot_xy(values, y, xlabels, args.key, target_metric, out_png)
    elif metric_mode == "agg":
        y = _series_safe(target_metric)
        out_png = os.path.join(sweep_root, f"agg_{target_metric}_vs_{key_sanitized}.png")
        plot_xy(values, y, xlabels, args.key, f"agg_{target_metric}", out_png)
    else:  # all
        for m in all_metrics:
            y = _series_safe(m)
            out_png = os.path.join(sweep_root, f"{m}_vs_{key_sanitized}.png")
            plot_xy(values, y, xlabels, args.key, m, out_png)

    # 8) 최적값 탐색 + 요약 저장 (목적지표 기준)
    best_info = {}
    def _best_of(col: str):
        if col not in df.columns: return None
        sub = df[["value", col]].dropna()
        if sub.empty: return None
        idx = sub[col].astype(float).idxmin()
        rec = df.loc[idx]
        return {"value": rec["value"], "metric": float(rec[col])}

    if metric_mode == "single":
        best_info[target_metric] = _best_of(target_metric)
    elif metric_mode == "agg":
        best_info[f"agg_{target_metric}"] = _best_of(target_metric)
    else:  # all + 참조용 집계도 함께
        for m in all_metrics + ["mean", "median", "min", "max"]:
            best_info[m] = _best_of(m)

    with open(os.path.join(sweep_root, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    print("\n[sweep] saved dir:", sweep_root)
    print("[sweep] metrics csv:", out_csv)
    if best_info:
        print("[sweep] best summary:")
        for k, v in best_info.items():
            if v is not None:
                print(f"  - {k}: value={v['value']}  metric={v['metric']:.6f}")

if __name__ == "__main__":
    main()
