# src/evaluate.py
# -*- coding: utf-8 -*-
"""
Evaluation (sequence targets for SOC / Temp_high/avg/low / P_cell)
- Metrics: MAE/RMSE/R^2, MAPE(+valid ratio), Pearson/Spearman, KS-test(optional)
- Sequence-aware: per-horizon arrays for all targets
- Crossing-time metrics for temperature thresholds (50, 58°C)
- Session / condition breakdown CSVs
- Speed summary (avg latency, throughput)
- Plots: per-horizon MAE, scatter (last-step), hist of errors, sample timeseries
- per_horizon_metrics.csv
- QCD report (qcd_report.json)

Outputs saved under: artifacts/<model_name>/<split>/
"""

import os
import json
import argparse
import time
import hashlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Headless-safe Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "savefig.dpi": 600,
    "figure.dpi": 120,
    "font.size": 12,
    "axes.labelweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.antialiased": True,
    "figure.constrained_layout.use": True
})

from src.config import load_config
from src.models.registry import build_model

try:
    from scipy.stats import ks_2samp
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ----------------------
# Dataset
# ----------------------
class MTSet(Dataset):
    def __init__(self, pack: Dict[str, np.ndarray], sid: np.ndarray):
        self.X = pack["X"].astype(np.float32)
        self.soc = pack["soc"].astype(np.float32)       # (N,H)
        self.th  = pack["th"].astype(np.float32)        # (N,H)
        self.ta  = pack["ta"].astype(np.float32)        # (N,H)
        self.tl  = pack["tl"].astype(np.float32)        # (N,H)
        self.p_cell = pack["p_cell"].astype(np.float32) # (N,H)
        self.sid = np.asarray(sid)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.soc[i]),
            torch.from_numpy(self.th[i]),
            torch.from_numpy(self.ta[i]),
            torch.from_numpy(self.tl[i]),
            torch.from_numpy(self.p_cell[i]),
            str(self.sid[i]),
        )


# ----------------------
# Metrics
# ----------------------
def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return mae, rmse

def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def mape_np(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    rel = np.abs(y_pred - y_true) / denom
    valid = (np.abs(y_true) >= eps)
    mape = float(np.mean(rel[valid]) * 100.0) if np.any(valid) else float("nan")
    valid_ratio = float(np.mean(valid.astype(np.float32)))
    return mape, valid_ratio

def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    sx = np.std(x); sy = np.std(y)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(x, y)
    return float(c[0,1])

def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    r1 = pd.Series(x).rank(method="average").values
    r2 = pd.Series(y).rank(method="average").values
    return pearson_np(r1, r2)

def seq_mae_rmse(P: np.ndarray, T: np.ndarray):
    """P,T: (N,H) -> overall MAE/RMSE + per-horizon vectors"""
    err = P - T
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae_h = np.mean(np.abs(err), axis=0)
    rmse_h = np.sqrt(np.mean(err**2, axis=0))
    return mae, rmse, mae_h, rmse_h

def crossing_time(seq: np.ndarray, thresh: float, dt: float):
    idx = np.where(seq >= thresh)[0]
    if len(idx) == 0:
        return False, float("inf")
    return True, float(idx[0] * dt)

def bin_idx(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(vals, edges, right=False) - 1, 0, len(edges)-2)

def _save_png_svg(path_png: str):
    plt.savefig(path_png, dpi=600, bbox_inches="tight")
    try:
        if path_png.lower().endswith(".png"):
            plt.savefig(path_png[:-4] + ".svg", bbox_inches="tight")
    except Exception:
        pass

def cfg_hash(cfg: dict) -> str:
    try:
        s = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "na"


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default="tcn", choices=["tcn","lstm","transformer"])
    parser.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    parser.add_argument("--sample_plots", type=int, default=6, help="예시 시퀀스 플롯 개수")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # out dir
    out_dir = os.path.join("artifacts", args.model_name, args.split)
    os.makedirs(out_dir, exist_ok=True)

    # 데이터
    npz_path = os.path.join(cfg["data"]["processed_dir"], "dataset_windows_labels.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    sid_all = np.array([str(s).strip() for s in npz["sid"]], dtype=object)

    split_key = dict(train="sessions_train", val="sessions_val", test="sessions_test")[args.split]
    if split_key not in cfg["data"]:
        raise KeyError(f"'{split_key}' not found in config.data.")
    want = [str(s).strip() for s in cfg["data"][split_key]]

    mask = np.isin(sid_all, np.array(want, dtype=object))
    if not mask.any():
        raise RuntimeError(f"No samples matched {split_key}={want}. Available: {sorted(set(sid_all))}")

    keys = ["X","soc","th","ta","tl","p_cell"]
    pack = {k: npz[k][mask] for k in keys}
    sid_sel = sid_all[mask]

    dt = float(cfg["window"]["dt"])
    horizon = pack["th"].shape[-1]
    in_feat = pack["X"].shape[-1]

    # 로더
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    loader = DataLoader(
        MTSet(pack, sid_sel),
        batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    # 모델/체크포인트
    model = build_model(in_feat, horizon, cfg, kind=args.model_name)
    ckpt = os.path.join("artifacts", args.model_name, "models", f"multitask_{args.model_name}_cellkW.pt")
    if not os.path.exists(ckpt):
        legacy_ckpt = "artifacts/models/multitask_tcn_cellkW.pt"
        if os.path.exists(legacy_ckpt):
            ckpt = legacy_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    # -------- 추론 --------
    preds = dict(soc=[], pcell=[], th=[], ta=[], tl=[], sid=[])
    trues = dict(soc=[], pcell=[], th=[], ta=[], tl=[], sid=[])

    use_amp = torch.cuda.is_available()
    autocast_dtype = torch.float16 if use_amp else torch.bfloat16
    tot_samples = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for Xb, soc, th, ta, tl, pc, ss in loader:
            Xb, soc, th, ta, tl, pc = Xb.to(device), soc.to(device), th.to(device), ta.to(device), tl.to(device), pc.to(device)
            with torch.autocast(device_type=("cuda" if use_amp else "cpu"), dtype=autocast_dtype):
                psoc, pth, pta, ptl, ppc = model(Xb)
            preds["soc"].append(psoc.cpu().numpy());     trues["soc"].append(soc.cpu().numpy())
            preds["pcell"].append(ppc.cpu().numpy());    trues["pcell"].append(pc.cpu().numpy())
            preds["th"].append(pth.cpu().numpy());       trues["th"].append(th.cpu().numpy())
            preds["ta"].append(pta.cpu().numpy());       trues["ta"].append(ta.cpu().numpy())
            preds["tl"].append(ptl.cpu().numpy());       trues["tl"].append(tl.cpu().numpy())
            preds["sid"].extend(list(ss));               trues["sid"].extend(list(ss))
            tot_samples += Xb.size(0)
    infer_s = time.perf_counter() - t0
    avg_latency_ms = float(1000.0 * infer_s / max(1, tot_samples))
    throughput_sps = float(tot_samples / max(infer_s, 1e-9))

    # 합치기
    def cat_seq(k): return np.concatenate(preds[k], axis=0), np.concatenate(trues[k], axis=0)
    soc_p, soc_t     = cat_seq("soc")
    pcell_p, pcell_t = cat_seq("pcell")
    th_p, th_t       = cat_seq("th")
    ta_p, ta_t       = cat_seq("ta")
    tl_p, tl_t       = cat_seq("tl")
    sid_used = np.array(preds["sid"])

    # -------- 회귀 지표 --------
    soc_mae, soc_rmse, soc_mae_h, soc_rmse_h = seq_mae_rmse(soc_p, soc_t)
    pcell_mae, pcell_rmse, pcell_mae_h, pcell_rmse_h = seq_mae_rmse(pcell_p, pcell_t)
    th_mae, th_rmse, th_mae_h, th_rmse_h = seq_mae_rmse(th_p, th_t)
    ta_mae, ta_rmse, ta_mae_h, ta_rmse_h = seq_mae_rmse(ta_p, ta_t)
    tl_mae, tl_rmse, tl_mae_h, tl_rmse_h = seq_mae_rmse(tl_p, tl_t)

    # R^2 (sequence flattened)
    soc_r2   = r2_score_np(soc_t.reshape(-1),   soc_p.reshape(-1))
    pcell_r2 = r2_score_np(pcell_t.reshape(-1), pcell_p.reshape(-1))
    th_r2    = r2_score_np(th_t.reshape(-1),    th_p.reshape(-1))
    ta_r2    = r2_score_np(ta_t.reshape(-1),    ta_p.reshape(-1))
    tl_r2    = r2_score_np(tl_t.reshape(-1),    tl_p.reshape(-1))

    # MAPE
    soc_mape, soc_mvalid       = mape_np(soc_t,   soc_p, eps=1e-3)
    pcell_mape, pcell_mvalid   = mape_np(pcell_t, pcell_p, eps=1e-6)
    th_mape, th_mvalid         = mape_np(th_t,    th_p,    eps=1e-2)
    ta_mape, ta_mvalid         = mape_np(ta_t,    ta_p,    eps=1e-2)
    tl_mape, tl_mvalid         = mape_np(tl_t,    tl_p,    eps=1e-2)

    # Correlation (flattened)
    soc_pear = pearson_np(soc_t.reshape(-1), soc_p.reshape(-1));       soc_spear = spearman_np(soc_t.reshape(-1), soc_p.reshape(-1))
    p_pear   = pearson_np(pcell_t.reshape(-1), pcell_p.reshape(-1));   p_spear   = spearman_np(pcell_t.reshape(-1), pcell_p.reshape(-1))
    th_pear  = pearson_np(th_t.reshape(-1), th_p.reshape(-1));         th_spear  = spearman_np(th_t.reshape(-1), th_p.reshape(-1))

    # KS-test
    ks = {}
    if HAVE_SCIPY:
        ks["soc"]   = float(ks_2samp(soc_t.reshape(-1),   soc_p.reshape(-1)).statistic)
        ks["pcell"] = float(ks_2samp(pcell_t.reshape(-1), pcell_p.reshape(-1)).statistic)
        ks["th"]    = float(ks_2samp(th_t.reshape(-1),    th_p.reshape(-1)).statistic)
    else:
        ks["note"]  = "scipy not installed; KS-test skipped."

    # 마지막 시점 포인트 비교(옵션 리포팅)
    soc_last_mae, _   = mae_rmse(soc_t[:, -1],   soc_p[:, -1])
    pcell_last_mae, _ = mae_rmse(pcell_t[:, -1], pcell_p[:, -1])
    th_last_mae, _    = mae_rmse(th_t[:, -1],    th_p[:, -1])

    # -------- 임계 온도 도달시간 --------
    def time_metrics(P, T, thr):
        n = P.shape[0]
        ok_pred, ok_true, tpe, tte = [], [], [], []
        for i in range(n):
            p_ok, p_t = crossing_time(P[i], thr, dt)
            t_ok, t_t = crossing_time(T[i], thr, dt)
            ok_pred.append(p_ok); ok_true.append(t_ok)
            tpe.append(p_t); tte.append(t_t)
        ok_pred = np.array(ok_pred); ok_true = np.array(ok_true)
        tp = int(np.sum(ok_pred & ok_true))
        tn = int(np.sum(~ok_pred & ~ok_true))
        fp = int(np.sum(ok_pred & ~ok_true))
        fn = int(np.sum(~ok_pred & ok_true))
        acc = float((tp+tn)/max(n,1))
        prec = float(tp/max(tp+fp,1)) if (tp+fp)>0 else 0.0
        rec = float(tp/max(tp+fn,1)) if (tp+fn)>0 else 0.0
        f1 = float(2*prec*rec/max(prec+rec,1e-9)) if (prec+rec)>0 else 0.0
        both = ok_pred & ok_true
        t_mae = float(np.mean(np.abs((np.array(tpe)[both]-np.array(tte)[both])))) if np.any(both) else float("nan")
        return dict(acc=acc, precision=prec, recall=rec, f1=f1, time_mae_sec=t_mae)

    th50 = time_metrics(th_p, th_t, 50.0)
    th58 = time_metrics(th_p, th_t, 58.0)

    # -------- 세션/조건 분해 --------
    sess_df = []
    for s in np.unique(sid_used):
        m = (sid_used == s)
        sess_df.append(dict(
            session=s,
            soc_seq_MAE=float(np.mean(np.abs(soc_p[m]-soc_t[m]))),
            pcell_seq_MAE=float(np.mean(np.abs(pcell_p[m]-pcell_t[m]))),
            th_seq_MAE=float(np.mean(np.abs(th_p[m]-th_t[m]))),
        ))
    pd.DataFrame(sess_df).sort_values("session").to_csv(
        os.path.join(out_dir, "session_breakdown.csv"),
        index=False, encoding="utf-8", float_format="%.6f"
    )

    # 초기 상태에 따른 bin (예: t=0 기준)
    edges_soc0 = np.array([0,20,40,60,80,100], dtype=float)
    edges_th0  = np.array([0,20,30,40,50,60,100], dtype=float)
    soc0_bin = bin_idx(soc_t[:,0], edges_soc0)
    th0_bin  = bin_idx(th_t[:,0],  edges_th0)

    bins = []
    def add_bin(group_name, m):
        if not np.any(m): return
        bins.append(dict(
            group=group_name,
            n=int(np.sum(m)),
            soc_seq_MAE=float(np.mean(np.abs(soc_p[m]-soc_t[m]))),
            pcell_seq_MAE=float(np.mean(np.abs(pcell_p[m]-pcell_t[m]))),
            th_seq_MAE=float(np.mean(np.abs(th_p[m]-th_t[m]))),
        ))

    for i in range(len(edges_soc0)-1):
        add_bin(f"soc0_{edges_soc0[i]}_{edges_soc0[i+1]}", soc0_bin == i)
    for i in range(len(edges_th0)-1):
        add_bin(f"th0_{edges_th0[i]}_{edges_th0[i+1]}", th0_bin == i)

    if bins:
        pd.DataFrame(bins).to_csv(
            os.path.join(out_dir, "condition_breakdown.csv"),
            index=False, encoding="utf-8", float_format="%.6f"
        )

    # -------- per-horizon CSV --------
    xs = np.arange(horizon) * dt
    per_h = []
    for i, (k_mae, k_rmse, label) in enumerate([
        (soc_mae_h,   soc_rmse_h,   "soc"),
        (pcell_mae_h, pcell_rmse_h, "pcell"),
        (th_mae_h,    th_rmse_h,    "th"),
        (ta_mae_h,    ta_rmse_h,    "ta"),
        (tl_mae_h,    tl_rmse_h,    "tl"),
    ]):
        for h in range(horizon):
            per_h.append(dict(target=label, h=h, sec=float(xs[h]),
                              MAE=float(k_mae[h]), RMSE=float(k_rmse[h])))
    pd.DataFrame(per_h).to_csv(
        os.path.join(out_dir, "per_horizon_metrics.csv"),
        index=False, encoding="utf-8", float_format="%.6f"
    )

    # -------- 플롯 --------
    # 1) per-horizon MAE
    try:
        plt.figure(figsize=(8,5))
        plt.plot(xs, soc_mae_h,   label="SOC")
        plt.plot(xs, pcell_mae_h, label="P_cell")
        plt.plot(xs, th_mae_h,    label="Temp_high")
        plt.plot(xs, ta_mae_h,    label="Temp_avg")
        plt.plot(xs, tl_mae_h,    label="Temp_low")
        plt.xlabel("seconds ahead"); plt.ylabel("MAE"); plt.grid(True, alpha=0.3); plt.legend()
        _save_png_svg(os.path.join(out_dir, "per_horizon_mae.png"))
        plt.close()
    except Exception:
        pass

    # 2) scatter (last step)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True)
        for ax, T, P, name in [
            (axes[0], soc_t[:,-1],   soc_p[:,-1],   "SOC (last)"),
            (axes[1], pcell_t[:,-1], pcell_p[:,-1], "P_cell (last)"),
            (axes[2], th_t[:,-1],    th_p[:,-1],    "Temp_high (last)")
        ]:
            ax.scatter(T, P, s=5, alpha=0.35)
            lo = np.nanpercentile(np.concatenate([T,P]), 1)
            hi = np.nanpercentile(np.concatenate([T,P]), 99)
            ax.plot([lo,hi],[lo,hi], lw=1.2)
            ax.set_title(name); ax.set_xlabel("True"); ax.set_ylabel("Pred"); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "scatter_last.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # 3) error hist (last step)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True)
        for ax, T, P, name in [
            (axes[0], soc_t[:,-1],   soc_p[:,-1],   "SOC err (last)"),
            (axes[1], pcell_t[:,-1], pcell_p[:,-1], "P_cell err (last)"),
            (axes[2], th_t[:,-1],    th_p[:,-1],    "Temp_high err (last)")
        ]:
            err = (P - T).reshape(-1)
            ax.hist(err, bins=60, alpha=0.8)
            ax.set_title(name); ax.set_xlabel("Pred - True"); ax.set_ylabel("Count"); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "error_hist_last.png"), dpi=600, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    # 4) sample timeseries
    try:
        n_show = max(1, int(args.sample_plots))
        idx = np.linspace(0, len(soc_t)-1, num=n_show, dtype=int)
        for i in idx:
            tt = np.arange(horizon) * dt
            fig, axes = plt.subplots(1, 3, figsize=(13,4), constrained_layout=True)
            axes[0].plot(tt, soc_t[i], label="True");  axes[0].plot(tt, soc_p[i], label="Pred")
            axes[0].set_title(f"SOC seq (i={i})"); axes[0].set_xlabel("sec"); axes[0].set_ylabel("SOC")
            axes[1].plot(tt, pcell_t[i], label="True"); axes[1].plot(tt, pcell_p[i], label="Pred")
            axes[1].set_title("P_cell seq (kW)"); axes[1].set_xlabel("sec"); axes[1].set_ylabel("kW")
            axes[2].plot(tt, th_t[i], label="True"); axes[2].plot(tt, th_p[i], label="Pred")
            axes[2].set_title("Temp_high seq (°C)"); axes[2].set_xlabel("sec"); axes[2].set_ylabel("°C")
            for ax in axes: ax.grid(True, alpha=0.3); ax.legend()
            fig.savefig(os.path.join(out_dir, f"sample_seq_{i}.png"), dpi=600, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass

    # -------- 메트릭/요약 저장 --------
    metrics = dict(
        # overall seq metrics
        soc_seq_MAE=soc_mae,   soc_seq_RMSE=soc_rmse,   soc_R2=soc_r2,
        pcell_seq_MAE=pcell_mae, pcell_seq_RMSE=pcell_rmse, pcell_R2=pcell_r2,
        th_seq_MAE=th_mae,     th_seq_RMSE=th_rmse,     th_R2=th_r2,
        ta_seq_MAE=ta_mae,     ta_seq_RMSE=ta_rmse,     ta_R2=ta_r2,
        tl_seq_MAE=tl_mae,     tl_seq_RMSE=tl_rmse,     tl_R2=tl_r2,

        # mape (with valid ratio)
        soc_MAPE=soc_mape, soc_MAPE_valid_ratio=soc_mvalid,
        pcell_MAPE=pcell_mape, pcell_MAPE_valid_ratio=pcell_mvalid,
        th_MAPE=th_mape, th_MAPE_valid_ratio=th_mvalid,
        ta_MAPE=ta_mape, ta_MAPE_valid_ratio=ta_mvalid,
        tl_MAPE=tl_mape, tl_MAPE_valid_ratio=tl_mvalid,

        # correlations
        soc_Pearson=soc_pear, soc_Spearman=soc_spear,
        pcell_Pearson=p_pear, pcell_Spearman=p_spear,
        th_Pearson=th_pear, th_Spearman=th_spear,

        # KS
        KS=ks,

        # last-step only
        soc_last_MAE=soc_last_mae,
        pcell_last_MAE=pcell_last_mae,
        th_last_MAE=th_last_mae,

        # crossing-time
        th50=th50,
        th58=th58,

        # runtime
        avg_latency_ms=avg_latency_ms,
        throughput_sps=throughput_sps,

        # meta
        horizon=int(horizon),
        dt=float(dt),
        samples=int(soc_t.shape[0]),
    )
    with open(os.path.join(out_dir, "metrics_ext.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # -------- QCD Report --------
    # 임시 기준(예시): 프로젝트 상황에 맞게 조정하세요.
    qcd_thresholds = dict(
        soc_seq_MAE=2.0,        # SOC 단위 기준
        pcell_seq_MAE=5.0,      # kW 기준
        th_seq_MAE=2.0,         # °C 기준
        latency_ms=5.0          # 평균 샘플당 지연(ms) 기준
    )
    fails = []
    if not np.isfinite(soc_mae) or soc_mae > qcd_thresholds["soc_seq_MAE"]:
        fails.append(f"soc_seq_MAE>{qcd_thresholds['soc_seq_MAE']}")
    if not np.isfinite(pcell_mae) or pcell_mae > qcd_thresholds["pcell_seq_MAE"]:
        fails.append(f"pcell_seq_MAE>{qcd_thresholds['pcell_seq_MAE']}")
    if not np.isfinite(th_mae) or th_mae > qcd_thresholds["th_seq_MAE"]:
        fails.append(f"th_seq_MAE>{qcd_thresholds['th_seq_MAE']}")
    if not np.isfinite(avg_latency_ms) or avg_latency_ms > qcd_thresholds["latency_ms"]:
        fails.append(f"latency_ms>{qcd_thresholds['latency_ms']}")

    qcd = dict(
        QCD_pass=(len(fails) == 0),
        reasons=fails,
        thresholds=qcd_thresholds,
        cost=dict(  # 참고용(원하면 KPI 비용화 가능)
            soc_seq_MAE=float(soc_mae),
            pcell_seq_MAE=float(pcell_mae),
            th_seq_MAE=float(th_mae),
            latency_ms=float(avg_latency_ms)
        )
    )
    with open(os.path.join(out_dir, "qcd_report.json"), "w", encoding="utf-8") as f:
        json.dump(qcd, f, ensure_ascii=False, indent=2)

    print(f"[done] saved to: {out_dir}")
    print(f"  avg_latency_ms={avg_latency_ms:.3f} | throughput_sps={throughput_sps:.1f}")
    print(f"  soc_seq_MAE={soc_mae:.4f} | pcell_seq_MAE={pcell_mae:.4f} | th_seq_MAE={th_mae:.4f}")

if __name__ == "__main__":
    main()
