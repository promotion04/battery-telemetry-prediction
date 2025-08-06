# scripts/bench_test_six_segments.py
# -*- coding: utf-8 -*-
"""
테스트 데이터(test split)를 6개의 연속 구간으로 나눠
각 구간의 추론 시간/처리량을 측정하고 평균을 산출합니다.

출력: artifacts/<model_name>/test/six_segments_speed.json
사용 예:
  python scripts/bench_test_six_segments.py --config config.yaml --model_name tcn --batch_size 256
"""
import os, json, argparse, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from src.config import load_config
from src.models.registry import build_model


class MTSet(Dataset):
    def __init__(self, pack, sid):
        self.X = pack["X"].astype(np.float32)
        self.soc = pack["soc"].astype(np.float32)
        self.th  = pack["th"].astype(np.float32)
        self.ta  = pack["ta"].astype(np.float32)
        self.tl  = pack["tl"].astype(np.float32)
        self.p_cell = pack["p_cell"].astype(np.float32)
        self.sid = np.asarray(sid)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.tensor(self.soc[i]),
            torch.from_numpy(self.th[i]),
            torch.from_numpy(self.ta[i]),
            torch.from_numpy(self.tl[i]),
            torch.tensor(self.p_cell[i]),
            str(self.sid[i]),
        )


@torch.no_grad()
def run_one_loader(model, loader, device: str):
    """로더 전체를 한 번 통과시키는 데 걸린 시간과 샘플 수를 반환."""
    use_cuda = device.startswith("cuda")
    use_amp = torch.cuda.is_available() and use_cuda
    autocast_dtype = torch.float16 if use_amp else torch.bfloat16

    # GPU 타이밍 정확도 개선: sync
    if use_cuda:
        torch.cuda.synchronize()

    n_samples = 0
    t0 = time.perf_counter()
    for X, *_ in loader:
        # pinned memory가 있을 때만 non_blocking=True가 이점 있음
        X = X.to(device, non_blocking=use_cuda)
        with torch.autocast(device_type=("cuda" if use_cuda else "cpu"), dtype=autocast_dtype):
            _ = model(X)
        n_samples += X.size(0)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    dur = max(t1 - t0, 1e-9)
    return dict(
        total_seconds=float(dur),
        samples=int(n_samples),
        avg_latency_ms=float(1000.0 * dur / max(1, n_samples)),
        throughput_sps=float(n_samples / dur)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--model_name", type=str, default="tcn", choices=["tcn","lstm","transformer"])
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === 데이터 로드 (test split만) ===
    npz_path = os.path.join(cfg["data"]["processed_dir"], "dataset_windows_labels.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)

    sid_all = np.array([str(s).strip() for s in npz["sid"]], dtype=object)
    want = [str(s).strip() for s in cfg["data"]["sessions_test"]]
    mask = np.isin(sid_all, np.array(want, dtype=object))
    if not mask.any():
        avail = sorted(set(sid_all))
        raise RuntimeError(f"No samples matched sessions_test={want}. Available: {avail}")

    keys = ["X","soc","th","ta","tl","p_cell"]
    pack = {k: npz[k][mask] for k in keys}
    sid_sel = sid_all[mask]
    dataset = MTSet(pack, sid_sel)

    in_feat = npz["X"].shape[-1]
    horizon = npz["th"].shape[-1]

    # === 모델/체크포인트 ===
    model = build_model(in_feat, horizon, cfg, kind=args.model_name).to(device).eval()
    ckpt = os.path.join("artifacts", args.model_name, "models", f"multitask_{args.model_name}_cellkW.pt")
    if not os.path.exists(ckpt):
        legacy = "artifacts/models/multitask_tcn_cellkW.pt"
        if os.path.exists(legacy):
            ckpt = legacy
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # === segments 개수로 연속 인덱스 분할 ===
    N = len(dataset)
    splits = np.array_split(np.arange(N), max(1, int(args.segments)))

    # === 결과 폴더 ===
    out_dir = os.path.join("artifacts", args.model_name, "test")
    os.makedirs(out_dir, exist_ok=True)

    # === 웜업 (한 미니배치) ===
    if N > 0 and len(splits[0]) > 0:
        warm_idx = splits[0][:min(len(splits[0]), args.batch_size)]
        warm_loader = DataLoader(
            Subset(dataset, warm_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        )
        _ = run_one_loader(model, warm_loader, device)

    # === 각 구간 타이밍 ===
    seg_stats = []
    for i, idxs in enumerate(splits, 1):
        if len(idxs) == 0:
            seg_stats.append({"segment": i, "samples": 0, "avg_latency_ms": None, "throughput_sps": None, "total_seconds": 0.0})
            continue
        loader = DataLoader(
            Subset(dataset, idxs),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        )
        st = run_one_loader(model, loader, device)
        st["segment"] = i
        seg_stats.append(st)
        print(f"[seg {i}/{len(splits)}] samples={st['samples']}  avg_latency_ms={st['avg_latency_ms']:.3f}  thr={st['throughput_sps']:.1f}  sec={st['total_seconds']:.3f}")

    # === 평균(유효 seg만) ===
    valid = [s for s in seg_stats if s["samples"] and s["samples"] > 0]
    if valid:
        total_samp = sum(s["samples"] for s in valid)
        total_sec  = sum(s["total_seconds"] for s in valid)
        overall = dict(
            segments=len(valid),
            samples=total_samp,
            total_seconds=float(total_sec),
            avg_latency_ms=float(1000.0 * total_sec / max(1, total_samp)),
            throughput_sps=float(total_samp / max(total_sec, 1e-9)),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers)
        )
    else:
        overall = dict(
            segments=0, samples=0, total_seconds=0.0,
            avg_latency_ms=None, throughput_sps=None,
            batch_size=int(args.batch_size), num_workers=int(args.num_workers)
        )

    # === 저장 ===
    out_path = os.path.join(out_dir, "six_segments_speed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"segments": seg_stats, "overall": overall}, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    if overall["avg_latency_ms"] is not None and overall["throughput_sps"] is not None:
        print(f"[overall] segments={overall['segments']}  samples={overall['samples']}  "
              f"avg_latency_ms={overall['avg_latency_ms']:.3f}  throughput_sps={overall['throughput_sps']:.1f}")
    else:
        print(f"[overall] segments=0  samples=0  avg_latency_ms=n/a  throughput_sps=n/a")
    print(f"[saved] {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
