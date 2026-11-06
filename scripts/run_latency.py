"""
ONE latency / throughput / memory harness for every system (replaces the
latency parts of benchmark.py, promptguard_*.py and batch_throughput.py).

Protocol
  * the same N test texts (seeded sample of the seed-42 test split) for all systems
  * warm-up calls before timing; GPU synchronised inside every timed call
  * single-request latency: every text timed individually, R passes;
    mean / median / p95 / p99 reported
  * single-stream throughput: measured wall clock for one pass over the N texts
  * batch throughput: at every batch size the SAME set of texts is processed
    (random permutations, so mean text length is identical across batch sizes)
  * memory: serialized model size (CPU) or parameter bytes (GPU), plus peak
    inference memory (tracemalloc for CPU systems, torch.cuda.max_memory_allocated for GPU)
  * per-stage breakdown for the risk-scorer systems

Close other heavy programs and keep the laptop plugged in while this runs.

    py scripts/run_latency.py              # CPU systems
    py scripts/run_latency.py --with-b3    # + PromptGuard on the GPU
"""
import sys
import gc
import json
import time
import pickle
import platform
import argparse
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import load_pool, make_split
from src.trained import load_many

RESULTS = ROOT / "results"
BATCH_SIZES = [1, 8, 16, 32, 64, 128]


def hardware_info():
    info = {"platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_vram_gb"] = round(p.total_memory / 2**30, 1)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def is_gpu(det):
    return getattr(det, "device", "cpu") == "cuda"


def sync(det):
    if is_gpu(det):
        import torch
        torch.cuda.synchronize()


def model_size_mb(det):
    if hasattr(det, "model") and hasattr(det.model, "parameters") and callable(det.model.parameters):
        try:
            return sum(p.numel() * p.element_size() for p in det.model.parameters()) / 2**20
        except TypeError:
            pass
    return len(pickle.dumps(det)) / 2**20


def peak_inference_mem_mb(det, texts):
    if is_gpu(det):
        import torch
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        for t in texts:
            det.score_one(t)
        sync(det)
        return (torch.cuda.max_memory_allocated() - base) / 2**20
    gc.collect()
    tracemalloc.start()
    for t in texts:
        det.score_one(t)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["B1", "v1", "v2", "v2cf", "v3", "B2"])
    ap.add_argument("--with-b3", action="store_true")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    hw = hardware_info()
    print(json.dumps(hw, indent=2))
    (RESULTS / "hardware.json").write_text(json.dumps(hw, indent=2))

    pool = load_pool(42)
    _, _, test = make_split(pool, 42, "random")
    rng = np.random.default_rng(42)
    texts = test.text.values[rng.choice(len(test), min(args.n, len(test)), replace=False)].tolist()
    lens = np.array([len(t) for t in texts])
    print(f"\n{len(texts)} identical texts for every system; mean length {lens.mean():.0f} chars, "
          f"median {np.median(lens):.0f}")

    names = args.systems + (["B3"] if args.with_b3 else [])
    dets = load_many(names)
    single, batch, stages, bylen = [], [], [], []
    quart = np.quantile(lens, [0.25, 0.5, 0.75])

    for name, det in dets.items():
        for t in texts[:args.warmup]:
            det.score_one(t)
        sync(det)
        lat = []
        for _ in range(args.passes):
            for t in texts:
                t0 = time.perf_counter()
                det.score_one(t)          # B3: .item() inside synchronises the GPU
                lat.append((time.perf_counter() - t0) * 1000.0)
        lat = np.array(lat)
        t0 = time.perf_counter()
        for t in texts:
            det.score_one(t)
        sync(det)
        thr = len(texts) / (time.perf_counter() - t0)
        row = {"system": name, "device": "cuda" if is_gpu(det) else "cpu", "n_texts": len(texts),
               "n_timed_calls": len(lat), "mean_ms": lat.mean(), "median_ms": np.median(lat),
               "p95_ms": np.percentile(lat, 95), "p99_ms": np.percentile(lat, 99),
               "throughput_single_stream_req_s": thr, "model_size_mb": model_size_mb(det),
               "peak_inference_mem_mb": peak_inference_mem_mb(det, texts[:100])}
        single.append(row)
        print(f"\n{name:5s} median {row['median_ms']:.2f} ms  mean {row['mean_ms']:.2f}  p95 {row['p95_ms']:.2f}  "
              f"thr {thr:.1f} req/s  model {row['model_size_mb']:.2f} MB  peak {row['peak_inference_mem_mb']:.2f} MB",
              flush=True)

        L = lat.reshape(args.passes, len(texts)).mean(axis=0)
        bucket = np.searchsorted(quart, lens)
        for b in range(4):
            bylen.append({"system": name, "length_quartile": b + 1,
                          "max_chars": int(lens[bucket == b].max()), "median_ms": float(np.median(L[bucket == b]))})

        for bs in BATCH_SIZES:
            per_item = []
            for rep in range(args.passes):
                order = np.random.default_rng(rep).permutation(len(texts))
                chunks = [[texts[i] for i in order[j:j + bs]] for j in range(0, len(texts), bs)]
                sync(det)
                t0 = time.perf_counter()
                for c in chunks:
                    if is_gpu(det):
                        det.scores(c, batch_size=bs)
                    else:
                        det.scores(c)
                sync(det)
                per_item.append((time.perf_counter() - t0) * 1000.0 / len(texts))
            batch.append({"system": name, "batch_size": bs, "per_item_ms_median": float(np.median(per_item)),
                          "throughput_req_s": 1000.0 / float(np.median(per_item)),
                          "mean_chars_per_item": float(lens.mean())})
        print("   batch per-item ms: " + "  ".join(
            f"bs{r['batch_size']}={r['per_item_ms_median']:.2f}" for r in batch if r["system"] == name), flush=True)

        if hasattr(det, "score_one_staged"):
            acc = []
            for t in texts:
                acc.append(det.score_one_staged(t)[1])
            s = pd.DataFrame(acc).mean().to_dict()
            s["system"] = name
            stages.append(s)
            print("   stages (mean ms): " + "  ".join(f"{k}={v:.2f}" for k, v in s.items() if k != "system"))

    pd.DataFrame(single).to_csv(RESULTS / "latency_single.csv", index=False)
    pd.DataFrame(batch).to_csv(RESULTS / "latency_batch.csv", index=False)
    pd.DataFrame(stages).to_csv(RESULTS / "latency_stages.csv", index=False)
    pd.DataFrame(bylen).to_csv(RESULTS / "latency_by_length.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n##### single request #####")
    print(pd.DataFrame(single).round(3).to_string(index=False))
    print("\n##### batch: per-item ms (same texts at every batch size) #####")
    print(pd.DataFrame(batch).pivot(index="system", columns="batch_size", values="per_item_ms_median").round(3).to_string())
    if stages:
        print("\n##### per-stage mean ms #####")
        print(pd.DataFrame(stages).set_index("system").round(3).to_string())


if __name__ == "__main__":
    main()
