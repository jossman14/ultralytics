"""Extra experiments for the applied paper (user-approved):
  1. Inference latency / FPS on GPU and CPU for proposed vs baselines.
  2. Statistical tests on per-fold mAP50-95: Wilcoxon (proposed vs baseline) and
     Friedman across the compared models.
Writes experiments.json. Real measurements."""
import json, statistics as st, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
RUNS = REPO / "judol_sweep/runs"
OUT = Path(__file__).resolve().parent.parent
PROPOSED = "yolov8-spd-shuffle-simam"
LAT_MODELS = ["yolov8n-base", "yolo11n-base", "yolov8-se-baseline",
              "yolov8-res2net-spd-ssga", PROPOSED]


def latency(name, device, n=60, warmup=10):
    cfg = yaml.safe_load((RUNS / f"{name}_v2-1_f0" / "args.yaml").read_text())["model"]
    m = YOLO(cfg, task="detect").model.to(device).eval()
    x = torch.rand(1, 3, 640, 640, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
    return round(dt * 1000, 2), round(1.0 / dt, 1)  # ms, fps


def run_latency():
    out = {}
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for name in LAT_MODELS:
        out[name] = {}
        for dev in devs:
            ms, fps = latency(name, dev)
            out[name][dev] = {"ms": ms, "fps": fps}
            print(f"  {name:26s} {dev:4s} {ms:7.2f} ms  {fps:6.1f} FPS")
    return out


def run_stats():
    from scipy import stats
    led = json.loads((REPO / "judol_sweep/ledger.json").read_text())
    bym = defaultdict(lambda: defaultdict(list))
    for k, v in led.items():
        if v.get("status") == "ok":
            n, ds, f = k.split("|")
            bym[n][ds].append((int(f[1:]), v.get("mAP50-95", 0)))
    def folds(n, ds):
        return [x for _, x in sorted(bym[n][ds])]
    prop = folds(PROPOSED, "v2-1"); base = folds("yolov8n-base", "v2-1")
    w = stats.wilcoxon(prop, base)
    out = {"wilcoxon_proposed_vs_baseline_v21": {
        "proposed_mean": round(st.mean(prop), 4), "baseline_mean": round(st.mean(base), 4),
        "statistic": round(float(w.statistic), 4), "pvalue": round(float(w.pvalue), 4), "n_folds": len(prop)}}
    # Friedman across the compared models on v2-1 (need same 5 folds each)
    names = [n for n in ["yolov8n-base", "yolo11n-base", "yolo12n-base", "yolov8-se-baseline",
                         "yolov8-res2net-spd-ssga", PROPOSED] if len(folds(n, "v2-1")) == 5]
    mat = [folds(n, "v2-1") for n in names]
    fr = stats.friedmanchisquare(*mat)
    out["friedman_v21"] = {"models": names, "statistic": round(float(fr.statistic), 4),
                           "pvalue": round(float(fr.pvalue), 4), "k": len(names), "n": 5}
    print("  Wilcoxon proposed vs baseline p =", out["wilcoxon_proposed_vs_baseline_v21"]["pvalue"])
    print("  Friedman across", len(names), "models p =", out["friedman_v21"]["pvalue"])
    return out


if __name__ == "__main__":
    print("=== latency ===")
    lat = run_latency()
    print("=== statistics ===")
    stt = run_stats()
    (OUT / "experiments.json").write_text(json.dumps({"latency": lat, "stats": stt}, indent=1))
    print("wrote experiments.json")
