"""Aggregate per-fold metrics.json into mean+/-std tables and a token map for the manuscript.

Reads revisi/runs/<dataset>/<model>/fold_*/metrics.json. Only folds that exist are
aggregated; the fold count is printed with every row so partial runs are never
mistaken for complete ones.
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/ftib/ultralytics")
RUNS = ROOT / "revisi/runs"
OUT = ROOT / "revisi/out"
KEYS = ["precision", "recall", "f1", "map50", "map"]


def load():
    d = defaultdict(list)
    for f in glob.glob(str(RUNS / "*/*/fold_*/metrics.json")):
        m = json.loads(Path(f).read_text())
        d[(m["dataset"], m["model"])].append(m)
    return d


def ms(vals):
    if not vals:
        return None
    return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(pair, pct=True, dec=2):
    if pair is None:
        return "n/a"
    m, s = pair
    if pct:
        return f"{m * 100:.{dec}f} +/- {s * 100:.{dec}f}"
    return f"{m:.{dec}f} +/- {s:.{dec}f}"


def main():
    d = load()
    lines = []
    tok = {}
    for (ds, model) in sorted(d):
        runs = d[(ds, model)]
        agg = {k: ms([r[k] for r in runs]) for k in KEYS}
        inf = ms([r["speed_ms"]["inference"] for r in runs])
        mins = ms([r["train_minutes"] for r in runs])
        lines.append(
            f"| {ds} | {model} | {len(runs)}/5 | {runs[0]['params']:,} | "
            + " | ".join(fmt(agg[k]) for k in KEYS)
            + f" | {fmt(inf, pct=False)} | {fmt(mins, pct=False, dec=1)} |"
        )
        tok[f"{ds}.{model}.n_folds"] = (float(len(runs)), 0.0)
        tok[f"{ds}.{model}.params"] = (float(runs[0]["params"]), 0.0)
        for k in KEYS:
            if agg[k]:
                tok[f"{ds}.{model}.{k}"] = agg[k]
        if inf:
            tok[f"{ds}.{model}.inference_ms"] = inf

        cls = defaultdict(lambda: defaultdict(list))
        for r in runs:
            for name, v in r.get("per_class", {}).items():
                for k in KEYS:
                    cls[name][k].append(v[k])
        for name in cls:
            for k in KEYS:
                tok[f"{ds}.{model}.class.{name}.{k}"] = ms(cls[name][k])

    hdr = ("| dataset | model | folds | params | precision | recall | F1 | mAP@50 | "
           "mAP@50-95 | inference (ms) | train (min) |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|")
    table = hdr + "\n" + "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results_tables.md").write_text(table + "\n")
    (OUT / "tokens.json").write_text(json.dumps({k: list(v) for k, v in tok.items()}, indent=2))
    print(table)
    print(f"\n{len(tok)} tokens -> {OUT / 'tokens.json'}")


if __name__ == "__main__":
    main()
