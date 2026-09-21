"""Map aggregated metrics onto the {{token}} names used in the manuscript markdown.

Reads revisi/out/tokens.json (from aggregate.py), xai_eval.json, and
leakage_random.json, and writes revisi/out/tokens_filled.json. A token is emitted
only when its source data covers all five folds, so build_docx.py keeps flagging
anything the runs have not yet produced.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path("/home/ftib/ultralytics")
OUT = ROOT / "revisi/out"
PRIMARY = "yolo12s"


def pct(pair, dec=2):
    m, s = pair
    return f"{m * 100:.{dec}f}% +/- {s * 100:.{dec}f}"


def num(pair, dec=2, unit=""):
    m, s = pair
    return f"{m:.{dec}f} +/- {s:.{dec}f}{unit}"



MODELS = ["yolov8s", "yolov10s", "yolo11s", "yolo12s"]
NICE = {"yolov8s": "YOLOv8s", "yolov10s": "YOLOv10s", "yolo11s": "YOLO11s", "yolo12s": "YOLOv12s"}
DSNICE = {"binary": "Binary, random folds", "binary_videofold": "Binary, video-disjoint folds",
          "multiclass": "Multi-class"}


def write_tables(src, partial):
    def done(ds, m):
        n = src.get(f"{ds}.{m}.n_folds")
        return bool(n) and (partial or n[0] >= 5)

    rows = ["| Dataset | Model | Params | Precision | Recall | F1 | mAP@50 | mAP@50-95 | Inference (ms) |",
            "|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for ds in ("binary", "binary_videofold", "multiclass"):
        for m in MODELS:
            if not done(ds, m):
                continue
            g = lambda k: src.get(f"{ds}.{m}.{k}")
            cells = [pct(tuple(g(k))) if g(k) else "n/a"
                     for k in ("precision", "recall", "f1", "map50", "map")]
            inf = g("inference_ms")
            pm = g("params")
            rows.append(f"| {DSNICE[ds]} | {NICE[m]} | {int(pm[0]):,} | " + " | ".join(cells) +
                        f" | {num(tuple(inf)) if inf else 'n/a'} |")
            any_row = True
    if any_row:
        (OUT / "table_arch.md").write_text("\n".join(rows) + "\n")

    if done("multiclass", PRIMARY):
        names = sorted({k.split(".class.")[1].rsplit(".", 1)[0]
                        for k in src if f"multiclass.{PRIMARY}.class." in k})
        pr = ["| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|---|"]
        for c in names:
            g = lambda k: src.get(f"multiclass.{PRIMARY}.class.{c}.{k}")
            pr.append(f"| {c} | " + " | ".join(
                pct(tuple(g(k))) if g(k) else "n/a"
                for k in ("precision", "recall", "f1", "map50", "map")) + " |")
        (OUT / "table_perclass.md").write_text("\n".join(pr) + "\n")

    if done("binary_videofold", PRIMARY):
        names = sorted({k.split(".class.")[1].rsplit(".", 1)[0]
                        for k in src if f"binary_videofold.{PRIMARY}.class." in k})
        pb = ["| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|---|"]
        for c in names:
            g = lambda k: src.get(f"binary_videofold.{PRIMARY}.class.{c}.{k}")
            pb.append(f"| {c} | " + " | ".join(
                pct(tuple(g(k))) if g(k) else "n/a"
                for k in ("precision", "recall", "f1", "map50", "map")) + " |")
        (OUT / "table_perclass_binary.md").write_text("\n".join(pb) + "\n")

    xf = OUT / "xai_eval.json"
    if xf.exists():
        data = json.loads(xf.read_text())
        rows2 = ["| Variant | Pointing game | Energy in box | Interpretation |",
                 "|---|---|---|---|"]
        allr = [r for rs in data.values() for r in rs]
        if allr:
            def agg(k):
                v = [r[k] for r in allr if r.get(k) is not None]
                return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else None
            spec = [("box", "CAM, box-conditioned (as deployed)",
                     "Upper bound; partly circular, see text"),
                    ("glob", "CAM, prediction-independent", "Localization independent of the detection output"),
                    ("rand", "Random-box control", "Score obtainable from map shape alone"),
                    ("sanity", "Randomized model weights", "Sanity check; should collapse")]
            for key, lab, note in spec:
                pg, eb = agg(f"pg_{key}"), agg(f"eb_{key}")
                if pg is None and eb is None:
                    continue
                rows2.append(f"| {lab} | {pct(pg, 1) if pg else 'n/a'} | "
                             f"{pct(eb, 1) if eb else 'n/a'} | {note} |")
            a = agg("area")
            if a:
                rows2.append(f"| Ground-truth box area (reference) | | {pct(a, 1)} | "
                             f"Energy a uniform map would place in the box |")
            (OUT / "table_xai.md").write_text("\n".join(rows2) + "\n")



def fold_rows(ds, model=PRIMARY):
    """Per-fold metrics.json plus the final-epoch losses from the same run's results.csv."""
    import csv
    out = {}
    for f in sorted((ROOT / "revisi/runs" / ds / model).glob("fold_*/metrics.json")):
        m = json.loads(f.read_text())
        csvf = f.parent / "results.csv"
        if csvf.exists():
            with csvf.open() as fh:
                last = list(csv.DictReader(fh))[-1]
            m["loss"] = {k.strip(): float(v) for k, v in last.items()
                         if k.strip().startswith(("train/", "val/"))}
        out[m["fold"]] = m
    return out


def stat_rows(vals, dec=2, scale=100):
    """Mean and SD rows in the same shape as the per-fold rows above them."""
    import statistics as _st
    mean = [f"{_st.mean(c) * scale:.{dec}f}" for c in zip(*vals)]
    sd = [f"{(_st.stdev(c) if len(c) > 1 else 0.0) * scale:.{dec}f}" for c in zip(*vals)]
    return mean, sd


def write_fold_tables():
    """The per-fold, per-class binary, and efficiency tables the submitted version carried.

    The aggregate tables elsewhere report mean +/- SD; these keep the individual folds
    visible, which is what a reader needs to check the variance claim in Section 3.2.
    """
    rnd, vid = fold_rows("binary"), fold_rows("binary_videofold")
    if len(rnd) == 5 and len(vid) == 5:
        rows = ["| Fold | Random mAP@50 | Random mAP@50-95 | Video-disjoint mAP@50 | Video-disjoint mAP@50-95 |",
                "|---|---|---|---|---|"]
        vals = [[rnd[k]["map50"], rnd[k]["map"], vid[k]["map50"], vid[k]["map"]] for k in range(5)]
        for k, v in enumerate(vals):
            rows.append(f"| {k} | " + " | ".join(f"{x * 100:.2f}" for x in v) + " |")
        mean, sd = stat_rows(vals)
        rows.append("| **Mean** | " + " | ".join(f"**{x}**" for x in mean) + " |")
        rows.append("| **SD** | " + " | ".join(f"**{x}**" for x in sd) + " |")
        (OUT / "table_folds_binary.md").write_text("\n".join(rows) + "\n")

    mc = fold_rows("multiclass")
    if len(mc) == 5:
        keys = ("precision", "recall", "f1", "map50", "map")
        rows = ["| Fold | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---|---|---|---|---|"]
        vals = [[mc[k][x] for x in keys] for k in range(5)]
        for k, v in enumerate(vals):
            rows.append(f"| {k} | " + " | ".join(f"{x * 100:.2f}" for x in v) + " |")
        mean, sd = stat_rows(vals)
        rows.append("| **Mean** | " + " | ".join(f"**{x}**" for x in mean) + " |")
        rows.append("| **SD** | " + " | ".join(f"**{x}**" for x in sd) + " |")
        (OUT / "table_folds_multiclass.md").write_text("\n".join(rows) + "\n")

    order = ["train/box_loss", "train/cls_loss", "train/dfl_loss",
             "val/box_loss", "val/cls_loss", "val/dfl_loss"]
    for ds, name in (("binary_videofold", "table_eff_binary.md"), ("multiclass", "table_eff_multiclass.md")):
        d = fold_rows(ds)
        if len(d) != 5 or not all("loss" in r for r in d.values()):
            continue
        rows = ["| Fold | Inference (ms) | Train box | Train cls | Train DFL | Val box | Val cls | Val DFL |",
                "|---|---|---|---|---|---|---|---|"]
        vals = [[d[k]["speed_ms"]["inference"]] + [d[k]["loss"][x] for x in order] for k in range(5)]
        for k, v in enumerate(vals):
            rows.append(f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
        mean, sd = stat_rows(vals, dec=3, scale=1)
        rows.append("| **Mean** | " + " | ".join(f"**{x}**" for x in mean) + " |")
        rows.append("| **SD** | " + " | ".join(f"**{x}**" for x in sd) + " |")
        (OUT / name).write_text("\n".join(rows) + "\n")


def main():
    src = json.loads((OUT / "tokens.json").read_text()) if (OUT / "tokens.json").exists() else {}
    t = {}

    partial = "--partial" in sys.argv

    def complete(ds, model=PRIMARY):
        n = src.get(f"{ds}.{model}.n_folds")
        return bool(n) and (partial or n[0] >= 5)

    def g(ds, key):
        if not complete(ds):
            return None
        v = src.get(f"{ds}.{PRIMARY}.{key}")
        return tuple(v) if v else None

    for ds, pre in (("binary", "bin"), ("binary_videofold", "vid"), ("multiclass", "mc")):
        for key, short in (("precision", "p"), ("recall", "r"), ("f1", "f1"),
                           ("map50", "map50"), ("map", "map")):
            v = g(ds, key)
            if v:
                t[f"{pre}.{short}"] = pct(v)
                # Sentences comparing spread need the deviation on its own,
                # without the mean that the combined token carries.
                t[f"{pre}.{short}.sd"] = f"{v[1] * 100:.2f}"
        pm = g(ds, "params")
        if pm:
            t[f"{pre}.params"] = f"{int(pm[0]):,}"

    b, v = g("binary", "map50"), g("binary_videofold", "map50")
    if b and v:
        for key, short in (("precision", "p"), ("recall", "r"), ("f1", "f1"),
                           ("map50", "map50"), ("map", "map")):
            bb, vv = g("binary", key), g("binary_videofold", key)
            if bb and vv:
                t[f"d.{short}"] = f"{(bb[0] - vv[0]) * 100:+.2f} points"
        t["LEAK_GAP"] = f"{(b[0] - v[0]) * 100:.2f}"
        t["BIN_MAP50"] = t["bin.map50"]
        t["VID_MAP50"] = t["vid.map50"]
        t["VID_MAP"] = t.get("vid.map", "")

    if g("multiclass", "map50"):
        t["MC_MAP50"] = t["mc.map50"]
    if g("multiclass", "map"):
        t["MC_MAP"] = t["mc.map"]

    # per-class best and worst on the multi-class task
    cls = {k.split(".class.")[1].rsplit(".", 1)[0]: None
           for k in src if f"multiclass.{PRIMARY}.class." in k and k.endswith(".map50")
           } if complete("multiclass") else {}
    if cls:
        m50 = {c: src[f"multiclass.{PRIMARY}.class.{c}.map50"][0] for c in cls}
        best, worst = max(m50, key=m50.get), min(m50, key=m50.get)
        t["mc.best"] = f"{m50[best] * 100:.2f}%"
        t["mc.worst"] = f"{m50[worst] * 100:.2f}%"
        t["MC_BEST"], t["MC_WORST"] = t["mc.best"], t["mc.worst"]
        t["mc.best.name"], t["mc.worst.name"] = best, worst
        rk = f"multiclass.{PRIMARY}.class.BK8.recall"
        if rk in src:
            t["mc.bk8.r"] = pct(tuple(src[rk]))

    lk = OUT / "leakage_random.json"
    if lk.exists():
        rows = json.loads(lk.read_text())
        t["LEAK_PCT"] = f"{st.mean(r['leak_pct'] for r in rows):.1f}%"

    xf = OUT / "xai_eval.json"
    if xf.exists():
        data = json.loads(xf.read_text())
        rows = [r for rs in data.values() for r in rs]
        if rows:
            def agg(k):
                v = [r[k] for r in rows if r.get(k) is not None]
                return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0) if v else None
            for k, name in (("pg_glob", "xai.pg_glob"), ("pg_box", "xai.pg_box"),
                            ("pg_rand", "xai.pg_rand"), ("pg_sanity", "xai.sanity"),
                            ("eb_glob", "xai.eb_glob"), ("eb_rand", "xai.eb_rand")):
                a = agg(k)
                if a:
                    t[name] = pct(a, 1)
            t["PG_GLOB"] = t.get("xai.pg_glob", "")
            t["PG_RAND"] = t.get("xai.pg_rand", "")
            t["PG_SANITY"] = t.get("xai.sanity", "")
            for k, name in (("det_ms", "det_ms"), ("cam_ms", "cam_ms")):
                a = agg(k)
                if a:
                    t[name] = num(a, 2, " ms")
            d, c = agg("det_ms"), agg("cam_ms")
            if d and c:
                t["total_ms"] = f"{d[0] + c[0]:.2f} ms"
                t["DET_MS"], t["CAM_MS"] = t["det_ms"], t["cam_ms"]

    write_tables(src, partial)
    write_fold_tables()

    (OUT / "tokens_filled.json").write_text(json.dumps(t, indent=2, sort_keys=True))
    print(f"{len(t)} tokens -> {OUT / 'tokens_filled.json'}")
    for k in sorted(t):
        print(f"  {k} = {t[k]}")


if __name__ == "__main__":
    main()
