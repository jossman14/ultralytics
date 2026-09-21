"""Rebuild the binary dataset into video-disjoint (GroupKFold) folds.

The published random 5-fold split puts frames of the same Instagram Reel in both
train and validation, so every validation frame has a sibling in training. Grouping
by source video shortcode removes that path and measures generalization to unseen posts.
"""
import re, shutil
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedGroupKFold

SRC = Path("/home/ftib/ultralytics/judol/dataset_judi_online_yolo_5fold/fold_0")
OUT = Path("/home/ftib/ultralytics/judol/dataset_judi_online_yolo_videofold")
VIDEO_RE = re.compile(r"_\d+$")


def collect():
    items = []
    for split in ("train", "valid"):
        for img in sorted((SRC / split / "images").glob("*.jpg")):
            lbl = SRC / split / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            cls = int(lbl.read_text().split()[0]) if lbl.read_text().strip() else -1
            items.append((img, lbl, VIDEO_RE.sub("", img.stem), cls))
    return items


def main():
    items = collect()
    groups = [i[2] for i in items]
    labels = [i[3] for i in items]
    print(f"{len(items)} frames, {len(set(groups))} source videos")

    if OUT.exists():
        shutil.rmtree(OUT)
    meta = yaml.safe_load((SRC / "data.yaml").read_text())

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(sgkf.split(items, labels, groups)):
        fdir = OUT / f"fold_{fold}"
        for split, idx in (("train", tr), ("valid", va)):
            (fdir / split / "images").mkdir(parents=True, exist_ok=True)
            (fdir / split / "labels").mkdir(parents=True, exist_ok=True)
            for i in idx:
                img, lbl, _, _ = items[i]
                shutil.copy2(img, fdir / split / "images" / img.name)
                shutil.copy2(lbl, fdir / split / "labels" / lbl.name)
        (fdir / "data.yaml").write_text(yaml.dump(
            {**meta, "train": "train/images", "val": "valid/images", "test": ""}, sort_keys=False))
        gtr = {groups[i] for i in tr}
        gva = {groups[i] for i in va}
        assert not (gtr & gva), f"fold {fold} leaks {len(gtr & gva)} videos"
        print(f"fold {fold}: {len(tr):>4} train / {len(va):>3} val frames | "
              f"{len(gtr):>3} / {len(gva):>3} videos | overlap {len(gtr & gva)}")


if __name__ == "__main__":
    main()
