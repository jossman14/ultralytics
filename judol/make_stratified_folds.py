"""Stratified 5-fold generator for the 3 Judol datasets.

Stratifies by each image's dominant (most frequent) class so every fold keeps the
class distribution. Uses symlinks (not copies) to save disk. Produces:
  judol/<ds>_strat5fold/fold_k/{train,valid}/{images,labels} + data.yaml (absolute path)
Idempotent: re-running rebuilds folds deterministically (seed 0).
"""
import shutil
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).parent
DATASETS = ["dataset_judi_online_yolo", "Judol-Detection-v2-1", "Judol-Detection-v2-9"]
K = 5
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dominant_class(label_path: Path) -> int:
    """Most frequent class id in a YOLO label file; -1 if empty/missing (background)."""
    if not label_path.exists():
        return -1
    ids = [int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip()]
    return Counter(ids).most_common(1)[0][0] if ids else -1


def collect(src: Path):
    items = []
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lbl = src / split / "labels" / (img.stem + ".txt")
            items.append((img, lbl, dominant_class(lbl)))
    return items


def main():
    for ds in DATASETS:
        src = BASE / ds
        out = BASE / f"{ds}_strat5fold"
        if out.exists():
            shutil.rmtree(out)
        items = collect(src)
        y = [c for _, _, c in items]
        meta = yaml.safe_load((src / "data.yaml").read_text())
        skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
        print(f"{ds}: {len(items)} images, class dist {Counter(y)}")
        for k, (tr_idx, va_idx) in enumerate(skf.split(items, y)):
            fold = out / f"fold_{k}"
            for split, idxs in (("train", tr_idx), ("valid", va_idx)):
                (fold / split / "images").mkdir(parents=True, exist_ok=True)
                (fold / split / "labels").mkdir(parents=True, exist_ok=True)
                for i in idxs:
                    img, lbl, _ = items[i]
                    (fold / split / "images" / img.name).symlink_to(img.resolve())
                    if lbl.exists():
                        (fold / split / "labels" / lbl.name).symlink_to(lbl.resolve())
            data = {
                "path": str(fold.resolve()),
                "train": "train/images",
                "val": "valid/images",
                "nc": meta["nc"],
                "names": meta["names"],
            }
            (fold / "data.yaml").write_text(yaml.dump(data, sort_keys=False))
            print(f"  fold_{k}: train {len(tr_idx)}, valid {len(va_idx)}")


if __name__ == "__main__":
    main()
