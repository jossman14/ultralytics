"""Generate real result figures from the trained YOLOv8-SE checkpoint.

Produces (all in figures/):
  - confusion_matrix.png : normalized confusion matrix on v2-1 fold-0 val set
  - detections.png       : model predictions on sample v2-1 images
  - samples_v21.png      : raw v2-1 dataset examples (gambling logos)
  - samples_judi3k.png   : raw judi3k dataset examples (full reel frames)

Uses the real fold-0 SE weights and the real val images. No synthetic data.
"""
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

REPO = Path("/home/ftib/ultralytics")
FIG = REPO / "conferencev1/paper/figures"
FIG.mkdir(exist_ok=True)

SE_V21 = REPO / "judol_sweep/runs/yolov8-se-baseline_v2-1_f0/weights/best.pt"
V21 = REPO / "judol/Judol-Detection-v2-1_strat5fold/fold_0"
JUDI = REPO / "judol/dataset_judi_online_yolo_strat5fold/fold_0"


def montage(img_paths, out, cols=3, cell=320, pad=6, bg=(255, 255, 255)):
    """Tile images into a simple grid and save."""
    imgs = [Image.open(p).convert("RGB") for p in img_paths]
    rows = (len(imgs) + cols - 1) // cols
    W = cols * cell + (cols + 1) * pad
    H = rows * cell + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), bg)
    for i, im in enumerate(imgs):
        im.thumbnail((cell, cell))
        r, c = divmod(i, cols)
        x = pad + c * cell + (cell - im.width) // 2
        y = pad + r * cell + (cell - im.height) // 2
        canvas.paste(im, (x, y))
    canvas.save(out)
    print("wrote", out)


def raw_samples():
    v21 = sorted((V21 / "valid/images").glob("*.jpg"))[:6]
    montage(v21, FIG / "samples_v21.png", cols=3)
    # judi3k: real jpg targets behind the symlinks; pick gambling + nongambling by label
    lbl = JUDI / "valid/labels"
    imgs = sorted((JUDI / "valid/images").glob("*.jpg"))
    gambling, nongambling = [], []
    for p in imgs:
        lf = lbl / (p.stem + ".txt")
        if not lf.exists():
            continue
        cls0 = lf.read_text().split()[0] if lf.read_text().strip() else "1"
        (gambling if cls0 == "0" else nongambling).append(p)
        if len(gambling) >= 3 and len(nongambling) >= 3:
            break
    montage(gambling[:3] + nongambling[:3], FIG / "samples_judi3k.png", cols=3)


def val_and_predict():
    model = YOLO(str(SE_V21))
    # confusion matrix via validation
    res = model.val(data=str(V21 / "data.yaml"), split="val", plots=True,
                    project=str(FIG / "_val"), name="se_v21", exist_ok=True)
    src = Path(res.save_dir) / "confusion_matrix_normalized.png"
    if src.exists():
        shutil.copy(src, FIG / "confusion_matrix.png")
        print("wrote", FIG / "confusion_matrix.png")
    # detection samples
    imgs = sorted((V21 / "valid/images").glob("*.jpg"))[:6]
    pr = model.predict(source=[str(p) for p in imgs], conf=0.25, save=True,
                       project=str(FIG / "_pred"), name="se_v21", exist_ok=True)
    saved = sorted(Path(pr[0].save_dir).glob("*.jpg"))[:6]
    montage(saved, FIG / "detections.png", cols=3)


if __name__ == "__main__":
    raw_samples()
    val_and_predict()
    print("done")
