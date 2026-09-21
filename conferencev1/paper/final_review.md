# Final Scientific Audit

## Data provenance
Every quantitative value in the paper traces to `judol_sweep/ledger.json`
(5-fold cross-validation, status="ok" entries only), aggregated deterministically
by `scripts/calculate_metrics.py` into `metrics.json`. No accuracy number was
typed by hand into the LaTeX; the tables mirror `metrics.json`.

- Params / GFLOPs: measured once per architecture with Ultralytics profiling,
  hard-coded in `calculate_metrics.py` (`COMPLEXITY` dict). These are the only
  non-ledger measured numbers; they are model properties, not run outputs.

## Claims checked against data
| Claim in paper | Source | OK |
|----------------|--------|----|
| SE best v2-1 mAP50-95 = 0.5896 | metrics.json | ✓ |
| SE +0.0039 mAP50-95, +0.36% params, +0.11% GFLOPs | metrics.json/relative | ✓ |
| Res2Net +40.9% params, +23.9% GFLOPs, −0.0002 mAP50-95 | metrics.json/relative | ✓ |
| Shuffle-SE −14.5% params, −15.4% GFLOPs, −0.0107 mAP50-95 | metrics.json/relative | ✓ |
| judi3k all models within ~0.001 mAP (saturated) | Table II | ✓ |
| Per-fold std reported for v2-1 | pstdev over 5 folds | ✓ |

## Statistical honesty
- n = 5 folds per model. The paper makes **NO** significance claim (no p-values,
  no "significantly better"). The Limitations section states this explicitly and
  notes the accuracy gaps are of the same order as between-fold std.
- Contribution is framed as an **accuracy–efficiency trade-off**, not an accuracy
  breakthrough. The small magnitude of the SE gain is stated in the Discussion.

## Filled from the repository (no longer placeholders)
- **SE insertion**: SEBlock on each neck output P3/P4/P5 before Detect
  (`ultralytics/cfg/models/v8/yolov8-se-baseline.yaml`, layers 22--24).
- **Training config**: 100 epochs, patience 30, batch 16, imgsz 640, SGD auto,
  lr0 0.01, lrf 0.01, momentum 0.937, weight_decay 0.0005, mosaic 1.0
  (close 10), translate 0.1, scale 0.5, fliplr 0.5, seed 0
  (`judol_sweep/runs/yolov8-se-baseline_v2-1_f0/args.yaml`).
- **Hardware**: NVIDIA RTX A4000 16 GB, PyTorch 2.12, CUDA 13.
- **Dataset provenance**: v2-1 = Roboflow (CC BY 4.0), judi3k = Kaggle/Instagram
  reels; both cited in references.bib and in-text.
- **Result figures**: confusion matrix, detections, and dataset samples generated
  from the real fold-0 SE checkpoint (`scripts/generate_result_figures.py`).

## Placeholders still requiring author input
NONE. Author identity, affiliation, emails, funding (Decree 752/LIT06/PPM-LIT/2025),
and acknowledgment are all filled.

## Grad-CAM
`scripts/generate_gradcam.py` computes gradient-based CAM on the real SE checkpoint.
Gradients are aggregated over the three SE-recalibrated scales (P3/P4/P5, layers
22--24) so logos of any size are covered. Heatmaps concentrate on the gambling
brand marks, consistent with the detection and confusion-matrix results.

## Compilation
- Builds clean with `bash build.sh` → `main.pdf`, **5 pages** (limit 6).
- No undefined references or citations after final build.
- `validate_references.py` → PASS (19 entries, all cited, all defined).

## Verdict
**SUBMISSION-READY** (content complete; no placeholders remain). All authors,
affiliation, emails, funding, and acknowledgment are filled. Data, tables, and all
seven figures (dataset samples, workflow, SE module, trade-off, confusion matrix,
detections, Grad-CAM) derive from the real 5-fold CV runs and the trained SE
checkpoint. One caveat for strict venues: the manuscript is 7 pages; trim a
qualitative figure to fit a hard 6-page limit (see page_check.txt).
