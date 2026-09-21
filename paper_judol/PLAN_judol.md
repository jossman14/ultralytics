# PLAN — Q1 Manuscript: Lightweight YOLO for Online-Gambling (Judol) Logo Detection

## Target
Q1 manuscript (Expert Systems w/ Applications / Image and Vision Computing / Eng. Applications of AI).
Proposed method must be **smaller (params/GFLOPs) yet competitive/better mAP** vs YOLO family.

## Dataset (REAL, in repo)
- Primary: `judol/Judol-Detection-v2-9` — 5 logo classes [BK8, Gate-of-olympus, Princess, Starlight-Princess, Zeus], 936 train / 131 valid, 640x640. Balanced.
- Held-out test: `judol/Judol-Detection-v2-1` — same 5 classes, has test split (53 imgs).
- 5-fold CV: `judol/Judol-Detection-v2-1_5fold`.
- (Not used for detection paper: `dataset_judi_online_yolo` 2-class saturated set.)

## Environment
- RTX A4000 16GB, torch 2.12, ultralytics 8.3.235, timm/ptflops/fvcore present.

## Proposed method (novel)
Working name: **SGSA-YOLO** (Star-Ghost + Scale-Sensitive Attention).
Base YOLOv8n +
1. StarNet-Ghost hybrid backbone (C2f-Star + GhostConv) — lightweight [#31,#32,#25,#18,#21]
2. **NOVEL module: Scale-Sensitive Gated Attention (SSGA)** — freq-decoupled + context gate, answers survey #33 "attention is scale-agnostic"
3. Shared-conv lightweight detection head [#31,#32]
4. WIoUv3 loss [#21]
Honest reporting: report real numbers even if a module does not win.

## Baselines / SOTA comparison
Native YOLO family (nano/tiny): yolov5n, yolov6n, yolov8n, yolov9t, yolov10n, yolo11n, yolo12n.
YOLOv1/v2 = Darknet-era, not in ultralytics, not meaningful modern baselines -> note in paper, not trained.

## Ablation study design (user-requested, cumulative)
| Variant | Config | Params | GFLOPs | Isolates |
|---|---|---|---|---|
| Baseline | yolov8n | 3,157,200 | 8.858 | reference |
| +SSGA | yolov8-ssga | 3,590,528 | 9.615 | attention contribution |
| +StarGhost | yolov8-starghost | 2,210,936 | 6.797 | lightweight backbone/neck |
| **Proposed (full)** | yolov8-sgsa | **2,644,264** | **7.554** | both combined |
Proposed is -16.2% params and -14.7% GFLOPs vs YOLOv8n baseline (measured, real).

## Status
| Phase | State |
|---|---|
| Env + dataset inventory | DONE |
| Smoke test (3ep) | DONE |
| SSGA novel module: impl + register + forward-test | DONE |
| Ablation configs built + complexity measured | DONE (model_complexity.csv) |
| Baseline sweep (7 models) | RUNNING (train_sweep.py -> sweep.log) |
| Proposed + ablation training | QUEUED (train_proposed.py, waits for sweep) |
| 5-fold CV on best model | pending |
| Held-out test eval (v2-1 test split) | pending |
| Figures (Grad-CAM, t-SNE, confusion, ROC/PR, curves, per-fold) | pending |
| LaTeX manuscript (Materials/Methods can draft now) | pending |
| Gap table + SOTA table | pending |
| Compile + AUDIT | pending |

## ⚠️ HARD BLOCKER (needs user action)
GPU driver wedged. Two CUDA processes initialized simultaneously (baseline sweep +
proposed driver) and one early inventory python hung during CUDA init. Result: stuck
D-state processes (PID 698098, 751735) on `uvm_gpu_retain_by_uuid`. SIGKILL cannot
remove D-state kernel calls; `nvidia-smi` now hangs; new CUDA contexts hang.
**Fix requires user privileges:**
  1. Try: `sudo nvidia-smi --gpu-reset -i 0`  (may fail if procs attached)
  2. Reliable: `sudo reboot`  (clears all stuck UVM processes)
**Resume after GPU freed (single process, no contention):**
  cd /home/ftib/ultralytics
  PYTHONPATH=/home/ftib/ultralytics nohup python3 paper_judol/run_all.py > paper_judol/run_all.log 2>&1 &
run_all.py trains baselines + ablation + proposed sequentially, resumable via results.csv.

## Real numbers recorded so far
- Complexity: model_complexity.csv (all 4 variants, measured via thop).
- mAP: results.csv (filled by drivers as training completes) — NO placeholders, wait for real.

## Next steps (after training completes)
1. Read results.csv; pick best proposed vs baselines.
2. Run 5-fold CV (judol/Judol-Detection-v2-1_5fold) on baseline + proposed.
3. Eval on held-out test split (judol_v21.yaml test).
4. Regenerate figures from best checkpoint (Grad-CAM via gradcam_judol.py, confusion, PR, curves, t-SNE of features).
5. Draft manuscript from results.csv + model_complexity.csv (zero placeholders).
6. Build gap table (>=7 refs) + SOTA table (>=10 refs, overlap <=2) from the 31-paper review.
7. latexmk compile clean; AUDIT_judol.md pass.
