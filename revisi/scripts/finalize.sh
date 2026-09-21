#!/bin/bash
# Waits for the training sweep to drain, then runs the full analysis chain on idle
# GPUs: explanation evaluation (timing needs an idle device), aggregation, token
# filling, figures, and the .docx build.
cd /home/ftib/ultralytics || exit 1
export PYTHONPATH=/home/ftib/ultralytics

# Training is complete; the wait this script performed matched its own launcher
# shell (whose cmdline contains the script text) and never exited. Removed.
echo "[finalize] training drained at $(date)"

python3 revisi/scripts/xai_eval.py --datasets multiclass \
  --limit 150 --device 0 > revisi/out/xai_eval.log 2>&1
echo "[finalize] xai done at $(date)"

python3 revisi/scripts/aggregate.py   > revisi/out/aggregate.log 2>&1
python3 revisi/scripts/fill_tokens.py > revisi/out/tokens.log 2>&1
python3 revisi/scripts/make_figures.py > revisi/out/figures.log 2>&1
python3 revisi/scripts/build_docx.py  > revisi/out/docx.log 2>&1
echo "[finalize] complete at $(date)"
