#!/bin/bash
# Launch N parallel sweep workers sharing GPU 0. Each claims jobs from the shared locked ledger.
cd /home/ftib/ultralytics
export PYTHONPATH=/home/ftib/ultralytics
N=${1:-2}
for w in $(seq 0 $((N-1))); do
  nohup python judol/run_sweep.py --worker "$w" --workers-per-job 2 \
    > "judol_sweep/worker_${w}.log" 2>&1 &
  echo "launched worker $w pid $!"
  sleep 8   # stagger so the two don't hit dataset caching at the exact same instant
done
