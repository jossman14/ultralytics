#!/bin/bash
# Wait for the phase-1 (random-split) sweep, then run the video-disjoint sweep.
cd /home/ftib/ultralytics
while pgrep -f "train_baselines.py --slot" > /dev/null; do sleep 120; done
echo "phase 1 finished at $(date)"
for slot in 0 1 2 3; do
  dev=$((slot % 2))
  PYTHONPATH=/home/ftib/ultralytics nohup python3 revisi/scripts/train_baselines.py \
    --slot $slot --nslots 4 --device $dev --workers 3 --datasets binary_videofold \
    > revisi/runs/p2_slot${slot}.log 2>&1 &
done
wait
echo "phase 2 finished at $(date)"
