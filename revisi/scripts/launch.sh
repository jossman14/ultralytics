#!/bin/bash
cd /home/ftib/ultralytics
for slot in 0 1 2 3; do
  dev=$((slot % 2))
  PYTHONPATH=/home/ftib/ultralytics nohup python3 revisi/scripts/train_baselines.py --slot $slot --nslots 4 --device $dev --workers 3 \
    > revisi/runs/slot${slot}.log 2>&1 &
  echo "slot $slot -> gpu $dev pid $!"
done
