#!/bin/bash
python test_alpha.py \
    --test_gt_path /data/vggss_4600/ \
    --test_data_path /data/vggss_4600 \
    --model_dir ./checkpoints \
    --experiment_name vggsound_144k_run1 \
    --testset 'vggss' \
    --alpha [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] \