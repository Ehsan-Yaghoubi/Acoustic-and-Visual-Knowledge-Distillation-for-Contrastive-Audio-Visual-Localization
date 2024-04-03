#!/bin/bash
python test_N_times.py \
    --test_data_path /data2/datasets/labeled_5k_flicker/Data/ \
    --test_gt_path /data2/datasets/labeled_5k_flicker/Annotations/ \
    --model_dir /checkpoints \
    --experiment_name flickr_144k_run1 \
    --testset 'flickr' \
    --alpha 0.4