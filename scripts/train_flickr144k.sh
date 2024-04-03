#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2

python train_N_times.py \
    --multiprocessing_distributed \
    --train_data_path /data2/datasets/274k_flicker/ \
    --test_data_path /data2/datasets/labeled_5k_flicker/Data/ \
    --test_gt_path /data2/datasets/labeled_5k_flicker/Annotations/ \
    --experiment_name flickr_144k_run1 \
    --trainset 'flickr_144k_random' \
    --testset 'flickr' \
    --epochs 50 \
    --batch_size 1024 \
    --init_lr 0.0001
