#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2
python train_N_times.py \
    --multiprocessing_distributed \
    --train_data_path /data/vggsound/ \
    --test_data_path /data/vggss/ \
    --test_gt_path /data/vggss/ \
    --experiment_name vggsound_144k_random \
    --trainset 'vggsound_144k_random' \
    --testset 'vggss' \
    --epochs 30 \
    --batch_size 1600 \
    --init_lr 0.0001
~
~