export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,3

python train_N_times.py \
    --multiprocessing_distributed \
    --train_data_path /data/vggsound/ \
    --test_data_path /data/vggss/ \
    --test_gt_path /data/vggss/ \
    --experiment_name vggsound_10k_random \
    --trainset 'vggsound_10k_random' \
    --testset 'vggss' \
    --epochs 1 \
    --batch_size 1600 \
    --init_lr 0.0001
