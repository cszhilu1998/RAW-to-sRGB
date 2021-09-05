#!/bin/bash
echo "Start to test the model...."

python test.py \
    --model zrrjoint    --name zrrjoint    --dataset_name eth   --pre_ispnet_coord False  --alignnet_coord True \
    --load_iter 56      --save_imgs True  --calc_psnr True      --gpu_id 0           --visual_full_imgs False \
    --dataroot /data/dataset/Zurich-RAW-to-DSLR
