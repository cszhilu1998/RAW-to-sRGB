#!/bin/bash
echo "Start to test the model...."

name="zrrjoint"

python test.py \
    --model zrrjoint    --name $name     --dataset_name zrr   --pre_ispnet_coord False  --gcm_coord True \
    --load_iter 56      --save_imgs True  --calc_metrics True      --gpu_id 0           --visual_full_imgs False \
    --dataroot /Data/dataset/Zurich-RAW-to-DSLR
