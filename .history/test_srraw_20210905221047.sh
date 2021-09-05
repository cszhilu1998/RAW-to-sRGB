#!/bin/bash
echo "Start to test the model...."

name="srrawjoint"
dataroot="/Data/dataset/SRARW"

python test.py \
    --model srrawjoint   --name $name      --dataset_name zrr   --pre_ispnet_coord False  --gcm_coord True \
    --load_iter 1      --save_imgs True  --calc_metrics True  --gpu_id 0        --visual_full_imgs False \
    --dataroot $dataroot

python metrics.py  --name $name --dataroot $dataroot
