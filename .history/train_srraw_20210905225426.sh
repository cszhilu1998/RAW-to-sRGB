#!/bin/bash

echo "Start to train the model...."

name="srrawjoint_try"
dataroot="/Data/dataset/SRARW"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


# x1 Super-Resolution
python train.py \
    --dataset_name srraw      --model srraw      --name $name          --gcm_coord True  \
    --pre_ispnet_coord False  --niter 2        --lr_decay_iters 2   --save_imgs False \
    --batch_size 4           --print_freq 100   --calc_metrics True   --lr 1e-4   -j 8  \
    --dataroot $dataroot      --scale 1   | tee $LOG    

# x2 Super-Resolution
python train.py \
    --dataset_name srraw      --model srraw      --name $name           --gcm_coord True  \
    --pre_ispnet_coord False  --niter 4        --lr_decay_iters 5   --save_imgs False \
    --batch_size 4           --print_freq 100   --calc_metrics True    --lr 1e-4   -j 8  \
    --dataroot $dataroot      --scale 2          --load_iter 2   | tee $LOG    

# x4 Super-Resolution
python train.py \
    --dataset_name srraw      --model srraw      --name $name           --gcm_coord True  \
    --pre_ispnet_coord False  --niter 6        --lr_decay_iters 4   --save_imgs False \
    --batch_size 4           --print_freq 100   --calc_metrics True    --lr 5e-5   -j 8  \
    --dataroot $dataroot      --scale 4          --load_iter 200  | tee $LOG   

# # x1 Super-Resolution
# python train.py \
#     --dataset_name srraw      --model srraw      --name $name          --gcm_coord True  \
#     --pre_ispnet_coord False  --niter 100        --lr_decay_iters 50   --save_imgs False \
#     --batch_size 16           --print_freq 100   --calc_metrics True   --lr 1e-4   -j 8  \
#     --dataroot $dataroot      --scale 1   | tee $LOG    

# # x2 Super-Resolution
# python train.py \
#     --dataset_name srraw      --model srraw      --name $name           --gcm_coord True  \
#     --pre_ispnet_coord False  --niter 200        --lr_decay_iters 150   --save_imgs False \
#     --batch_size 16           --print_freq 100   --calc_metrics True    --lr 1e-4   -j 8  \
#     --dataroot $dataroot      --scale 2          --load_iter 100   | tee $LOG    

# # x4 Super-Resolution
# python train.py \
#     --dataset_name srraw      --model srraw      --name $name           --gcm_coord True  \
#     --pre_ispnet_coord False  --niter 300        --lr_decay_iters 250   --save_imgs False \
#     --batch_size 16           --print_freq 100   --calc_metrics True    --lr 5e-5   -j 8  \
#     --dataroot $dataroot      --scale 4          --load_iter 200  | tee $LOG   