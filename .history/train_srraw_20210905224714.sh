#!/bin/bash

echo "Start to train the model...."

name="srrawjoint"
dataroot="/Data/dataset/SRARW"


build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name srraw      --model srraw      --name $name       --gcm_coord True  \
    --pre_ispnet_coord False  --niter 100        --lr_decay_iters 50   --save_imgs False \
    --batch_size 16           --print_freq 300   --calc_metrics True   --lr 1e-4   -j 8  \
    --dataroot $dataroot      --scale 1 | tee $LOG    



build_dir="./ckpt/"$name_x2

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi
LOG=./ckpt/$name_x2/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name srraw      --model srraw      --name $name_x2       --gcm_coord True  \
    --pre_ispnet_coord False  --niter 100        --lr_decay_iters 50   --save_imgs False \
    --batch_size 16           --print_freq 300   --calc_metrics True   --lr 1e-4   -j 8  \
    --dataroot $dataroot      --scale 2 | tee $LOG    



build_dir="./ckpt/"$name_x4

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi
LOG=./ckpt/$name_x4/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name srraw      --model srraw      --name $name_x4       --gcm_coord True  \
    --pre_ispnet_coord False  --niter 100        --lr_decay_iters 50   --save_imgs False \
    --batch_size 16           --print_freq 300   --calc_metrics True   --lr 5e-5   -j 8  \
    --dataroot $dataroot      --scale 4 | tee $LOG   