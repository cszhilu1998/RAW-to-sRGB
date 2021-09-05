#!/bin/bash

echo "Start to train the model...."

name_x1="srrawjoint_x1"
name_x2="srrawjoint_x2"
name_x4="srrawjoint"
dataroot="/Data/dataset/SRARW"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name srraw      --model srraw      --name $name_x1         --gcm_coord True  \
    --pre_ispnet_coord False  --niter 80         --lr_decay_iters 40   --save_imgs False \
    --batch_size 16           --print_freq 300   --calc_metrics True  --lr 1e-4   -j 8  \
    --dataroot $dataroot  --scale 4 | tee $LOG    
