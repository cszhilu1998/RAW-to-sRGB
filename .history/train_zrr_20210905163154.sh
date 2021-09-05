#!/bin/bash

echo "Start to train the model...."

name="zrrjoint_try"

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name zrr        --model zrrjoint    --name $name         --gcm_coord True  \
    --pre_ispnet_coord False  --niter 80         --lr_decay_iters 40   --save_imgs False \
    --batch_size 16           --print_freq 300    --calc_metrics True  --lr 1e-4   -j 8  \
    --dataroot /Data/dataset/Zurich-RAW-to-DSLR | tee $LOG    
