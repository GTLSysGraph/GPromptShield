#!/bin/bash

# 定义三个参数的值
ptb_rate=(0.2)
shot_num=(1 5 10)
run_split=(1)

# 遍历所有参数的组合
for shot in "${shot_num[@]}"; do
    for run in "${run_split[@]}"; do
        for ptb in "${ptb_rate[@]}"; do
            echo "运行顺序: $shot $run $ptb"
            CUDA_VISIBLE_DEVICES=1 python generate_few_shot_attack.py --dataset 'Citeseer' --model 'Meta_Self' --ptb_rate $ptb --shot_num $shot --run_split $run
        done
    done
done