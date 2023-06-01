#!/bin/bash
name=one_step_finetune_nerf
prev_exp_name=one_step_nerf_only
declare -a Scenes=("s00" "s10" "s20" "s30" "s40" "s50" "s60" "s70" "s80" "s90")
for i in "${!Scenes[@]}"; do
    python scripts/train_finetune.py --exp cfg/exp/one_step_finetune_nerf/${Scenes[i]}_lr1e-5.yml --project_name $name --prev_exp_name $prev_exp_name 
done