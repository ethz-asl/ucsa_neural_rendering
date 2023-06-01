#!/bin/bash
name=one_step_joint
declare -a Scenes=("s00" "s10" "s20" "s30" "s40" "s50" "s60" "s70" "s80" "s90")
for i in "${!Scenes[@]}"; do
    python scripts/train_joint.py --exp cfg/exp/one_step_joint/${Scenes[i]}_lr1e-5.yml --exp_name $name --project_name $name --nerf_train_epoch 10 --joint_train_epoch 50
done