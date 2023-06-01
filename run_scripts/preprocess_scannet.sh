#!/bin/bash
root_dir="data/scannet/scans"
for i in $(ls  -d  $root_dir/*/); 
do 
    echo ${i};
    python preprocessing_scripts/scannet2transform.py --scene_folder $i --scaled_image --semantics
    python preprocessing_scripts/scannet2nerf.py --scene_folder $i --transform_train $i/transforms_train_scaled_semantics_40_raw.json \
    --transform_test $i/transforms_test_scaled_semantics_40_raw.json --interval 10
done 