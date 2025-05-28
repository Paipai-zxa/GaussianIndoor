#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$2]}
dataset_name=scannetpp
depth_l1_weight_init=100000
depth_l1_weight_final=1000

# 添加额外的训练参数
extra_args="--is_train_on_all_images"

current_time=$(date "+%Y%m%d_%H%M%S")
# base_exp_name=train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final
# base_exp_name=debug_sem_wogeo_semantic_guidance_instancetrain

# 遍历每个weight的所有组合
exp_name="${base_exp_name}_${current_time}"
exp_name="train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final_20250512_155336"
output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}

python eval_segmentation_scannet.py \
    --scene_idx ${scene} \
    --data_root ./data \
    --debug \
    --is_use_remap_instance \
    --result_root ${output_path}