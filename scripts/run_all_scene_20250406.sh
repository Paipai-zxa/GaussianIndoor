#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_depth_1000000_100000_wo_detachall

# 遍历每个场景
for scene in "${scene_list[@]}"; do
    if [[ " ${scene_list[@]:0:8} " =~ " ${scene} " ]]; then
        dataset_name=scannetv2
    else
        dataset_name=scannetpp
    fi

    data_path=data/${scene}
    exp_name=${base_exp_name}_${current_time}
    output_path=output/${dataset_name}/${scene}/${exp_name}
    mkdir -p ${output_path}

    # 执行训练命令
    python train.py \
        -s ${data_path} \
        -m ${output_path} \
        --use_depth_regularization \
        --depth_l1_weight_init 1000000 \
        --depth_l1_weight_final 100000 \
        --enable_geo_mlp \
        --opt_geo_mlp_iteration 7000 \
        --feat_dim 32 \
        --iterations ${iterations} \
        --eval

    # 执行渲染命令
    python render.py \
        -m ${output_path} \
        --iteration ${iterations} \
        --eval \
        --skip_train \
        --mesh_res 512 \
        --depth_trunc 5.0 

    # 执行评估命令
    python ./eval_mesh/exp_evaluation.py \
        --mode eval_3D_mesh_metrics \
        --dir_dataset data \
        --dir_results_baseline ${output_path} \
        --path_mesh_pred ${output_path}/fuse_post.ply \
        --scene_name ${scene}

    python metrics.py \
        -m ${output_path}
done