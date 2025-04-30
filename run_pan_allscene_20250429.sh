#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0087_02 0088_00 0420_01 0628_02)

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_crossview_7000_1.5_numneighbors1_depth_100000_1000

# 遍历每个场景
for scene in "${scene_list[@]}"; do
    dataset_name=scannetv2_pan

    data_path=data/${scene}
    exp_name=${base_exp_name}_${current_time}
    output_path=output/${dataset_name}/${scene}/${exp_name}
    mkdir -p ${output_path}

    # 执行训练命令
    python train.py -s data/${scene} -m ${output_path} \
        --use_scale_flatten --scale_flatten_iteration 0 --scale_flatten_weight 1 \
        --use_cross_view_constraint --cross_view_constraint_iteration 7000 --cross_view_constraint_weight 1.5 \
        --num_neighbors_views 1 \
        --use_depth_regularization \
        --depth_l1_weight_init 100000 \
        --depth_l1_weight_final 1000 \
        --iterations ${iterations} --eval 

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

    blender --background --python visualize_normal_blender.py -- \
    /data1/wxb/indoor/GaussianIndoor/data/${scene} \
    /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
    /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
    > /dev/null
done