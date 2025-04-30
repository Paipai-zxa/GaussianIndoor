#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$2]}
if [ $2 -le 7 ]; then
    dataset_name=scannetv2
else
    dataset_name=scannetpp
fi

    # --detach_geo_rasterizer_input_shs \
    # --detach_geo_rasterizer_input_means3D \
    # --detach_geo_rasterizer_input_means2D \
    # --detach_geo_rasterizer_input_means2D_abs \
    # --detach_geo_rasterizer_input_opacity \
    # --detach_geo_rasterizer_input_shs \
    # --detach_geo_rasterizer_input_input_all_map \

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train
options=(
    "detach_geo_mlp_input_feat"
    "detach_scales_ori"
    "detach_rotations_ori"
    "detach_geo_rasterizer_input_shs"
    "detach_geo_rasterizer_input_means3D"
    "detach_geo_rasterizer_input_means2D"
    "detach_geo_rasterizer_input_means2D_abs"
    "detach_geo_rasterizer_input_opacity"
    "detach_geo_rasterizer_input_input_all_map"
)

switche_list=(000000000 111000000 111100000 111100010 111000010 100100000 100100010 100000010 011100000 011100010 011000010 000100000 000100010 000000010 111111111)
switche_list=("${switche_list[@]:$3:$4-$3+1}")

# 遍历每个开关组合
for switches in "${switche_list[@]}"; do
    exp_name=${base_exp_name}
    # 根据开关参数决定是否添加选项
    option_flags=""
    for i in {0..8}; do
        if [ "${switches:$i:1}" -eq 1 ]; then
            option_flags+=" --${options[$i]}"
            exp_name+="_${options[$i]}"
        fi
    done

    exp_name=${exp_name}_${current_time}
    output_path=output/${dataset_name}/${scene}/crossview_7000_1.5_numneighbors1_depth_100000_1000_geo_mlp/${exp_name}
    mkdir -p ${output_path}

    # 构建命令
    command="python train.py -s data/${scene} -m ${output_path} \
    --use_scale_flatten --scale_flatten_iteration 0 --scale_flatten_weight 1 \
    --use_cross_view_constraint --cross_view_constraint_iteration 7000 --cross_view_constraint_weight 1.5 \
    --num_neighbors_views 1 \
    --use_depth_regularization \
    --depth_l1_weight_init 100000 \
    --depth_l1_weight_final 1000 \
    --enable_geo_mlp \
    --opt_geo_mlp_iteration 7000 \
    --feat_dim 32 \
    --iterations ${iterations} --eval \
    ${option_flags}"

    # 执行训练命令
    eval $command

    # 执行后续命令
    python render.py \
        -m ${output_path} \
        --iteration ${iterations} \
        --eval \
        --skip_train \
        --mesh_res 512 \
        --depth_trunc 5.0 

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