#!/bin/bash

iterations=50000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 1ada7a0617 f6659a3107 5748ce6f01 0087_02 0088_00 0420_01 0628_02)

scene=${scene_list[$2]}
if [ $2 -le 7 ]; then
    dataset_name=scannetv2
    depth_l1_weight_init=100000
    depth_l1_weight_final=1000
elif [ $2 -le 12 ]; then
    dataset_name=scannetpp
    depth_l1_weight_init=100000
    depth_l1_weight_final=1000
else
    dataset_name=scannetv2_pan
    depth_l1_weight_init=100000
    depth_l1_weight_final=1000
fi

# 添加额外的训练参数
extra_args=""
if [ $2 -gt 9 ]; then
    extra_args="--is_train_on_all_images --enable_semantic"
fi

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_onlygradsdf_Omega_0.0002_StartIter_15000_EndIter_25000_Interval_100

    # --detach_geo_mlp_input_feat \
    # --detach_geo_rasterizer_input_shs \
    # --enable_geo_mlp \
    # --opt_geo_mlp_iteration 7000 \
    # --feat_dim 32 \

# 遍历每个weight的所有组合
exp_name="${base_exp_name}_${current_time}"
output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}

command="python train.py -s data/${scene} -m ${output_path} \
    --use_scale_flatten --scale_flatten_iteration 0 --scale_flatten_weight 1 \
    --use_cross_view_constraint --cross_view_constraint_iteration 7000 --cross_view_constraint_weight 1.5 \
    --num_neighbors_views 1 \
    --use_depth_regularization \
    --depth_l1_weight_init ${depth_l1_weight_init} \
    --depth_l1_weight_final ${depth_l1_weight_final} \
    --densify_until_iter 25000 \
    --sdf_guidance_start_iter 15000 \
    --sdf_guidance_end_iter 25000 \
    --sdf_guidance_interval 100 \
    --grad_sdf_omega 0.0002 \
    --is_apply_grad_sdf_omega \
    --detach_geo_mlp_input_feat \
    --detach_geo_rasterizer_input_shs \
    --enable_geo_mlp \
    --opt_geo_mlp_iteration 7000 \
    --feat_dim 32 \
    --iterations ${iterations} --eval ${extra_args}" 

# 执行训练命令
eval $command

# 执行后续命令
python render.py \
    -m ${output_path} \
    --iteration ${iterations} \
    --eval \
    --skip_train \
    --mesh_res 512 \
    --depth_trunc 5.0 \
    ${extra_args}

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