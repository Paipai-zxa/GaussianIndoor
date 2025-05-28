#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(8b5caf3398 8d563fc2cc)

scene=${scene_list[$2]}
dataset_name=scannetpp
depth_l1_weight_init=100000
depth_l1_weight_final=1000


current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_wogeo
# base_exp_name=debug_sem_wogeo_semantic_guidance_instancetrain

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
    --iterations ${iterations}" 

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

# python eval_segmentation_scannet.py \
#     --scene_idx ${scene} \
#     --data_root ./data \
#     --debug \
#     --is_use_remap_instance \
#     --result_root ${output_path}

blender --background --python visualize_normal_blender.py -- \
/data1/wxb/indoor/GaussianIndoor/data/${scene} \
/data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
/data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
> /dev/null

# blender --background --python visualize_blender.py -- \
# /data1/wxb/indoor/GaussianIndoor/data/${scene} \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_semantic_post.ply \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_semantic \
# > /dev/null