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
base_exp_name=train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final

# 遍历每个weight的所有组合
exp_name="${base_exp_name}_${current_time}"
# exp_name="train_sem_wogeo_semantic_guidance_start4000_20250511_091223"
output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}

command="python train.py -s data/${scene} -m ${output_path} \
    --use_scale_flatten --scale_flatten_iteration 0 --scale_flatten_weight 1 \
    --num_neighbors_views 1 \
    --use_depth_regularization \
    --depth_l1_weight_init ${depth_l1_weight_init} \
    --depth_l1_weight_final ${depth_l1_weight_final} \
    --densify_until_iter 15000 \
    --sdf_guidance_start_iter 12000 \
    --sdf_guidance_end_iter 15000 \
    --sdf_guidance_interval 100 \
    --grad_sdf_omega 0.000002 \
    --is_apply_grad_sdf_omega \
    --enable_semantic \
    --opt_semantic_mlp_iteration 0 \
    --semantic_mlp_dim 64 \
    --instance_query_distance_mode 2 \
    --semantic_warping_weight 0.0 \
    --load_semantic_from_pcd \
    --apply_semantic_guidance \
    --iterations ${iterations} --eval ${extra_args}" 
    # --use_cross_view_constraint --cross_view_constraint_iteration 7000 --cross_view_constraint_weight 1.5 \

    # 
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

python eval_segmentation_scannet.py \
    --scene_idx ${scene} \
    --data_root ./data \
    --debug \
    --is_use_remap_instance \
    --result_root ${output_path}

# blender --background --python visualize_normal_blender.py -- \
# /data1/wxb/indoor/GaussianIndoor/data/${scene} \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
# > /dev/null