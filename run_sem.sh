#!/bin/bash

iterations=30000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0087_02 0088_00 0420_01 0628_02)

dataset_name=scannetv2_pan
depth_l1_weight_init=100000
depth_l1_weight_final=1000

# 添加额外的训练参数
extra_args="--is_train_on_all_images"


# 遍历每个weight的所有组合
exp_name="${base_exp_name}_${current_time}"
exp_name="train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final_20250512_154805"
output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}

script_name=$(basename "$0" .sh)

# 执行训练命令
# eval $command

# 执行后续命令
python render.py \
    -m ${output_path} \
    --iteration ${iterations} \
    --eval \
    --skip_train \
    --skip_test \
    --skip_mesh \
    --render_traj \
    --traj_json data/${scene}/interpolated_poses.json \
    --mesh_res 512 \
    --depth_trunc 5.0 

# python ./eval_mesh/exp_evaluation.py \
#     --mode eval_3D_mesh_metrics \
#     --dir_dataset data \
#     --dir_results_baseline ${output_path} \
#     --path_mesh_pred ${output_path}/fuse_post.ply \
#     --scene_name ${scene}

# python metrics.py \
#     -m ${output_path}

python eval_segmentation_scannet.py \
    --scene_idx ${scene} \
    --data_root ./data \
    --debug \
    --is_use_remap_instance \
    --save_traj \
    --save_traj_path render_traj \
    --result_root ${output_path}

# python merge_video.py \
# --input_dir ${output_path}/render_traj \
# --fps 24

# blender --background --python visualize_normal_blender.py -- \
# /data1/wxb/indoor/GaussianIndoor/data/${scene} \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
# /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
# > /dev/null

# blender --background --python visualize_normal_blender.py -- \
# /data1/wxb/indoor/GaussianIndoor/data/8b5caf3398 \
# /data1/wxb/indoor/GaussianIndoor/data/8b5caf3398/8b5caf3398_vh_clean_2_clean.ply \
# /data1/wxb/indoor/GaussianIndoor/data/8b5caf3398/mesh_render_normal \
# > /dev/null

# blender --background --python visualize_normal_blender.py -- \
# /data1/wxb/indoor/GaussianIndoor/data/8d563fc2cc \
# /data1/wxb/indoor/GaussianIndoor/data/8d563fc2cc/8d563fc2cc_vh_clean_2_clean.ply \
# /data1/wxb/indoor/GaussianIndoor/data/8d563fc2cc/mesh_render_normal \
# > /dev/null