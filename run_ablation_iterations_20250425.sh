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

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_fulldetachgeo
options=(
    "detach_geo_mlp_input_feat"
    "detach_scales_ori"
    "detach_rotations_ori"
    "detach_geo_rasterizer_input"
    "scales_geo_after_activation"
    "rotations_geo_after_activation"
)

# 固定应用所有的detach选项
option_flags=""
for option in "${options[@]}"; do
    option_flags+=" --${option}"
done

opt_geo_mlp_iteration_values=(0 7000 15000)
plane_constraint_iteration_values=(0 7000 15000)
scale_flatten_iteration_values=(0 7000 15000)

# 遍历每个参数的所有组合
for opt_iter in "${opt_geo_mlp_iteration_values[@]}"; do
    for plane_iter in "${plane_constraint_iteration_values[@]}"; do
        for scale_iter in "${scale_flatten_iteration_values[@]}"; do
            exp_name="${base_exp_name}_optIter_${opt_iter}_planeIter_${plane_iter}_scaleIter_${scale_iter}_${current_time}"
            output_path=output/${dataset_name}/${scene}/${exp_name}
            mkdir -p ${output_path}

            # 构建命令
            command="python train.py -s data/${scene} -m ${output_path} --use_depth_regularization --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01 --use_plane_constraint --plane_constraint_iteration ${plane_iter} --plane_constraint_weight 0.015 --use_scale_flatten --scale_flatten_iteration ${scale_iter} --scale_flatten_weight 100 --enable_geo_mlp --opt_geo_mlp_iteration ${opt_iter} --feat_dim 32 --iterations ${iterations} --eval${option_flags}"

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
        done
    done
done