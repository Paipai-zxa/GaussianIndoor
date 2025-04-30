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
base_exp_name=train_wo_detachall

plane_constraint_weight_values=(0.000001 0.015 0.15 1.5)
scale_flatten_weight_values=(0.000001 1 100 10000)
cross_view_constraint_weight_values=(0.000001 0.15 1.5 15)

# 遍历每个weight的所有组合
for plane_weight in "${plane_constraint_weight_values[@]}"; do
    for scale_weight in "${scale_flatten_weight_values[@]}"; do
        for cross_view_weight in "${cross_view_constraint_weight_values[@]}"; do
            exp_name="${base_exp_name}_planeWeight_${plane_weight}_scaleWeight_${scale_weight}_crossViewWeight_${cross_view_weight}_${current_time}"
            output_path=output/${dataset_name}/${scene}/wogeomlp/${exp_name}
            mkdir -p ${output_path}

            # 构建命令 换行？
            command="python train.py -s data/${scene} -m ${output_path} \
            --use_plane_constraint --plane_constraint_iteration 7000 --plane_constraint_weight ${plane_weight} \
            --use_scale_flatten --scale_flatten_iteration 0 --scale_flatten_weight ${scale_weight} \
            --use_cross_view_constraint --cross_view_constraint_iteration 7000 --cross_view_constraint_weight ${cross_view_weight} \
            --num_neighbors_views 1 \
            --use_depth_regularization \
            --depth_l1_weight_init 100000 \
            --depth_l1_weight_final 1000 \
            --iterations ${iterations} --eval"

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
    done
done