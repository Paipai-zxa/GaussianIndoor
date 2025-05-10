#!/bin/bash

iterations=50000
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$2]}
if [ $2 -le 7 ]; then
    dataset_name=scannetv2
else
    dataset_name=scannetpp
fi

current_time=$(date "+%Y%m%d_%H%M%S")
base_exp_name=train_onlygradsdf

grad_sdf_omega_values=(0.002 0.0002 0.00002 0.000002)
sdf_guidance_start_iter_values=(15000 20000)
sdf_guidance_end_iter_values=(25000 30000)
sdf_guidance_interval_values=(50 100 500)

# 遍历每个weight的所有组合
for grad_sdf_omega in "${grad_sdf_omega_values[@]}"; do
    for sdf_guidance_start_iter in "${sdf_guidance_start_iter_values[@]}"; do
        for sdf_guidance_end_iter in "${sdf_guidance_end_iter_values[@]}"; do
            for sdf_guidance_interval in "${sdf_guidance_interval_values[@]}"; do
                exp_name="${base_exp_name}_Omega_${grad_sdf_omega}_StartIter_${sdf_guidance_start_iter}_EndIter_${sdf_guidance_end_iter}_Interval_${sdf_guidance_interval}_${current_time}"
                output_path=output/${dataset_name}/${scene}/onlygradsdf/${exp_name}
                mkdir -p ${output_path}

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
                    --detach_geo_mlp_input_feat \
                    --detach_geo_rasterizer_input_shs \
                    --densify_until_iter ${sdf_guidance_end_iter} \
                    --sdf_guidance_start_iter ${sdf_guidance_start_iter} \
                    --sdf_guidance_end_iter ${sdf_guidance_end_iter} \
                    --sdf_guidance_interval ${sdf_guidance_interval} \
                    --grad_sdf_omega ${grad_sdf_omega} \
                    --is_apply_grad_sdf_omega \
                    --enable_sdf_guidance \
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
done