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
base_exp_name=train_depthloss_1.0_0.01_planecons_7000_0.015_scaleflatten_0_100
options=(
    "detach_geo_mlp_input_feat"
    "detach_scales_ori"
    "detach_rotations_ori"
    "detach_geo_rasterizer_input"
    "scales_geo_after_activation"
    "rotations_geo_after_activation"
)
switche_list=(000000 111111 100000 010000 001000 000100 000010 000001 111000 111100 011111 000100 000111)

# 遍历每个开关组合
for switches in "${switche_list[@]}"; do
    exp_name=${base_exp_name}
    # 根据开关参数决定是否添加选项
    option_flags=""
    for i in {0..5}; do
        if [ "${switches:$i:1}" -eq 1 ]; then
            option_flags+=" --${options[$i]}"
            exp_name+="_${options[$i]}"
        fi
    done

    # 从0085_00文件夹中查找对应的工程
    for output_path in $(find output/${dataset_name}/${scene} -type d -name "${exp_name}_20250425*"); do
        # 检查是否找到对应的工程
        if [ -z "${output_path}" ]; then
            echo "No matching output path found for ${exp_name}."
            continue
        fi

        echo "Output path: ${output_path}"

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
            --dir_results_baseline "${output_path}" \
            --path_mesh_pred "${output_path}/fuse_post.ply" \
            --scene_name ${scene}

        python metrics.py \
            -m "${output_path}"
    done
done