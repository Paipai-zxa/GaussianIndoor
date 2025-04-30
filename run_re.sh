iterations=30000
# checkpoint_iterations=(5000 10000 15000)
export CUDA_VISIBLE_DEVICES=$1

scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$2]}
if [ $2 -le 7 ]; then
    dataset_name=scannetv2
else
    dataset_name=scannetpp
fi


current_time=$(date "+%Y%m%d_%H%M%S")
exp_name=train_re
# exp_name=train_planecons_7000_0.015_detachall
# exp_name=train_planecons_7000_0.01
# exp_name=train_detachshs_featbank_20250429_194238

data_path=data/${scene}
exp_name=${exp_name}_${current_time}
# exp_name=train_detachshs_re_20250429_213347


output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}

python train.py -s data/${scene} -m ${output_path} \
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
    --iterations ${iterations} --eval 

    # --enable_geo_mlp \
    # --opt_geo_mlp_iteration 7000 \
    # --feat_dim 32 \
    # --detach_geo_mlp_input_feat \
    # --detach_scales_ori \
    # --detach_rotations_ori 


    # --detach_geo_rasterizer_input_shs \
    # --detach_geo_rasterizer_input_means3D \
    # --detach_geo_rasterizer_input_means2D \
    # --detach_geo_rasterizer_input_means2D_abs \
    # --detach_geo_rasterizer_input_opacity \
    # --detach_geo_rasterizer_input_shs \
    # --detach_geo_rasterizer_input_input_all_map \

    # --use_render_geo \
    # --lambda_geo 1.0 \

    # --use_scale_flatten \
    # --scale_flatten_iteration 0 \
    # --scale_flatten_weight 10000 \

    # --use_depth_regularization \
    # --depth_l1_weight_init 1.0 \
    # --depth_l1_weight_final 0.01 \

    # --use_plane_constraint \
    # --plane_constraint_iteration 7000 \
    # --plane_constraint_weight 0.01 \

    # --use_cross_view_constraint \
    # --cross_view_constraint_iteration 7000 \
    # --cross_view_constraint_weight 0.01 \

    # --enable_geo_mlp \
    # --feat_dim 32 \
    # --detach_geo_mlp_input_feat \
    # --detach_scales_ori \
    # --detach_rotations_ori \
    # --detach_geo_rasterizer_input \
    # --scales_geo_after_activation \
    # --rotations_geo_after_activation \

    # --enable_training_exposure \

    # --n_offsets 10 \
    # --plane_constraint_weight 1.0 \
    # --plane_constraint_iteration ${plane_constraint_iteration} \
    # --cross_view_constraint_iteration ${cross_view_constraint_iteration} \
    # --cross_view_constraint_weight 0.0005 \
    # --num_neighbors_views 2 \
    # --start_checkpoint ${start_checkpoint}
    # --checkpoint_iterations ${checkpoint_iterations[@]}


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
