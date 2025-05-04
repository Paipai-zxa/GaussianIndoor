iterations=30000
plane_constraint_iteration=5000
cross_view_constraint_iteration=15000
use_plane_constraint=False
use_cross_view_constraint=False
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
exp_name=train_scaffold_featurebank_n10_planecons_7000_0.015

data_path=data/${scene}
exp_name=${exp_name}_${current_time}
# exp_name=train_scaffold_featurebank_20250425_155333

output_path=output/${dataset_name}/${scene}/${exp_name}
mkdir -p ${output_path}
python train.py \
    -s ${data_path} \
    -m ${output_path} \
    --iterations ${iterations} \
    --enable_scaffold \
    --n_offsets 10 \
    --use_feat_bank \
    --use_plane_constraint \
    --plane_constraint_iteration 7000 \
    --plane_constraint_weight 0.015 \
    --eval

    # --use_depth_regularization \
    # --depth_l1_weight_init 1000000 \
    # --depth_l1_weight_final 100000 \

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