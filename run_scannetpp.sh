iterations=30000
plane_constraint_iteration=5000
cross_view_constraint_iteration=15000
use_plane_constraint=False
use_cross_view_constraint=False
# checkpoint_iterations=(5000 10000 15000)
export CUDA_VISIBLE_DEVICES=$1

scene_list=(8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)
# scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00)

scene=${scene_list[$2]}

output_path=output/${scene}
if [ ${use_plane_constraint} = True ]; then
    output_path=${output_path}_use_plane_constraint
fi
if [ ${use_cross_view_constraint} = True ]; then
    output_path=${output_path}_use_cross_view_constraint
fi
data_path=data/${scene}

output_path=output/${scene}
python train.py \
    -s ${data_path} \
    -m ${output_path} \
    --iterations ${iterations} \
    --eval

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

python ./eval_mesh/exp_evaluation.py \
    --mode eval_3D_mesh_metrics \
    --dir_dataset data \
    --dir_results_baseline ${output_path} \
    --path_mesh_pred ${output_path}/fuse_unbounded.ply \
    --scene_name ${scene}

python metrics.py \
    -m ${output_path}

