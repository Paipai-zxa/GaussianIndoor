scene_list=(0087_02 0088_00 0420_01 0628_02)

for scene in ${scene_list[@]}
do 
    python eval_segmentation_scannet.py \
        --scene_idx $scene \
        --data_root ./data \
        --result_root ./panoptic_results/$scene \
        --is_use_remap_instance \
        --debug \
        # --save_remap_instance
done
