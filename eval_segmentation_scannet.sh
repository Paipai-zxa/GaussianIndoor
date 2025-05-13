scene_list=(0087_02 0088_00 0420_01 0628_02 1ada7a0617 5748ce6f01 f6659a3107)
# scene_list=(0420_01)

for scene in ${scene_list[@]}
do 
    python eval_segmentation_scannet.py \
        --scene_idx $scene \
        --data_root ./data \
        --result_root ./data/$scene \
        --debug \
        --save_gt \
        --is_use_remap_instance
        # --save_remap_instance
done
        # --is_use_remap_instance
        # --debug \
        # --save_gt \
        # --debug \