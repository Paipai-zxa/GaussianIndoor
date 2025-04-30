# export CUDA_VISIBLE_DEVICES=$1

# scene_list=(0087_02 0088_00 0420_01 0628_02 5748ce6f01 1ada7a0617 f6659a3107)

# scene=${scene_list[$2]}

# python groundsam_mask.py \
# --dataset_root ./data \
# --dataset scannet++ \
# --scene $scene 

# export CUDA_VISIBLE_DEVICES=7

# scene_list=(0087_02 0088_00 0420_01 0628_02)

# for scene in ${scene_list[@]}
# do 
#     python groundsam_mask.py \
#     --dataset_root ./data \
#     --dataset scannet \
#     --scene $scene 
# done