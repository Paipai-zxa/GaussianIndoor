export CUDA_VISIBLE_DEVICES=7

scene_list=(5748ce6f01 1ada7a0617 f6659a3107)

for scene in ${scene_list[@]}
do 
    python groundsam_mask.py \
    --dataset_root ./data \
    --dataset scannet++ \
    --scene $scene \
    --box_threshold 0.15 \
    --text_threshold 0.20 \
    --instance_threshold 0.20
done

# export CUDA_VISIBLE_DEVICES=7

# scene_list=(0087_02 0088_00 0420_01 0628_02)
# scene_list=(0087_02)

# for scene in ${scene_list[@]}
# do 
#     python groundsam_mask.py \
#     --dataset_root ./data \
#     --dataset scannet \
#     --scene $scene \
#     --box_threshold 0.15 \
#     --text_threshold 0.20 \
#     --instance_threshold 0.20
# done