scene_list=(0087_02 0088_00 0420_01 0628_02 1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$1]}

echo "Processing scene: ${scene}"

python utils/semantic2point_utils.py \
    --scene_dir data/${scene} \
    --semantic_dir semantic \

