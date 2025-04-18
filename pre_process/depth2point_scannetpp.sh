scene_list=(8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

scene=${scene_list[$1]}

echo "Processing scene: ${scene}"

python utils/depth2point_utils.py \
    --scene_dir data/${scene} 

