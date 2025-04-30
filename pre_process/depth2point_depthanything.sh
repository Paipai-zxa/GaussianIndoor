scene_list=(0085_00)

scene=${scene_list[$1]}

echo "Processing scene: ${scene}"

python utils/depth2point_utils.py \
    --is_depth_anything \
    --scene_dir data/${scene} 

