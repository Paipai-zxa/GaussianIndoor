# scene_list=(0087_02 0088_00 0420_01 0628_02)
scene_list=(1ada7a0617 5748ce6f01 f6659a3107)
# scene_list=(1ada7a0617)

for scene in ${scene_list[@]}; do
    python gen_traj.py \
        --scene $scene \
        --num_interp 10 \
        --intrinsics data/$scene/color_intrinsics.txt \
        --width 640 \
        --height 480 
done