# scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00 8b5caf3398 8d563fc2cc 41b00feddb b20a261fdf 1ada7a0617 5748ce6f01 f6659a3107)

# for scene in ${scene_list[@]}; do
#     blender --background --python visualize_normal_blender.py -- \
#     /data1/wxb/indoor/GaussianIndoor/data/${scene} \
#     /data1/wxb/indoor/GaussianIndoor/data/${scene}/${scene}_vh_clean_2.ply \
#     /data1/wxb/indoor/GaussianIndoor/data/${scene}/mesh_render_normal \
#     > /dev/null
# done

scene=0085_00
output_path=output/scannetv2/0085_00/arxiv/3dgs_depth_1000000_100000_20250425_161623

blender --background --python visualize_normal_blender.py -- \
/data1/wxb/indoor/GaussianIndoor/data/${scene} \
/data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
/data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
> /dev/null