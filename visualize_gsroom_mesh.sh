# scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00)
scene_list=(8b5caf3398 8d563fc2cc)

for scene in ${scene_list[@]}
do
    # blender --background --python visualize_normal_blender.py -- \
    # /data1/wxb/indoor/GaussianIndoor/data/${scene} \
    # /data1/wxb/indoor/GaussianIndoor/gsroom_results/sdf_output/neus/scene${scene}/scene${scene}-full/meshes/00030000_reso512_scene${scene}_world_clean_bbox_faces.ply \
    # /data1/wxb/indoor/GaussianIndoor/gsroom_results/sdf_output/neus/scene${scene}/scene${scene}-full/mesh_render_normal \
    # > /dev/null
    blender --background --python visualize_normal_blender.py -- \
    /data1/wxb/indoor/GaussianIndoor/data/${scene} \
    /data1/wxb/indoor/GaussianIndoor/gsroom_results/sdf_output/neus/${scene}/${scene}-full/meshes/00030000_reso512_${scene}_world_clean_bbox_faces.ply \
    /data1/wxb/indoor/GaussianIndoor/gsroom_results/sdf_output/neus/${scene}/${scene}-full/mesh_render_normal \
    > /dev/null
done