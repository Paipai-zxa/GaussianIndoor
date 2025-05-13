scene_list=(0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00)

root_path=output/recon_bkp/scannetv2/

for scene in ${scene_list[@]}; do
    # 遍历每个实验目录
    for experiment_dir in $(find ${root_path}/${scene} -mindepth 1 -maxdepth 1 -type d); do
        output_path=${experiment_dir}
        blender --background --python visualize_normal_blender.py -- \
        /data1/wxb/indoor/GaussianIndoor/data/${scene} \
        /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
        /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
        > /dev/null
    done
done

scene_list=(8b5caf3398 8d563fc2cc)

root_path=output/recon_bkp/scannetpp/

for scene in ${scene_list[@]}; do
    # 遍历每个实验目录
    for experiment_dir in $(find ${root_path}/${scene} -mindepth 1 -maxdepth 1 -type d); do
        output_path=${experiment_dir}
        blender --background --python visualize_normal_blender.py -- \
        /data1/wxb/indoor/GaussianIndoor/data/${scene} \
        /data1/wxb/indoor/GaussianIndoor/${output_path}/fuse_post.ply \
        /data1/wxb/indoor/GaussianIndoor/${output_path}/mesh_render_normal \
        > /dev/null
    done
done