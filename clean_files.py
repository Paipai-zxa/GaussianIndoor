import os
import shutil

def clean_experiment_files(base_path):
    """
    清理实验文件夹中的特定文件和目录
    
    Args:
        base_path: 实验文件夹的基础路径
    """
    # 要删除的文件列表
    files_to_delete = [
        'fuse_post_clean_bbox_faces.ply',
        'fuse_post_clean_bbox_faces_mask.ply',
        'fuse_post_clean_bbox.ply',
        'input.ply', 
        'fuse_gaussian_tsdf.ply'
    ]
    
    # 要删除的目录
<<<<<<< HEAD
    dirs_to_delete = ['test/ours_50000/gt', 'test/ours_30000/gt']
=======
    dirs_to_delete = ['mesh_render_normal', 'test/ours_50000/gt', 'test/ours_30000/gt', 'render_traj']
>>>>>>> f3b4367b1af5aac9177357d49c7c10f12fafc5e6
    
    # 遍历所有实验文件夹
    for root, dirs, files in os.walk(base_path):
        # 检查并删除指定目录
        for dir_name in dirs_to_delete:
            dir_path = os.path.join(root, dir_name)
            if os.path.exists(dir_path):
                # print(f"删除目录: {dir_path}")
                shutil.rmtree(dir_path)
        
        # 检查并删除指定文件
        for file_name in files_to_delete:
            file_path = os.path.join(root, file_name)
            if os.path.exists(file_path):
                # print(f"删除文件: {file_path}")
                os.remove(file_path)
        
        # 处理point_cloud文件夹
        if os.path.basename(root) == 'point_cloud':
            # 获取所有iteration_开头的文件夹
            iteration_dirs = [d for d in os.listdir(root) if d.startswith('iteration_')]
            if iteration_dirs:
                # 按迭代次数排序
                iteration_dirs.sort(key=lambda x: int(x.split('_')[1]))
                # 保留最新的一个，删除其他的
                for old_dir in iteration_dirs[:-1]:
                    old_path = os.path.join(root, old_dir)
                    # print(f"删除旧的点云文件夹: {old_path}")
                    shutil.rmtree(old_path)

if __name__ == "__main__":
    # scene_list = ['8b5caf3398', '8d563fc2cc', '1ada7a0617', 'f6659a3107', '5748ce6f01']
    # for scene in scene_list:
    #     base_path = "./output/scannetpp/" + scene
    #     clean_experiment_files(base_path)


    # scene_list = ['0050_00', '0085_00', '0114_02', '0580_00', '0603_00', '0616_00', '0617_00', '0721_00']
<<<<<<< HEAD
    # for scene in scene_list:
    #     base_path = "./output/scannetv2/" + scene
    #     clean_experiment_files(base_path)

    scene_list = ['0087_02', '0088_00', '0420_01', '0628_02']
    for scene in scene_list:
=======
    scene_list = ['0087_02', '0088_00', '0420_01', '0628_02']
    for scene in scene_list:
        # base_path = "./output/scannetv2/" + scene
>>>>>>> f3b4367b1af5aac9177357d49c7c10f12fafc5e6
        base_path = "./output/scannetv2_pan/" + scene
        clean_experiment_files(base_path)
    
    # scene_list = ['0087_02', '0088_00', '0420_01', '0628_02']
    # for scene in scene_list:
    #     base_path = "./output/scannetv2_pan/" + scene
    #     clean_experiment_files(base_path)
    # base_path = "/data1/wxb/indoor/GaussianIndoor/output/scannetv2/0085_00/arxiv/onlygradsdf"
    # clean_experiment_files(base_path)
