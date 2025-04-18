import numpy as np
import open3d as o3d
import cv2
import os
from argparse import ArgumentParser
import tqdm
import torch
def reproject_depth_to_color(depth_image, K_depth, K_color):
    """
    将深度图从深度相机坐标系重投影到颜色相机坐标系。
    """
    height, width = depth_image.shape
    depth = torch.from_numpy(depth_image).float().cuda()

    # 创建像素坐标网格
    v, u = torch.meshgrid(
        torch.arange(height, device='cuda'),
        torch.arange(width, device='cuda'),
        indexing='ij'
    )
    # 找到有效深度值的位置
    valid_depth = depth > 0
    
    # 计算3D点坐标
    Z = depth[valid_depth]
    X = (u[valid_depth] - K_depth[0, 2]) * Z / K_depth[0, 0]
    Y = (v[valid_depth] - K_depth[1, 2]) * Z / K_depth[1, 1]
    
    # 投影到颜色相机坐标系
    u_color = ((K_color[0, 0] * X / Z) + K_color[0, 2]).long()
    v_color = ((K_color[1, 1] * Y / Z) + K_color[1, 2]).long()
    
    # 创建输出深度图
    reprojected_depth = torch.zeros_like(depth)
    
    # 找到有效的投影点
    valid_proj = (u_color >= 0) & (u_color < width) & (v_color >= 0) & (v_color < height)
    
    # 更新重投影深度图
    v_valid = v_color[valid_proj]
    u_valid = u_color[valid_proj]
    z_valid = Z[valid_proj]
    
    # 使用scatter_操作更新深度值
    reprojected_depth.index_put_((v_valid, u_valid), z_valid)
    
    # 转回CPU并转换为numpy数组
    return reprojected_depth.cpu().numpy().astype(np.uint16)    


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_path, "reprojected_depths"), exist_ok=True)
    depth_path = os.path.join(args.data_path, "depths")
    depth_intrinsics_path = os.path.join(args.data_path, "depth_intrinsics.txt")
    color_intrinsics_path = os.path.join(args.data_path, "color_intrinsics.txt")
    
    # 预先加载内参矩阵
    K_depth = np.loadtxt(depth_intrinsics_path).reshape(4, 4)
    K_color = np.loadtxt(color_intrinsics_path).reshape(4, 4)
    K_depth = torch.from_numpy(K_depth[:3, :3]).float().cuda()
    K_color = torch.from_numpy(K_color[:3, :3]).float().cuda()

    depth_files = os.listdir(depth_path)
    scene_name = depth_path.split("/")[5]

    # 使用tqdm显示进度
    with tqdm.tqdm(total=len(depth_files), desc=f"Scene: {scene_name}") as t:
        for depth_file in depth_files:
            # 读取深度图
            depth_image = cv2.imread(
                os.path.join(depth_path, depth_file), 
                cv2.IMREAD_UNCHANGED
            )
            
            # CUDA重投影
            reprojected_depth = reproject_depth_to_color(
                depth_image, 
                K_depth, 
                K_color
            )
            
            # 保存为npy格式
            output_path = os.path.join(
                args.data_path, 
                "reprojected_depths", 
                depth_file.replace(".png", ".npy")
            )
            np.save(output_path, reprojected_depth)
            t.update(1)

if __name__ == "__main__":
    main()