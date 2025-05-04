import open3d as o3d
import numpy as np
from PIL import Image
import os
import sys
from argparse import ArgumentParser
import time
import cv2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pre_process.label import get_labels

def orient_normals_safely(pcd, k=30, max_attempts=3, initial_noise=0.001):
    """安全地进行法向量一致化"""
    for attempt in range(max_attempts):
        try:
            # 尝试直接计算
            if attempt == 0:
                pcd.orient_normals_consistent_tangent_plane(k=k)
                return pcd
            
            # 如果失败，增加噪声并重试
            noise_level = initial_noise * (2 ** attempt)  # 逐渐增加噪声级别
            points = np.asarray(pcd.points)
            noise = np.random.normal(0, noise_level, points.shape)
            
            # 只对z坐标添加噪声，保持xy平面的结构
            noise[:, :2] = 0  
            points += noise
            
            # 更新点云并重试
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.orient_normals_consistent_tangent_plane(k=k)
            return pcd
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Warning: Failed to orient normals after {max_attempts} attempts")
                # 使用备选方案：朝向质心的法向量
                points = np.asarray(pcd.points)
                centroid = np.mean(points, axis=0)
                normals = points - centroid
                normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
                pcd.normals = o3d.utility.Vector3dVector(normals)
                return pcd
            continue

def create_point_cloud(color_path, depth_path, intrinsic, extrinsic, downsample_factor=8, scene=None):

    color = np.array(Image.open(color_path))
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    else:
        depth = np.array(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED))
    
    h, w = depth.shape[:2]
    new_h, new_w = h//downsample_factor, w//downsample_factor
    
    color_small = Image.fromarray(color).resize((new_w, new_h), Image.BILINEAR)
    color = np.array(color_small)
    
    depth_small = Image.fromarray(depth).resize((new_w, new_h), Image.NEAREST)
    depth = np.array(depth_small)

    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)

    # 相应地调整内参矩阵
    intrinsic_scaled = intrinsic.copy()
    intrinsic_scaled[0, 0] = intrinsic[0, 0] / downsample_factor  # fx
    intrinsic_scaled[1, 1] = intrinsic[1, 1] / downsample_factor  # fy
    intrinsic_scaled[0, 2] = intrinsic[0, 2] / downsample_factor  # cx
    intrinsic_scaled[1, 2] = intrinsic[1, 2] / downsample_factor  # cy

    # 将语义ID转换为对应的颜色
    id2label = get_labels(scene)
    # 使用向量化操作替代循环
    v_colors = np.vstack([id2label[semID].color if semID in id2label else (100, 100, 100) 
                         for semID in color.reshape(-1).tolist()])
    semantic_color = v_colors.reshape(color.shape[0], color.shape[1], 3).astype(np.uint8)

    color_o3d = o3d.geometry.Image(semantic_color)
    depth_o3d = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=new_w,
        height=new_h,
        fx=intrinsic_scaled[0, 0],
        fy=intrinsic_scaled[1, 1],
        cx=intrinsic_scaled[0, 2],
        cy=intrinsic_scaled[1, 2]
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic_o3d,
        extrinsic
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        ),
        fast_normal_computation=False
    )
    try:
        pcd = orient_normals_safely(pcd, k=30)
    except Exception as e:
        print(f"Normal orientation failed: {e}")

    return pcd

def process_scene_point_clouds(scene_dir, output_dir, is_depth_anything=False):
    # 读取内参
    intrinsic = np.loadtxt(os.path.join(scene_dir, "color_intrinsics.txt"))

    combined_pcd = o3d.geometry.PointCloud()

    start_time = time.time()
    colors_path = os.path.join(scene_dir, "semantic_image")
    colors_list = os.listdir(colors_path)
    colors_list = sorted(colors_list, key=lambda x: int(x.replace("DSC", "").split(".")[0]))
    for color in colors_list:
        color_path = os.path.join(colors_path, color)
        name = color.split(".")[0]
        if is_depth_anything:
            depth_path = os.path.join(scene_dir, "video_depth_anything", f"{name}.png")
        else:
            if os.path.exists(os.path.join(scene_dir, "reprojected_depths", f"{name}.npy")):
                depth_path = os.path.join(scene_dir, "reprojected_depths", f"{name}.npy")
            else:
                depth_path = os.path.join(scene_dir, "depths", f"{name}.png")
        extrinsic_path = os.path.join(scene_dir, "poses", f"{name}.txt")
        
        if not all(os.path.exists(p) for p in [color_path, depth_path, extrinsic_path]):
            continue
        
        extrinsic = np.loadtxt(extrinsic_path)
        extrinsic = np.linalg.inv(extrinsic)
        pcd = create_point_cloud(color_path, depth_path, intrinsic, extrinsic, scene=os.path.basename(scene_dir))
        # pcd = pcd.voxel_down_sample(voxel_size=0.1)
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_pcd += pcd
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        ),
        fast_normal_computation=False
    )
    
    out_pcd_path = os.path.join(output_dir, f"points3D_semantic.ply")
    if is_depth_anything:
        out_pcd_path = os.path.join(output_dir, f"points3D_depth_anything.ply")

    o3d.io.write_point_cloud(
        out_pcd_path, 
        combined_pcd,
        write_ascii=False, 
        compressed=False
    )

    end_time = time.time()
    print(f"Create point cloud time taken: {end_time - start_time} seconds")

    print("Point cloud created successfully\nNumber of points: ", len(combined_pcd.points))
    return combined_pcd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene_dir", type=str)
    parser.add_argument("--is_depth_anything", action="store_true")
    args = parser.parse_args()
    process_scene_point_clouds(args.scene_dir, args.scene_dir, args.is_depth_anything)
