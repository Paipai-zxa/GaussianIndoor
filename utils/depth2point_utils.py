import open3d as o3d
import numpy as np
from PIL import Image
import os
from argparse import ArgumentParser
import time

def create_point_cloud(color_path, depth_path, intrinsic, extrinsic, downsample_factor=8):

    color = np.array(Image.open(color_path))
    depth = np.load(depth_path)
    
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

    color_o3d = o3d.geometry.Image(color)
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

    pcd.orient_normals_consistent_tangent_plane(k=30)
    camera_location = np.linalg.inv(extrinsic)[:3, 3]
    pcd.orient_normals_towards_camera_location(camera_location)

    return pcd

def process_scene_point_clouds(scene_dir, output_dir):
    # 读取内参
    intrinsic = np.loadtxt(os.path.join(scene_dir, "color_intrinsics.txt"))

    combined_pcd = o3d.geometry.PointCloud()

    start_time = time.time()
    for frame_id in range(0, 2197, 6):
        color_path = os.path.join(scene_dir, "images", f"{frame_id}.png")
        depth_path = os.path.join(scene_dir, "reprojected_depths", f"{frame_id}.npy")
        extrinsic_path = os.path.join(scene_dir, "poses", f"{frame_id}.txt")
        
        if not all(os.path.exists(p) for p in [color_path, depth_path, extrinsic_path]):
            continue
        
        extrinsic = np.loadtxt(extrinsic_path)
        extrinsic = np.linalg.inv(extrinsic)
        pcd = create_point_cloud(color_path, depth_path, intrinsic, extrinsic)
        # pcd = pcd.voxel_down_sample(voxel_size=0.1)
        # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        combined_pcd += pcd
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.025)
    combined_pcd, _ = combined_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    combined_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        ),
        fast_normal_computation=False
    )
    
    o3d.io.write_point_cloud(
        os.path.join(output_dir, f"points3D.ply"), 
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
    args = parser.parse_args()
    process_scene_point_clouds(args.scene_dir, args.scene_dir)
