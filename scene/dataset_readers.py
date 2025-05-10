#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
import json
from typing import NamedTuple
from pre_process.label import get_labels
# from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
#     read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud
from utils.depth2point_utils import process_scene_point_clouds

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    # depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    instance_train_path: str
    semantic_train_path: str
    instance_gt_path: str
    semantic_gt_path: str

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    semantic_point_cloud: BasicPointCloud
    instance_point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
    id2label: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchInstancePly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchSemanticPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def load_scene(path, images, eval, llffhold=10, use_video_depth_anything=False, is_train_on_all_images=False):
    poses_path = os.path.join(path, "poses")
    images_path = os.path.join(path, images)
    if use_video_depth_anything:
        depths_path = os.path.join(path, "video_depth_anything")
    else:
        depths_path = os.path.join(path, "reprojected_depths")
        if not os.path.exists(depths_path):
            depths_path = os.path.join(path, "depths")
    point_cloud_path = os.path.join(path, "points3D.ply")
    intrinsics_path = os.path.join(path, "color_intrinsics.txt")

    instances_train_path = os.path.join(path, "instance_image")
    semantics_train_path = os.path.join(path, "semantic_image")
    instances_gt_path = os.path.join(path, "instance_remap")
    if not os.path.exists(instances_gt_path):
        instances_gt_path = os.path.join(path, "instance")
    semantics_gt_path = os.path.join(path, "semantic")

    intrinsics = np.loadtxt(intrinsics_path).reshape(4, 4)
    image_names = os.listdir(images_path)
    image_names = sorted(image_names, key=lambda x: int(x.replace("DSC", "").split('.')[0]))

    # 划分测试集
    if os.path.exists(os.path.join(path, "train_test_lists.json")):
        with open(os.path.join(path, "train_test_lists.json"), "r") as f:
            train_test_lists = json.load(f)
        test_cam_names_list = train_test_lists["test"]
    else:
        test_cam_names_list = []
        if eval:
            print(f"Test set ratio: {1/llffhold}")
        test_cam_names_list = image_names[::llffhold]

    focal_length_x = intrinsics[0, 0]
    focal_length_y = intrinsics[1, 1]
    height, width = np.array(Image.open(os.path.join(images_path, image_names[0]))).shape[:2]
    FovY = focal2fov(focal_length_y, height)
    FovX = focal2fov(focal_length_x, width)           

    cam_infos = []
    for image_name in image_names:
        uid = 0
        pose_path = os.path.join(poses_path, image_name.split('.')[0] + '.txt')
        pose = np.loadtxt(pose_path).reshape(4, 4)
        if pose[0, 0] == -np.inf:
            continue
        pose = np.linalg.inv(pose)
        R = pose[:3, :3]
        R = np.transpose(R)
        T = pose[:3, 3]

        image_path = os.path.join(images_path, image_name)
        depth_path = os.path.join(depths_path, f"{image_name.split('.')[0]}.npy")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(depths_path, f"{image_name.split('.')[0]}.png")
            if not os.path.exists(depth_path):
                depth_path = ''
        
        instance_train_path = os.path.join(instances_train_path, f"{image_name.split('.')[0]}.png")
        if not os.path.exists(instance_train_path):
            instance_train_path = ''
        semantic_train_path = os.path.join(semantics_train_path, f"{image_name.split('.')[0]}.png")
        if not os.path.exists(semantic_train_path):
            semantic_train_path = ''
        instance_gt_path = os.path.join(instances_gt_path, f"{image_name.split('.')[0]}.png")
        if not os.path.exists(instance_gt_path):
            instance_gt_path = ''
        semantic_gt_path = os.path.join(semantics_gt_path, f"{image_name.split('.')[0]}.png")
        if not os.path.exists(semantic_gt_path):
            semantic_gt_path = ''

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              instance_train_path=instance_train_path, semantic_train_path=semantic_train_path,
                              instance_gt_path=instance_gt_path, semantic_gt_path=semantic_gt_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    if is_train_on_all_images:
        train_cam_infos = cam_infos
    else:
        train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if not os.path.exists(point_cloud_path):
        print(f"Generating initial point cloud.")
        process_scene_point_clouds(path, path)
    try:        
        pcd = fetchPly(point_cloud_path)
    except:
        pcd = None
    try:
        semantic_pcd = fetchSemanticPly(point_cloud_path.replace(".ply", "_semantic.ply"))
    except:
        semantic_pcd = None
    try:
        instance_pcd = fetchInstancePly(point_cloud_path.replace(".ply", "_instance.ply"))
    except:
        instance_pcd = None
    id2label = get_labels(os.path.basename(path))

    scene_info = SceneInfo(point_cloud=pcd,
                           semantic_point_cloud=semantic_pcd,
                           instance_point_cloud=instance_pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=point_cloud_path,
                           is_nerf_synthetic=False,
                           id2label=id2label)

    return scene_info