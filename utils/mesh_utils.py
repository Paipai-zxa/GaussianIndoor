#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import colorsys

def id2rgb(id):
    # Convert ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.uint8)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)
    return rgb

def save_tsdf_as_pointcloud(tsdf, samples):
    """
    将 TSDF 保存为点云文件
    
    Args:
        tsdf: TSDF 值数组
        samples: 采样点坐标
        filename: 输出文件名
    """
    # 将 TSDF 值转换为颜色
    # 使用颜色映射：负值（内部）为蓝色，正值（外部）为红色，0（表面）为绿色
    colors = np.zeros((len(tsdf), 3))
    
    # 归一化 TSDF 值到 [0,1] 用于颜色映射
    tsdf_normalized = (tsdf - tsdf.min()) / (tsdf.max() - tsdf.min())
    
    # 创建颜色映射
    colors[tsdf < 0, 2] = 1 - tsdf_normalized[tsdf < 0]  # 蓝色通道
    colors[tsdf > 0, 0] = tsdf_normalized[tsdf > 0]      # 红色通道
    # 绿色
    colors[tsdf == 0, 1] = 1
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云
    return pcd
    # o3d.io.write_point_cloud(filename, pcd)
    
    # 也可以保存为其他格式
    # o3d.io.write_point_cloud("tsdf.xyz", pcd)  # XYZ 格式
    # o3d.io.write_point_cloud("tsdf.pcd", pcd)  # PCD 格式

def save_raw_sdf_as_pointcloud(sdf, samples, densify_mask=None, prune_mask=None, filename="raw_sdf.ply"):
    """
    改进颜色映射，增加 densify 和 prune mask 的可视化
    """
    # 将 tensor 转换为 numpy
    if torch.is_tensor(sdf):
        sdf = sdf.cpu().numpy()
    if torch.is_tensor(samples):
        samples = samples.cpu().numpy()
    if torch.is_tensor(densify_mask):
        densify_mask = densify_mask.cpu().numpy()
    if torch.is_tensor(prune_mask):
        prune_mask = prune_mask.cpu().numpy()
    
    colors = np.ones((len(sdf), 3)) * 0.5  # 初始化为灰色
    
    # 归一化 SDF，但保持符号
    sdf_max = np.abs(sdf).max()
    sdf_normalized = sdf / sdf_max  # 范围在 [-1, 1]
    
    # 基础 SDF 颜色映射
    # 内部点（负值）：蓝色
    mask_inside = sdf < 0
    colors[mask_inside] = np.array([0, 0, 1])  # 基础蓝色
    brightness = 0.5 + 0.5 * np.abs(sdf_normalized[mask_inside])
    colors[mask_inside] *= brightness[:, None]
    
    # 外部点（正值）：红色
    mask_outside = sdf > 0
    colors[mask_outside] = np.array([1, 0, 0])  # 基础红色
    brightness = 0.5 + 0.5 * sdf_normalized[mask_outside]
    colors[mask_outside] *= brightness[:, None]
    
    # 表面点：绿色
    mask_surface = np.abs(sdf) < sdf_max * 0.01
    colors[mask_surface] = np.array([0, 1, 0])
    
    # 添加 densify 和 prune mask 的可视化
    if densify_mask is not None:
        # 需要加密的点：黄色
        colors[densify_mask] = np.array([1, 1, 0])  # 黄色
    
    if prune_mask is not None:
        # 需要剪枝的点：紫色
        colors[prune_mask] = np.array([1, 0, 1])  # 紫色
    
    # 打印统计信息
    print(f"SDF 统计:")
    print(f"最大值: {sdf.max():.6f}")
    print(f"最小值: {sdf.min():.6f}")
    print(f"内部点数量: {np.sum(mask_inside)}")
    print(f"外部点数量: {np.sum(mask_outside)}")
    print(f"表面点数量: {np.sum(mask_surface)}")
    if densify_mask is not None:
        print(f"需要加密的点数量: {np.sum(densify_mask)}")
    if prune_mask is not None:
        print(f"需要剪枝的点数量: {np.sum(prune_mask)}")
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(samples)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存点云
    o3d.io.write_point_cloud(filename, pcd)


def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        intrinsic=o3d.camera.PinholeCameraIntrinsic(width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    cx = viewpoint_cam.image_width/2,
                    cy = viewpoint_cam.image_height/2,
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.)))

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None, extract_semantic=False):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()
        self.extract_semantic = extract_semantic


    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.viewpoint_stack = []
        self.instance_maps = []
        self.semantic_maps = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack, id2label=[]):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            # turn camera data to cuda
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to(device="cuda")
            viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.to(device="cuda")
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.to(device="cuda")
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.to(device="cuda")

            render_pkg = self.render(viewpoint_cam, self.gaussians)

            if self.extract_semantic:
                # instance_map = viewpoint_cam.instance_train.cpu().numpy()
                instance_map = viewpoint_cam.instance_gt.cpu().numpy()
                instance_color = np.array([id2rgb(ID) for ID in instance_map.reshape(-1).tolist()])
                instance_color = instance_color.reshape(instance_map.shape[1], instance_map.shape[2], 3)

                # semantic_map = viewpoint_cam.semantic_train.cpu().numpy()
                semantic_map = viewpoint_cam.semantic_gt.cpu().numpy()
                semantic_color = np.array([id2label[semID].color for semID in semantic_map.reshape(-1).tolist()])
                semantic_color = semantic_color.reshape(semantic_map.shape[1], semantic_map.shape[2], 3)

                self.instance_maps.append(instance_color)
                self.semantic_maps.append(semantic_color)
            
            rgb = render_pkg['render']
            depth = render_pkg['depth']
            self.rgbmaps.append(rgb.to(device="cuda"))
            self.depthmaps.append(depth.to(device="cuda"))
        
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        

    def estimate_bounding_sphere(self, radius=None):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        # self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        if radius is None:
            self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        else:
            self.radius = radius
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_semantic_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """

        volume_semantic = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        volume_instance = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            depth = self.depthmaps[i]
            semantic_color = self.semantic_maps[i]
            instance_color = self.instance_maps[i]

            # make open3d rgbd
            rgbd_semantic = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(semantic_color, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            rgbd_instance = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(instance_color, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume_semantic.integrate(rgbd_semantic, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
            volume_instance.integrate(rgbd_instance, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh_semantic = volume_semantic.extract_triangle_mesh()
        mesh_instance = volume_instance.extract_triangle_mesh()
        return mesh_semantic, mesh_instance

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024, radius=None, only_visualize=False):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_raw_sdf(samples, inv_contraction):
            """
            只计算原始的 SDF 值，不进行 TSDF 融合
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)

            # 初始化 SDF 数组
            sdf_values = torch.zeros_like(samples[:,0])

            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="Raw SDF computation"):
                sdf, _, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )
                
                # 只保留有效的 SDF 值
                sdf = sdf.flatten()
                sdf_values[mask_proj] = sdf[mask_proj]
            
            return sdf_values

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                samples = inv_contraction(samples)
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs
        self.estimate_bounding_sphere(radius)
        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        
        gaussians_tsdf = compute_unbounded_tsdf(self.gaussians.get_xyz, None, voxel_size)
        pcd = save_tsdf_as_pointcloud(gaussians_tsdf.cpu().numpy(), self.gaussians.get_xyz.cpu().numpy())
        if only_visualize:
            return pcd
        # gaussians_sdf = compute_raw_sdf(self.gaussians.get_xyz, None)
        # save_raw_sdf_as_pointcloud(gaussians_sdf.cpu().numpy(), self.gaussians.get_xyz.cpu().numpy(), filename=f"sdf_test.ply")
        # breakpoint()

        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh


def extract_gaussians_sdf(rgbmaps, depthmaps, viewpoint_stack, samples):
    """
    提取高斯分布的 SDF 值
    """

    def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
        """
            compute per frame sdf
        """
        new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
        z = new_points[..., -1:]
        pix_coords = (new_points[..., :2] / new_points[..., -1:])
        mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
        sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
        sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
        sdf = (sampled_depth-z)
        return sdf, sampled_rgb, mask_proj

    def compute_raw_sdf(samples, inv_contraction):
        """
        只计算原始的 SDF 值，不进行 TSDF 融合
        """
        if inv_contraction is not None:
            samples = inv_contraction(samples)

        # 初始化 SDF 数组
        sdf_values = torch.zeros_like(samples[:,0])

        for i, viewpoint_cam in enumerate(viewpoint_stack):
            sdf, _, mask_proj = compute_sdf_perframe(i, samples,
                depthmap = depthmaps[i],
                rgbmap = rgbmaps[i],
                viewpoint_cam=viewpoint_stack[i],
            )
            
            # 只保留有效的 SDF 值
            sdf = sdf.flatten()
            sdf_values[mask_proj] = sdf[mask_proj]
        
        return sdf_values

    gaussians_sdf = compute_raw_sdf(samples, None)

    return gaussians_sdf


@torch.no_grad()
def reconstruction(viewpoint_stack, gaussians, render, pipe, bg):
    """
    reconstruct radiance field given cameras
    """
    rgbmaps = []
    depthmaps = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        # turn camera data to cuda
        viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to(device="cuda")
        viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.to(device="cuda")
        viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.to(device="cuda")
        viewpoint_cam.camera_center = viewpoint_cam.camera_center.to(device="cuda")

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        
        rgb = render_pkg['render']
        depth = render_pkg['depth']
        rgbmaps.append(rgb.to(device="cuda"))
        depthmaps.append(depth.to(device="cuda"))
    
    rgbmaps = torch.stack(rgbmaps, dim=0)
    depthmaps = torch.stack(depthmaps, dim=0)

    return rgbmaps, depthmaps

@torch.no_grad()
def extract_sdf_guidance(viewpoint_stack, gaussians, render, pipe, bg):
    rgbmaps, depthmaps = reconstruction(viewpoint_stack, gaussians, render, pipe, bg)
    gaussians_sdf = extract_gaussians_sdf(rgbmaps, depthmaps, viewpoint_stack, gaussians.get_xyz)

    def gaussian_fun(s, sigma):
        return torch.exp((-s**2)/(2*torch.square(sigma)))

    sdf_guidance =  gaussian_fun(gaussians_sdf, gaussians.get_opacity.squeeze())

    return gaussians_sdf, sdf_guidance
