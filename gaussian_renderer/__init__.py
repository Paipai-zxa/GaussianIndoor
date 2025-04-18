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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import os
import cv2 as cv
import numpy as np

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=True,
        debug=pipe.debug,
        # antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    means2D_abs = screenspace_points_abs
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3] + viewpoint_camera.world_view_transform[3, :3]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)
    
    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]

    # return_dict =  {"render": rendered_image,
    #                 "viewspace_points": screenspace_points,
    #                 "viewspace_points_abs": screenspace_points_abs,
    #                 "visibility_filter" : radii > 0,
    #                 "radii": radii,
    #                 "out_observe": out_observe,
    #                 "rendered_normal": rendered_normal,
    #                 "plane_depth": plane_depth,
    #                 "rendered_distance": rendered_distance
    #                 }    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "normal": rendered_normal,
                    "depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "alpha": rendered_alpha
                    }   
    
    return return_dict
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # rendered_image, radii, depth_image = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    # if use_trained_exp:
    #     exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
    #     rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # rendered_image = rendered_image.clamp(0, 1)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if pipe.compute_normals and iteration > plane_constraint_iteration:
    #     normal_map = compute_normal_map(depth_image, viewpoint_camera)
    # else:
    #     normal_map = None    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # out = {
    #     "render": rendered_image,
    #     "viewspace_points": screenspace_points,
    #     "visibility_filter" : (radii > 0).nonzero(),
    #     "radii": radii,
    #     "depth" : depth_image,
    #     "normal_map" : normal_map
    #     }
    
    # return out

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# def compute_normal_map(depth, camera):
#     """从深度图计算法向量图，首先将深度图转换为三维空间中的点云，然后通过计算点云中相邻点的差值来估算法向量。叉积操作确保了法向量是垂直于表面的
#     Args:
#         depth: 深度图
#         camera: 相机
#     Returns:
#         normals: 法向量图
#     """
#     # 获取视场角
#     fovx = camera.FoVx
#     fovy = camera.FoVy
#     aspect_ratio = camera.image_width / camera.image_height
#     tan_fovX_half  = torch.tan(torch.tensor(fovx * 0.5))
#     tan_fovY_half = torch.tan(torch.tensor(fovy * 0.5))

#     # 生成归一化像素坐标
#     y, x = torch.meshgrid(
#         torch.linspace(-tan_fovY_half, tan_fovY_half, camera.image_height),
#         torch.linspace(-tan_fovX_half * aspect_ratio, tan_fovX_half * aspect_ratio, camera.image_width)
#     )
#     x = x.to(depth.device)
#     y = y.to(depth.device)

#     # 计算每个像素的3D位置
#     X = x * depth
#     Y = y * depth
#     Z = depth

#     # 计算相邻点的差值来估计法向量
#     dX = torch.roll(X, -1, dims=1) - X
#     dY = torch.roll(Y, -1, dims=0) - Y
#     dZ = torch.roll(Z, -1, dims=1) - Z
    
#     # 叉乘得到法向量
#     normals = torch.cross(
#         torch.stack([dX, dY, dZ], dim=-1),
#         torch.stack([dX, dY, torch.roll(dZ, -1, dims=0)], dim=-1)
#     )
    
#     # 归一化
#     normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7)
    
#     return normals
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
