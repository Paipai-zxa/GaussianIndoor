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
from scene.gaussian_model_vanilla import GaussianModel as VanillaGaussianModel
from scene.gaussian_model_scaffold import GaussianModel as ScaffoldGaussianModel
from utils.sh_utils import eval_sh
import os
import cv2 as cv
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix
from einops import repeat
from utils.graphics_utils import quaternion_multiply
from utils.loss_utils import depth_to_normal
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def generate_neural_gaussians(viewpoint_camera, pc : ScaffoldGaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_xyz.shape[0], dtype=torch.bool, device = pc.get_xyz.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_xyz[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def voxelize_sample(data=None, voxel_size=0.01):
    np.random.shuffle(data)
    data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
    return data


def get_rotation_matrix(rot):
    return quaternion_to_matrix(rot)
        
def get_smallest_axis(rot, scaling, return_idx=False):
    rotation_matrices = get_rotation_matrix(rot)
    smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
    if return_idx:
        return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
    return smallest_axis.squeeze(dim=2)    
    
def get_normal(view_cam, xyz, rot, scaling):
    # torch.cuda.synchronize()
    normal_global = get_smallest_axis(rot, scaling)
    gaussian_to_cam_global = view_cam.camera_center - xyz
    neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
    normal_global[neg_mask] = -normal_global[neg_mask]
    return normal_global
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def scaffold_render(viewpoint_camera, pc : ScaffoldGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, iteration = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    visible_mask = get_filter(viewpoint_camera, pc, pipe, bg_color)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    
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
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)

    shs = None

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    means3D = xyz
    global_normal = get_normal(viewpoint_camera, xyz, rot, scaling)
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
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)
    
    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
  
    if is_training:
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe,
                        "normal": rendered_normal,
                        "depth": plane_depth,
                        "rendered_distance": rendered_distance,
                        "alpha": rendered_alpha,
                        "neural_opacity": neural_opacity,
                        "selection_mask": mask,
                        "scaling": scaling,
                        "visible_mask": visible_mask,
                        }   
    else:
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "normal": rendered_normal,
                        "depth": plane_depth,
                        "rendered_distance": rendered_distance,
                        "alpha": rendered_alpha,
                        } 
    depth_normal = (depth_to_normal(viewpoint_camera, plane_depth).permute(2,0,1) * (rendered_alpha)).detach()
    return_dict.update({"depth_normal": depth_normal})
    return return_dict

def get_filter(viewpoint_camera, pc : ScaffoldGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    means2D = screenspace_points
    means2D_abs = screenspace_points_abs

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
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_geo=True,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    cov3D_precomp = None
    shs = None
    anchor = pc.get_xyz
    color = torch.ones_like(anchor, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rot = pc.get_rotation
    means3D = anchor
    global_normal = torch.ones_like(anchor, dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0
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
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)
    return radii > 0.0


def vanilla_render(viewpoint_camera, pc : VanillaGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, iteration = None):
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

    if pc.enable_training_exposure:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        if exposure is not None:
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ] 

    apply_geo_mlp = False
    if iteration != None:
        if iteration > pc.opt_geo_mlp_iteration:
            apply_geo_mlp = True
    else:
        # for test
        apply_geo_mlp = pc.enable_geo_mlp

    if pc.enable_geo_mlp and apply_geo_mlp:

        visible_mask = radii > 0
        
        feat = pc.get_features[visible_mask]
        feat = feat.reshape(feat.shape[0], -1)
        anchor = pc.get_xyz[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if pc.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            
            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            # TODO qu houmian 32
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

        if pc.detach_geo_mlp_input_feat:
            cat_local_view = cat_local_view.detach()
            cat_local_view_wodist = cat_local_view_wodist.detach()

        if pc.add_cov_dist:
            scale_rot = pc.get_geo_mlp_cov(cat_local_view)
        else:
            scale_rot = pc.get_geo_mlp_cov(cat_local_view_wodist)


        if pc.scales_geo_after_activation:
            scales_ = pc._scaling[visible_mask]
            if pc.detach_scales_ori:
                scales_ = scales_.detach()
            scales_geo = pc.scaling_activation(scales_ + scale_rot[:,:3])
        else:
            scales_ = pc.get_scaling[visible_mask]
            if pc.detach_scales_ori:
                scales_ = scales_.detach()
            scales_geo = scales_ * torch.sigmoid(scale_rot[:,:3])
        
        if pc.rotations_geo_after_activation:
            rotations_ = pc._rotation[visible_mask]
            if pc.detach_rotations_ori:
                rotations_ = rotations_.detach()
            rotations_geo = pc.rotation_activation(rotations_ + scale_rot[:,3:7])
        else:
            rotations_ = pc.get_rotation[visible_mask]
            if pc.detach_rotations_ori:
                rotations_ = rotations_.detach()
            rotations_geo = pc.rotation_activation(scale_rot[:,3:7])
            rotations_geo = quaternion_multiply(rotations_geo, rotations_)

        if pc.detach_geo_rasterizer_input_means3D:
            means3D_geo = means3D.detach()
        else:
            means3D_geo = means3D

        if pc.detach_geo_rasterizer_input_means2D:
            means2D_geo = means2D.detach()
        else:
            means2D_geo = means2D
        
        if pc.detach_geo_rasterizer_input_means2D_abs:
            means2D_abs_geo = means2D_abs.detach()
        else:
            means2D_abs_geo = means2D_abs

        if pc.detach_geo_rasterizer_input_opacity:
            opacity_geo = opacity.detach()
        else:
            opacity_geo = opacity
        
        if pc.detach_geo_rasterizer_input_shs:
            shs_geo = shs.detach()
        else:
            shs_geo = shs
        
        if pc.detach_geo_rasterizer_input_input_all_map:
            input_all_map_geo = input_all_map.detach()
        else:
            input_all_map_geo = input_all_map

        rendered_image_geo, _, _, out_all_map, plane_depth = rasterizer(
            means3D = means3D_geo[visible_mask],
            means2D = means2D_geo[visible_mask],
            means2D_abs = means2D_abs_geo[visible_mask],
            shs = shs_geo[visible_mask],
            colors_precomp = None,
            opacities = opacity_geo[visible_mask],
            scales = scales_geo,
            rotations = rotations_geo,
            all_map = input_all_map_geo[visible_mask],
            cov3D_precomp = None)

        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4, ]
        rendered_distance = out_all_map[4:5, ]
        # import torchvision
        # torchvision.utils.save_image(rendered_image_geo, "rendered_image_geo.png")
        # torchvision.utils.save_image(viewpoint_camera.depth_mask.float(), "depth_mask.png")
        # breakpoint()


    return_dict =  {"render": rendered_image,
                    "render_geo": rendered_image_geo if pc.enable_geo_mlp and apply_geo_mlp else None,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "normal": rendered_normal,
                    "depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "alpha": rendered_alpha,
                    "scales": scales_geo if pc.enable_geo_mlp and apply_geo_mlp else scales,
                    }   

    depth_normal = (depth_to_normal(viewpoint_camera, plane_depth).permute(2,0,1) * (rendered_alpha)).detach()
    return_dict.update({"depth_normal": depth_normal})

    return return_dict