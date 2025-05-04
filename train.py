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
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, cross_view_constraint
from gaussian_renderer import get_filter, vanilla_render, scaffold_render
import sys
from scene import Scene, ScaffoldGaussianModel, VanillaGaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import get_img_grad_weight
from torchmetrics import PearsonCorrCoef
from utils.loss_utils import multiview_loss

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torchvision
import matplotlib.pyplot as plt
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(args, dataset, opt, pipe):

    testing_iterations = args.test_iterations
    saving_iterations = args.save_iterations
    debug_from = args.debug_from

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    prepare_output_and_logger(dataset)
    if dataset.enable_scaffold:
        gaussians = ScaffoldGaussianModel(
                            dataset.sh_degree, 
                            dataset.feat_dim, 
                            dataset.n_offsets, 
                            dataset.voxel_size, 
                            dataset.update_depth, 
                            dataset.update_init_factor,
                            dataset.update_hierachy_factor, 
                            dataset.use_feat_bank, 
                            dataset.appearance_dim,
                            dataset.ratio, 
                            dataset.add_opacity_dist, 
                            dataset.add_cov_dist, 
                            dataset.add_color_dist)    
    else:
        gaussians = VanillaGaussianModel(
                            dataset.sh_degree,
                            opt.optimizer_type,
                            dataset.enable_training_exposure,
                            dataset.enable_geo_mlp,
                            dataset.feat_dim,
                            dataset.use_feat_bank,
                            dataset.add_cov_dist,
                            dataset.detach_geo_mlp_input_feat,
                            dataset.detach_scales_ori,
                            dataset.detach_rotations_ori,
                            dataset.detach_geo_rasterizer_input_means3D,
                            dataset.detach_geo_rasterizer_input_means2D,
                            dataset.detach_geo_rasterizer_input_means2D_abs,
                            dataset.detach_geo_rasterizer_input_opacity,
                            dataset.detach_geo_rasterizer_input_shs,
                            dataset.detach_geo_rasterizer_input_input_all_map,
                            dataset.scales_geo_after_activation,
                            dataset.rotations_geo_after_activation,
                            dataset.opt_geo_mlp_iteration)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", position=0, leave=True)
    first_iter += 1

    pearson_loss = PearsonCorrCoef().cuda()

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if not dataset.enable_scaffold:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_func = scaffold_render if dataset.enable_scaffold else vanilla_render
        render_pkg = render_func(viewpoint_cam, gaussians, pipe, bg, iteration=iteration)
        if dataset.enable_scaffold:
            image, viewspace_point_tensor, visibility_filter, radii, offset_selection_mask, scaling, opacity, voxel_visible_mask = \
                render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], \
                render_pkg["selection_mask"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["visible_mask"]
        else:
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        # torchvision.utils.save_image(image, "./output/test/image.png")
        # valid_mask = torch.isfinite(depth_map) & (depth_map > 0)
        # valid_depth_map = depth_map[valid_mask]
        # min_depth = valid_depth_map.min()
        # max_depth = valid_depth_map.max()
        # normalized_depth_map = torch.zeros_like(depth_map)
        # normalized_depth_map[valid_mask] = (depth_map[valid_mask] - min_depth) / (max_depth - min_depth)
        # cmap = plt.get_cmap("viridis")
        # depth_np = normalized_depth_map.squeeze(0).detach().cpu().numpy()
        # colored_depth = cmap(depth_np)
        # colored_depth = torch.from_numpy(colored_depth[:, :, 0:3]).permute(2, 0, 1).float()
        # torchvision.utils.save_image(colored_depth, "./output/test/depth_map.png")
        # torchvision.utils.save_image(normal_map, "./output/test/normal_map.png")
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        render_geo = render_pkg["render_geo"] if "render_geo" in render_pkg.keys() else None
        if opt.use_render_geo and render_geo is not None:
            render_geo *= alpha_mask
            Ll1_geo = l1_loss(render_geo, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value_geo = fused_ssim(render_geo.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value_geo = ssim(render_geo, gt_image)
            render_geo_loss = (1.0 - opt.lambda_dssim) * Ll1_geo + opt.lambda_dssim * (1.0 - ssim_value_geo)
            loss += opt.lambda_geo * render_geo_loss

        # scale flatten loss
        scale_flatten_loss = torch.tensor(0.0, device="cuda")
        if iteration > opt.scale_flatten_iteration and opt.use_scale_flatten:
            scales = render_pkg["scales"]
            sorted_scale, _ = torch.sort(scales, dim=-1)
            scale_flatten_loss = sorted_scale[...,0].mean()
            loss += opt.scale_flatten_weight * scale_flatten_loss

        # Plane normalconstraint
        plane_loss = torch.tensor(0.0, device="cuda")
        if iteration > opt.plane_constraint_iteration and opt.use_plane_constraint:
            depth_normal_map, normal_map = render_pkg["depth_normal"], render_pkg["normal"]
            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            plane_loss = ((depth_normal_map - normal_map).abs().sum(0) * image_weight).mean()
            loss += opt.plane_constraint_weight * plane_loss

        #Cross view constraint
        cross_view_loss = torch.tensor(0.0, device="cuda")
        if iteration > opt.cross_view_constraint_iteration and opt.use_cross_view_constraint:
            neighbor_cams = scene.get_neighbor_cameras(viewpoint_cam, opt.num_neighbors_views)
            for neighbor_cam in neighbor_cams:
                cross_view_loss += multiview_loss(render_pkg, viewpoint_cam, neighbor_cam, render_func, gaussians, pipe, bg, iteration, opt.multi_view_pixel_noise_th)
            loss += opt.cross_view_constraint_weight * cross_view_loss
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Depth regularization
        depth_loss = torch.tensor(0.0, device="cuda")
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable and opt.use_depth_regularization:
            invDepth = render_pkg["depth"]  / float(2**16)
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            if dataset.use_video_depth_anything:
                pred_depth = invDepth * float(2**16)
                mono_depth = mono_invdepth * float(2**16)
                scale_mono_depth = (mono_depth[depth_mask] - mono_depth[depth_mask].min()) / (mono_depth[depth_mask].max() - mono_depth[depth_mask].min())
                scale_pred_depth = (pred_depth[depth_mask] - pred_depth[depth_mask].min()) / (pred_depth[depth_mask].max() - pred_depth[depth_mask].min())

                depth_loss = (1 - pearson_loss(scale_mono_depth, scale_pred_depth))
                loss += depth_l1_weight(iteration) * depth_loss
            else:
                depth_loss = torch.abs((invDepth  - mono_invdepth)).mean()
                loss += depth_l1_weight(iteration) * depth_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                loss_for_log = loss.item()
                plane_loss_for_log = plane_loss.item() * opt.plane_constraint_weight
                cross_view_loss_for_log = cross_view_loss.item() * opt.cross_view_constraint_weight
                depth_loss_for_log = depth_loss.item() * depth_l1_weight(iteration)
                scale_flatten_loss_for_log = scale_flatten_loss.item() * opt.scale_flatten_weight
                anchor_num = gaussians.get_xyz.shape[0]
                progress_bar.set_postfix({"Loss": f"{loss_for_log:.{5}f}", "D Loss": f"{depth_loss_for_log:.{4}f}", \
                                          "P Loss": f"{plane_loss_for_log:.{4}f}", "Cro Loss": f"{cross_view_loss_for_log:.{4}f}", "Sca Loss": f"{scale_flatten_loss_for_log:.{4}f}", "Anchor Num": f"{anchor_num}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize()
            render_func = scaffold_render if dataset.enable_scaffold else vanilla_render
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_func, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if dataset.enable_scaffold:
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                    
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, \
                                                min_opacity=opt.min_opacity)    
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()
            else:
                if opt.enable_sdf_guidance:
                    if iteration < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if opt.densify_from_iter < iteration < opt.sdf_guidance_start_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)

                        if opt.sdf_guidance_start_iter <= iteration < opt.sdf_guidance_end_iter and iteration % opt.sdf_guidance_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.sdf_densify_and_prune(max_grad=opt.densify_grad_threshold, min_opacity=0.005, extent=scene.cameras_extent, max_screen_size=size_threshold, radii=radii, \
                                                            viewpoint_stack=scene.getTrainCameras(), render=render_func, pipe=pipe, bg=bg, \
                                                            densification_threshold=opt.densification_threshold, \
                                                            pruning_threshold=opt.pruning_threshold, \
                                                            is_recal_split=opt.is_recal_split, \
                                                            is_recal_prune=opt.is_recal_prune, \
                                                            grad_sdf_omega=opt.grad_sdf_omega, \
                                                            is_apply_grad_sdf_omega=opt.is_apply_grad_sdf_omega, \
                                                            enable_sdf_guidance=opt.enable_sdf_guidance)
                        
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()
                else:
                    if iteration < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if dataset.enable_training_exposure:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, iteration=iteration)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--warmup', action='store_true', default=False)  
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
