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
import scipy.optimize
import monai
import torch.nn.functional as F
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
from utils.loss_utils import create_virtual_gt_with_linear_assignment
import colorsys
import cv2
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
                            dataset.opt_geo_mlp_iteration,
                            dataset.enable_semantic,
                            dataset.opt_semantic_mlp_iteration,
                            dataset.semantic_feature_dim,
                            dataset.instance_feature_dim,
                            dataset.semantic_mlp_dim,
                            dataset.instance_query_num,
                            dataset.instance_query_feat_dim,
                            dataset.load_semantic_from_pcd,
                            dataset.use_geo_mlp_scales,
                            dataset.use_geo_mlp_rotations,
                            dataset.instance_query_gaussian_sigma,
                            dataset.instance_query_distance_mode,
                            dataset.apply_semantic_guidance)

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

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Train", position=0, leave=True)
    first_iter += 1

    pearson_loss = PearsonCorrCoef().cuda()
    # dice_loss = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
    # dice_loss = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, sigmoid=True)
    dice_loss = monai.losses.DiceLoss(include_background=False, to_onehot_y=True, sigmoid=False)
    # dice_loss = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, sigmoid=True)

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
        semantic_warping_loss = torch.tensor(0.0, device="cuda")
        if iteration > opt.cross_view_constraint_iteration and opt.use_cross_view_constraint:
            neighbor_cams = scene.get_neighbor_cameras(viewpoint_cam, opt.num_neighbors_views)
            for neighbor_cam in neighbor_cams:
                cross_view_loss_, semantic_warping_loss_ = multiview_loss(render_pkg, viewpoint_cam, neighbor_cam, render_func, gaussians, \
                                                                          pipe, bg, iteration, opt.multi_view_pixel_noise_th, dataset.enable_semantic)
                cross_view_loss += cross_view_loss_
                if semantic_warping_loss_ is not None:
                    semantic_warping_loss += semantic_warping_loss_
            loss += opt.cross_view_constraint_weight * cross_view_loss
            loss += opt.semantic_warping_weight * semantic_warping_loss
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
        
        semantic_map = render_pkg["semantic_map"]
        if dataset.enable_semantic:
            if opt.use_semantic_train:
                gt_semantic_map = viewpoint_cam.semantic_train.cuda()
            else:
                gt_semantic_map = viewpoint_cam.semantic_gt.cuda()

        semantic_ce_loss = torch.tensor(0.0, device="cuda")
        if dataset.enable_semantic and semantic_map is not None:
            semantic_ce_loss = F.cross_entropy(semantic_map.unsqueeze(0), gt_semantic_map.long())
            loss += opt.semantic_ce_weight * semantic_ce_loss
        
        instance_map = render_pkg["instance_map"]
        if dataset.enable_semantic:
            if opt.use_instance_train:
                gt_instance_map = viewpoint_cam.instance_train.cuda()
            else:
                gt_instance_map = viewpoint_cam.instance_gt.cuda()

        if instance_map is not None:
            virtual_instance_map, virtual_instance_map_ind = create_virtual_gt_with_linear_assignment(gt_instance_map, instance_map)

        # def id2rgb(id):
        #     # Convert ID into a hue value
        #     golden_ratio = 1.6180339887
        #     h = ((id * golden_ratio) % 1)
        #     s = 0.5 + (id % 2) * 0.5
        #     l = 0.5

        #     rgb = np.zeros((3,), dtype=np.float32)
        #     if id==0:
        #         return rgb
        #     r, g, b = colorsys.hls_to_rgb(h, l, s)

        #     rgb[0], rgb[1], rgb[2] = r, g, b
        #     return rgb
        # if iteration % 100 == 0:
        #     lut = np.array([id2rgb(i) for i in range(30)])  # shape: [max_id+1, 3]
        #     instance_ids = torch.argmax(torch.softmax(instance_map, dim=0), dim=0).detach().cpu().numpy().astype(np.int32)
        #     instance_rgb = lut[instance_ids]  # shape: [H, W, 3]
        #     instance_rgb = (instance_rgb * 255).astype(np.uint8)
        #     cv2.imwrite("./instance_map.png", instance_rgb)

        #     virtual_instance_ids = virtual_instance_map.detach().cpu().numpy().astype(np.int32)
        #     virtual_instance_rgb = lut[virtual_instance_ids]  # shape: [H, W, 3]
        #     virtual_instance_rgb = (virtual_instance_rgb * 255).astype(np.uint8)
        #     cv2.imwrite("./virtual_instance_map.png", virtual_instance_rgb[0])

        # torchvision.utils.save_image((gt_instance_map==1).float(), "gt_instance_map.png")

        # breakpoint()
        instance_bce_loss = torch.tensor(0.0, device="cuda")
        # instance_dice_loss = torch.tensor(0.0, device="cuda")
        if dataset.enable_semantic and instance_map is not None:
            for lidx in virtual_instance_map_ind:
                # instance_bce_loss_ = F.binary_cross_entropy_with_logits(instance_map[lidx:lidx+1], (virtual_instance_map==lidx).float())
                # instance_dice_loss_ = dice_loss(instance_map[lidx:lidx+1], (virtual_instance_map==lidx).float())
                instance_bce_loss_ = F.binary_cross_entropy(instance_map[lidx:lidx+1], (virtual_instance_map==lidx).float())
                instance_bce_loss += instance_bce_loss_
                # instance_dice_loss += instance_dice_loss_
            instance_bce_loss /= len(virtual_instance_map_ind)
            # instance_dice_loss /= len(virtual_instance_map_ind)
            loss += opt.instance_bce_weight * instance_bce_loss
            # loss += opt.instance_dice_weight * instance_dice_loss
        
        instance_dice_loss = torch.tensor(0.0, device="cuda")
        if dataset.enable_semantic and instance_map is not None:
            instance_dice_loss = dice_loss(instance_map.unsqueeze(0), virtual_instance_map.unsqueeze(0))
            loss += opt.instance_dice_weight * instance_dice_loss

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
                semantic_ce_loss_for_log = semantic_ce_loss.item() * opt.semantic_ce_weight
                semantic_warping_loss_for_log = semantic_warping_loss.item() * opt.semantic_warping_weight
                instance_bce_loss_for_log = instance_bce_loss.item() * opt.instance_bce_weight
                instance_dice_loss_for_log = instance_dice_loss.item() * opt.instance_dice_weight
                anchor_num = gaussians.get_xyz.shape[0] / 1000000
                progress_bar.set_postfix({"N": f"{anchor_num:.{2}f}M", "L": f"{loss_for_log:.{2}f}", "D": f"{depth_loss_for_log:.{3}f}", \
                                          "P": f"{plane_loss_for_log:.{3}f}", "Cro": f"{cross_view_loss_for_log:.{3}f}", "Sca": f"{scale_flatten_loss_for_log:.{3}f}", \
                                            "Se": f"{semantic_ce_loss_for_log:.{3}f}", "SeW": f"{semantic_warping_loss_for_log:.{3}f}", "InB": f"{instance_bce_loss_for_log:.{3}f}", \
                                            "InD": f"{instance_dice_loss_for_log:.{3}f}"})
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
                # if opt.enable_sdf_guidance:
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
                # else:
                #     if iteration < opt.densify_until_iter:
                #         # Keep track of max radii in image-space for pruning
                #         gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #         gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #         if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #             size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #             gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                        
                #         if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #             gaussians.reset_opacity()

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
