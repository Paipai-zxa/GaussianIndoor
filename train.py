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
from random import randint
from utils.loss_utils import l1_loss, ssim, plane_constraint_loss, cross_view_constraint
from gaussian_renderer import render, get_filter
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torchvision
import matplotlib.pyplot as plt
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, plane_constraint_iteration, cross_view_constraint_iteration, ply_path=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians = GaussianModel(dataset.sh_degree, 
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
    scene = Scene(dataset, gaussians, ply_path=ply_path)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_plane_loss_for_log = 0.0
    ema_cross_view_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=200)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        voxel_visible_mask = get_filter(viewpoint_cam, gaussians, pipe, bg)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        image, viewspace_point_tensor, visibility_filter, radii, offset_selection_mask, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["selection_mask"], render_pkg["scaling"], render_pkg["neural_opacity"]
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

        # Plane normalconstraint
        plane_loss = torch.tensor(0.0, device="cuda")
        if iteration > plane_constraint_iteration and pipe.use_plane_constraint:
            depth_map, normal_map = render_pkg["depth"], render_pkg["normal"]
            plane_loss = plane_constraint_loss(depth_map, normal_map)
            loss += opt.plane_constraint_weight * plane_loss

        #Cross view constraint
        cross_view_loss = torch.tensor(0.0, device="cuda")
        if iteration > cross_view_constraint_iteration and pipe.use_cross_view_constraint:
        # if True:
            neighbor_cams = scene.get_neighbor_cameras(viewpoint_cam, opt.num_neighbors_views)
            neighbor_depths = []
            for neighbor_cam in neighbor_cams:
                neighbor_render_pkg = render(neighbor_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
                neighbor_depths.append(neighbor_render_pkg["depth"])
            depth_map = render_pkg["depth"]
            cross_view_loss = cross_view_constraint(viewpoint_cam, depth_map, neighbor_cams, neighbor_depths)
            loss += opt.cross_view_constraint_weight * cross_view_loss
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_plane_loss_for_log = (0.4 * plane_loss.item() + 0.6 * ema_plane_loss_for_log) * opt.plane_constraint_weight
                ema_cross_view_loss_for_log = (0.4 * cross_view_loss.item() + 0.6 * ema_cross_view_loss_for_log) * opt.cross_view_constraint_weight
                # point_num = gaussians.get_xyz.shape[0]
                anchor_num = gaussians.get_anchor.shape[0]
                ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "D Loss": f"{ema_Ll1depth_for_log:.{4}f}", "P Loss": f"{ema_plane_loss_for_log:.{4}f}", "Cro Loss": f"{ema_cross_view_loss_for_log:.{4}f}", "Anchor Num": f"{anchor_num}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)    
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # save checkpoint
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--plane_constraint_iteration", type=int, default=10000)
    parser.add_argument("--cross_view_constraint_iteration", type=int, default=5000)
    parser.add_argument('--warmup', action='store_true', default=False)  
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # if not args.disable_viewer:
    #     network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), \
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, \
                args.debug_from, args.plane_constraint_iteration, args.cross_view_constraint_iteration)

    # All done
    print("\nTraining complete.")
