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
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
import numpy as np
from gaussian_renderer import render, get_filter
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, post_process_mesh
import open3d as o3d
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def visualize_depth(depth):
    depth = depth.squeeze().cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    depth_viz = (depth - depth_min) / (depth_max - depth_min)
    depth_viz = (depth_viz * 255.0).astype(np.uint8)
    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    return depth_viz

def visualize_normal(normal):
    normal_viz = ((normal.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
    normal_viz = cv2.cvtColor(normal_viz, cv2.COLOR_RGB2BGR)
    return normal_viz

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depths_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depths")
    render_normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normals")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(render_depths_path, exist_ok=True)
    makedirs(render_normals_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        bg = background
        # voxel_visible_mask = get_filter(view, gaussians, pipeline, bg)
        voxel_visible_mask = None
        renderpkg = render(view, gaussians, pipeline, bg, visible_mask=voxel_visible_mask, retain_grad=False)
        img_name = view.image_name.split(".")[0]
        rendering = renderpkg["render"]
        depth = renderpkg["depth"]
        normal = renderpkg["normal"]
        depth_viz = visualize_depth(depth)
        normal_viz = visualize_normal(normal)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, img_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, img_name + ".png"))
        cv2.imwrite(os.path.join(render_depths_path, img_name + ".png"), depth_viz)
        cv2.imwrite(os.path.join(render_normals_path, img_name + ".png"), normal_viz)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, skip_mesh: bool, \
                voxel_size: float, depth_trunc: float, sdf_trunc: float, num_cluster: int, mesh_res: int):
    with torch.no_grad():
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
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.eval()
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_mesh:
            gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=bg_color)    
            print("export mesh ...")
            # set the active_sh to 0 to export only diffuse texture
            # gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(scene.getTrainCameras(), gaussians)
            # gaussExtractor.reconstruction(scene.getTestMeshCameras(), gaussians)
            # extract the mesh and save

            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=mesh_res)

            # name = 'fuse.ply'
            # depth_trunc = (gaussExtractor.radius * 2.0) if depth_trunc < 0  else depth_trunc
            # voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
            # sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
            # mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name), mesh)
            print("mesh saved at {}".format(os.path.join(dataset.model_path, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(dataset.model_path, name.replace('.ply', '_post.ply'))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")

    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--voxel_size_TSDF", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--mesh_res", default=256, type=int, help='Mesh: resolution for unbounded mesh extraction')


    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, \
                args.skip_mesh, args.voxel_size_TSDF, args.depth_trunc, args.sdf_trunc, args.num_cluster, args.mesh_res)