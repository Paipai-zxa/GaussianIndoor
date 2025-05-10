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
import random
import numpy as np
import os
import cv2
from tqdm import tqdm
from os import makedirs
import numpy as np
from gaussian_renderer import vanilla_render, scaffold_render, get_filter
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import ScaffoldGaussianModel, VanillaGaussianModel
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

def render_set(dataset, name, iteration, views, gaussians, pipeline, background):
    model_path = dataset.model_path
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
        render_func = scaffold_render if dataset.enable_scaffold else vanilla_render
        renderpkg = render_func(view, gaussians, pipeline, bg)

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

def render_semantic_set(dataset, views, gaussians, pipeline, background):
    model_path = dataset.model_path
    render_semantic_path = os.path.join(model_path, "semantic_image")
    render_instance_path = os.path.join(model_path, "instance_image")

    makedirs(render_semantic_path, exist_ok=True)
    makedirs(render_instance_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        bg = background
        render_func = scaffold_render if dataset.enable_scaffold else vanilla_render
        renderpkg = render_func(view, gaussians, pipeline, bg)

        img_name = view.image_name.split(".")[0]
        semantic_map = renderpkg["semantic_map"]
        semantic_map = torch.argmax(torch.softmax(semantic_map, dim=0), dim=0)
        instance_map = renderpkg["instance_map"]
        instance_map = torch.argmax(torch.softmax(instance_map, dim=0), dim=0)
        semantic_map = semantic_map.cpu().numpy().astype(np.uint8)
        instance_map = instance_map.cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(render_semantic_path, img_name + ".png"), semantic_map)
        cv2.imwrite(os.path.join(render_instance_path, img_name + ".png"), instance_map)

def render_sets(args, dataset : ModelParams, iteration : int, pipeline : PipelineParams):

    with torch.no_grad():
        if args.enable_scaffold:
            gaussians = ScaffoldGaussianModel(dataset.sh_degree, 
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
            gaussians = VanillaGaussianModel(dataset.sh_degree,
                                            enable_training_exposure = dataset.enable_training_exposure,
                                            enable_geo_mlp = dataset.enable_geo_mlp,
                                            feat_dim = dataset.feat_dim,
                                            use_feat_bank = dataset.use_feat_bank,
                                            add_cov_dist = dataset.add_cov_dist,
                                            detach_geo_mlp_input_feat = dataset.detach_geo_mlp_input_feat,
                                            detach_scales_ori = dataset.detach_scales_ori,
                                            detach_rotations_ori = dataset.detach_rotations_ori,
                                            detach_geo_rasterizer_input_means3D = dataset.detach_geo_rasterizer_input_means3D,
                                            detach_geo_rasterizer_input_means2D = dataset.detach_geo_rasterizer_input_means2D,
                                            detach_geo_rasterizer_input_means2D_abs = dataset.detach_geo_rasterizer_input_means2D_abs,
                                            detach_geo_rasterizer_input_opacity = dataset.detach_geo_rasterizer_input_opacity,
                                            detach_geo_rasterizer_input_shs = dataset.detach_geo_rasterizer_input_shs,
                                            detach_geo_rasterizer_input_input_all_map = dataset.detach_geo_rasterizer_input_input_all_map,
                                            scales_geo_after_activation = dataset.scales_geo_after_activation,
                                            rotations_geo_after_activation = dataset.rotations_geo_after_activation,
                                            opt_geo_mlp_iteration = dataset.opt_geo_mlp_iteration,
                                            enable_semantic = dataset.enable_semantic,
                                            opt_semantic_mlp_iteration = dataset.opt_semantic_mlp_iteration,
                                            semantic_feature_dim = dataset.semantic_feature_dim,
                                            instance_feature_dim = dataset.instance_feature_dim,
                                            semantic_mlp_dim = dataset.semantic_mlp_dim,
                                            instance_query_num = dataset.instance_query_num,
                                            instance_query_feat_dim = dataset.instance_query_feat_dim,
                                            load_semantic_from_pcd = dataset.load_semantic_from_pcd,
                                            use_geo_mlp_scales = dataset.use_geo_mlp_scales,
                                            use_geo_mlp_rotations = dataset.use_geo_mlp_rotations,
                                            instance_query_gaussian_sigma = dataset.instance_query_gaussian_sigma,
                                            instance_query_distance_mode = dataset.instance_query_distance_mode)

        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.eval()
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
             render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not args.skip_test:
             render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            
        if dataset.enable_semantic:
            render_semantic_set(dataset, scene.getTrainCameras(), gaussians, pipeline, background)
            render_semantic_set(dataset, scene.getTestCameras(), gaussians, pipeline, background)

        if not args.skip_mesh:
            render_func = scaffold_render if dataset.enable_scaffold else vanilla_render
            gaussExtractor = GaussianExtractor(gaussians, render_func, pipeline, bg_color=bg_color, extract_semantic=dataset.enable_semantic)    
            print("export mesh ...")
            # set the active_sh to 0 to export only diffuse texture
            # gaussExtractor.gaussians.active_sh_degree = 0
            
            gaussExtractor.reconstruction(scene.getTrainCameras(), scene.id2label)
            # gaussExtractor.reconstruction(scene.getTestCameras())
            # extract the mesh and save

            name = 'fuse.ply'

            if args.is_unbounded:
                radius = None if args.depth_trunc < 0  else args.depth_trunc / 2
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res, radius=radius)
            else:
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size_TSDF < 0 else args.voxel_size_TSDF
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

                if dataset.enable_semantic:
                    mesh_semantic, mesh_instance = gaussExtractor.extract_semantic_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name.replace('.ply', '_semantic.ply')), mesh_semantic)
                    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name.replace('.ply', '_instance.ply')), mesh_instance)

                    mesh_semantic_post = post_process_mesh(mesh_semantic, cluster_to_keep=args.num_cluster)
                    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name.replace('.ply', '_semantic_post.ply')), mesh_semantic_post)

                    mesh_instance_post = post_process_mesh(mesh_instance, cluster_to_keep=args.num_cluster)
                    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name.replace('.ply', '_instance_post.ply')), mesh_instance_post)

                # radius = None if args.depth_trunc < 0  else args.depth_trunc / 2
                # pcd = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res, radius=radius, only_visualize=True)
                # o3d.io.write_point_cloud(os.path.join(dataset.model_path, name.replace('.ply', '_gaussian_tsdf.ply')), pcd)

            o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, name), mesh)
            print("mesh saved at {}".format(os.path.join(dataset.model_path, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
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
    parser.add_argument("--is_unbounded", action="store_true")
    parser.add_argument("--voxel_size_TSDF", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--mesh_res", default=256, type=int, help='Mesh: resolution for unbounded mesh extraction')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    render_sets(args, model.extract(args), args.iteration, pipeline.extract(args))