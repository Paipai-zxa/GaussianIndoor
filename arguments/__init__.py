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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    if value is False:
                        group.add_argument("--" + key, default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, default=value, action="store_false")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.enable_scaffold = False
        self.enable_training_exposure = False
        self.enable_geo_mlp = False
        self.opt_geo_mlp_iteration = 7000
        self.detach_geo_mlp_input_feat = False
        self.detach_scales_ori = False
        self.detach_rotations_ori = False
        self.detach_geo_rasterizer_input_means3D = False
        self.detach_geo_rasterizer_input_means2D = False
        self.detach_geo_rasterizer_input_means2D_abs = False
        self.detach_geo_rasterizer_input_opacity = False
        self.detach_geo_rasterizer_input_shs = False
        self.detach_geo_rasterizer_input_input_all_map = False
        self.scales_geo_after_activation = False
        self.rotations_geo_after_activation = False
        self.use_video_depth_anything = False
        self.is_train_on_all_images = False


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  -1        
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.use_feat_bank = False

        self.lod = 0
        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        #### semantic
        self.enable_semantic = False
        self.opt_semantic_mlp_iteration = 0
        self.semantic_feature_dim = 16
        self.instance_feature_dim = 16
        self.semantic_mlp_dim = 64
        self.instance_query_num = 35
        self.instance_query_feat_dim = 16
        self.load_semantic_from_pcd = False
        self.use_geo_mlp_scales = False
        self.use_geo_mlp_rotations = False
        self.instance_query_gaussian_sigma = 0.01
        self.instance_query_distance_mode = 0
        self.apply_semantic_guidance = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"

        # semantic
        self.semantic_features_lr = 0.0025
        self.instance_query_pos_lr = 0.00016
        self.instance_query_rotation_lr = 0.001
        self.instance_query_scaling_lr = 0.005
        self.instance_query_features_lr = 0.0025
        self.instance_features_lr = 0.0025

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.plane_constraint_weight = 0.015
        self.cross_view_constraint_weight = 0.015
        self.semantic_ce_weight = 0.5
        self.semantic_warping_weight = 1.5
        self.instance_bce_weight = 0.5
        self.instance_dice_weight = 0.5
        self.use_instance_train = False
        self.use_semantic_train = False
        self.scale_flatten_weight = 100.0
        self.num_neighbors_views = 1
        self.use_plane_constraint = False
        self.use_cross_view_constraint = False
        self.use_depth_regularization = False
        self.use_scale_flatten = False
        self.plane_constraint_iteration = 7000
        self.cross_view_constraint_iteration = 7000
        self.multi_view_pixel_noise_th = 0.01
        self.multi_view_geo_weight = 1.0
        self.scale_flatten_iteration = 0

        self.use_render_geo = False
        self.lambda_geo = 1.0

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000    

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.geo_mlp_lr_init = 0.004
        self.geo_mlp_lr_final = 0.0004
        self.geo_mlp_lr_delay_mult = 0.01
        self.geo_mlp_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000   

        # semantic
        self.semantic_mlp_lr_init = 0.004
        self.semantic_mlp_lr_final = 0.0004
        self.semantic_mlp_lr_delay_mult = 0.01
        self.semantic_mlp_lr_max_steps = 30_000

        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002      

        #### sdf
        self.enable_sdf_guidance = False
        self.is_recal_split = False
        self.is_recal_prune = False
        self.densification_threshold = 0.95
        self.pruning_threshold = 0.002
        self.sdf_guidance_start_iter = 15000
        self.sdf_guidance_end_iter = 25000
        self.sdf_guidance_interval = 100
        self.grad_sdf_omega = 0.0002
        self.is_apply_grad_sdf_omega = False

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
