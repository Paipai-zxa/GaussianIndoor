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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from pytorch3d.transforms import quaternion_to_matrix
from scene.cameras import Camera
from typing import List
from torch.nn import init
from utils.mesh_utils import extract_sdf_guidance
from torch.nn import functional as F
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class  GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 sh_degree, 
                 optimizer_type="default",
                 enable_training_exposure=False,
                 enable_geo_mlp=False,
                 feat_dim=32,
                 use_feat_bank=False,
                 add_cov_dist=False,
                 detach_geo_mlp_input_feat=False,
                 detach_scales_ori=False,
                 detach_rotations_ori=False,
                 detach_geo_rasterizer_input_means3D=False,
                 detach_geo_rasterizer_input_means2D=False,
                 detach_geo_rasterizer_input_means2D_abs=False,
                 detach_geo_rasterizer_input_opacity=False,
                 detach_geo_rasterizer_input_shs=False,
                 detach_geo_rasterizer_input_input_all_map=False,
                 scales_geo_after_activation=False,
                 rotations_geo_after_activation=False,
                 opt_geo_mlp_iteration=7000,
                 enable_semantic=False,
                 opt_semantic_mlp_iteration=7000,
                 semantic_feature_dim=16,
                 instance_feature_dim=16,
                 semantic_mlp_dim=128,
                 instance_query_num=100,
                 instance_query_feat_dim=16,
                 load_semantic_from_pcd=False,
                 use_geo_mlp_scales=False,
                 use_geo_mlp_rotations=False,
                 instance_query_gaussian_sigma=0.01,
                 instance_query_distance_mode=0,
                 apply_semantic_guidance=False):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  

        # geo_mlp
        self.enable_geo_mlp = enable_geo_mlp
        self.use_feat_bank = use_feat_bank
        self.opt_geo_mlp_iteration = opt_geo_mlp_iteration
        if self.enable_geo_mlp:
            self.feat_dim = feat_dim
            self.add_cov_dist = add_cov_dist
            self.detach_geo_mlp_input_feat = detach_geo_mlp_input_feat
            self.detach_scales_ori = detach_scales_ori
            self.detach_rotations_ori = detach_rotations_ori
            self.detach_geo_rasterizer_input_means3D = detach_geo_rasterizer_input_means3D
            self.detach_geo_rasterizer_input_means2D = detach_geo_rasterizer_input_means2D
            self.detach_geo_rasterizer_input_means2D_abs = detach_geo_rasterizer_input_means2D_abs
            self.detach_geo_rasterizer_input_opacity = detach_geo_rasterizer_input_opacity
            self.detach_geo_rasterizer_input_shs = detach_geo_rasterizer_input_shs
            self.detach_geo_rasterizer_input_input_all_map = detach_geo_rasterizer_input_input_all_map
            self.scales_geo_after_activation = scales_geo_after_activation
            self.rotations_geo_after_activation = rotations_geo_after_activation
            self.cov_dist_dim = 1 if self.add_cov_dist else 0
        
        self.enable_semantic = enable_semantic
        self.load_semantic_from_pcd = load_semantic_from_pcd
        self.opt_semantic_mlp_iteration = opt_semantic_mlp_iteration
        if self.enable_semantic:
            self.semantic_feature_dim = semantic_feature_dim
            self.instance_feature_dim = instance_feature_dim
            self.semantic_mlp_dim = semantic_mlp_dim
            self.instance_query_num = instance_query_num
            self.instance_query_feat_dim = instance_query_feat_dim
            self.use_geo_mlp_scales = use_geo_mlp_scales
            self.use_geo_mlp_rotations = use_geo_mlp_rotations
            self.instance_query_gaussian_sigma = instance_query_gaussian_sigma
            self.instance_query_distance_mode = instance_query_distance_mode
            self.apply_semantic_guidance = apply_semantic_guidance
            self._semantic_features = torch.empty(0)
            self._instance_features = torch.empty(0)
            self._instance_query_pos = torch.empty(0)
            self._instance_query_rotation = torch.empty(0)
            self._instance_query_scaling = torch.empty(0)
            self._instance_query_features = torch.empty(0)
        
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.enable_training_exposure = enable_training_exposure
        self.setup_functions()

    def construct_mlp(self, semantic_class_num=1):
        if self.load_semantic_from_pcd:
            self.semantic_feature_dim = semantic_class_num

        if self.enable_geo_mlp:
            shs_feature_dim = (self.max_sh_degree + 1) ** 2 * 3
            self.geo_mlp_cov = nn.Sequential(
                nn.Linear(3+self.cov_dist_dim+shs_feature_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 7),
            ).cuda()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        if self.enable_semantic:
            self.semantic_mlp = nn.Sequential(
                nn.Linear(self.semantic_feature_dim+3, self.semantic_mlp_dim),
                nn.ReLU(True),
                nn.Linear(self.semantic_mlp_dim, semantic_class_num),
            ).cuda()
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize geo_mlp_cov layers
        if self.enable_geo_mlp:
            for layer in self.geo_mlp_cov:
                if isinstance(layer, nn.Linear):
                    init.constant_(layer.weight, 0)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)

        # Initialize mlp_feature_bank layers if it exists
        if self.use_feat_bank:
            for layer in self.mlp_feature_bank:
                if isinstance(layer, nn.Linear):
                    init.constant_(layer.weight, 0)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)
        
        if self.enable_semantic:
            for layer in self.semantic_mlp:
                if isinstance(layer, nn.Linear):
                    init.constant_(layer.weight, 0)
                    if layer.bias is not None:
                        init.constant_(layer.bias, 0)

    def set_appearance(self, num_cameras):
        pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        if self.enable_training_exposure:
            return self._exposure
        else:
            return None

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_geo_mlp_cov(self):
        return self.geo_mlp_cov

    @property
    def get_semantic_features(self):
        return self._semantic_features

    @property
    def get_instance_features(self):
        return self._instance_features

    @property
    def get_instance_query_pos(self):
        return self._instance_query_pos

    @property
    def get_instance_query_rotation(self):
        return self.rotation_activation(self._instance_query_rotation)

    @property
    def get_instance_query_scaling(self):
        return self.scaling_activation(self._instance_query_scaling)

    @property
    def get_instance_query_features(self):
        return self._instance_query_features

    @property
    def get_semantic_mlp(self):
        return self.semantic_mlp

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            if image_name in self.exposure_mapping:
                return self._exposure[self.exposure_mapping[image_name]]
            else:
                return None
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_instance_query_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_instance_query_scaling, scaling_modifier, self.get_instance_query_rotation)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        return data


    def eval(self):
        if self.enable_geo_mlp:
            self.geo_mlp_cov.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.enable_semantic:
            self.semantic_mlp.eval()

    def train(self):
        if self.enable_geo_mlp:
            self.geo_mlp_cov.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.enable_semantic:
            self.semantic_mlp.train()

    def get_rotation_matrix(self):
        return quaternion_to_matrix(self.get_rotation)
        
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)    
    
    def get_normal(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : List[Camera], spatial_lr_scale : float, semantic_pcd=None, instance_pcd=None):
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        if self.enable_semantic:
            if self.load_semantic_from_pcd:
                semantic_features = F.one_hot(torch.tensor(np.asarray(semantic_pcd.colors[:,0])).long(), num_classes=self.semantic_feature_dim).float().cuda()
                semantic_features[semantic_features==0] = np.log(0.1)
                semantic_features[semantic_features==1] = np.log(0.9)
                self._semantic_features = nn.Parameter(semantic_features.requires_grad_(True))
                
                instance_inds = np.unique(np.asarray(instance_pcd.colors[:,0]))
                self.instance_query_num = min(len(instance_inds), 34)
                self.instance_feature_dim = self.instance_query_num
                instance_inds = instance_inds[:self.instance_query_num]
                instance_query_pos = []
                instance_query_features = []
                instance_query_rotation = []
                instance_query_scaling = []
                instance_features = np.log(0.1) * torch.ones((instance_pcd.colors.shape[0], self.instance_feature_dim), dtype=torch.float32).cuda()

                for ind, instance_ind in enumerate(instance_inds):
                    instance_mask = np.asarray(instance_pcd.colors[:,0]) == instance_ind
                    instance_query_pos.append(np.mean(instance_pcd.points[instance_mask], axis=0))
                    instance_query_feature = np.array([1.0 if i == ind else 0.0 for i in range(self.instance_feature_dim)])
                    instance_query_features.append(instance_query_feature)
                    instance_query_rotation.append(np.array([1.0, 0.0, 0.0, 0.0]))
                    instance_features[instance_mask, ind] = np.log(0.9)
                    # 计算边界框
                    min_coords = np.min(instance_pcd.points[instance_mask], axis=0)
                    max_coords = np.max(instance_pcd.points[instance_mask], axis=0)
                    # 计算长宽高的一半
                    half_size = (max_coords - min_coords) / 2.0
                    instance_query_scaling.append(half_size)
                    
                instance_query_pos = torch.tensor(np.stack(instance_query_pos)).float().cuda()
                instance_query_features = torch.tensor(np.stack(instance_query_features)).float().cuda()
                instance_query_rotation = torch.tensor(np.stack(instance_query_rotation)).float().cuda()
                instance_query_scaling = torch.tensor(np.stack(instance_query_scaling)).float().cuda()
                self._instance_query_pos = nn.Parameter(instance_query_pos.requires_grad_(True))
                self._instance_query_features = nn.Parameter(instance_query_features.requires_grad_(True))
                self._instance_query_rotation = nn.Parameter(instance_query_rotation.requires_grad_(True))
                self._instance_query_scaling = nn.Parameter(instance_query_scaling.requires_grad_(True))
                # instance_features = F.one_hot(torch.tensor(np.asarray(instance_pcd.colors[:,0])).long(), num_classes=self.instance_feature_dim).float().cuda()
                # instance_features[instance_features==0] = np.log(0.1)
                # instance_features[instance_features==1] = np.log(0.9)
                self._instance_features = nn.Parameter(instance_features.requires_grad_(True))

            else:
                instance_query_pos = torch.tensor(np.zeros((self.instance_query_num, 3))).float().cuda()
                instance_query_rotation = torch.tensor(np.zeros((self.instance_query_num, 4))).float().cuda()
                instance_query_rotation[:, 0] = 1
                instance_query_scaling = 0.1 * torch.tensor(np.ones((self.instance_query_num, 3))).float().cuda()
                instance_query_features = torch.tensor(np.zeros((self.instance_query_num, self.instance_feature_dim))).float().cuda()
                instance_features = torch.tensor(np.zeros((pcd.colors.shape[0], self.instance_feature_dim))).float().cuda()
                
                self._instance_query_pos = nn.Parameter(instance_query_pos.requires_grad_(True))
                self._instance_query_rotation = nn.Parameter(instance_query_rotation.requires_grad_(True))
                self._instance_query_scaling = nn.Parameter(instance_query_scaling.requires_grad_(True))
                self._instance_query_features = nn.Parameter(instance_query_features.requires_grad_(True))
                self._instance_features = nn.Parameter(instance_features.requires_grad_(True))

                semantic_features = torch.tensor(np.zeros((pcd.colors.shape[0], self.semantic_feature_dim))).float().cuda()
                self._semantic_features = nn.Parameter(semantic_features.requires_grad_(True))
        # breakpoint()
        # fused_color = RGB2SH(torch.tensor(np.random.randn(pcd.colors.shape[0], 3)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.enable_training_exposure:
            self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
            self.pretrained_exposures = None
            exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
            self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]

        if self.enable_geo_mlp:
            l.append({'params': list(self.geo_mlp_cov.parameters()), 'lr': training_args.geo_mlp_lr_init, "name": "geo_mlp_cov"})
        if self.use_feat_bank:
            l.append({'params': list(self.mlp_feature_bank.parameters()), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})
        
        if self.enable_semantic:
            l.append({'params': [self._semantic_features], 'lr': training_args.semantic_features_lr, "name": "semantic_features"})
            l.append({'params': [self._instance_query_pos], 'lr': training_args.instance_query_pos_lr, "name": "instance_query_pos"})
            l.append({'params': [self._instance_query_rotation], 'lr': training_args.instance_query_rotation_lr, "name": "instance_query_rotation"})
            l.append({'params': [self._instance_query_scaling], 'lr': training_args.instance_query_scaling_lr, "name": "instance_query_scaling"})
            l.append({'params': [self._instance_query_features], 'lr': training_args.instance_query_features_lr, "name": "instance_query_features"})
            l.append({'params': [self._instance_features], 'lr': training_args.instance_features_lr, "name": "instance_features"})
            l.append({'params': list(self.semantic_mlp.parameters()), 'lr': training_args.semantic_mlp_lr_init, "name": "semantic_mlp"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)
        if self.enable_training_exposure:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
            self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        
        if self.enable_geo_mlp:
            self.geo_mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.geo_mlp_lr_init,
                                                        lr_final=training_args.geo_mlp_lr_final,
                                                        lr_delay_mult=training_args.geo_mlp_lr_delay_mult,
                                                        max_steps=training_args.iterations)
        if self.use_feat_bank:
            self.mlp_feature_bank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                    max_steps=training_args.iterations)
        
        if self.enable_semantic:
            self.semantic_mlp_scheduler_args = get_expon_lr_func(lr_init=training_args.semantic_mlp_lr_init,
                                                    lr_final=training_args.semantic_mlp_lr_final,
                                                    lr_delay_mult=training_args.semantic_mlp_lr_delay_mult,
                                                    max_steps=training_args.semantic_mlp_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.enable_training_exposure:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.enable_geo_mlp and param_group["name"] == "geo_mlp_cov":
                lr = self.geo_mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_feature_bank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.enable_semantic and param_group["name"] == "semantic_mlp":
                lr = self.semantic_mlp_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_instance_query_attributes(self):
        l = ['x', 'y', 'z']
        for i in range(self._instance_query_rotation.shape[1]):
            l.append('instance_query_rotation_{}'.format(i))
        for i in range(self._instance_query_scaling.shape[1]):
            l.append('instance_query_scaling_{}'.format(i))
        for i in range(self._instance_query_features.shape[1]):
            l.append('instance_query_features_{}'.format(i))
        return l

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.enable_semantic:
            for i in range(self._semantic_features.shape[1]):
                l.append('semantic_features_{}'.format(i))
            for i in range(self._instance_features.shape[1]):
                l.append('instance_features_{}'.format(i))
        return l


    def save_instance_query_ply(self, path):
        mkdir_p(os.path.dirname(path))
        instance_query_pos = self._instance_query_pos.detach().cpu().numpy()
        instance_query_rotation = self._instance_query_rotation.detach().cpu().numpy()
        instance_query_scaling = self._instance_query_scaling.detach().cpu().numpy()
        instance_query_features = self._instance_query_features.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_instance_query_attributes()]
        elements = np.empty(instance_query_pos.shape[0], dtype=dtype_full)
        attributes = np.concatenate((instance_query_pos, instance_query_rotation, instance_query_scaling, instance_query_features), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        if self.enable_semantic:
            semantic_features = self._semantic_features.detach().cpu().numpy()
            instance_features = self._instance_features.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.enable_semantic:
            attributes = np.concatenate((attributes, semantic_features, instance_features), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.enable_training_exposure:
            exposure_dict = {
                image_name: self.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.exposure_mapping
            }

            with open(os.path.join(os.path.dirname(path), "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)
        
        if self.enable_semantic:
            self.save_instance_query_ply(os.path.join(os.path.dirname(path), "instance_query.ply"))
        self.save_mlp_checkpoints(os.path.dirname(path))

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_instance_query_ply(self, path):
        plydata = PlyData.read(path)
        instance_query_pos = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        instance_query_rotation_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("instance_query_rotation")]
        instance_query_rotation_names = sorted(instance_query_rotation_names, key = lambda x: int(x.split('_')[-1]))
        instance_query_rotation = np.zeros((instance_query_pos.shape[0], len(instance_query_rotation_names)))
        for i, attr_name in enumerate(instance_query_rotation_names):
            instance_query_rotation[:, i] = np.asarray(plydata.elements[0][attr_name])

        instance_query_scaling_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("instance_query_scaling")]
        instance_query_scaling_names = sorted(instance_query_scaling_names, key = lambda x: int(x.split('_')[-1]))
        instance_query_scaling = np.zeros((instance_query_pos.shape[0], len(instance_query_scaling_names)))
        for i, attr_name in enumerate(instance_query_scaling_names):
            instance_query_scaling[:, i] = np.asarray(plydata.elements[0][attr_name])
        
        instance_query_feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("instance_query_features")]
        instance_query_feature_names = sorted(instance_query_feature_names, key = lambda x: int(x.split('_')[-1]))
        instance_query_features = np.zeros((instance_query_pos.shape[0], len(instance_query_feature_names)))
        for i, attr_name in enumerate(instance_query_feature_names):
            instance_query_features[:, i] = np.asarray(plydata.elements[0][attr_name])
        
        self._instance_query_pos = nn.Parameter(torch.tensor(instance_query_pos, dtype=torch.float, device="cuda").requires_grad_(True))
        self._instance_query_rotation = nn.Parameter(torch.tensor(instance_query_rotation, dtype=torch.float, device="cuda").requires_grad_(True))
        self._instance_query_scaling = nn.Parameter(torch.tensor(instance_query_scaling, dtype=torch.float, device="cuda").requires_grad_(True))
        self._instance_query_features = nn.Parameter(torch.tensor(instance_query_features, dtype=torch.float, device="cuda").requires_grad_(True))

    def load_ply(self, path):
        plydata = PlyData.read(path)
        if self.enable_training_exposure:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")

            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        if self.enable_semantic:
            semantic_feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("semantic_features")]
            semantic_feature_names = sorted(semantic_feature_names, key = lambda x: int(x.split('_')[-1]))
            semantic_features = np.zeros((xyz.shape[0], len(semantic_feature_names)))
            for i, attr_name in enumerate(semantic_feature_names):
                semantic_features[:, i] = np.asarray(plydata.elements[0][attr_name])
            instance_feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("instance_features")]
            instance_feature_names = sorted(instance_feature_names, key = lambda x: int(x.split('_')[-1]))
            instance_features = np.zeros((xyz.shape[0], len(instance_feature_names)))
            for i, attr_name in enumerate(instance_feature_names):
                instance_features[:, i] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.enable_semantic:
            self._semantic_features = nn.Parameter(torch.tensor(semantic_features, dtype=torch.float, device="cuda").requires_grad_(True))
            self._instance_features = nn.Parameter(torch.tensor(instance_features, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree   
        if self.enable_semantic:
            self.load_instance_query_ply(os.path.join(os.path.dirname(path), "instance_query.ply"))
        self.load_mlp_checkpoints(os.path.dirname(path))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "geo_mlp_cov" or group["name"] == "mlp_featurebank" or \
                group["name"] == "semantic_mlp" or group["name"] == "instance_query_features" or \
                group["name"] == "instance_query_pos" or group["name"] == "instance_query_rotation" or \
                group["name"] == "instance_query_scaling":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.enable_semantic:
            self._semantic_features = optimizable_tensors["semantic_features"]
            self._instance_features = optimizable_tensors["instance_features"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "geo_mlp_cov" or group["name"] == "mlp_featurebank" or \
                group["name"] == "semantic_mlp" or group["name"] == "instance_query_features" or \
                group["name"] == "instance_query_pos" or group["name"] == "instance_query_rotation" or \
                group["name"] == "instance_query_scaling":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,
                              new_semantic_features=None, new_instance_features=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if self.enable_semantic:
            d["semantic_features"] = new_semantic_features
            d["instance_features"] = new_instance_features

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.enable_semantic:
            self._semantic_features = optimizable_tensors["semantic_features"]
            self._instance_features = optimizable_tensors["instance_features"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, sdf_densify_mask=None, grad_sdf_omega=0.0002, is_apply_grad_sdf_omega=False, sdf_guidance=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        if is_apply_grad_sdf_omega:
            padded_grad[:sdf_guidance.shape[0]] += sdf_guidance.squeeze() * grad_sdf_omega
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if sdf_densify_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, sdf_densify_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        if self.enable_semantic:
            new_semantic_features = self._semantic_features[selected_pts_mask].repeat(N,1)
            new_instance_features = self._instance_features[selected_pts_mask].repeat(N,1)
        else:
            new_semantic_features = None
            new_instance_features = None

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii,
                                   new_semantic_features, new_instance_features)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, sdf_densify_mask=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if sdf_densify_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, sdf_densify_mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        if self.enable_semantic:
            new_semantic_features = self._semantic_features[selected_pts_mask]
            new_instance_features = self._instance_features[selected_pts_mask]
        else:
            new_semantic_features = None
            new_instance_features = None

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii,
                                   new_semantic_features, new_instance_features)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()


    def sdf_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, densification_threshold=0.95, pruning_threshold=0.002, \
                            viewpoint_stack=None, render=None, pipe=None, bg=None, is_recal_split=False, is_recal_prune=False,
                            grad_sdf_omega=0.0002, is_apply_grad_sdf_omega=False, enable_sdf_guidance=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        # densification_threshold 阈值设置越大（比如0.95），意味着：
        # 只有非常接近表面的点才会被选中进行加密（densify）
        # 被选中的点会更少
        # 加密操作会更保守

        gaussians_sdf, sdf_guidance = extract_sdf_guidance(viewpoint_stack, self, render, pipe, bg)
        if self.enable_semantic and self.load_semantic_from_pcd and self.apply_semantic_guidance:
            # exclude background class
            with torch.no_grad():
                semantic_probs = torch.softmax(self._semantic_features, dim=1)  # [N, C]
                semantic_confidence, semantic_class = torch.max(semantic_probs, dim=1)
                # set background class to 0
                semantic_confidence[semantic_class == 0] = 0.0
                sdf_guidance = sdf_guidance * semantic_confidence

        sdf_densify_mask = (sdf_guidance > densification_threshold).squeeze()
        self.densify_and_clone(grads if not is_apply_grad_sdf_omega else grads + sdf_guidance.unsqueeze(-1) * grad_sdf_omega, 
                               max_grad, extent, sdf_densify_mask=sdf_densify_mask if enable_sdf_guidance else None)

        if is_recal_split:
            gaussians_sdf, sdf_guidance = extract_sdf_guidance(viewpoint_stack, self, render, pipe, bg)
            if self.enable_semantic and self.load_semantic_from_pcd and self.apply_semantic_guidance:
                # exclude background class
                semantic_probs = torch.softmax(self._semantic_features, dim=1)  # [N, C]
                semantic_confidence, semantic_class = torch.max(semantic_probs, dim=1)
                # set background class to 0
                semantic_confidence[semantic_class == 0] = 0.0
                sdf_guidance = sdf_guidance * semantic_confidence.unsqueeze(-1)

        sdf_densify_mask = (sdf_guidance > densification_threshold).squeeze()
        self.densify_and_split(grads, max_grad, extent, sdf_densify_mask=sdf_densify_mask if enable_sdf_guidance else None,
                               grad_sdf_omega=grad_sdf_omega, is_apply_grad_sdf_omega=is_apply_grad_sdf_omega, sdf_guidance=sdf_guidance)

        #-----------prune------------------
        if enable_sdf_guidance:
            if is_recal_prune:
                gaussians_sdf, sdf_guidance = extract_sdf_guidance(viewpoint_stack, self, render, pipe, bg)
            sdf_prune_guidance = 1 - sdf_guidance
            sdf_prune_mask = (sdf_prune_guidance > pruning_threshold).squeeze()
            self.prune_points(sdf_prune_mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            if self.enable_geo_mlp:
                self.geo_mlp_cov.eval()
                shs_feature_dim = (self.max_sh_degree + 1) ** 2 * 3
                geo_mlp_cov = torch.jit.trace(self.geo_mlp_cov, (torch.rand(1, 3+self.cov_dist_dim+shs_feature_dim).cuda()))
                geo_mlp_cov.save(os.path.join(path, 'geo_mlp_cov.pt'))
                self.geo_mlp_cov.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()
            
            if self.enable_semantic:
                self.semantic_mlp.eval()
                semantic_mlp = torch.jit.trace(self.semantic_mlp, (torch.rand(1, self.semantic_feature_dim+3).cuda()))
                semantic_mlp.save(os.path.join(path, 'semantic_mlp.pt'))
                self.semantic_mlp.train()
            
        elif mode == 'unite':
            save_dict = {}
            if self.enable_geo_mlp:
                save_dict['geo_mlp_cov'] = self.geo_mlp_cov.state_dict()
            if self.use_feat_bank:
                save_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.enable_semantic:
                save_dict['semantic_mlp'] = self.semantic_mlp.state_dict()
            if len(save_dict.keys()) > 0:
                torch.save(save_dict, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError

    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            if self.enable_geo_mlp:
                self.geo_mlp_cov = torch.jit.load(os.path.join(path, 'geo_mlp_cov.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.enable_semantic:
                self.semantic_mlp = torch.jit.load(os.path.join(path, 'semantic_mlp.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            if self.enable_geo_mlp:
                self.geo_mlp_cov.load_state_dict(checkpoint['geo_mlp_cov'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.enable_semantic:
                self.semantic_mlp.load_state_dict(checkpoint['semantic_mlp'])
        else:
            raise NotImplementedError

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1):
        st = max(int(scale/2)-1,0)
        depth_view = depth[None,:,st::scale,st::scale]
        W, H = int(fov_camera.image_width/scale), int(fov_camera.image_height/scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
                        [points_in_camera_space[:,0] * fov_camera.Fx / points_in_camera_space[:,2] + fov_camera.Cx,
                         points_in_camera_space[:,1] * fov_camera.Fy / points_in_camera_space[:,2] + fov_camera.Cy], -1).float()/scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) &\
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:,2] > 0.1)

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        return map_z, mask
    
    def get_points_from_depth(self, fov_camera, depth, scale=1):
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        return pts

    def get_points_from_depth_with_values(self, fov_camera, depth, values, scale=1):
        """
        从深度图和对应的值生成带值的点云
        Args:
            fov_camera: 相机参数
            depth: 深度图 [H, W]
            values: 对应的值图 [C, H, W] 或 [H, W]
            scale: 下采样比例
        Returns:
            points: 3D点坐标 [N, 3]
            point_values: 对应的值 [N, C] 或 [N]
        """
        st = int(max(int(scale/2)-1,0))
        depth_view = depth.squeeze()[st::scale,st::scale]
        rays_d = fov_camera.get_rays(scale=scale)
        depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        
        # 获取3D点坐标
        pts = (rays_d * depth_view[..., None]).reshape(-1,3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts-T)@R.transpose(-1,-2)
        
        # 处理值
        if values.dim() == 3:  # [C, H, W]
            values = values[:, st::scale, st::scale]
            values = values[:, :rays_d.shape[0], :rays_d.shape[1]]
            point_values = values.permute(1,2,0).reshape(-1, values.shape[0])
        else:  # [H, W]
            values = values[st::scale, st::scale]
            values = values[:rays_d.shape[0], :rays_d.shape[1]]
            point_values = values.reshape(-1)
            
        return pts, point_values
