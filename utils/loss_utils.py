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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import scipy
import cv2
import numpy as np
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def get_img_grad_weight(img):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def points_to_normal(points):
    """
        points: points
    """
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def cross_view_constraint(viewpoint_cam, depth, normal, neighbor_cams, neighbor_depths, neighbor_normals):
    """计算跨视图约束损失
    
    Args:
        viewpoint_cam: 参考视角相机
        neighbor_cams: 相邻视角相机列表
        gaussians: 高斯模型
        pipe: 渲染管线
        background: 背景颜色
        
    Returns:
        cross_view_loss: 跨视图约束损失
    """
    total_loss = 0.0
    count = 0
    depth = depth.squeeze()
    dtype = depth.dtype
    device = depth.device
    # 对于每个相邻视图
    for i in range(len(neighbor_cams)):
        neighbor_depth = neighbor_depths[i].squeeze()
        neighbor_normal = neighbor_normals[i].squeeze()
        # 计算从参考视图到相邻视图的单应性矩阵
        H_rn = compute_homography(viewpoint_cam, neighbor_cams[i], depth, normal)
        
        # 计算从相邻视图到参考视图的单应性矩阵
        H_nr = compute_homography(neighbor_cams[i], viewpoint_cam, neighbor_depth, neighbor_normal)
        
        # 生成参考视图像素坐标网格
        height, width = depth.shape
        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype)
        )
        ref_pixels = torch.stack([x_coords, y_coords, torch.ones_like(x_coords, device=device, dtype=dtype)], dim=-1)
        
        # 前向-后向投影
        warped_pixels = warp_pixels(ref_pixels, H_nr @ H_rn)
        # 计算投影误差
        pixel_diff = ref_pixels[..., :2] - warped_pixels[..., :2]
        error = torch.norm(pixel_diff, dim=-1)
        
        # 创建有效区域掩码（排除深度为0或无效的区域）
        valid_mask = (depth > 0) & (neighbor_depth > 0)
        
        # 计算损失
        if valid_mask.sum() > 0:
            loss = (error * valid_mask.float()).sum() / valid_mask.sum()
            total_loss += loss
            count += 1
    
    # 返回平均损失
    return total_loss / max(count, 1)

def compute_homography(src_cam, dst_cam, depth, normal):
    """计算单应性矩阵
    
    Args:
        src_cam: 源相机
        dst_cam: 目标相机
        depth: 源视图深度图
        
    Returns:
        H: 单应性矩阵
    """
    # 从视场角计算内参
    device = depth.device
    dtype = depth.dtype
    src_height, src_width = depth.shape
    dst_height, dst_width = depth.shape
    # 源相机
    src_focal_x = src_width / (2 * torch.tan(torch.tensor(src_cam.FoVx / 2, device=device, dtype=dtype)))
    src_focal_y = src_height / (2 * torch.tan(torch.tensor(src_cam.FoVy / 2, device=device, dtype=dtype)))
    src_cx = src_width / 2
    src_cy = src_height / 2
    
    # 目标相机
    dst_focal_x = dst_width / (2 * torch.tan(torch.tensor(dst_cam.FoVx / 2, device=device, dtype=dtype)))
    dst_focal_y = dst_height / (2 * torch.tan(torch.tensor(dst_cam.FoVy / 2, device=device, dtype=dtype)))
    dst_cx = dst_width / 2
    dst_cy = dst_height / 2
    # 获取相机内参
    K_src = torch.tensor([
        [src_focal_x, 0, src_cx],
        [0, src_focal_y, src_cy],
        [0, 0, 1]
    ], device=device, dtype=dtype)
    
    K_dst = torch.tensor([
        [dst_focal_x, 0, dst_cx],
        [0, dst_focal_y, dst_cy],
        [0, 0, 1]
    ], device=device, dtype=dtype)
    
    # 获取相机外参 (R, t)
    R_src = torch.tensor(src_cam.R, device=device, dtype=dtype)
    t_src = torch.tensor(src_cam.T, device=device, dtype=dtype)
    
    R_dst = torch.tensor(dst_cam.R, device=device, dtype=dtype)
    t_dst = torch.tensor(dst_cam.T, device=device, dtype=dtype)
    
    # 计算相对旋转和平移
    R_rel = R_dst @ R_src.transpose(-2, -1)
    t_rel = t_dst - R_rel @ t_src
    
    # 计算单应性矩阵
    avg_depth = torch.mean(depth[depth > 0]) if (depth > 0).any() else torch.tensor(1.0, device=device, dtype=dtype)
    # 使用平面-单应性公式: H = K2 * (R - t*n^T/d) * K1^(-1)
    breakpoint()
    depth_mask = (depth > 0)
    depth = depth[depth_mask]
    normal = normal[:,depth_mask]

    H = K_dst @ (R_rel - (t_rel.reshape(3, 1) @ normal.reshape(1, 3)) / avg_depth) @ torch.inverse(K_src)
    
    return H

def sample_semantics_from_neighbor(pts_in_nearest_cam, nearest_cam, nearest_sem_map):
    """
    pts_in_nearest_cam: [N, 3]
    nearest_cam: 相机对象，需有 Fx, Fy, Cx, Cy, 图像尺寸
    nearest_sem_map: [C, H, W]
    返回: [N, C]，每个点在相邻视角采样到的语义分布
    """
    N = pts_in_nearest_cam.shape[0]
    C, H, W = nearest_sem_map.shape

    # 投影到像素平面
    proj_x = pts_in_nearest_cam[:, 0] * nearest_cam.Fx / pts_in_nearest_cam[:, 2] + nearest_cam.Cx
    proj_y = pts_in_nearest_cam[:, 1] * nearest_cam.Fy / pts_in_nearest_cam[:, 2] + nearest_cam.Cy

    # 归一化到[-1, 1]，适配grid_sample
    norm_x = (proj_x / (W - 1)) * 2 - 1
    norm_y = (proj_y / (H - 1)) * 2 - 1
    grid = torch.stack([norm_x, norm_y], dim=-1)  # [N, 2]
    grid = grid.view(1, 1, N, 2)  # [1, 1, N, 2]，适配grid_sample

    # [C, H, W] -> [1, C, H, W]
    sem_map = nearest_sem_map.unsqueeze(0)
    # grid_sample: [1, C, H, W] + [1, 1, N, 2] -> [1, C, 1, N]
    sampled_sem = F.grid_sample(sem_map, grid, mode='bilinear', align_corners=True)
    sampled_sem = sampled_sem.squeeze(0).squeeze(1).permute(1, 0)  # [N, C]

    return sampled_sem

def multiview_loss(render_pkg, viewpoint_cam, nearest_cam, render, gaussians, pipe, bg, iteration, pixel_noise_th, enable_semantic):
    ## compute geometry consistency mask and loss
    H, W = render_pkg['depth'].squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['depth'].device)

    nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, iteration=iteration)
    if enable_semantic:
        pts, pts_sem = gaussians.get_points_from_depth_with_values(viewpoint_cam, render_pkg['depth'], render_pkg['semantic_map'])
    else:
        pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['depth'])
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['depth'], pts_in_nearest_cam)
    
    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
    R = torch.tensor(nearest_cam.R).float().cuda()
    T = torch.tensor(nearest_cam.T).float().cuda()
    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
    pts_projections = torch.stack(
                [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
    d_mask = d_mask & (pixel_noise < pixel_noise_th)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~d_mask] = 0

    geo_loss = torch.zeros_like(pixel_noise).sum()
    sem_loss = torch.zeros_like(pixel_noise).sum()
    if d_mask.sum() > 0:
        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
        # if enable_semantic:
        #     # 将nearest_render_pkg['semantic_map']从[C,H,W]转换为[N,C]格式
        #     nearest_sem_map = nearest_render_pkg['semantic_map']  # [C, H, W]
        #     sampled_sem = sample_semantics_from_neighbor(pts_in_nearest_cam, nearest_cam, nearest_sem_map)  # [N, C]
        #     l2_norm_diff = torch.norm(pts_sem - sampled_sem, dim=-1)
        #     sem_loss = ((weights * l2_norm_diff)[d_mask]).mean()
        
    return geo_loss, sem_loss

def warp_pixels(pixels, H):
    """使用单应性矩阵变换像素坐标
    
    Args:
        pixels: 像素坐标 (batch_size, height, width, 3)
        H: 单应性矩阵 (3, 3)
        
    Returns:
        warped_pixels: 变换后的像素坐标
    """
    # 应用变换
    warped_homogeneous = torch.matmul(pixels, H.transpose(-2, -1))
    
    # 归一化
    warped_pixels = warped_homogeneous / (warped_homogeneous[..., 2:] + 1e-8)
    
    return warped_pixels

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@torch.no_grad()
def create_virtual_gt_with_linear_assignment(gt_instance_map, instance_map):
    labels = sorted(torch.unique(gt_instance_map).cpu().tolist())
    # L x N
    cost_matrix = np.zeros([len(labels), instance_map.shape[0]])
    for lidx, label in enumerate(labels):
        # # N 对于每一行
        # with torch.no_grad():
        #     bce_cost = F.binary_cross_entropy(instance_map[lidx], gt_instance_map==label)
        cost_matrix[lidx, :] = -(instance_map[:, (gt_instance_map == label).squeeze()].sum(dim=1) / ((gt_instance_map == label).sum() + 1e-4)).cpu().numpy()
    assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
    new_labels = torch.zeros_like(gt_instance_map)
    for aidx, lidx in enumerate(assignment[0]):
        new_labels[gt_instance_map == labels[lidx]] = assignment[1][aidx]
    new_labels_ind = torch.unique(new_labels).cpu().tolist()
    return new_labels, new_labels_ind