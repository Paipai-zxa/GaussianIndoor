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
import cv2
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
def plane_constraint_loss(depth_map, normal_map):
    """计算平面约束损失
    Args:
        depth_map: 渲染得到的深度图 (H, W)
        normal_map: 从深度图计算得到的法向量图 (H, W, 3)
    Returns:
        loss: 平面约束损失
    """
    # 计算深度图的梯度来得到局部法向量
    def get_depth_gradients(depth):
        dy = depth[1:, :] - depth[:-1, :]  # 垂直梯度
        dx = depth[:, 1:] - depth[:, :-1]  # 水平梯度
        return dx, dy
    
    # 从深度梯度计算法向量
    def depth_to_normals(depth):
        dx, dy = get_depth_gradients(depth)
        # 填充使得大小与输入相同
        dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
        dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
        
        # 构建法向量 [-dx, -dy, 1]
        normals = torch.stack([-dx, -dy, torch.ones_like(dx)], dim=-1)
        # 归一化
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-7)
        return normals
    depth_map = depth_map.squeeze()
    normal_map = normal_map.squeeze()
    # 计算从深度图得到的法向量
    computed_normals = depth_to_normals(depth_map)
    
    # 计算法向量差异的L1损失
    loss = torch.abs(computed_normals - normal_map).mean()
    
    return loss

def cross_view_constraint(viewpoint_cam, depth, neighbor_cams, neighbor_depths):
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
        # 计算从参考视图到相邻视图的单应性矩阵
        H_rn = compute_homography(viewpoint_cam, neighbor_cams[i], depth)
        
        # 计算从相邻视图到参考视图的单应性矩阵
        H_nr = compute_homography(neighbor_cams[i], viewpoint_cam, neighbor_depth)
        
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

def compute_homography(src_cam, dst_cam, depth):
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
    normal = torch.tensor([0, 0, 1], device=device, dtype=dtype).reshape(3, 1)
    # 使用平面-单应性公式: H = K2 * (R - t*n^T/d) * K1^(-1)
    H = K_dst @ (R_rel - (t_rel.reshape(3, 1) @ normal.reshape(1, 3)) / avg_depth) @ torch.inverse(K_src)
    
    return H

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