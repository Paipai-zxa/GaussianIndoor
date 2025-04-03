from utils.loss_utils import cross_view_constraint, compute_homography, warp_pixels
import torch
import cv2


path1 = '/data1/zxa/Data/ScanNetV2/scans/scene0085_00/depth/0.png'
path2 = '/data1/zxa/Data/ScanNetV2/scans/scene0085_00/depth/10.png'

depth1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
depth2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

print(depth1.shape)
print(depth2.shape)





