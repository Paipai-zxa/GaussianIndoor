import numpy as np
import os

path = "/data1/zxa/GaussianIndoor/data/0085_00/colmap_poses"
poses_dir = os.listdir(path)
k = 0
poses = []
for pose_dir in poses_dir:
    k+=1
    pose_path = os.path.join(path, pose_dir)
    pose = np.loadtxt(pose_path)
    if k%20==0:
        poses.append(pose)

poses = np.array(poses)
np.save("colmap_poses_inv.npy", poses)
