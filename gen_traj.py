import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R, Slerp
import argparse

def read_pose(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        mat = np.array([[float(x) for x in line.strip().split()] for line in lines])
    return mat

def read_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        mat = [[float(x) for x in line.strip().split()] for line in lines]
    return mat

def interpolate_poses(pose1, pose2, n):
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    # 检查是否有无效数值
    if (np.isnan(R1).any() or np.isnan(R2).any() or
        np.isinf(R1).any() or np.isinf(R2).any()):
        print("检测到无效旋转矩阵，跳过该段插值。")
        return []
    # 正交化
    U1, _, Vt1 = np.linalg.svd(R1)
    U2, _, Vt2 = np.linalg.svd(R2)
    R1 = U1 @ Vt1
    R2 = U2 @ Vt2
    q1 = R.from_matrix(R1)
    q2 = R.from_matrix(R2)
    key_times = [0, 1]
    key_rots = R.from_matrix([R1, R2])
    slerp = Slerp(key_times, key_rots)
    result = []
    for i in range(n+1):
        alpha = i / n
        rot = slerp([alpha]).as_matrix()[0]
        trans = (1 - alpha) * t1 + alpha * t2
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = trans
        result.append(mat)
    return result

def main():
    parser = argparse.ArgumentParser(description='Interpolate camera poses')
    parser.add_argument('--scene', type=str, required=True, help='scene name, e.g. 0087_02')
    parser.add_argument('--num_interp', type=int, default=10, help='number of interpolation frames between two poses')
    parser.add_argument('--intrinsics', type=str, required=True, help='path to camera intrinsics txt')
    parser.add_argument('--width', type=int, default=640, help='image width')
    parser.add_argument('--height', type=int, default=480, help='image height')
    args = parser.parse_args()

    pose_dir = f'data/{args.scene}/poses'
    out_path = f'data/{args.scene}/interpolated_poses.json'

    pose_files = [f for f in os.listdir(pose_dir) if f.endswith('.txt')]
    pose_files_sorted = sorted(pose_files, key=lambda x: int(x.replace("DSC","").split('.')[0]))

    poses = [read_pose(os.path.join(pose_dir, f)) for f in pose_files_sorted]

    all_results = []
    frame_idx = 0

    for i in range(len(poses)-1):
        inter_poses = interpolate_poses(poses[i], poses[i+1], args.num_interp)
        for j, mat in enumerate(inter_poses):
            all_results.append({
                "frame": frame_idx,
                "matrix": mat.tolist()
            })
            frame_idx += 1

    # 最后一个关键帧
    all_results.append({
        "frame": frame_idx,
        "matrix": poses[-1].tolist()
    })

    # 读取内参
    intrinsics = read_intrinsics(args.intrinsics)
    intrinsics[0][2] = args.width / 2
    intrinsics[1][2] = args.height / 2

    output = {
        "intrinsics": intrinsics,
        "poses": all_results
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"插值完成，结果已写入 {out_path}")

if __name__ == '__main__':
    main()