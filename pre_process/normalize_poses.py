import os
import numpy as np
from pathlib import Path

def normalize_pose(pose):
    """归一化pose的平移部分"""
    # 提取平移向量
    translation = pose[:3, 3]
    
    # 计算平移向量的范数
    norm = np.linalg.norm(translation)
    
    # 归一化平移向量
    if norm > 0:
        normalized_translation = translation / norm
    else:
        normalized_translation = translation
    
    # 创建新的pose矩阵
    normalized_pose = pose.copy()
    normalized_pose[:3, 3] = normalized_translation
    
    return normalized_pose

def process_pose_file(file_path):
    """处理单个pose文件"""
    # 读取pose文件
    pose = np.loadtxt(file_path)
    
    # 归一化pose
    normalized_pose = normalize_pose(pose)
    
    # 写回文件
    np.savetxt(file_path, normalized_pose, fmt='%.6f')

def main():
    # 设置pose文件夹路径
    pose_dir = Path("data/0088_00/poses")
    
    # 确保文件夹存在
    if not pose_dir.exists():
        print(f"Error: {pose_dir} does not exist!")
        return
    
    # 处理所有pose文件
    for pose_file in pose_dir.glob("*.txt"):
        try:
            process_pose_file(pose_file)
            print(f"Processed {pose_file.name}")
        except Exception as e:
            print(f"Error processing {pose_file.name}: {str(e)}")

if __name__ == "__main__":
    main() 