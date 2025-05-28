import os
import re

# scene_list = ['1ada7a0617', '5748ce6f01', 'f6659a3107']
# image_dir = ['semantic_image', 'instance_image']

source_dir = '/data1/wxb/indoor/GaussianIndoor/data/5748ce6f01/instance_image'
target_dir = '/data1/wxb/indoor/GaussianIndoor/panoptic_results/5748ce6f01/instance_image'

# 假设 source_files 是 ['DSC03462.png', 'DSC03818.png', ...]
def extract_number(filename):
    nums = re.findall(r'\d+', filename)
    return int(nums[-1]) if nums else -1

# 获取源目录文件名并按数字排序
source_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
source_files_sorted = sorted(source_files, key=extract_number)

# 获取目标目录文件名并按数字排序
target_files = [f for f in os.listdir(target_dir) if f.endswith('.png')]
target_files_sorted = sorted(target_files, key=lambda x: int(x.split('.')[0]))


if len(source_files_sorted) != len(target_files_sorted):
    print(f"数量不一致，源目录{len(source_files_sorted)}，目标目录{len(target_files_sorted)}")
else:
    # 为避免重名冲突，先全部改为临时名
    for old_name in target_files_sorted:
        old_path = os.path.join(target_dir, old_name)
        tmp_path = os.path.join(target_dir, f"tmp_{old_name}")
        os.rename(old_path, tmp_path)
    # 再改为目标名
    for tmp_name, new_name in zip(target_files_sorted, source_files_sorted):
        tmp_path = os.path.join(target_dir, f"tmp_{tmp_name}")
        new_path = os.path.join(target_dir, new_name)
        os.rename(tmp_path, new_path)
        print(f"{tmp_name} -> {new_name}")
    print("重命名完成！")