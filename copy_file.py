import os
import shutil

# 要处理的目录
base_path = "/data1/wxb/indoor/GaussianIndoor/data/f6659a3107"

# 遍历目录
def process_directory(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        # 检查是否为符号链接
        if os.path.islink(item_path):
            # 获取符号链接指向的实际文件路径
            target_path = os.readlink(item_path)
            
            # 删除符号链接
            os.unlink(item_path)
            
            # 复制实际文件
            if os.path.isdir(target_path):
                shutil.copytree(target_path, item_path)
            else:
                shutil.copy2(target_path, item_path)

# 处理指定目录
process_directory(base_path)