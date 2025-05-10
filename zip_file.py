import os
import zipfile

# 要压缩的文件夹列表
folders_to_zip = ['0087_02', '0088_00', '0420_01', '0628_02']
data_dir = 'data'  # 数据目录
zip_filename = 'selected_data.zip'  # 输出 zip 文件名

# 创建 zip 文件
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for folder in folders_to_zip:
        folder_path = os.path.join(data_dir, folder)
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 将文件添加到 zip 中，保持目录结构
                zipf.write(file_path, os.path.relpath(file_path, data_dir))