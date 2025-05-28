import os
import zipfile

zip_filename = 'selected_exp.zip'  # 输出 zip 文件名
directories = [
    "/lustre/xiaobao.wei/indoor/GaussianIndoor/output/scannetv2_pan/0087_02/train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final_20250512_154740",
    "/lustre/xiaobao.wei/indoor/GaussianIndoor/output/scannetv2_pan/0088_00/train_sem_semantic_guidance_start4000_20250511_024331",
    "/lustre/xiaobao.wei/indoor/GaussianIndoor/output/scannetv2_pan/0420_01/train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final_20250512_154802",
    "/lustre/xiaobao.wei/indoor/GaussianIndoor/output/scannetv2_pan/0628_02/train_sem_wogeo_semantic_guidance_start12000_omega0.000002_final_20250512_154805"
]

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(directory))
                zipf.write(file_path, arcname)