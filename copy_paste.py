import os
import json
from plyfile import PlyData

# 定义目录和输出文件
base_dir = 'output/scannetv2_pan/0628_02'
output_file = 'summary_metrics.txt'
if os.path.exists(output_file):
    os.remove(output_file)

# 遍历每个实验文件夹
for folder in sorted(os.listdir(base_dir)):
    if "arxiv" in folder:
        continue
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # 初始化指标变量
        ssim = psnr = lpips = "N/A"
        accuracy = completion = precision = recall = f_score = "N/A"
        pc_size = pc_points = "N/A"

        # 检查点云文件
        pc_path = os.path.join(folder_path, 'point_cloud/iteration_50000/point_cloud.ply')
        if not os.path.exists(pc_path):
            pc_path = os.path.join(folder_path, 'point_cloud/iteration_30000/point_cloud.ply')
        if os.path.exists(pc_path):
            # 获取文件大小（MB）
            pc_size = round(os.path.getsize(pc_path) / (1024 * 1024), 2)
            # 获取点数
            try:
                plydata = PlyData.read(pc_path)
                pc_points = len(plydata['vertex'])
            except:
                pc_points = "Error"

        # 检查并读取 results.json
        results_path = os.path.join(folder_path, 'results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_data = json.load(f)
                if "ours_50000" in results_data:
                    ssim = results_data["ours_50000"].get("SSIM", "N/A")
                    psnr = results_data["ours_50000"].get("PSNR", "N/A")
                    lpips = results_data["ours_50000"].get("LPIPS", "N/A")
                elif "ours_30000" in results_data:
                    ssim = results_data["ours_30000"].get("SSIM", "N/A")
                    psnr = results_data["ours_30000"].get("PSNR", "N/A")
                    lpips = results_data["ours_30000"].get("LPIPS", "N/A")

        # 检查并读取 mesh_results.json
        mesh_results_path = os.path.join(folder_path, 'mesh_results.json')
        if os.path.exists(mesh_results_path):
            with open(mesh_results_path, 'r') as f:
                mesh_data = json.load(f)
                accuracy = mesh_data.get("Accuracy", "N/A")
                completion = mesh_data.get("Completion", "N/A")
                precision = mesh_data.get("Precision", "N/A")
                recall = mesh_data.get("Recall", "N/A")
                f_score = mesh_data.get("F-score", "N/A")

        # 将结果写入输出文件
        with open(output_file, 'a') as f:
            f.write(f"{folder}\t{ssim}\t{psnr}\t{lpips}\t{accuracy}\t{completion}\t{precision}\t{recall}\t{f_score}\t{pc_size}\t{pc_points}\n")

print(f"Metrics summary has been written to {output_file}")
