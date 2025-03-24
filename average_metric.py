import os
import json

path = "output"
scene_list = os.listdir(path)
scene_list = [scene for scene in scene_list if not scene.endswith(".json")]
scene_list = ['0050_00', '0085_00', '0114_02', '0580_00']
metrics = []
for scene in scene_list:
    metrics_path = os.path.join(path, scene, "results.json")
    with open(metrics_path, "r") as f:
        data = json.load(f)
        for key, value in data.items():
            metrics.append(value)

#计算平均值
average_ssim = sum([metric["SSIM"] for metric in metrics]) / len(metrics)
average_psnr = sum([metric["PSNR"] for metric in metrics]) / len(metrics)
average_lpips = sum([metric["LPIPS"] for metric in metrics]) / len(metrics)

print(f"SSIM: {average_ssim}\nPSNR: {average_psnr}\nLPIPS: {average_lpips}")
# 存储为json文件，该文件不一定存在
if not os.path.exists("output/average_metric.json"):
    with open("output/average_metric.json", "w") as f:
        json.dump({"SSIM": average_ssim, "PSNR": average_psnr, "LPIPS": average_lpips}, f)