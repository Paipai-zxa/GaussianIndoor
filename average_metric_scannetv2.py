import os
import json

path = "output"
scene_list = os.listdir(path)
scene_list = [scene for scene in scene_list if not scene.endswith(".json")]
scene_list = ['0050_00', '0085_00', '0114_02', '0580_00', '0603_00', '0616_00', '0617_00', '0721_00']
metrics = []
metrics_mesh = []
for scene in scene_list:
    metrics_path = os.path.join(path, scene, "results.json")
    mesh_path = os.path.join(path, scene, "mesh_results.json")
    with open(metrics_path, "r") as f:
        data = json.load(f)
        for key, value in data.items():
            metrics.append(value)
    with open(mesh_path, "r") as f:
        data = json.load(f)
        metrics_mesh.append(data)

#计算平均值
average_ssim = sum([metric["SSIM"] for metric in metrics]) / len(metrics)
average_psnr = sum([metric["PSNR"] for metric in metrics]) / len(metrics)
average_lpips = sum([metric["LPIPS"] for metric in metrics]) / len(metrics)
average_accuracy = sum([metric["Accuracy"] for metric in metrics_mesh]) / len(metrics_mesh)
average_completion = sum([metric["Completion"] for metric in metrics_mesh]) / len(metrics_mesh)
average_precision = sum([metric["Precision"] for metric in metrics_mesh]) / len(metrics_mesh)
average_recall = sum([metric["Recall"] for metric in metrics_mesh]) / len(metrics_mesh)
average_fscore = sum([metric["F-score"] for metric in metrics_mesh]) / len(metrics_mesh)

print(f"SSIM: {average_ssim}\nPSNR: {average_psnr}\nLPIPS: {average_lpips}")
print(f"Accuracy: {average_accuracy}\nCompletion: {average_completion}\nPrecision: {average_precision}\nRecall: {average_recall}\nF-score: {average_fscore}")
# 存储为json文件，该文件不一定存在
if not os.path.exists("output/average_metric.json"):
    with open("output/average_metric.json", "w") as f:
        json.dump({"SSIM": average_ssim, "PSNR": average_psnr, "LPIPS": average_lpips, "Accuracy": average_accuracy, "Completion": average_completion, "Precision": average_precision, "Recall": average_recall, "F-score": average_fscore}, f, indent=4)