import os
import json
exp_name = "train_crossview_7000_1.5_numneighbors1_depth_"
path = "output/scannetv2_pan"
scene_list = os.listdir(path)
scene_list = [scene for scene in scene_list if not scene.endswith(".json")]
scene_list = ['0087_02', '0088_00', '0420_01', '0628_02']
metrics = []
metrics_mesh = []
for scene in scene_list:
    exp_list = os.listdir(os.path.join(path, scene))
    for exp in exp_list:
        if exp_name in exp:
            metrics_path = os.path.join(path, scene, exp, "results.json")
            mesh_path = os.path.join(path, scene, exp, "mesh_results.json")
            with open(metrics_path, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    metrics.append(value)
            with open(mesh_path, "r") as f:
                data = json.load(f)
                metrics_mesh.append(data)
            break

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
with open(f"output/average_metric_pan_scannetv2_{exp_name}.json", "w") as f:
    json.dump({"SSIM": average_ssim, "PSNR": average_psnr, "LPIPS": average_lpips, "Accuracy": average_accuracy, "Completion": average_completion, "Precision": average_precision, "Recall": average_recall, "F-score": average_fscore}, f, indent=4)
    f.write(f"\n{average_ssim}\t{average_psnr}\t{average_lpips}\t{average_accuracy}\t{average_completion}\t{average_precision}\t{average_recall}\t{average_fscore}")