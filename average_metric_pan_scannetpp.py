import os
import json
exp_name = "train_sem_wogeo_semantic_guidance_start12000_"
path = "output/scannetpp"
scene_list = os.listdir(path)
scene_list = [scene for scene in scene_list if not scene.endswith(".json")]
scene_list = ['1ada7a0617', 'f6659a3107', '5748ce6f01']
metrics = []
metrics_mesh = []
metrics_semantic = []
for scene in scene_list:
    exp_list = os.listdir(os.path.join(path, scene))
    for exp in exp_list:
        if exp_name in exp:
            metrics_path = os.path.join(path, scene, exp, "results.json")
            mesh_path = os.path.join(path, scene, exp, "mesh_results.json")
            semantic_path = os.path.join(path, scene, exp, "semantic_results.json")
            with open(metrics_path, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    metrics.append(value)
            with open(mesh_path, "r") as f:
                data = json.load(f)
                metrics_mesh.append(data)
            with open(semantic_path, "r") as f:
                data = json.load(f)
                metrics_semantic.append(data)
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
average_PQ = sum([metric["panoptic"]["total"]["PQ"] for metric in metrics_semantic]) / len(metrics_semantic)
average_SQ = sum([metric["panoptic"]["total"]["SQ"] for metric in metrics_semantic]) / len(metrics_semantic)
average_RQ = sum([metric["panoptic"]["total"]["RQ"] for metric in metrics_semantic]) / len(metrics_semantic)
average_mIoU = sum([metric["semantic"]["total"]["S_iou"] for metric in metrics_semantic]) / len(metrics_semantic)
average_mAcc = sum([metric["semantic"]["total"]["S_acc"] for metric in metrics_semantic]) / len(metrics_semantic)
average_mCov = sum([metric["instance"]["mCov"] for metric in metrics_semantic]) / len(metrics_semantic)
average_mWCov = sum([metric["instance"]["mWCov"] for metric in metrics_semantic]) / len(metrics_semantic)




print(f"SSIM: {average_ssim}\nPSNR: {average_psnr}\nLPIPS: {average_lpips}")
print(f"Accuracy: {average_accuracy}\nCompletion: {average_completion}\nPrecision: {average_precision}\nRecall: {average_recall}\nF-score: {average_fscore}")
print(f"PQ: {average_PQ}\nSQ: {average_SQ}\nRQ: {average_RQ}\nmIoU: {average_mIoU}\nmAcc: {average_mAcc}\nmCov: {average_mCov}\nmWCov: {average_mWCov}")
# 存储为json文件，该文件不一定存在
with open(f"output/average_metric_pan_scannetpp_{exp_name}.json", "w") as f:
    json.dump({"SSIM": average_ssim, "PSNR": average_psnr, "LPIPS": average_lpips, "Accuracy": average_accuracy, "Completion": average_completion, "Precision": average_precision, "Recall": average_recall, "F-score": average_fscore, "PQ": average_PQ, "SQ": average_SQ, "RQ": average_RQ, "mIoU": average_mIoU, "mAcc": average_mAcc, "mCov": average_mCov, "mWCov": average_mWCov}, f, indent=4)
    f.write(f"\n{average_ssim}\t{average_psnr}\t{average_lpips}\t{average_accuracy}\t{average_completion}\t{average_precision}\t{average_recall}\t{average_fscore}\t{average_PQ}\t{average_SQ}\t{average_RQ}\t{average_mIoU}\t{average_mAcc}\t{average_mCov}\t{average_mWCov}")