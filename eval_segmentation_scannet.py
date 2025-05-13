"""
全景分割评估器
用于评估全景分割(panoptic segmentation)的性能，包括语义分割、实例分割和全景分割的评估
主要评估指标包括：
- PQ (Panoptic Quality)
- SQ (Segmentation Quality)
- RQ (Recognition Quality)
- mIoU (mean Intersection over Union)
- mAcc (mean Accuracy)
"""

import os
import csv
import json
import pickle
import numpy as np
import pandas as pd
import torch
import cv2
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from rich.table import Table
from rich.console import Console
from rich import print as rprint
from collections import defaultdict
from pre_process.label import get_labels
import colorsys

# 常量定义
OFFSET = 256 * 256 * 256  # 用于实例ID编码的偏移量
VOID = 0  # 无效区域标签

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='全景分割评估')
    parser.add_argument('--scene_idx', type=str, required=True, help='场景ID，例如：0087_02')
    parser.add_argument('--data_root', type=str, required=True, help='ScanNet数据集根目录')
    parser.add_argument('--result_root', type=str, required=True, help='结果保存根目录')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    parser.add_argument('--stride', type=int, default=1, help='评估步长')
    parser.add_argument('--save_gt', action='store_true', help='是否保存真实标签')
    parser.add_argument('--save_remap_instance', action='store_true', help='是否保存实例标签')
    parser.add_argument('--is_use_remap_instance', action='store_true', help='是否使用remap实例标签')
    return parser.parse_args()

def id2rgb(id):
    # Convert ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5

    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3,), dtype=np.float32)
    if id==0:
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    # rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    rgb[0], rgb[1], rgb[2] = r, g, b
    return rgb

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 场景配置
    scene_idx = args.scene_idx
    scene_name = scene_idx
    print(f"当前评估场景: {scene_idx}")

    # 路径配置
    gt_folder = os.path.join(args.data_root, scene_name)
    gt_semantic_path = os.path.join(gt_folder, "semantic")
    is_use_remap_instance = args.is_use_remap_instance
    if is_use_remap_instance:
        gt_instance_path = os.path.join(gt_folder, "instance_remap")
    else:
        gt_instance_path = os.path.join(gt_folder, "instance")

    # 结果路径配置
    result_folder = os.path.join(args.result_root)
    pre_semantic_path = os.path.join(result_folder, "semantic_image")
    pre_instance_path = os.path.join(result_folder, "instance_image")

    # 评估参数配置
    debug_flag = args.debug
    stride = args.stride
    save_gt = args.save_gt
    save_remap_instance = args.save_remap_instance
    vis_path = os.path.join(result_folder)
    label_map_path = os.path.join(gt_folder, f"{scene_idx}_map.csv")
    label_map = pd.read_csv(label_map_path)
    color_label_map = get_labels(scene_idx)

    # -----
    # Adapt from https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
    from collections import defaultdict
    OFFSET = 256 * 256 * 256
    VOID = 0 # or -1

    class PanopticStatCat:
        """
        每个类别的全景分割统计信息类
        用于存储单个类别的评估指标
        """
        def __init__(self):
            # 全景分割评估指标
            self.iou = 0.0  # 交并比
            self.tp = 0     # 真正例数
            self.fp = 0     # 假正例数
            self.fn = 0     # 假负例数

            # 语义分割评估指标
            self.semantic = {'iou': 0.0, 'acc': 0.0}  # 原始语义分割指标
            self.semantic_denoised = {'iou': 0.0, 'acc': 0.0}  # 去噪后的语义分割指标
            self.semantic_n = 0  # 语义分割样本数

        def __iadd__(self, panoptic_stat_cat):
            """
            重载+=运算符，用于累加统计信息
            """
            self.iou += panoptic_stat_cat.iou
            self.tp += panoptic_stat_cat.tp
            self.fp += panoptic_stat_cat.fp
            self.fn += panoptic_stat_cat.fn
            self.semantic['iou'] += panoptic_stat_cat.semantic['iou']
            self.semantic['acc'] += panoptic_stat_cat.semantic['acc']
            self.semantic_denoised['iou'] += panoptic_stat_cat.semantic_denoised['iou']
            self.semantic_denoised['acc'] += panoptic_stat_cat.semantic_denoised['acc']
            self.semantic_n += panoptic_stat_cat.semantic_n
            return self


    class PanopticStat:
        """
        全景分割统计信息类
        用于存储所有类别的评估指标和实例级别的统计信息
        """
        def __init__(self):
            self.panoptic_per_cat = defaultdict(PanopticStatCat)  # 每个类别的统计信息
            self.instance_stat = {
                'coverage': [],  # 实例覆盖度
                'gt_inst_area': [],  # 真实实例面积
                'num_pred_inst': 0,  # 预测实例数
                'num_gt_inst': 0,  # 真实实例数
            }
            self.panoptic_miou = 0  # 全景分割平均交并比

        def __getitem__(self, i):
            """获取指定类别的统计信息"""
            return self.panoptic_per_cat[i]

        def __iadd__(self, panoptic_stat):
            """累加统计信息"""
            for label, panoptic_stat_cat in panoptic_stat.panoptic_per_cat.items():
                self.panoptic_per_cat[label] += panoptic_stat_cat
            self.instance_stat['coverage'].extend(panoptic_stat.instance_stat['coverage'])
            self.instance_stat['gt_inst_area'].extend(panoptic_stat.instance_stat['gt_inst_area'])
            self.instance_stat['num_pred_inst'] += panoptic_stat.instance_stat['num_pred_inst']
            self.instance_stat['num_gt_inst'] += panoptic_stat.instance_stat['num_gt_inst']
            return self

        def pq_average(self, categories, label_thing_mapping, instance_type='all', verbose=False):
            """
            计算全景分割质量(PQ)的平均值
            
            Args:
                categories: 类别列表
                label_thing_mapping: 类别到thing/stuff的映射
                instance_type: 实例类型('all'/'thing'/'stuff')
                verbose: 是否输出详细信息
                
            Returns:
                总体结果和每个类别的结果
            """
            pq, sq, rq, n = 0, 0, 0, 0
            per_class_results = {}
            tp_all, fp_all, fn_all = 0, 0, 0
            
            for label in categories:
                iou = self.panoptic_per_cat[label].iou
                tp = self.panoptic_per_cat[label].tp
                fp = self.panoptic_per_cat[label].fp
                fn = self.panoptic_per_cat[label].fn
                
                if tp + fp + fn == 0:
                    n += 1
                    if verbose:
                        per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False, 'tp': tp, 'fp': fp, 'fn': fn}
                    else:
                        per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'valid': False}
                    continue
                
                pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
                sq_class = iou / tp if tp != 0 else 0
                rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
                
                if verbose:
                    per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True, 'tp': tp, 'fp': fp, 'fn': fn}
                else:
                    per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'valid': True}
                
                # 根据实例类型过滤
                if label_thing_mapping is not None:
                    if instance_type == 'thing' and label_thing_mapping[label] != 1:
                        continue
                    if instance_type == 'stuff' and label_thing_mapping[label] != 0:
                        continue

                pq += pq_class
                sq += sq_class
                rq += rq_class
                tp_all += tp
                fp_all += fp
                fn_all += fn
                n += 1

            if verbose:
                return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 
                        'tp': tp_all / n, 'fp': fp_all / n, 'fn': fn_all / n}, per_class_results
            else:
                return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n, 'miou': self.panoptic_miou}, per_class_results
        
        def instance_average(self, iou_threshold=0.5):
            """
            计算实例分割的平均指标
            
            Args:
                iou_threshold: IoU阈值
                
            Returns:
                实例分割评估结果
            """
            stat_coverage = np.array(self.instance_stat['coverage'])
            stat_gt_inst_area = np.array(self.instance_stat['gt_inst_area'])
            coverage = np.mean(stat_coverage)
            weighted_coverage = np.sum((stat_gt_inst_area / stat_gt_inst_area.sum()) * stat_coverage)
            prec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_pred_inst']
            rec = (stat_coverage > iou_threshold).sum() / self.instance_stat['num_gt_inst']
            return {'mCov': coverage, 'mWCov': weighted_coverage, 'mPrec': prec, 'mRec': rec}
        
        def semantic_average(self, categories):
            """
            计算语义分割的平均指标
            
            Args:
                categories: 类别列表
                
            Returns:
                语义分割评估结果
            """
            iou, acc, iou_d, acc_d, n = 0, 0, 0, 0, 0
            per_class_results = {}
            for label in categories:
                if self.panoptic_per_cat[label].semantic_n == 0:
                    per_class_results[label] = {'iou': 0.0, 'acc': 0.0, 'iou_d': 0.0, 'acc_d': 0.0, 'valid': False}
                    n += 1
                    continue
                n += 1
                iou_class = self.panoptic_per_cat[label].semantic['iou'] / self.panoptic_per_cat[label].semantic_n
                acc_class = self.panoptic_per_cat[label].semantic['acc'] / self.panoptic_per_cat[label].semantic_n
                iou_d_class = self.panoptic_per_cat[label].semantic_denoised['iou'] / self.panoptic_per_cat[label].semantic_n
                acc_d_class = self.panoptic_per_cat[label].semantic_denoised['acc'] / self.panoptic_per_cat[label].semantic_n
                per_class_results[label] = {'iou': iou_class, 'acc': acc_class, 'iou_d': iou_d_class, 'acc_d': acc_d_class, 'valid': True}
                iou += iou_class
                acc += acc_class
                iou_d += iou_d_class
                acc_d += acc_d_class
            return {'iou': iou / n, 'acc': acc / n, 'iou_d': iou_d / n, 'acc_d': acc_d / n, 'n': n}, per_class_results


    class OpenVocabEvaluator:
        """
        开放词汇评估器基类
        提供基础的评估功能和配置
        """
        def __init__(self,
                     device='cuda',
                     name="model",
                     features=None,
                     checkpoint=None,
                     debug=False,
                     stride=1,
                     save_figures=None,
                     time=False):
            """
            初始化评估器
            
            Args:
                device: 计算设备
                name: 模型名称
                features: 特征提取器
                checkpoint: 模型检查点
                debug: 是否启用调试模式
                stride: 评估步长
                save_figures: 可视化结果保存路径
                time: 是否计时
            """
            self.device = device
            self.name = name
            self.debug = debug
            self.stride = stride
            self.model = None
            self.label_id_map = None
            self.label_map = None
            self.features = features
            self.save_figures = save_figures
            self.time = time

        def reset(self, label_map, figure_path):
            """
            重置评估器状态
            
            Args:
                label_map: 标签映射
                figure_path: 可视化结果保存路径
            """
            self.label_map = label_map
            self.label_id_map = torch.tensor(self.label_map['idx'].values).to(self.device)
            self.label_mapping = {0: 'void'}
            self.label_to_color_id = np.zeros((label_map['idx'].max() + 1), dtype=int)
            self.our_label = np.zeros((label_map['idx'].max() + 1), dtype=int)
            self.label_thing_mapping = None
            
            if 'thing' in self.label_map:
                self.label_thing_mapping = {0: -1}
                for index, (i, prompt, thing, new_label) in enumerate(
                        zip(label_map['idx'], label_map['class'], label_map['thing'], label_map['label'])):
                    self.label_mapping[new_label] = prompt
                    self.label_to_color_id[new_label] = index + 1
                    self.label_thing_mapping[new_label] = thing
                    self.our_label[i] = new_label
            else:
                for index, (i, prompt) in enumerate(zip(label_map['idx'], label_map['class'])):
                    self.label_mapping[i] = prompt
                    self.label_to_color_id[i] = index + 1
                
            self.save_figures = figure_path
            os.makedirs(self.save_figures, exist_ok=True)
            
            if 'evaluated' in self.label_map:
                self.evaluated_labels = np.unique(label_map[label_map['evaluated']==1]['label'].values)
            else:
                self.evaluated_labels = label_map['id'].values

        def eval(self, dataset, visualize=False):
            """评估方法，由子类实现"""
            raise NotImplementedError()


    class OpenVocabInstancePQEvaluator(OpenVocabEvaluator):
        """
        实例级别的全景分割评估器
        继承自OpenVocabEvaluator，实现具体的评估逻辑
        """
        def __init__(self, 
                     device='cuda', 
                     name="model", 
                     debug=False, 
                     stride=1, 
                     save_figures=None, 
                     time=False,
                     save_gt=False,
                     save_remap_instance=False,
                     is_use_remap_instance=False):
            super().__init__(device=device, name=name, debug=debug, stride=stride, save_figures=save_figures, time=time)
            self.save_gt = save_gt
            self.save_remap_instance = save_remap_instance
            self.is_use_remap_instance = is_use_remap_instance
        def eval(self):
            """
            执行评估流程
            
            Returns:
                PanopticStat: 评估统计结果
            """
            # self.debug = False
            self.panoptic_stat = PanopticStat()

            # 处理所有帧
            pred_semantics, pred_instances, gt_semantics, gt_instances, indices = [], [], [], [], []
            names = []
            for frame in tqdm(sorted(os.listdir(pre_semantic_path), key=lambda x: int(x.replace("DSC", "").split('.')[0]))):
                # 读取预测结果
                pred_semantic = np.array(Image.open(os.path.join(pre_semantic_path, frame))).astype(np.int64)
                pred_instance = np.array(Image.open(os.path.join(pre_instance_path, frame))).astype(np.int64)
                
                pred_semantics.append(pred_semantic)
                pred_instances.append(pred_instance)

                # 读取真实标签
                idx = int(frame.split('.')[0].replace("DSC", ""))
                indices.append(idx)
                names.append(frame)
                gt_semantic = np.array(Image.open(os.path.join(gt_semantic_path, frame))).astype(np.int64)
                gt_instance = np.array(Image.open(os.path.join(gt_instance_path, frame))).astype(np.int64)
                
                # gt_semantic_remapping = self.our_label[gt_semantic]
                gt_semantic_remapping = gt_semantic
                gt_semantics.append(gt_semantic_remapping)
                gt_instances.append(gt_instance)
                
            # 堆叠所有帧的结果
            pred_semantics = np.stack(pred_semantics, axis=0)
            pred_instances = np.stack(pred_instances, axis=0)
            gt_semantics = np.stack(gt_semantics, axis=0)
            gt_instances = np.stack(gt_instances, axis=0)
            indices = np.array(indices)

            if self.save_remap_instance:
                gt_instance_remap_path = os.path.join(self.save_figures, 'instance_remap')
                gt_instances, gt_thing_ids = self._instance_label_remapping(gt_instances, gt_semantics)
                os.makedirs(gt_instance_remap_path, exist_ok=True)
                for i, (name, gt_instance) in enumerate(zip(names, gt_instances)):
                    cv2.imwrite(os.path.join(gt_instance_remap_path, name), gt_instance.astype(np.uint8))
                np.save(os.path.join(self.save_figures, 'gt_thing_ids.npy'), gt_thing_ids)
                exit()
            # 执行各项评估
            # self._evaluate_semantic(gt_semantics, pred_semantics, indices, names, save_gt=self.save_gt)

            if self.is_use_remap_instance:
                gt_thing_ids = np.load(os.path.join(gt_folder, 'gt_thing_ids.npy'))
            else:
                gt_instances, gt_thing_ids = self._instance_label_remapping(gt_instances, gt_semantics)

            # self._evaluate_instance(gt_instances, gt_thing_ids, pred_instances, indices, names, save_gt=self.save_gt)
            pred_instances, pred_thing_ids = self._instance_label_remapping(pred_instances, pred_semantics)
            self._evaluate_panoptic(
                pred_instances=pred_instances,
                pred_semantics=pred_semantics,
                gt_semantics=gt_semantics,
                gt_instances=gt_instances,
                indices=indices,
                names=names,
                save_gt=self.save_gt
            )

            return self.panoptic_stat

        def _instance_label_remapping(self, instances, semantics):
            """
            重新映射实例标签
            
            Args:
                instances: 实例标签图
                semantics: 语义标签图
                
            Returns:
                重映射后的实例标签和thing类别ID列表
            """
            if 'thing' not in self.label_map:
                return instances
            
            stuff_id_mapping = {}
            thing_id_list = []
            instance_ids = np.unique(instances)
            new_instance_id = np.max(instance_ids) + 1

            # 处理无效区域
            void_mask = np.isin(instances, [VOID])
            if void_mask.sum() != 0:
                s_labels = np.unique(semantics[void_mask])
                for s_id in s_labels:
                    if s_id not in self.evaluated_labels:
                        continue
                    else:
                        instances[np.logical_and(void_mask, semantics == s_id)] = new_instance_id
                        new_instance_id += 1

            # 处理每个实例
            for ins_id in instance_ids:
                if ins_id == VOID:
                    continue
                s_labels = semantics[instances == ins_id]
                s_ids, cnts = np.unique(s_labels, return_counts=True)
                s_id = s_ids[np.argmax(cnts)]
                
                if s_id not in self.evaluated_labels:
                    instances[instances == ins_id] = VOID
                elif s_id in self.evaluated_labels and self.label_thing_mapping[s_id] == 0:
                    if s_id not in stuff_id_mapping.keys():
                        stuff_id_mapping[s_id] = ins_id
                    else:
                        instances[instances == ins_id] = stuff_id_mapping[s_id]
                elif s_id in self.evaluated_labels and self.label_thing_mapping[s_id] == 1:
                    thing_id_list.append(ins_id)
                
            return instances, thing_id_list
        
        def _read_gt_panoptic_segmentation(self, semantic, instance):
            """
            读取真实全景分割结果
            
            Args:
                semantic: 语义标签图
                instance: 实例标签图
                
            Returns:
                真实全景分割结果字典
            """
            gt_segms = {}
            labels, labels_cnt = np.unique(instance, return_counts=True)

            for label, label_cnt in zip(labels, labels_cnt):
                if label == VOID:
                    continue
                semantic_ids = semantic[instance == label]
                ids, cnts = np.unique(semantic_ids, return_counts=True)
                gt_segms[label] = {
                    'area': label_cnt,
                    'category_id': ids[np.argmax(cnts)]
                }
            return gt_segms
        
        def _predict_panoptic_segmentation(self, pred_instance, pred_semantic):
            """
            预测全景分割结果
            
            Args:
                pred_instance: 预测的实例标签图
                pred_semantic: 预测的语义标签图
                
            Returns:
                预测的全景分割结果字典
            """
            pred_segms = {}
            labels, labels_cnt = np.unique(pred_instance, return_counts=True)

            for label, label_cnt in zip(labels, labels_cnt):
                if label == VOID:
                    continue
                semantic_ids = pred_semantic[pred_instance == label]
                ids, cnts = np.unique(semantic_ids, return_counts=True)
                pred_segms[label] = {
                    'area': label_cnt,
                    'category_id': ids[np.argmax(cnts)]
                }

            return pred_segms

        def _evaluate_semantic(self, gt_semantics, pred_semantics, indices, names, save_gt=False):
            """
            评估语义分割性能
            
            Args:
                gt_semantics: 真实语义标签
                pred_semantics: 预测语义标签
                indices: 帧索引
            """
            if self.debug:
                semantic_label_color_mapping = {}
                labels = np.unique(np.append(np.unique(gt_semantics), np.unique(pred_semantics)))
                for label in labels:
                    color = np.array(color_label_map[label].color) / 255.0
                    semantic_label_color_mapping[label] = color

            for gt_semantic, pred_semantic, index, name in tqdm(
                list(zip(gt_semantics, pred_semantics, indices, names)), desc="Evaluating semantic segmentation"):

                mask = np.isin(gt_semantic, self.evaluated_labels)
                labels = np.unique(gt_semantic)
                for label in labels:
                    if label not in self.evaluated_labels:
                        continue
                    object_mask = gt_semantic[mask] == label

                    # 计算语义分割指标
                    pred_mask = pred_semantic[mask] == label
                    true_positive = np.bitwise_and(pred_mask, object_mask).sum()
                    false_positive = np.bitwise_and(pred_mask, object_mask == False).sum()
                    false_negative = np.bitwise_and(pred_mask == False, object_mask).sum()

                    class_iou = float(true_positive) / (true_positive + false_positive + false_negative)
                    self.panoptic_stat[label].semantic['iou'] += class_iou
                    self.panoptic_stat[label].semantic['acc'] += float(true_positive) / (true_positive + false_negative)
                    self.panoptic_stat[label].semantic_n += 1

                if self.debug:
                    self._visualize_semantic_results(pred_semantic, gt_semantic, semantic_label_color_mapping, name, save_gt=save_gt)

        def _visualize_semantic_results(self, pred_semantic, gt_semantic, semantic_label_color_mapping, name, save_gt=False):
            """
            可视化语义分割结果并使用 cv2 保存

            Args:
                pred_semantic: 预测语义标签
                gt_semantic: 真实语义标签
                index: 帧索引
                semantic_label_color_mapping: 标签颜色映射
            """
            # 创建预测结果图像
            p_s = np.zeros((pred_semantic.shape[0], pred_semantic.shape[1], 3), dtype=np.uint8)
            labels = np.unique(pred_semantic)
            for label in labels:
                color = semantic_label_color_mapping[label] * 255
                p_s[pred_semantic == label] = color

            # 保存预测结果
            pred_save_path = os.path.join(self.save_figures, 'semantic_image_visualization', name)
            os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
            cv2.imwrite(pred_save_path, cv2.cvtColor(p_s, cv2.COLOR_RGB2BGR))

            if save_gt:
                # 创建真实标签图像
                gt_s = np.zeros((gt_semantic.shape[0], gt_semantic.shape[1], 3), dtype=np.uint8)
                labels = np.unique(gt_semantic)
                for label in labels:
                    color = semantic_label_color_mapping[label] * 255
                    gt_s[gt_semantic == label] = color
                # 保存真实标签
                gt_save_path = os.path.join(self.save_figures, 'semantic_visualization', name)
                os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
                cv2.imwrite(gt_save_path, cv2.cvtColor(gt_s, cv2.COLOR_RGB2BGR))

        def _evaluate_instance(self, gt_instances, gt_thing_ids, pred_instances, indices, names, save_gt=False):
            """
            评估实例分割性能
            
            Args:
                gt_instances: 真实实例标签
                gt_thing_ids: 真实thing类别ID列表
                pred_instances: 预测实例标签
                indices: 帧索引
            """
            if self.debug:
                pred_instance_label_color_mapping = {}
                gt_instance_label_color_mapping = {}

            print("Evaluating instance segmentation ...")
            gt_inst_ids, gt_inst_areas = np.unique(gt_instances, return_counts=True)
            for gt_inst_id, gt_inst_area in zip(gt_inst_ids, gt_inst_areas):
                if gt_inst_id not in gt_thing_ids:
                    continue
                gt_inst_mask = gt_instances == gt_inst_id
                pred_inst_ids, pred_gt_intersections = np.unique(pred_instances[np.logical_and(gt_inst_mask, pred_instances>0)], return_counts=True)
                if len(pred_gt_intersections) == 0:
                    self.panoptic_stat.instance_stat['coverage'].append(0)
                else:
                    index = np.argmax(pred_gt_intersections)
                    matched_pred_inst_id = pred_inst_ids[index]
                    matched_pred_gt_intersection = pred_gt_intersections[index]
                    matched_pred_inst_mask = pred_instances == matched_pred_inst_id
                    iou = matched_pred_gt_intersection / (np.sum(matched_pred_inst_mask) + np.sum(gt_inst_mask) - matched_pred_gt_intersection)
                    self.panoptic_stat.instance_stat['coverage'].append(iou)
                self.panoptic_stat.instance_stat['gt_inst_area'].append(gt_inst_area)
                
                if self.debug:
                    color = id2rgb(matched_pred_inst_id)
                    pred_instance_label_color_mapping[matched_pred_inst_id] = color
                    gt_instance_label_color_mapping[gt_inst_id] = color
            
            gt_inst_mask = np.isin(gt_instances, gt_thing_ids)
            pred_inst_ids = np.unique(pred_instances[gt_inst_mask])
            self.panoptic_stat.instance_stat['num_pred_inst'] += len(pred_inst_ids)
            self.panoptic_stat.instance_stat['num_gt_inst'] += len(gt_thing_ids)
            if self.debug:
                self._visualize_instance_results(gt_instances, pred_instances, names, 
                                              pred_instance_label_color_mapping, gt_instance_label_color_mapping, save_gt=save_gt)

        def _visualize_instance_results(self, gt_instances, pred_instances, names, 
                                        pred_instance_label_color_mapping, gt_instance_label_color_mapping, save_gt=False):
            """
            可视化实例分割结果并使用 cv2 保存

            Args:
                gt_instances: 真实实例标签
                pred_instances: 预测实例标签
                indices: 帧索引
                pred_instance_label_color_mapping: 预测实例颜色映射
                gt_instance_label_color_mapping: 真实实例颜色映射
            """
            for gt_instance, pred_instance, name in tqdm(
                list(zip(gt_instances, pred_instances, names)), desc="[DEBUG] visualizing"):

                # 创建预测结果图像
                p_ins = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3), dtype=np.uint8)
                labels = np.unique(pred_instance)
                for label in labels:
                    color = pred_instance_label_color_mapping.get(label, np.zeros((3,))) * 255
                    p_ins[pred_instance == label] = color

                # 保存预测结果
                pred_save_path = os.path.join(self.save_figures, 'instance_image_visualization', name)
                os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                cv2.imwrite(pred_save_path, cv2.cvtColor(p_ins, cv2.COLOR_RGB2BGR))

                if save_gt:
                    # 创建真实标签图像
                    gt_ins = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3), dtype=np.uint8)
                    labels = np.unique(gt_instance)
                    for label in labels:
                        color = gt_instance_label_color_mapping.get(label, np.zeros((3,))) * 255
                        gt_ins[gt_instance == label] = color

                    # 保存真实标签
                    gt_save_path = os.path.join(self.save_figures, 'instance_visualization', name)
                    os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
                    cv2.imwrite(gt_save_path, cv2.cvtColor(gt_ins, cv2.COLOR_RGB2BGR))

        def _evaluate_panoptic(self, pred_instances, pred_semantics, gt_semantics, gt_instances, indices, names, save_gt=False):
            """
            评估全景分割性能
            
            Args:
                pred_instances: 预测实例标签
                pred_semantics: 预测语义标签
                gt_semantics: 真实语义标签
                gt_instances: 真实实例标签
                indices: 帧索引
            """
            print("Evaluating panoptic quality ...")
            miou, nn = 0, 0
            
            gt_segms = self._read_gt_panoptic_segmentation(gt_semantics, gt_instances)
            pred_segms = self._predict_panoptic_segmentation(pred_instances, pred_semantics)
            
            if self.debug:
                pred_panoptic_label_color_mapping = {}
                gt_panoptic_label_color_mapping = {}
            
            # 计算混淆矩阵
            gt_pred_instance = gt_instances.astype(np.uint64) * OFFSET + pred_instances.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(gt_pred_instance, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # 统计匹配对
            gt_matched = set()
            pred_matched = set()

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms or pred_label not in pred_segms:
                    continue

                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue

                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
                iou = intersection / union
                if iou > 0.1:
                    miou += iou
                    nn += 1
                    self.panoptic_stat[gt_segms[gt_label]['category_id']].tp += 1
                    self.panoptic_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    if self.debug:
                        if self.label_thing_mapping[gt_segms[gt_label]['category_id']] == 0:
                            color = np.array(color_label_map[gt_segms[gt_label]['category_id']].color) / 255.0
                        elif self.label_thing_mapping[gt_segms[gt_label]['category_id']] == 1:
                            color = id2rgb(gt_label)
                        else:
                            breakpoint()
                        # green
                        # color = np.array([0, 1, 0])
                        pred_panoptic_label_color_mapping[pred_label] = color
                        gt_panoptic_label_color_mapping[gt_label] = color

            self.panoptic_stat.panoptic_miou = miou / nn if nn>0 else 0

            # 统计假负例
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                self.panoptic_stat[gt_info['category_id']].fn += 1

                if self.debug:
                    if self.label_thing_mapping[gt_info['category_id']] == 0:
                        color = np.array(color_label_map[gt_info['category_id']].color) / 255.0
                    elif self.label_thing_mapping[gt_info['category_id']] == 1:
                        color = id2rgb(gt_label)
                    else:
                        breakpoint()
                    # blue
                    # color = np.array([0, 0, 1])
                    gt_panoptic_label_color_mapping[gt_label] = color

            thing_idx = max(gt_segms.keys()) + 1
            # 统计假正例
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                if intersection / pred_info['area'] > 0.5:
                    continue
                self.panoptic_stat[pred_info['category_id']].fp += 1

                if self.debug:
                    if self.label_thing_mapping[pred_info['category_id']] == 0:
                        color = np.array(color_label_map[pred_info['category_id']].color) / 255.0
                    elif self.label_thing_mapping[pred_info['category_id']] == 1:
                        color = id2rgb(thing_idx)
                        thing_idx += 1
                    else:
                        breakpoint()
                    # red
                    # color = np.array([1, 0, 0])
                    pred_panoptic_label_color_mapping[pred_label] = color
            
            if self.debug:
                self._visualize_panoptic_results(pred_instances, pred_segms, gt_instances, gt_segms, indices,
                                              pred_panoptic_label_color_mapping, gt_panoptic_label_color_mapping, names, save_gt)
                # exit()

        def _visualize_panoptic_results(self, pred_instances, pred_segms, gt_instances, gt_segms, indices,
                                      pred_panoptic_label_color_mapping, gt_panoptic_label_color_mapping, names, save_gt=False):
            """
            可视化全景分割结果
            
            Args:
                pred_instances: 预测实例标签
                pred_segms: 预测全景分割结果
                gt_instances: 真实实例标签
                gt_segms: 真实全景分割结果
                indices: 帧索引
                pred_panoptic_label_color_mapping: 预测全景分割颜色映射
                gt_panoptic_label_color_mapping: 真实全景分割颜色映射
            """
            os.makedirs(os.path.join(self.save_figures, "panoptic_image_visualization"), exist_ok=True)
            for i, name in enumerate(tqdm(names, desc="[DEBUG] visualizing")):
                # 创建预测结果图像
                pred_instance = pred_instances[i]
                p_panop = np.zeros((pred_instance.shape[0], pred_instance.shape[1], 3), dtype=np.uint8)
                labels = np.unique(pred_instance)
                for label in labels:
                    if label == VOID:
                        continue
                    color = (pred_panoptic_label_color_mapping.get(label, np.zeros((3, ))) * 255).astype(np.uint8)
                    p_panop[pred_instance == label] = color

                # 保存预测结果
                pred_save_path = os.path.join(self.save_figures, 'panoptic_image_visualization', name)
                os.makedirs(os.path.dirname(pred_save_path), exist_ok=True)
                cv2.imwrite(pred_save_path, cv2.cvtColor(p_panop, cv2.COLOR_RGB2BGR))

                if save_gt:
                    # 创建真实标签图像
                    gt_instance = gt_instances[i]
                    gt_panop = np.zeros((gt_instance.shape[0], gt_instance.shape[1], 3), dtype=np.uint8)
                    labels = np.unique(gt_instance)
                    for label in labels:
                        if label == VOID:
                            continue
                        color = (gt_panoptic_label_color_mapping.get(label, np.zeros((3, ))) * 255).astype(np.uint8)
                        gt_panop[gt_instance == label] = color

                    # 保存真实标签
                    gt_save_path = os.path.join(self.save_figures, 'panoptic_visualization', name)
                    os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
                    cv2.imwrite(gt_save_path, cv2.cvtColor(gt_panop, cv2.COLOR_RGB2BGR))


    def print_panoptic_results(panoptic_stat, categories, label_mapping, label_thing_mapping, verbose=False):
        """
        打印全景分割评估结果
        
        Args:
            panoptic_stat: 全景分割统计信息
            categories: 类别列表
            label_mapping: 标签映射
            label_thing_mapping: 标签到thing/stuff的映射
            verbose: 是否输出详细信息
            
        Returns:
            打印的表格和JSON格式的结果
        """
        json_result = {}
        print_tables = []

        def percentage_to_string(num):
            """将数值转换为百分比字符串"""
            if num is None:
                return "N/A"
            else:
                v = num * 100
                return f"{v:.1f}"

        console = Console()
        
        # 打印全景分割结果
        pq_total_result, pq_per_class_result = panoptic_stat.pq_average(categories, label_thing_mapping, verbose=verbose)
        table = Table(show_lines=True, caption_justify='left')
        table.add_column('Class')
        table.add_column('PQ')
        table.add_column('SQ')
        table.add_column('RQ')
        table.add_column('mIoU')
        if verbose:
            table.add_column('tp')
            table.add_column('fp')
            table.add_column('fn')

        table.title = "Panoptic Evaluation"
        json_result['panoptic'] = {}
        per_class_result = {}
        for category_id in categories:
            pq_info = pq_per_class_result[category_id]
            if pq_info['valid']:
                if verbose:
                    table.add_row(label_mapping[category_id], 
                            percentage_to_string(pq_info['pq']),
                            percentage_to_string(pq_info['sq']),
                            percentage_to_string(pq_info['rq']),
                            str(pq_info['tp']),
                            str(pq_info['fp']),
                            str(pq_info['fn']))
                    per_class_result[label_mapping[category_id]] = {
                        'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100,
                        'tp': pq_info['tp'], 'fp': pq_info['fp'], 'fn': pq_info['fn']
                    }
                else:
                    table.add_row(label_mapping[category_id], 
                            percentage_to_string(pq_info['pq']),
                            percentage_to_string(pq_info['sq']),
                            percentage_to_string(pq_info['rq']))
                    per_class_result[label_mapping[category_id]] = {
                        'PQ': pq_info['pq'] * 100, 'SQ': pq_info['sq'] * 100, 'RQ': pq_info['rq'] * 100
                    }
        json_result['panoptic']['per_class_result'] = per_class_result
        
        if verbose:
            table.add_row('Total:\n{} valid panoptic categories.'.format(pq_total_result['n']),
                      percentage_to_string(pq_total_result['pq']), 
                      percentage_to_string(pq_total_result['sq']), 
                      percentage_to_string(pq_total_result['rq']),
                      '{:.1f}'.format(pq_total_result['tp']),
                      '{:.1f}'.format(pq_total_result['fp']),
                      '{:.1f}'.format(pq_total_result['fn']))
            json_result['panoptic']['total'] = {
                'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100,
                'tp': pq_total_result['tp'], 'fp': pq_total_result['fp'], 'fn': pq_total_result['fn']
            }
        else:
            table.add_row('Total:\n{} valid panoptic categories.'.format(pq_total_result['n']),
                      percentage_to_string(pq_total_result['pq']), 
                      percentage_to_string(pq_total_result['sq']), 
                      percentage_to_string(pq_total_result['rq']),
                      percentage_to_string(pq_total_result['miou']))
            json_result['panoptic']['total'] = {
                'PQ': pq_total_result['pq'] * 100, 'SQ': pq_total_result['sq'] * 100, 'RQ': pq_total_result['rq'] * 100, 
                'mIoU': pq_total_result['miou'] * 100
            }
        console.print(table)
        print_tables.append(table)

        # 打印语义分割结果
        semantic_total_result, semantic_per_class_result = panoptic_stat.semantic_average(categories)
        table = Table(show_lines=True, caption_justify='left')
        table.add_column('Class')
        table.add_column('S_iou')
        table.add_column('S_acc')
        table.add_column('S_iou_d')
        table.add_column('S_acc_d')

        table.title = "Semantic Evaluation"
        json_result['semantic'] = {}
        per_class_result = {}
        for category_id in categories:
            semantic = semantic_per_class_result[category_id]
            if semantic['valid']:
                table.add_row(label_mapping[category_id],
                        percentage_to_string(semantic['iou']),
                        percentage_to_string(semantic['acc']),
                        percentage_to_string(semantic['iou_d']),
                        percentage_to_string(semantic['acc_d']))
                per_class_result[label_mapping[category_id]] = {
                    'S_iou': semantic['iou'] * 100, 'S_acc': semantic['acc'] * 100, 
                    'S_iou_d': semantic['iou_d'] * 100, 'S_acc_d': semantic['acc_d'] * 100
                }
        json_result['semantic']['per_class_result'] = per_class_result

        table.add_row('Total:\n{} valid semantic categories'.format(semantic_total_result['n']),
                    percentage_to_string(semantic_total_result['iou']),
                    percentage_to_string(semantic_total_result['acc']),
                    percentage_to_string(semantic_total_result['iou_d']),
                    percentage_to_string(semantic_total_result['acc_d']))
        json_result['semantic']['total'] = {
            'S_iou': semantic_total_result['iou'] * 100, 'S_acc': semantic_total_result['acc'] * 100, 
            'S_iou_d': semantic_total_result['iou_d'] * 100, 'S_acc_d': semantic_total_result['acc_d'] * 100
        }
        console.print(table)
        print_tables.append(table)

        # 打印实例分割结果
        instance_result = panoptic_stat.instance_average(iou_threshold=0.1)
        table = Table(show_lines=True, caption_justify='left')
        table.add_column('mCov')
        table.add_column('mWCov')
        table.add_column('mPrec')
        table.add_column('mRec')

        table.title = "Instance Evaluation"
        table.add_row(
            percentage_to_string(instance_result['mCov']),
            percentage_to_string(instance_result['mWCov']),
            percentage_to_string(instance_result['mPrec']),
            percentage_to_string(instance_result['mRec']))
        json_result['instance'] = {
            'mCov': instance_result['mCov'] * 100, 'mWCov': instance_result['mWCov'] * 100,
            'mPrec': instance_result['mPrec'] * 100, 'mRec': instance_result['mRec'] * 100
        }
        console.print(table)
        print_tables.append(table)
        return print_tables, json_result


    def print_iou_acc_results(ious, accs, table_title="Direct"):
        """
        打印IoU和准确率结果
        
        Args:
            ious: IoU结果列表
            accs: 准确率结果列表
            table_title: 表格标题
            
        Returns:
            打印的表格
        """
        table = Table()
        table.add_column('Class')
        table.add_column('mIoU')
        table.add_column('mAcc')
        table.title = table_title

        def percentage_to_string(iou):
            """将数值转换为百分比字符串"""
            if iou is None:
                return "N/A"
            else:
                v = iou * 100
                return f"{v:.1f}"

        reduced_iou = {}
        for iou in ious:
            for key, value in iou.items():
                if key not in reduced_iou:
                    reduced_iou[key] = []
                if value is None:
                    continue
                reduced_iou[key].append(value)
        reduced_acc = {}
        for acc in accs:
            for key, value in acc.items():
                if key not in reduced_acc:
                    reduced_acc[key] = []
                if value is None:
                    continue
                reduced_acc[key].append(value)
        for key, values in reduced_iou.items():
            if key == 'total':
                continue
            mIoU = np.mean(values)
            mAcc = np.mean(reduced_acc[key])
            table.add_row(key, percentage_to_string(mIoU),
                          percentage_to_string(mAcc))

        scene_total = percentage_to_string(np.mean([r['total'] for r in ious if 'total' in r]))
        scene_total_acc = percentage_to_string(np.mean([r['total'] for r in accs if 'total' in r]))
        table.add_row('Total', scene_total, scene_total_acc)

        console = Console()
        console.print(table)
        return table


    class NumpyEncoder(json.JSONEncoder):
        """用于JSON序列化numpy类型的编码器"""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)


    def write_results(out, tables, json_result, panoptic_stat=None):
        """
        将评估结果写入文件
        
        Args:
            out: 输出目录
            tables: 评估表格
            json_result: JSON格式的结果
            panoptic_stat: 全景分割统计信息
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        dumped = json.dumps(json_result, cls=NumpyEncoder, indent=2)
        with open(out / 'semantic_results.json', 'w') as f:
            f.write(dumped)

        # with open(out / 'table.txt', 'w') as f:
        #     for table in tables:
        #         rprint(table, file=f)
        #         rprint('\n\n', file=f)


    if __name__=='__main__':
        # 创建评估器
        evaluator = OpenVocabInstancePQEvaluator(
                    name=scene_name,
                    debug=debug_flag,
                    stride=stride,
                    save_figures=vis_path,
                    time=False,
                    save_gt=save_gt,
                    save_remap_instance=save_remap_instance,
                    is_use_remap_instance=is_use_remap_instance
        )

        # 执行评估
        evaluator.reset(label_map, vis_path)
        panoptic_stat = evaluator.eval()
        print(f"语义评估类别: {evaluator.evaluated_labels}")
        
        # 打印并保存结果
        tables, json_result = print_panoptic_results(panoptic_stat, 
                                    categories=evaluator.evaluated_labels,
                                    label_mapping=evaluator.label_mapping,
                                    label_thing_mapping=evaluator.label_thing_mapping,
                                    verbose=False)
        write_results(os.path.join(result_folder), tables, json_result, panoptic_stat)

if __name__=='__main__':
    main()
