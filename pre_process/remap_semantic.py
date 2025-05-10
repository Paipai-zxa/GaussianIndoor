import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

def remap_semantic(source_path, scene):
    semantic_dir = os.path.join(source_path, scene, "semantic_src")
    output_dir = os.path.join(source_path, scene, "semantic")
    os.makedirs(output_dir, exist_ok=True)

    label_map_path = os.path.join(source_path, scene, f"{scene}_map.csv")
    if not os.path.exists(label_map_path):
        print(f"Label map not found: {label_map_path}")
        return

    label_map_df = pd.read_csv(label_map_path)
    label_map = np.zeros((label_map_df['idx'].max() + 1), dtype=int)
    for idx, new_label in zip(label_map_df['idx'], label_map_df['label']):
        label_map[idx] = new_label

    for fname in os.listdir(semantic_dir):
        if not fname.endswith(".png"):
            continue
        semantic_path = os.path.join(semantic_dir, fname)
        semantic_gt = np.array(Image.open(semantic_path))
        remapped = label_map[semantic_gt]
        out_path = os.path.join(output_dir, fname)
        Image.fromarray(remapped.astype(np.uint8)).save(out_path)
        print(f"Remapped {fname} -> {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="数据根目录")
    parser.add_argument("--scene", type=str, required=True, help="场景名")
    args = parser.parse_args()
    remap_semantic(args.source_path, args.scene)