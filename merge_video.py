import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import subprocess

def get_sorted_filenames(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort()
    return files

def resize_img(img, max_width=1280):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='render_traj目录')
    parser.add_argument('--scene_name', type=str, required=True, help='场景名称')
    parser.add_argument('--fps', type=int, default=24, help='视频帧率')
    parser.add_argument('--max_width', type=int, default=1280, help='最大宽度，适配网页')
    args = parser.parse_args()

    subfolders = [
        "rgb_locate",
        'rgb',
        'depth',
        'normal',
        "semantic_locate",
        'instance_image_visualization',
        'semantic_image_visualization',
        'panoptic_image_visualization',
    ]
    subfolders = [os.path.join(args.input_dir, s) for s in subfolders]

    frame_names = get_sorted_filenames(subfolders[0])

    # 读取第一帧确定尺寸
    imgs = [cv2.imread(os.path.join(subfolders[i], frame_names[0])) for i in range(8)]
    h, w = imgs[0].shape[:2]
    row1 = np.hstack([resize_img(img, args.max_width//2) for img in imgs[:4]])
    row2 = np.hstack([resize_img(img, args.max_width//2) for img in imgs[4:8]])
    grid_img = np.vstack([row1, row2])
    h, w = grid_img.shape[:2]

    # 保存第一帧为 poster
    poster_path = os.path.join(args.input_dir, f'{args.scene_name}.jpg')
    cv2.imwrite(poster_path, grid_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # 视频写入器，H.264编码
    video_path = os.path.join(args.input_dir, f'{args.scene_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 若支持H264可用'X264'或'avc1'
    out = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

    for name in tqdm(frame_names):
        imgs = []
        for folder in subfolders:
            img_path = os.path.join(folder, name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((h//2, w//2, 3), dtype=np.uint8)
            img = resize_img(img, args.max_width//2)
            imgs.append(img)
        row1 = np.hstack(imgs[:4])
        row2 = np.hstack(imgs[4:8])
        grid_img = np.vstack([row1, row2])
        out.write(grid_img)

    out.release()

    # 自动用ffmpeg压缩并转码为H264格式
    web_video_path = video_path.replace('.mp4', '_web.mp4')
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-vcodec', 'libx264', '-crf', '28', '-preset', 'veryfast',
        web_video_path
    ])

if __name__ == '__main__':
    main()