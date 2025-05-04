import os
from argparse import ArgumentParser
import cv2 as cv
import numpy as np

parser = ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--scene", type=str, required=True)

args = parser.parse_args()
image_path = os.path.join(args.data_path, args.scene, "color")
instance_path = os.path.join(args.data_path, args.scene, "instance")
semantic_path = os.path.join(args.data_path, args.scene, "sematic")
depth_path = os.path.join(args.data_path, args.scene, "depth")
color_intrinsics_path = os.path.join(args.data_path, args.scene, "intrinsic", "intrinsic_color.txt")
depth_intrinsics_path = os.path.join(args.data_path, args.scene, "intrinsic", "intrinsic_depth.txt")
pose_path = os.path.join(args.data_path, args.scene, "pose")

os.makedirs(os.path.join(args.output_path, args.scene), exist_ok=True)
os.makedirs(os.path.join(args.output_path, args.scene, "images"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, args.scene, "depths"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, args.scene, "poses"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, args.scene, "instance"), exist_ok=True)
os.makedirs(os.path.join(args.output_path, args.scene, "semantic"), exist_ok=True)
# os.system(f"touch {os.path.join(args.output_path, args.scene, 'color_intrinsics.txt')}")

image = cv.imread(os.path.join(image_path, "0.jpg"))
origin_width, origin_height = image.shape[1], image.shape[0]
crop_width, crop_height = 1248, 936
width, height = 640, 480
reso = crop_width / width
crop_width_half = (origin_width - crop_width) // 2
crop_height_half = (origin_height - crop_height) // 2

color_intrinsics = np.loadtxt(color_intrinsics_path).reshape(4, 4)
# 主点偏移
color_intrinsics[0, 2] = color_intrinsics[0, 2] - crop_width_half
color_intrinsics[1, 2] = color_intrinsics[1, 2] - crop_height_half
color_intrinsics_help = color_intrinsics[0:2, 0:3] / reso
color_intrinsics[0:2, 0:3] = color_intrinsics_help
# 存储intrinsics
# np.savetxt(os.path.join(args.output_path, args.scene, "color_intrinsics.txt"), color_intrinsics, fmt="%.6f")
# os.system(f"cp {depth_intrinsics_path} {os.path.join(args.output_path, args.scene, 'depth_intrinsics.txt')}")

# 每6张图片选择一张到output
image_list = os.listdir(image_path)
for image in image_list:
    if image.endswith(".jpg"):
        image_index = int(image.split(".")[0])
        if image_index % 6 == 0:
            # 读取该图片
            # image_file = os.path.join(image_path, image)
            # image_origin = cv.imread(image_file)
            # image_crop = image_origin[crop_height_half:crop_height_half + crop_height, crop_width_half:crop_width_half + crop_width]
            # image_resize = cv.resize(image_crop, (width, height))
            # cv.imwrite(os.path.join(args.output_path, args.scene, "images", f"{image_index}.png"), image_resize)

            instance_file = os.path.join(instance_path, image.replace("jpg", "png"))
            instance_origin = cv.imread(instance_file, cv.IMREAD_UNCHANGED)
            instance_crop = instance_origin[crop_height_half:crop_height_half + crop_height, crop_width_half:crop_width_half + crop_width]
            instance_resize = cv.resize(instance_crop, (width, height), interpolation=cv.INTER_NEAREST)
            cv.imwrite(os.path.join(args.output_path, args.scene, "instance", f"{image_index}.png"), instance_resize)

            semantic_file = os.path.join(semantic_path, image.replace("jpg", "png"))
            semantic_origin = cv.imread(semantic_file, cv.IMREAD_UNCHANGED)
            semantic_crop = semantic_origin[crop_height_half:crop_height_half + crop_height, crop_width_half:crop_width_half + crop_width]
            semantic_resize = cv.resize(semantic_crop, (width, height), interpolation=cv.INTER_NEAREST)
            cv.imwrite(os.path.join(args.output_path, args.scene, "semantic", f"{image_index}.png"), semantic_resize)

            # # 复制pose与depth
            # pose_file = f"{image_index}.txt"
            # depth_file = f"{image_index}.png"
            # pose_ori_path = os.path.join(pose_path, pose_file)
            # depth_ori_path = os.path.join(depth_path, depth_file)
            # pose_des_path = os.path.join(args.output_path, args.scene, "poses", pose_file)
            # depth_des_path = os.path.join(args.output_path, args.scene, "depths", depth_file)
            # os.system(f"cp {pose_ori_path} {pose_des_path}")
            # os.system(f"cp {depth_ori_path} {depth_des_path}")




















