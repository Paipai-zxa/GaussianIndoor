import open3d as o3d
import numpy as np
import os
import sys

# 获取命令行参数
argv = sys.argv
argv = argv[argv.index("--") + 1:]
scene_path = argv[0]
mesh_path = argv[1]
output_path = argv[2]

# 加载PLY文件
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()

# 设置相机内参
intrinsics = np.loadtxt(os.path.join(scene_path, 'color_intrinsics.txt'))
width, height = 640, 480
fx, fy = intrinsics[0, 0], intrinsics[1, 1]
cx, cy = intrinsics[0, 2], intrinsics[1, 2]

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(width=width, height=height, visible=False)
vis.add_geometry(mesh)

# 设置渲染选项
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # 黑色背景

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

# 循环渲染每一帧
poses_files = sorted(os.listdir(os.path.join(scene_path, 'poses')), key=lambda x: int(x.replace("DSC", "").split('.')[0]))
image_names = sorted(os.listdir(os.path.join(scene_path, 'images')), key=lambda x: int(x.replace("DSC", "").split('.')[0]))

for pose_file, image_name in zip(poses_files, image_names):
    extrinsics = np.loadtxt(os.path.join(scene_path, 'poses', pose_file))
    extrinsics[:3, 1:3] *= -1

    # 设置相机参数
    ctr = vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = extrinsics
    param.intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    ctr.convert_from_pinhole_camera_parameters(param)

    # 渲染并保存图像
    out_file = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.png")
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_file)

vis.destroy_window()