import bpy
# 删除所有对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

import numpy as np
from mathutils import Matrix
import sys
argv = sys.argv
# print(argv)
argv = argv[argv.index("--") + 1:]  # get all args after "--"
import os

scene_path = argv[0]
mesh_path = argv[1]
output_path = argv[2]

# 相机内参
intrinsics = np.loadtxt(os.path.join(scene_path, 'color_intrinsics.txt'))
poses_files = sorted(os.listdir(os.path.join(scene_path, 'poses')), key=lambda x: int(x.replace("DSC", "").split('.')[0]))
image_names = sorted(os.listdir(os.path.join(scene_path, 'images')), key=lambda x: int(x.replace("DSC", "").split('.')[0]))

# 创建相机
bpy.ops.object.camera_add(location=(0, 0, 0))
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = intrinsics[0, 0]
cam.data.sensor_width = 2 * intrinsics[0, 2]
cam.data.sensor_height = 2 * intrinsics[1, 2]
cam.scale = (1, 1, 1)

# 添加点光源
bpy.ops.object.light_add(type='POINT', location=(2.5, 2.5, 10))
light = bpy.context.active_object
light.data.energy = 5000  # 调整光源强度

# 渲染设置
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_depth = '16'
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.render.film_transparent = True
bpy.context.scene.view_settings.view_transform = 'Raw'

bpy.ops.import_mesh.ply(filepath=mesh_path)

mesh = bpy.context.selected_objects[1]

# 创建材质
material = bpy.data.materials.new(name="VertexColorMaterial")
material.use_nodes = True
bsdf = material.node_tree.nodes.get("Principled BSDF")

# 添加 Vertex Color 节点
vertex_color_node = material.node_tree.nodes.new(type='ShaderNodeVertexColor')
vertex_color_node.layer_name = "Col"  # 确保名称与 PLY 文件中的颜色属性一致

# 连接节点
material.node_tree.links.new(vertex_color_node.outputs['Color'], bsdf.inputs['Base Color'])

# 将材质赋予网格
if mesh.data.materials:
    mesh.data.materials[0] = material
else:
    mesh.data.materials.append(material)

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

# 循环渲染每一帧
for pose_file, image_name in zip(poses_files, image_names):
    extrinsics = np.loadtxt(os.path.join(scene_path, 'poses', pose_file))
    extrinsics[:3, 1:3] *= -1
    cam.matrix_world = Matrix(extrinsics)
    out_file = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.png")
    bpy.context.scene.render.filepath = out_file
    bpy.ops.render.render(write_still=True)
