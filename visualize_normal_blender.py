#import bpy
#import numpy as np
#import os
#from mathutils import Matrix

#COLOR_DEPTH = 8
#FORMAT = 'PNG'

## 创建相机对象
#bpy.ops.object.camera_add(location=(0, 0, 0))
#cam = bpy.context.active_object
#cam.name = "Camera"

## 设置相机为活动相机
#bpy.context.scene.camera = cam

## 设置相机内参
#intrinsics = np.loadtxt(r'C:\Users\lenovo\Downloads\color_intrinsics.txt')
#cam.data.lens = intrinsics[0, 0]  # fx
#cam.data.sensor_width = 2 * intrinsics[0, 2]  # 2 * cx
#cam.data.sensor_height = 2 * intrinsics[1, 2]  # 2 * cy

## 设置相机外参
#extrinsics = np.loadtxt(r'C:\Users\lenovo\Downloads\poses\0.txt')
#extrinsics[:3, 1:3] *= -1
#cam.matrix_world = Matrix(extrinsics)

## 设置渲染分辨率
#bpy.context.scene.render.resolution_x = 640
#bpy.context.scene.render.resolution_y = 480

## 设置输出路径
#output_base_path = r'C:\Users\lenovo\Downloads'
##bpy.context.scene.render.filepath = os.path.join(output_base_path, "normal_render")

## 设置渲染引擎为 Cycles
#bpy.context.scene.render.engine = 'CYCLES'

## 启用 GPU 渲染
#bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
#bpy.context.scene.cycles.device = 'GPU'

## Background
#bpy.context.scene.render.dither_intensity = 0.0
#bpy.context.scene.render.film_transparent = True

## 启用节点树
#bpy.context.scene.use_nodes = True
#tree = bpy.context.scene.node_tree
#links = tree.links

## 清除现有节点
#for n in tree.nodes:
#    tree.nodes.remove(n)

## 添加渲染层节点
#render_layers = tree.nodes.new('CompositorNodeRLayers')
#render_layers.label = 'Custom Outputs'
#render_layers.name = 'Custom Outputs'

## 启用法线通道
#bpy.context.view_layer.use_pass_normal = True

## 设置法线输出
#normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
#normal_file_output.label = 'Normal Output'
#normal_file_output.name = 'Normal Output'
#normal_file_output.base_path = output_base_path
#normal_file_output.file_slots[0].path = "normal_"


## 添加分离RGB节点
#separate_rgb = tree.nodes.new(type="CompositorNodeSepRGBA")
#combine_rgb = tree.nodes.new(type="CompositorNodeCombRGBA")

## 对R、G、B分别做 (x+1)/2
#add_r = tree.nodes.new(type="CompositorNodeMath")
#add_r.operation = 'ADD'
#add_r.inputs[1].default_value = 1.0
#div_r = tree.nodes.new(type="CompositorNodeMath")
#div_r.operation = 'DIVIDE'
#div_r.inputs[1].default_value = 2.0

#add_g = tree.nodes.new(type="CompositorNodeMath")
#add_g.operation = 'ADD'
#add_g.inputs[1].default_value = 1.0
#div_g = tree.nodes.new(type="CompositorNodeMath")
#div_g.operation = 'DIVIDE'
#div_g.inputs[1].default_value = 2.0

#add_b = tree.nodes.new(type="CompositorNodeMath")
#add_b.operation = 'ADD'
#add_b.inputs[1].default_value = 1.0
#div_b = tree.nodes.new(type="CompositorNodeMath")
#div_b.operation = 'DIVIDE'
#div_b.inputs[1].default_value = 2.0

## 连接节点
#links.new(render_layers.outputs['Normal'], separate_rgb.inputs[0])

## R通道
#links.new(separate_rgb.outputs['R'], add_r.inputs[0])
#links.new(add_r.outputs[0], div_r.inputs[0])
#links.new(div_r.outputs[0], combine_rgb.inputs['R'])

## G通道
#links.new(separate_rgb.outputs['G'], add_g.inputs[0])
#links.new(add_g.outputs[0], div_g.inputs[0])
#links.new(div_g.outputs[0], combine_rgb.inputs['G'])

## B通道
#links.new(separate_rgb.outputs['B'], add_b.inputs[0])
#links.new(add_b.outputs[0], div_b.inputs[0])
#links.new(div_b.outputs[0], combine_rgb.inputs['B'])

## 合并后输出
#links.new(combine_rgb.outputs[0], normal_file_output.inputs[0])

## 删除不需要的对象
#objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
#bpy.ops.object.delete({"selected_objects": objs})

## 渲染图像
#bpy.ops.render.render(write_still=True)

import bpy
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

# 创建法线材质
mat = bpy.data.materials.new(name="NormalViewspaceMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
for n in nodes:
    nodes.remove(n)
geometry = nodes.new(type="ShaderNodeNewGeometry")
vec_transform = nodes.new(type="ShaderNodeVectorTransform")
vec_transform.vector_type = 'NORMAL'
vec_transform.convert_from = 'WORLD'
vec_transform.convert_to = 'CAMERA'
invert_y = nodes.new(type="ShaderNodeVectorMath")
invert_y.operation = 'MULTIPLY'
invert_y.inputs[1].default_value = (1, 1, -1)
add = nodes.new(type="ShaderNodeVectorMath")
add.operation = 'ADD'
add.inputs[1].default_value = (1, 1, 1)
divide = nodes.new(type="ShaderNodeVectorMath")
divide.operation = 'DIVIDE'
divide.inputs[1].default_value = (2, 2, 2)
emission = nodes.new(type="ShaderNodeEmission")
output = nodes.new(type="ShaderNodeOutputMaterial")
links.new(geometry.outputs['Normal'], vec_transform.inputs['Vector'])
links.new(vec_transform.outputs['Vector'], invert_y.inputs[0])
links.new(invert_y.outputs[0], add.inputs[0])
links.new(add.outputs[0], divide.inputs[0])
links.new(divide.outputs[0], emission.inputs['Color'])
links.new(emission.outputs['Emission'], output.inputs['Surface'])

# 给所有Mesh物体赋上该材质
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(mat)

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
