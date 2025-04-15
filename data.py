import os

source_path = "/data1/zxa/Data/ScanNetV2/scans"
target_path = "/data1/zxa/GaussianIndoor/data"

# scene_list = ['0050_00', '0085_00', '0114_02', '0580_00', '0603_00', '0616_00', '0617_00']
scene_list = ['0085_00']


for scene in scene_list:
    scenes_path = os.path.join(source_path, f"scene{scene}")
    images_path = os.path.join(scenes_path, "color")
    # poses_path = os.path.join(scenes_path, "pose")
    target_scenes_path = os.path.join(target_path, scene)
    target_images_path = os.path.join(target_scenes_path, "images")
    # target_poses_path = os.path.join(target_scenes_path, "poses")
    os.makedirs(target_scenes_path, exist_ok=True)
    os.makedirs(target_images_path, exist_ok=True)
    # os.makedirs(target_poses_path, exist_ok=True)

    image_list = os.listdir(images_path)
    # pose_list = os.listdir(poses_path)
    for image in image_list:
        image_path = os.path.join(images_path, image)
        target_image_path = os.path.join(target_images_path, image)
        image_name = image.split(".")[0]
        if int(image_name) % 6 == 0:
            os.system(f"cp {image_path} {target_image_path}")
        
    # for pose in pose_list:
    #     pose_path = os.path.join(poses_path, pose)
    #     target_pose_path = os.path.join(target_poses_path, pose)
    #     pose_name = pose.split(".")[0]
    #     if int(pose_name) % 6 == 0:
    #         os.system(f"cp {pose_path} {target_pose_path}")