import os

depths_path = "/data1/zxa/GaussianIndoor/data/0721_00/depths"
poses_path = "/data1/zxa/GaussianIndoor/data/0721_00/poses"

depth_list = os.listdir(depths_path)
pose_list = os.listdir(poses_path)

for depth in depth_list:
    depth_path = os.path.join(depths_path, depth)
    depth_name = depth.split(".")[0]
    depth_name = int(depth_name)
    depth_name = f"{depth_name}.png"
    target_depth_path = os.path.join(depths_path, depth_name)
    os.system(f"mv {depth_path} {target_depth_path}")

for pose in pose_list:
    pose_path = os.path.join(poses_path, pose)
    pose_name = pose.split(".")[0]
    pose_name = int(pose_name)
    pose_name = f"{pose_name}.txt"
    target_pose_path = os.path.join(poses_path, pose_name)
    os.system(f"mv {pose_path} {target_pose_path}")