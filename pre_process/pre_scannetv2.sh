scene_list=(0087_02 0088_00 0420_01 0628_02)

for scene in ${scene_list[@]}
do 
    # rm -rf ./data/$scene
    python ./pre_process/pre_scannetv2.py \
    --data_path /data1/zxa/Data/selected_pan_ScanNetV2 \
    --output_path ./data \
    --scene scene$scene
    # mv ./data/scene$scene ./data/$scene
    # python ./pre_process/depth_reproject.py --data_path ./data/$scene 
    # python ./utils/depth2point_utils.py --scene_dir ./data/$scene 
    # cp -r "/data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/scene${scene}_vh_clean_2.ply" "./data/$scene/${scene}_vh_clean_2.ply"

    # rm -rf /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/instance
    # rm -rf /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/sematic
    # unzip /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/scene${scene}_2d-instance-filt.zip -d /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/
    # unzip /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/scene${scene}_2d-label-filt.zip -d /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/
    # mv /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/instance-filt /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/instance
    # mv /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/label-filt /data1/zxa/Data/selected_pan_ScanNetV2/scene$scene/sematic
done
