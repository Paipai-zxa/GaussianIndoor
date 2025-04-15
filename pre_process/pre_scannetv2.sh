for scene in 0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00
do 
    # python pre_scannetv2.py --data_path /data1/zxa/Data/ScanNetV2/scans --output_path /data1/zxa/GaussianIndoor/data --scene scene$scene
    # mv /data1/zxa/GaussianIndoor/data/scene$scene /data1/zxa/GaussianIndoor/data/$scene
    python depth_reproject.py --data_path /data1/zxa/GaussianIndoor/data/$scene
done
