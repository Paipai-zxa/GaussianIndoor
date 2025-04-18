for scene in 0721_00
do 
    python pre_scannetv2.py --data_path /data1/zxa/Data/ScanNetV2/scans --output_path /data1/zxa/GaussianIndoor/data --scene scene$scene
    mv /data1/zxa/GaussianIndoor/data/scene$scene /data1/zxa/GaussianIndoor/data/$scene
    # python depth_reproject.py --data_path /data1/zxa/GaussianIndoor/data/$scene
done
