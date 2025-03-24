iterations=30000
export CUDA_VISIBLE_DEVICES=0

# for scene in 0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00
for scene in 0603_00 0616_00 0617_00
do 
    data_path=data/${scene}
    output_path=output/${scene}
    python train.py \
        -s ${data_path} \
        -m ${output_path} \
        --iterations ${iterations} \
        --eval
    python render.py \
        -m ${output_path} \
        --iteration ${iterations} \
        --skip_train \
        --eval
    python metrics.py \
        -m ${output_path}
done

