#!/bin/bash
cd /home/users/acc123/InstrumentSegmentationSurgery
source /usr/project/xtmp/acc123/venv/bin/activate

BASE="--processed_dir /home/users/acc123/InstrumentSegmentationSurgery/processed \
      --output_dir /home/users/acc123/InstrumentSegmentationSurgery/runs \
      --num_workers 2 --batch_size 4 --img_size 384 --epochs 10"

echo "=== SegFormer AdamW ==="
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --model segformer --optimizer adamw --lr 1e-4 $BASE

echo "=== SegFormer CE only ==="
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --model segformer --optimizer adamw --lr 1e-4 \
    --ce_weight 1.0 --dice_weight 0.0 $BASE

echo "=== SegFormer frozen ==="
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
    --model segformer --optimizer adamw --lr 1e-4 \
    --freeze_encoder $BASE

echo "=== ALL DONE ==="
