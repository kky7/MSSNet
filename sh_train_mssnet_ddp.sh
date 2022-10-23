#!/bin/bash

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export NODE_RANK=0
export WORLD_SIZE=$(($NUM_NODES * $NUM_GPUS_PER_NODE))

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    run_ddp.py  --train_datalist './datalist/datalist_gopro_train.txt'\
                --data_root_dir './dataset/GOPRO_Large/train'\
                --checkdir './checkpoint/MSSNet'\
                --max_epoch 3000\
                --wf 54\
                --scale 42\
                --vscale 42
