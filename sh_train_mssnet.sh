#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_dp.py \
                --train_datalist './datalist/datalist_gopro_train.txt'\
                --data_root_dir './dataset/GOPRO_Large/train'\
                --checkdir './checkpoint/MSSNet'\
                --max_epoch 3000\
                --wf 54\
                --scale 42\
                --vscale 42\
                --mgpu
