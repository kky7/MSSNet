#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
                --test_datalist './datalist/datalist_gopro_testset.txt'\
                --data_root_dir './dataset'\
                --load_dir './checkpoint/MSSNet/model_03000E.pt'\
                --outdir './result/MSSNet'\
                --wf 54\
                --scale 42\
                --vscale 42\
                --is_eval\
                --is_save
