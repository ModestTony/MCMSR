# !/bin/bash

# Train
CUDA_VISIBLE_DEVICES=0 python mmsr/train.py -opt "options/train/stage3_restoration_mse.yml"
# Test
#CUDA_VISIBLE_DEVICES=0 python mmsr/test.py -opt "options/test/test_C2_matching_mse.yml"
