#!/bin/bash

model_path="liuhaotian/llava-v1.5-7b"
image_folder="/mnt/HDD_1/walker/dlcv_test_data/test_images"
output_json="output.json"

base_model="liuhaotian/llava-v1.5-7b"

CUDA_VISIBLE_DEVICES=1,2 \
python LLaVA/llava/eval/gen_car_output.py \
    --model-path $model_path \
    --image-folder $image_folder \
    --output_file $output_json \