#!/bin/bash

model_path="dlcv_checkpoints/llava-v1.5-7b-finetune-with-pretrain-lora-1223"
# model_path="liuhaotian/llava-v1.5-7b"
image_folder="dlcv_test_data/test_images"
output_json="submission.json"

base_model="liuhaotian/llava-v1.5-7b"

CUDA_VISIBLE_DEVICES=1,2 \
python LLaVA/llava/eval/gen_car_output.py \
    --model-base $base_model \
    --model-path $model_path \
    --image-folder $image_folder \
    --output_file $output_json \
