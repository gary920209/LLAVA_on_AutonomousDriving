#!/bin/bash
export TMPDIR=/mnt/HDD_1/walker/

CUDA_VISIBLE_DEVICES=1,2 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /mnt/HDD_1/walker/dlcv_json_files/train.json \
    --val_data_path /mnt/HDD_1/walker/dlcv_json_files/val.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --tune_mm_mlp_adapter True \
    --bb_projector_type mlp2x_gelu \
    --bb_input_dim 35 \
    --tune_bbox_encoder True \
    --bb_encoder_lr 1e-3 \
    --bf16 True \
    --output_dir /mnt/HDD_1/walker/dlcv_checkpoints/llava-v1.5-7b-pretrain-lora-1222 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
