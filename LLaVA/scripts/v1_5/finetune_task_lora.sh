#!/bin/bash
export TMPDIR=/mnt/HDD_1/walker/
# --data_path ntudlcv/dlcv_2024_final1

CUDA_VISIBLE_DEVICES=0,1,2 \
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 4 --lora_alpha 64 --use_dora False --mm_projector_lr 5e-5 --bb_encoder_lr 5e-5 \
    --deepspeed ./scripts/zero2.json\
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /mnt/HDD_1/walker/dlcv_json_files/train.json \
    --val_data_path /mnt/HDD_1/walker/dlcv_json_files/val.json \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --bb_projector_type mlp2x_gelu \
    --bb_input_dim 35 \
    --tune_bbox_encoder False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --pretrain_bbox_encoder /mnt/HDD_1/walker/dlcv_checkpoints/llava-v1.5-7b-pretrain-1224/checkpoint-900/bbox_encoder.bin \
    --output_dir /mnt/HDD_1/walker/dlcv_checkpoints/llava-v1.5-7b-finetune-lora-1225 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --max_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb