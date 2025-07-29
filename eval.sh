#!/bin/bash

module load cuda/cuda-12.1.0

cd /home/ja882177/dso/gits/paligemma-video

# # with mm_proj only
# python evaluate_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_mm_proj \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_mm_proj"

# with mm_proj only random 1000 samples
python evaluate_rand_AG_v5_3_paligemma_finetune.py \
    --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_mm_proj \
    --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_mm_proj_1000samples" \
    --samples_to_process=1000