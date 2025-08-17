#!/bin/bash

module load cuda/cuda-12.1.0

cd /home/ja882177/dso/gits/paligemma-video

# # with mm_proj only and vision 1000 samples
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_vision_mm_proj \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_mm_proj_vision_1000samples"

# with mm_proj only random 1000 samples
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_mm_proj \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_mm_proj_1000samples" \
#     --samples_to_process=1000

# with attn layers lora random 1000 samples
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_attn_layers \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_attn_layers_1000samples" \
#     --samples_to_process=1000


# with just lora random 1000 samples
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5 \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_justlora_1000samples" \
#     --samples_to_process=1000

# with just vision and lora
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_vision \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_vision_lora_1000samples" \
#     --samples_to_process=1000

# with vision lora ep02
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_vision_e02 \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_vision_lora_1000samples_e02" \
#     --samples_to_process=1000


# # with vision + mm_proj + attn lora ep01
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_lora_vision_mmproj_attn \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_lora_vision_mmproj_attn_1000samples" \
#     --samples_to_process=1000

# ## not passing <image> token explicitly 
# python evaluate_rand_AG_v5_3_paligemma_finetune.py \
#     --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_lora_wo_custom_imgtoken \
#     --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_lora_wo_custom_imgtoken_1000samples" \
#     --samples_to_process=1000

## /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_lora_vision_mmproj_attn_proj
python evaluate_rand_AG_v5_3_paligemma_finetune.py \
    --model-path /home/ja882177/dso/gits/paligemma-video/checkpoints/paligemma2-lora_r256_a512_lr1e-5_with_lora_vision_mmproj_attn_proj \
    --output_dir="/home/ja882177/dso/gits/paligemma-video/results/outputs_with_lora_vision_mmproj_attn_proj_1000samples" \
    --samples_to_process=1000