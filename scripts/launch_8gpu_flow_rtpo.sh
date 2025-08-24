#!/bin/bash

# Launch script for 8-GPU Flow RTPO with device isolation
# GPUs 0-3: SD3.5 pipeline + LoRA under DDP
# GPUs 4-5: Reward VLM (LLaVA-7B) sharded
# GPU 6: Prompt-editor aux models (SBERT + vec2text)
# GPU 7: spare / eval

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch with 4 DDP ranks for SD3 on GPUs 0-3
accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port 29500 \
    scripts/train_flow_rtpo.py \
    --config config/flow_rtpo.py 