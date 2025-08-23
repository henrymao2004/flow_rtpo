#!/bin/bash
# Flow-RTPO Training Script
# Multi-GPU training for Flow-RTPO hierarchical red teaming
set -e

# Configuration
CONFIG_NAME="flow_rtpo_sd3"  # or "flow_rtpo_debug" for testing
# ACCELERATE_CONFIG="./accelerate_configs/multi_gpu.yaml"

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Three GPUs
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo"

# Optional: Set master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# Create output directory
OUTPUT_DIR="logs/flow_rtpo/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO Multi-GPU Training..."
echo "Config: $CONFIG_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: 7"

# Run training with accelerate (multi-GPU)
accelerate launch \
    --config_file=$ACCELERATE_CONFIG \
    --num_processes=7 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29501 \
    --multi_gpu \
    /root/autodl-tmp/flow_grpo/scripts/train_flow_rtpo.py \
    --config=/root/autodl-tmp/flow_grpo/config/flow_rtpo.py:$CONFIG_NAME \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Logs saved to: $OUTPUT_DIR"