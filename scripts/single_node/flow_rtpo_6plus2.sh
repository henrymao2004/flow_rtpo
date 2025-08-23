#!/bin/bash
# Flow-RTPO Training Script for 6+2 GPU Configuration
# 6 GPUs for training + 2 GPUs for reward computation
set -e

# Configuration
CONFIG_NAME="flow_rtpo_sd3"  # or "flow_rtpo_debug" for testing
ACCELERATE_CONFIG="workspace/flow_grpo/scripts/accelerate_configs/flow_rtpo_6plus2.yaml"

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 GPUs (6 for training, 2 for reward)
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo_6plus2"

# Optional: Set master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29501

# Create output directory
OUTPUT_DIR="logs/flow_rtpo_6plus2/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO 6+2 GPU Training..."
echo "Config: $CONFIG_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training GPUs: 0,1,2,3,4,5 (6 GPUs)"
echo "Reward GPUs: 6,7 (2 GPUs)"
echo "Number of training processes: 6"

# Run training with accelerate (6 GPUs for training)
accelerate launch \
    --config_file=$ACCELERATE_CONFIG \
    --num_processes=6 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_port=29501 \
    --multi_gpu \
    /root/autodl-tmp/flow_grpo/scripts/train_flow_rtpo.py \
    --config=/root/autodl-tmp/flow_grpo/config/flow_rtpo.py:$CONFIG_NAME \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Logs saved to: $OUTPUT_DIR" 