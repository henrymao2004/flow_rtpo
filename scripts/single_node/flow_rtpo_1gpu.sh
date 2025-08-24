#!/bin/bash
# Flow-RTPO Training Script (1-GPU version for debugging)
# Single-GPU training for Flow-RTPO hierarchical red teaming
set -e

# Configuration
CONFIG_NAME="flow_rtpo_debug"  # or "flow_rtpo_debug" for testing

# Memory optimization: Reduce startup memory pressure
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=0
export CUDA_MODULE_LOADING=LAZY
export TORCH_USE_CUDA_DSA=1
# Additional memory optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Set environment variables for single-GPU
export CUDA_VISIBLE_DEVICES=0  # Single GPU only
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo"

# Optional: Set master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=0

# Create output directory
OUTPUT_DIR="logs/flow_rtpo_1gpu/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO Single-GPU Training (for debugging)..."
echo "Config: $CONFIG_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: 1"

# Check GPU status before training
echo "=== GPU Status Before Training ==="
nvidia-smi
echo "=================================="

# Run training with accelerate (single-GPU)
# All accelerate parameters must come before the script path
accelerate launch \
    --num_processes=1 \
    --multi_gpu=false \
    --gpu_ids=0 \
    --mixed_precision=bf16 \
    /workspace/flow_rtpo/scripts/train_flow_rtpo.py \
    --config=/workspace/flow_rtpo/config/flow_rtpo.py:$CONFIG_NAME \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Logs saved to: $OUTPUT_DIR" 