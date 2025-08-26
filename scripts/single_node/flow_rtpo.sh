#!/bin/bash
# Flow-RTPO Training Script
# Multi-GPU training for Flow-RTPO hierarchical red teaming
set -e

# Configuration
CONFIG_NAME="flow_rtpo_debug"  # or "flow_rtpo_debug" for testing
ACCELERATE_CONFIG="/workspace/flow_rtpo/scripts/accelerate_configs/multi_gpu.yaml"

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

# NCCL Configuration - Fix timeout issues
export NCCL_TIMEOUT=1800  # 30 minutes timeout
export NCCL_IB_TIMEOUT=1800
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_CROSS_NIC=0
export NCCL_BUFFSIZE=2097152
export NCCL_NTHREADS=8
export NCCL_RINGS=4

# Set environment variables for multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Eight GPUs
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo"

# Optional: Set master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=0

# Create output directory
OUTPUT_DIR="logs/flow_rtpo/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO Multi-GPU Training..."
echo "Config: $CONFIG_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: 8"

# Check GPU status before training
echo "=== GPU Status Before Training ==="
nvidia-smi
echo "=================================="

# Run training with accelerate (multi-GPU)
# All accelerate parameters must come before the script path
accelerate launch \
    --num_processes=8 \
    --multi_gpu \
    --gpu_ids=0,1,2,3,4,5,6,7 \
    --mixed_precision=bf16 \
    /workspace/flow_rtpo/scripts/train_flow_rtpo.py \
    --config=/workspace/flow_rtpo/config/flow_rtpo.py:$CONFIG_NAME \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Logs saved to: $OUTPUT_DIR"