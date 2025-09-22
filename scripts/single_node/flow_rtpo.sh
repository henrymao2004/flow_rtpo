#!/bin/bash
# Flow-RTPO Training Script
# Single Node 8-GPU training for Flow-RTPO hierarchical red teaming
set -e

# Configuration - choose between debug and large scale
CONFIG_NAME="flow_rtpo_large"  # Options: "flow_rtpo_debug", "flow_rtpo_large", "flow_rtpo_sd3"

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

# NCCL Configuration - Extended timeout for long training
export NCCL_TIMEOUT=10800      # 3 hours timeout (aligned with InitProcessGroupKwargs)
export NCCL_IB_TIMEOUT=10800   # InfiniBand timeout
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export NCCL_IB_HCA=mlx5
export NCCL_IB_DISABLE=1       # Disable InfiniBand for single node
export NCCL_P2P_DISABLE=0      # Enable P2P for better GPU communication
export NCCL_SHM_DISABLE=0      # Enable shared memory
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_CROSS_NIC=0
export NCCL_BUFFSIZE=2097152
export NCCL_RINGS=4

# Additional NCCL debugging and reliability settings
export NCCL_DEBUG=INFO          # Enable NCCL debugging to track communication issues
export NCCL_DEBUG_SUBSYS=COLL   # Debug collective operations specifically
export NCCL_SOCKET_NTHREADS=8   # Increase socket threads
export NCCL_NSOCKS_PERTHREAD=8  # Increase sockets per thread

# Set environment variables for 8-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All 8 GPUs
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo"


# Create output directory
OUTPUT_DIR="logs/flow_rtpo/$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "Starting Flow-RTPO Single Node 8-GPU Training..."
echo "Config: $CONFIG_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: 8"
echo "NCCL Timeout: $NCCL_TIMEOUT seconds (3 hours)"

# Check GPU status before training
echo "=== GPU Status Before Training ==="
nvidia-smi
echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "=================================="

# Run training with accelerate (8-GPU single node)
accelerate launch \
    --num_machines=1 \
    --num_processes=8 \
    --gpu_ids=0,1,2,3,4,5,6,7 \
    --mixed_precision=fp16 \
    --main_process_port=$MASTER_PORT \
    scripts/train_flow_rtpo.py \
    --config_file="config/flow_rtpo.py:$CONFIG_NAME" \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed. Logs saved to: $OUTPUT_DIR"

# Final GPU status check
echo "=== GPU Status After Training ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "=================================="