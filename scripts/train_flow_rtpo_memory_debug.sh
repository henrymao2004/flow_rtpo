#!/bin/bash

# Set CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0

# Clear CUDA cache before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

echo "Starting Flow RTPO training with memory optimization and CUDA debugging..."
echo "CUDA_LAUNCH_BLOCKING=1"
echo "TORCH_USE_CUDA_DSA=1"
echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"

# Run the training script with memory-optimized configuration
python train_flow_rtpo.py \
    --config=flow_rtpo_debug \
    --num_epochs=5 \
    --save_freq=2 \
    --eval_freq=2 