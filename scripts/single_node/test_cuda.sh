#!/bin/bash
# CUDA Test Script
set -e

echo "=== CUDA Test Script ==="

# Set the same environment variables as the training script
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=0
export CUDA_MODULE_LOADING=LAZY
export TORCH_USE_CUDA_DSA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Environment variables set:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"

echo ""
echo "Running CUDA test..."
python3 /workspace/flow_rtpo/scripts/single_node/test_cuda.py

echo ""
echo "=== Test Complete ===" 