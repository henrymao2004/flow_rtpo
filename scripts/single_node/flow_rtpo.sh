#!/bin/bash
# Flow-RTPO Single Node 8-GPU Training Script
set -e

# 安装依赖
/usr/local/bin/python3 -m pip install -r /mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/requirements.txt
export PYTHONPATH="/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com

# NCCL Configuration for single node 8-GPU
export NCCL_DEBUG=INFO                  
export NCCL_DEBUG_SUBSYS=ALL            
export NCCL_TIMEOUT=10800      # NCCL timeout (3 hours)
export NCCL_IB_TIMEOUT=10800   # InfiniBand timeout
export NCCL_IB_DISABLE=1       # Disable InfiniBand for single node
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=0      # Enable P2P for single node
export NCCL_SHM_DISABLE=0      # Enable shared memory

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Single node 8-GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="flow_rtpo"


echo "Starting Flow-RTPO Single Node 8-GPU Training..."
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "NCCL Timeout: $NCCL_TIMEOUT seconds"

# Check GPU status
echo "=== GPU Status ==="
nvidia-smi
echo "=================="

# 启动单节点8卡训练
accelerate launch \
    --num_processes=8 \
    --num_machines=1 \
    --machine_rank=0 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    /mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/scripts/train_flow_rtpo.py \
    --config_file "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/config/flow_rtpo.py:flow_rtpo_large"
