#!/bin/bash
# Node 3 (Rank 3) - Worker node
export PYTHONPATH="/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo:$PYTHONPATH"
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export RANK=3

echo "Starting training on node with machine_rank=${RANK}"

# Launch command (parameters automatically read from accelerate_multi_node.yaml)
/etc/dsw/runtime/bin/python -m accelerate.commands.launch --config_file ${PYTHONPATH}/scripts/accelerate_configs/flow_rtpo_multi_node.yaml \
    --num_machines 6 --num_processes 48 \
    --machine_rank ${RANK} \
    ${PYTHONPATH}/scripts/train_flow_rtpo.py \
    --config_file ${PYTHONPATH}/config/flow_rtpo.py:flow_rtpo_large 