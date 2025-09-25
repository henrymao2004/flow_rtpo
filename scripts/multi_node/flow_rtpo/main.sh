# 安装依赖
/usr/local/bin/python3 -m pip install -r /mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/requirements.txt
export PYTHONPATH="/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo:$PYTHONPATH"
export HF_ENDPOINT=https://alpha.hf-mirror.com
export NCCL_DEBUG=INFO                  
export NCCL_DEBUG_SUBSYS=ALL   
export NCCL_TIMEOUT=10800      # NCCL timeout
export NCCL_IB_TIMEOUT=10800   # InfiniBand timeout
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

# 使用 DeepSpeed ZeRO-2 的 accelerate 参数设置
accelerate launch \
    --config_file /mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/scripts/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes=$((${WORLD_SIZE} * 8)) \
    --num_machines=${WORLD_SIZE} \
    --machine_rank=${RANK} \
    --main_process_ip=${MASTER_ADDR} \
    --main_process_port=${MASTER_PORT} \
    /mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/scripts/train_flow_rtpo.py \
    --config_file "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/config/flow_rtpo.py:flow_rtpo_large"