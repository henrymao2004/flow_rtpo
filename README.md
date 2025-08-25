# main.sh - Flow-RTPO Master Node Script

## Overview
This script launches the master node (RANK=0) for multi-node Flow-RTPO training using Accelerate.

## Configuration
- **Master Node IP**: 10.82.139.22
- **Master Port**: 19001
- **Total Machines**: 4
- **Total Processes**: 32 (8 per node)
- **Configuration**: flow_rtpo_large

## Environment Setup
The script configures NCCL for optimal InfiniBand performance:
- Enables InfiniBand communication
- Uses Mellanox HCA (mlx5)
- Sets appropriate GID index and debug level

## Usage
```bash
/scripts/multi_node/main.sh
```

swanlab api key: YiUzV5i2rB0pybueoH8A8

## Prerequisites
- Must be run on the master node (IP: 10.82.139.22)
- All worker nodes must be running their respective scripts (main1.sh, main2.sh, main3.sh)
- Accelerate configuration file must exist at `scripts/accelerate_configs/flow_rtpo_multi_node.yaml`

## What it does
1. Sets up NCCL environment variables for multi-node communication
2. Launches the Flow-RTPO training process using Accelerate
3. Coordinates with worker nodes for distributed training
4. Uses the `flow_rtpo_large` configuration for large-scale training 