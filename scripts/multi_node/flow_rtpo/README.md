# Flow-RTPO Multi-Node Training

This directory contains scripts for running Flow-RTPO training across multiple nodes.

## Setup

The multi-node setup follows the same pattern as the SD3 multi-node implementation:

- **4 nodes** with **32 total processes** (8 processes per node)
- Uses the `flow_rtpo_large` configuration for large-scale training
- Master node IP: `10.82.139.22`
- Master port: `19001`

## Scripts

- `main.sh` - Master node (RANK=0)
- `main1.sh` - Worker node 1 (RANK=1)
- `main2.sh` - Worker node 2 (RANK=2)
- `main3.sh` - Worker node 3 (RANK=3)

## Configuration

The scripts use the `flow_rtpo_large` configuration which includes:
- Full dataset (132 prompts)
- Large batch settings (batch_size=16)
- Extended training (200 epochs)
- Aggressive toxicity optimization

## Usage

1. **On the master node** (IP: 10.82.139.22):
   ```bash
   ./main.sh
   ```

2. **On worker node 1**:
   ```bash
   ./main1.sh
   ```

3. **On worker node 2**:
   ```bash
   ./main2.sh
   ```

4. **On worker node 3**:
   ```bash
   ./main3.sh
   ```

## Environment Variables

The scripts set the following NCCL environment variables for optimal multi-node performance:
- `NCCL_IB_DISABLE=0` - Enable InfiniBand
- `NCCL_IB_HCA=mlx5` - Use Mellanox HCA
- `NCCL_DEBUG=WARN` - Set debug level
- `NCCL_IB_GID_INDEX=3` - GID index for InfiniBand

## Accelerate Configuration

The scripts use `scripts/accelerate_configs/flow_rtpo_multi_node.yaml` for distributed training configuration.

## Notes

- All nodes must be on the same network and able to communicate
- The master node IP address must be accessible from all worker nodes
- Make sure all nodes have the same codebase and dependencies installed
- The training will automatically handle distributed data loading and gradient synchronization 