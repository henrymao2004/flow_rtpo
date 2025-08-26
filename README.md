# Flow-RTPO: Flow-based Reinforcement Learning for Text-to-Pose Optimization

## Quick Start Guide

### Download & Setup
```bash
# Clone the repository
git clone <repository-url> flow_rtpo

# Navigate to project directory
cd flow_rtpo

# Install dependencies
pip install -r requirements.txt
```

### Model & Dataset Download

#### Option 1: HuggingFace Models (Recommended for Quick Start)
```bash
# Models will be automatically downloaded from HuggingFace
# No manual download required - set in config:
# config.model_loading.use_local = False
```

**Models to be downloaded:**
- **SD3**: `stabilityai/stable-diffusion-3.5-medium`
- **LLaVA**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **CLIP**: `openai/clip-vit-large-patch14`
- **SBERT**: `sentence-transformers/all-MiniLM-L6-v2`
- **GTR**: `sentence-transformers/gtr-t5-base`

**Datasets to be downloaded:**
- **RTP**: `allenai/real-toxicity-prompts`

#### Option 2: Local Models (For Production/Offline Use)
```bash
# Create local model directory
mkdir -p /mnt/data/group/zhaoliangjie/ICLR-work/

# Download models to local directory
cd /mnt/data/group/zhaoliangjie/ICLR-work/

# Download SD3
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

# Download LLaVA
git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf

# Download CLIP
git clone https://huggingface.co/openai/clip-vit-large-patch14

# Download SBERT
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

# Download GTR
git clone https://huggingface.co/sentence-transformers/gtr-t5-base

# Download RTP dataset
git clone https://huggingface.co/datasets/allenai/real-toxicity-prompts
```

**Directory Structure:**
```
/mnt/data/group/zhaoliangjie/ICLR-work/
├── stable-diffusion-3.5-medium/
├── llava-v1.6-mistral-7b-hf/
├── clip-vit-large-patch14/
├── all-MiniLM-L6-v2/
├── gtr-t5-base/
└── real-toxicity-prompts/
```

**Configuration:**
```python
# Set in config/flow_rtpo.py
config.model_loading.use_local = True
config.dataset_loading.use_local = True
```

### Project Structure
```
flow_rtpo/
├── config/           # Configuration files
├── dataset/          # Training datasets
├── flow_grpo/        # Core Flow-GRPO implementation
├── models/           # Model checkpoints
├── scripts/          # Training and evaluation scripts
└── setup.py          # Package setup
```

### Single Node Training
```bash
# Basic training
bash scripts/single_node/flow_rtpo.sh

# Multi-GPU training
bash scripts/single_node/flow_rtpo_4gpu.sh
```

### Multi-Node Training
```bash
# Master node (RANK=0)
bash scripts/multi_node/flow_rtpo/main.sh

# Worker nodes (RANK=1,2,3)
bash scripts/multi_node/flow_rtpo/main1.sh
bash scripts/multi_node/flow_rtpo/main2.sh
bash scripts/multi_node/flow_rtpo/main3.sh
```

### SwanLab Integration
```bash
# API Key for experiment tracking
export SWANLAB_API_KEY=YiUzV5i2rB0pybueoH8A8
```

---

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