# Flow-RTPO

Flow-RTPO: Real-Time Prompt Optimization for Stable Diffusion models using Flow-based Reinforcement Learning.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Setup Local Models and Datasets



**Required Local Models:**
- **Stable Diffusion 3.5 Medium**: `models/local/stable-diffusion-3.5-medium/`
- **LLaVA v1.6 Mistral 7B**: `models/local/llava-v1.6-mistral-7b-hf/`
- **GTR-Base-T5**: `models/local/gtr-base-t5/`
- **all-MiniLM-L6-v2**: `models/local/all-MiniLM-L6-v2/`

**Required Local Dataset:**
- **Real Toxicity Prompts**: `dataset/local/real-toxicity-prompts/`

### Login to SwanLab

```bash
# Login using environment variables, Swanlab api key: YiUzV5i2rB0pybueoH8A8
swanlab login --api-key $SWANLAB_API_KEY
```




### Training

#### Single-Node Training with Local Models

For training with local models and datasets:

```bash
# Run the local training script
./scripts/train_flow_rtpo_local.sh
```

This script will:
- Check for all required local models and datasets
- Use the `flow_rtpo_local.py` configuration
- Train with local SD3, LLaVA, GTR-T5, and RTP dataset

#### Multi-Node Training

**Main Node (rank 0):** 
```bash
cd scripts/multi_node/flow_rtpo
./main.sh
```

**Worker Nodes (rank 1, 2, 3...):**
```bash
cd scripts/multi_node/flow_rtpo
./main1.sh  # for rank 1
./main2.sh  # for rank 2
./main3.sh  # for rank 3
```

#### Monitor Training

- **SwanLab**: [View experiments](https://swanlab.ai)
- **HuggingFace**: [Model Hub](https://huggingface.co/models)

### Configuration

#### Local Models Configuration


```python
# Key local model paths
config.pretrained.model = "models/local/stable-diffusion-3.5-medium"
config.target_vlm = "models/local/llava-v1.6-mistral-7b-hf"
config.gtr_t5_model = "models/local/gtr-base-t5"
config.sbert_model = "models/local/all-MiniLM-L6-v2"
config.rtp_dataset_path = "dataset/local/real-toxicity-prompts"
```

Available configurations:
- `flow_rtpo_sd3_local`: Standard configuration
- `flow_rtpo_debug_local`: Debug configuration with smaller settings
- `flow_rtpo_large_local`: Large-scale configuration
- `flow_rtpo_memory_optimized_local`: Memory-optimized configuration

#### Standard Configuration

Edit `config/flow_rtpo.py` to customize:
- Model architecture
- Training parameters
- Reward functions
- Dataset settings

### Supported Models

- Stable Diffusion 3
- FLUX
- FLUX Kontext
- WAN2.1

## Links

- **SwanLab**: [https://swanlab.ai](https://swanlab.ai) - Experiment tracking
- **HuggingFace**: [https://huggingface.co](https://huggingface.co) - Model repository
- **Requirements**: See `requirements.txt` for dependencies


