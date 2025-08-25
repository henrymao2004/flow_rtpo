# Flow-RTPO

Flow-RTPO: Real-Time Prompt Optimization for Stable Diffusion models using Flow-based Reinforcement Learning.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```


# Login using environment variables, Swanlab api key: YiUzV5i2rB0pybueoH8A8
swanlab login --api-key $SWANLAB_API_KEY




### Multi-Node Training


#### 1. Launch Multi-Node Training

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

#### 2. Monitor Training

- **SwanLab**: [View experiments](https://swanlab.ai)
- **HuggingFace**: [Model Hub](https://huggingface.co/models)

### Configuration

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


