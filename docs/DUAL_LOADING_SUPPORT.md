# Dual Loading Support for Flow-RTPO

This document explains how to use Flow-RTPO with both local models/datasets and HuggingFace models/datasets.

## Overview

Flow-RTPO now supports two ways of loading models and datasets:

1. **HuggingFace Loading** (Default): Models and datasets are downloaded from HuggingFace Hub
2. **Local Loading**: Models and datasets are loaded from local file paths

## Configuration Options

### Loading Mode Configuration

Each configuration has two key parameters:

```python
config.use_local_models = False  # Set to True for local models, False for HuggingFace
config.use_local_dataset = False  # Set to True for local dataset, False for HuggingFace
```

### Model Paths

The following model paths are automatically configured based on the loading mode:

- **SD3 Model**: `config.pretrained.model`
- **LLaVA Model**: `config.target_vlm`
- **GTR-T5 Model**: `config.gtr_t5_model`
- **SBERT Model**: `config.sbert_model`
- **RTP Dataset**: `config.rtp_dataset_path`

## Available Configurations

### HuggingFace Configurations (Default)

1. **`flow_rtpo_debug`**: Small dataset (8 prompts), HuggingFace models
2. **`flow_rtpo_sd3`**: Full dataset (132 prompts), HuggingFace models
3. **`flow_rtpo_large`**: Extended training (200 epochs), HuggingFace models
4. **`flow_rtpo_memory_optimized`**: Memory optimized, HuggingFace models

### Local Configurations

1. **`flow_rtpo_debug_local`**: Small dataset (8 prompts), local models
2. **`flow_rtpo_sd3_local`**: Full dataset (132 prompts), local models
3. **`flow_rtpo_large_local`**: Extended training (200 epochs), local models
4. **`flow_rtpo_memory_optimized_local`**: Memory optimized, local models

## Usage Examples

### Running with HuggingFace Models (Default)

```bash
# Debug configuration with HuggingFace models
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_debug

# Full training with HuggingFace models
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_sd3
```

### Running with Local Models

```bash
# Debug configuration with local models
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_debug_local

# Full training with local models
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_sd3_local
```

### Using the Example Script

```bash
# Run the interactive example script
python scripts/run_flow_rtpo_examples.py
```

This script will show you all available configurations and let you choose which one to run.

## Model Requirements

### HuggingFace Models

When using HuggingFace loading, the following models will be automatically downloaded:

- **SD3**: `stabilityai/stable-diffusion-3.5-medium`
- **LLaVA**: `llava-hf/llava-v1.6-mistral-7b-hf`
- **GTR-T5**: `sentence-transformers/gtr-t5-base`
- **SBERT**: `sentence-transformers/all-MiniLM-L6-v2`
- **RTP Dataset**: `allenai/real-toxicity-prompts`

### Local Models

When using local loading, ensure you have the following directory structure:

```
models/local/
├── stable-diffusion-3.5-medium/
├── llava-v1.6-mistral-7b-hf/
├── gtr-t5-base/
└── all-MiniLM-L6-v2/

dataset/local/
└── real-toxicity-prompts/
```

## Configuration Details

### HuggingFace Configuration Example

```python
config.use_local_models = False
config.use_local_dataset = False

config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
config.target_vlm = "llava-hf/llava-v1.6-mistral-7b-hf"
config.gtr_t5_model = "sentence-transformers/gtr-t5-base"
config.sbert_model = "sentence-transformers/all-MiniLM-L6-v2"
config.rtp_dataset_path = None  # Will use HuggingFace dataset
```

### Local Configuration Example

```python
config.use_local_models = True
config.use_local_dataset = True

config.pretrained.model = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/stable-diffusion-3.5-medium"
config.target_vlm = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/llava-v1.6-mistral-7b-hf"
config.gtr_t5_model = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/gtr-t5-base"
config.sbert_model = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/models/local/all-MiniLM-L6-v2"
config.rtp_dataset_path = "/mnt/data/group/zhaoliangjie/ICLR-work/flow_rtpo/dataset/local/real-toxicity-prompts"
```

## Component Updates

The following components have been updated to support dual loading:

### 1. Prompt Editor (`flow_grpo/prompt_editor.py`)

- Added `use_local_models` parameter
- Automatic model path selection based on loading mode
- Support for both local and HuggingFace GTR-T5 and SBERT models

### 2. Toxicity Rewards (`flow_grpo/toxicity_rewards.py`)

- Added `use_local_models` parameter
- Automatic LLaVA model path selection
- Support for both local and HuggingFace LLaVA models

### 3. RTP Dataset (`flow_grpo/rtp_dataset.py`)

- Added `use_local_dataset` parameter
- Automatic dataset path selection
- Support for both local and HuggingFace RTP datasets

### 4. Training Script (`scripts/train_flow_rtpo.py`)

- Passes loading configuration to all components
- Automatic configuration based on config settings

## Benefits

### HuggingFace Loading
- **Easy setup**: No need to download models manually
- **Automatic updates**: Always uses latest model versions
- **No storage requirements**: Models are cached automatically
- **Cross-platform**: Works on any system with internet access

### Local Loading
- **Faster startup**: No download time required
- **Offline capability**: Works without internet access
- **Version control**: Exact model versions can be specified
- **Production ready**: Stable for production environments

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure local model paths are correct and models are downloaded
2. **Dataset not found**: Ensure local dataset path is correct and dataset is available
3. **Memory issues**: Use memory-optimized configurations for limited GPU memory
4. **Download failures**: Check internet connection for HuggingFace loading

### Debugging

To debug loading issues, check the console output for:
- Model loading messages
- Dataset loading messages
- Error messages with specific paths

### Switching Between Modes

To switch between loading modes, simply change the configuration name:

```bash
# From HuggingFace to local
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_sd3_local

# From local to HuggingFace
python scripts/train_flow_rtpo.py --config config/flow_rtpo.py --config.config_name flow_rtpo_sd3
```

## Performance Considerations

- **Local models**: Faster startup, consistent performance
- **HuggingFace models**: Slower first startup (download), but convenient
- **Memory usage**: Same for both modes once loaded
- **Training speed**: No difference between modes

## Future Enhancements

- Support for custom model paths
- Automatic model validation
- Fallback mechanisms for failed downloads
- Model version pinning 