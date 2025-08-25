#!/usr/bin/env python3
"""
Setup script for local models and datasets in Flow-RTPO.

This script creates the necessary directory structure and provides instructions
for downloading and organizing local model checkpoints and datasets.
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the necessary directory structure for local models and datasets."""
    
    # Define the base directories
    base_dirs = [
        "models/local",
        "dataset/local",
        "logs/flow_rtpo",
        "dataset/cache"
    ]
    
    print("Creating directory structure...")
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print("\nDirectory structure created successfully!")

def create_model_instructions():
    """Create instructions for downloading and organizing models."""
    
    instructions = """
# Local Model Setup Instructions

## 1. Stable Diffusion 3.5 Medium
Download from: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
Save to: models/local/stable-diffusion-3.5-medium/

```bash
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-medium models/local/stable-diffusion-3.5-medium
```

## 2. LLaVA v1.6 Mistral 7B
Download from: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
Save to: models/local/llava-v1.6-mistral-7b-hf/

```bash
git lfs install
git clone https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf models/local/llava-v1.6-mistral-7b-hf
```

## 3. GTR-Base-T5
Download from: https://huggingface.co/sentence-transformers/gtr-base-t5
Save to: models/local/gtr-base-t5/

```bash
git lfs install
git clone https://huggingface.co/sentence-transformers/gtr-base-t5 models/local/gtr-base-t5
```

## 4. all-MiniLM-L6-v2 (SBERT)
Download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
Save to: models/local/all-MiniLM-L6-v2/

```bash
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/local/all-MiniLM-L6-v2
```

## 5. Real Toxicity Prompts Dataset
Download from: https://huggingface.co/datasets/allenai/real-toxicity-prompts
Save to: dataset/local/real-toxicity-prompts/

```bash
# Option 1: Using datasets library
python -c "from datasets import load_dataset; dataset = load_dataset('allenai/real-toxicity-prompts'); dataset.save_to_disk('dataset/local/real-toxicity-prompts')"

# Option 2: Manual download and extract
# Download the dataset files and extract to dataset/local/real-toxicity-prompts/
```

## Directory Structure After Setup:
```
flow_rtpo/
├── models/
│   └── local/
│       ├── stable-diffusion-3.5-medium/
│       ├── llava-v1.6-mistral-7b-hf/
│       ├── gtr-base-t5/
│       └── all-MiniLM-L6-v2/
├── dataset/
│   ├── local/
│   │   └── real-toxicity-prompts/
│   └── cache/
└── logs/
    └── flow_rtpo/
```
"""
    
    with open("MODEL_SETUP_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("✓ Created: MODEL_SETUP_INSTRUCTIONS.md")

def create_config_example():
    """Create an example configuration file for local models."""
    
    config_example = '''# Example configuration for local models
# Copy this to your training script or modify config/flow_rtpo_local.py

# Local model paths
config.pretrained.model = "models/local/stable-diffusion-3.5-medium"
config.target_vlm = "models/local/llava-v1.6-mistral-7b-hf"
config.gtr_t5_model = "models/local/gtr-base-t5"
config.sbert_model = "models/local/all-MiniLM-L6-v2"

# Local dataset path
config.rtp_dataset_path = "dataset/local/real-toxicity-prompts"

# Local save directory
config.save_dir = "logs/flow_rtpo/your_experiment_name"
'''
    
    with open("LOCAL_CONFIG_EXAMPLE.txt", "w") as f:
        f.write(config_example)
    
    print("✓ Created: LOCAL_CONFIG_EXAMPLE.txt")

def check_existing_models():
    """Check if any local models already exist."""
    
    models_to_check = [
        "models/local/stable-diffusion-3.5-medium",
        "models/local/llava-v1.6-mistral-7b-hf", 
        "models/local/gtr-base-t5",
        "models/local/all-MiniLM-L6-v2",
        "dataset/local/real-toxicity-prompts"
    ]
    
    print("\nChecking for existing local models and datasets...")
    
    for model_path in models_to_check:
        if os.path.exists(model_path):
            print(f"✓ Found: {model_path}")
        else:
            print(f"✗ Missing: {model_path}")

def main():
    """Main setup function."""
    
    print("Flow-RTPO Local Models Setup")
    print("=" * 40)
    
    # Create directory structure
    create_directory_structure()
    
    # Check for existing models
    check_existing_models()
    
    # Create instructions
    create_model_instructions()
    create_config_example()
    
    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Follow the instructions in MODEL_SETUP_INSTRUCTIONS.md")
    print("2. Download the required models and datasets")
    print("3. Use config/flow_rtpo_local.py for training with local models")
    print("4. Run your training script with the local configuration")

if __name__ == "__main__":
    main() 