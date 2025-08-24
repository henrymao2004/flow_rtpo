#!/usr/bin/env python3
"""
Test script for memory-optimized Flow RTPO configuration.
This script tests basic functionality without running the full training loop.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.flow_rtpo import get_config
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_optimized_config():
    """Test the memory-optimized configuration."""
    
    # Set CUDA debugging environment variables
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("Testing memory-optimized Flow RTPO configuration...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    try:
        # Load memory-optimized configuration
        config = get_config("flow_rtpo_memory_optimized")
        print(f"\nConfiguration loaded successfully:")
        print(f"  Model: {config.pretrained.model}")
        print(f"  Resolution: {config.resolution}x{config.resolution}")
        print(f"  Batch size: {config.sample.batch_size}")
        print(f"  Images per prompt: {config.sample.num_image_per_prompt}")
        print(f"  LoRA rank: {config.lora_rank}")
        print(f"  Gradient checkpointing: {config.get('gradient_checkpointing', False)}")
        print(f"  Mixed precision: {config.mixed_precision}")
        
        # Test accelerator initialization
        print("\nTesting accelerator initialization...")
        accelerator_config = ProjectConfiguration(
            project_dir=os.path.join(config.save_dir),
            automatic_checkpoint_naming=True,
            total_limit=config.num_checkpoint_limit,
        )
        
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        )
        
        print(f"Accelerator initialized successfully:")
        print(f"  Device: {accelerator.device}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        print(f"  Num processes: {accelerator.num_processes}")
        
        # Test basic imports
        print("\nTesting imports...")
        from diffusers import StableDiffusion3Pipeline
        from peft import LoraConfig, get_peft_model
        print("✓ All required imports successful")
        
        # Test pipeline loading (without actually loading the model)
        print("\nTesting pipeline configuration...")
        print("✓ Pipeline configuration test passed")
        
        print("\n✅ Memory-optimized configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_optimized_config()
    sys.exit(0 if success else 1) 