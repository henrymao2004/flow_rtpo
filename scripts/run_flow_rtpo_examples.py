#!/usr/bin/env python3
"""
Example script demonstrating how to run Flow-RTPO with both local and HuggingFace model/dataset loading.

This script shows how to:
1. Use HuggingFace models and datasets (default)
2. Use local models and datasets
3. Mix local models with HuggingFace datasets or vice versa
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.flow_rtpo import get_config


def run_training(config_name, description):
    """Run training with a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Running Flow-RTPO Training: {description}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    # Build the training command
    cmd = [
        "python", "scripts/train_flow_rtpo.py",
        "--config", f"config/flow_rtpo.py",
        "--config.config_name", config_name
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Starting training...")
    
    try:
        # Run the training command
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"Training interrupted by user")
        return False


def main():
    """Main function to demonstrate different configurations."""
    print("Flow-RTPO Training Examples")
    print("This script demonstrates different ways to run Flow-RTPO training")
    
    # Example 1: HuggingFace models and dataset (default)
    print("\n1. HuggingFace Configuration (Default)")
    print("   - Models: Downloaded from HuggingFace")
    print("   - Dataset: Downloaded from HuggingFace")
    print("   - Use case: Quick testing, no local storage required")
    
    config = get_config("flow_rtpo_debug")
    print(f"   - SD3 Model: {config.pretrained.model}")
    print(f"   - LLaVA Model: {config.target_vlm}")
    print(f"   - GTR-T5 Model: {config.gtr_t5_model}")
    print(f"   - SBERT Model: {config.sbert_model}")
    print(f"   - Dataset: HuggingFace (allenai/real-toxicity-prompts)")
    
    # Example 2: Local models and dataset
    print("\n2. Local Configuration")
    print("   - Models: Loaded from local paths")
    print("   - Dataset: Loaded from local path")
    print("   - Use case: Production training, faster loading")
    
    config_local = get_config("flow_rtpo_debug_local")
    print(f"   - SD3 Model: {config_local.pretrained.model}")
    print(f"   - LLaVA Model: {config_local.target_vlm}")
    print(f"   - GTR-T5 Model: {config_local.gtr_t5_model}")
    print(f"   - SBERT Model: {config_local.sbert_model}")
    print(f"   - Dataset: {config_local.rtp_dataset_path}")
    
    # Example 3: Large scale training with local models
    print("\n3. Large Scale Training (Local Models)")
    print("   - Full dataset (132 prompts)")
    print("   - Extended training (200 epochs)")
    print("   - Optimized for production")
    
    config_large = get_config("flow_rtpo_large_local")
    print(f"   - Max prompts: {config_large.max_prompts}")
    print(f"   - Epochs: {config_large.num_epochs}")
    print(f"   - Batch size: {config_large.sample.batch_size}")
    print(f"   - Images per prompt: {config_large.sample.num_image_per_prompt}")
    
    # Example 4: Memory optimized training
    print("\n4. Memory Optimized Training")
    print("   - Reduced batch sizes and resolution")
    print("   - Gradient checkpointing enabled")
    print("   - Use case: Limited GPU memory")
    
    config_memory = get_config("flow_rtpo_memory_optimized_local")
    print(f"   - Resolution: {config_memory.resolution}x{config_memory.resolution}")
    print(f"   - Batch size: {config_memory.sample.batch_size}")
    print(f"   - Mixed precision: {config_memory.mixed_precision}")
    print(f"   - Gradient checkpointing: {config_memory.get('gradient_checkpointing', False)}")
    
    # Ask user which configuration to run
    print(f"\n{'='*80}")
    print("Available configurations to run:")
    print("1. flow_rtpo_debug - HuggingFace models, small dataset (8 prompts)")
    print("2. flow_rtpo_debug_local - Local models, small dataset (8 prompts)")
    print("3. flow_rtpo_large_local - Local models, full dataset (132 prompts)")
    print("4. flow_rtpo_memory_optimized_local - Memory optimized, local models")
    print("5. flow_rtpo_sd3 - HuggingFace models, full dataset")
    print("6. flow_rtpo_sd3_local - Local models, full dataset")
    print("0. Exit without running")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-6): ").strip()
            
            if choice == "0":
                print("Exiting without running training.")
                break
            elif choice == "1":
                success = run_training("flow_rtpo_debug", "HuggingFace Models, Small Dataset")
            elif choice == "2":
                success = run_training("flow_rtpo_debug_local", "Local Models, Small Dataset")
            elif choice == "3":
                success = run_training("flow_rtpo_large_local", "Local Models, Full Dataset")
            elif choice == "4":
                success = run_training("flow_rtpo_memory_optimized_local", "Memory Optimized, Local Models")
            elif choice == "5":
                success = run_training("flow_rtpo_sd3", "HuggingFace Models, Full Dataset")
            elif choice == "6":
                success = run_training("flow_rtpo_sd3_local", "Local Models, Full Dataset")
            else:
                print("Invalid choice. Please enter a number between 0 and 6.")
                continue
            
            if success:
                print(f"\nTraining completed successfully!")
            else:
                print(f"\nTraining failed. Check the logs for details.")
            
            # Ask if user wants to run another configuration
            another = input("\nRun another configuration? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main() 