#!/usr/bin/env python3
"""
Flow-RTPO Training Script for Accelerate Launch
Multi-GPU training for Flow-RTPO hierarchical red teaming
"""

import os
import sys
from datetime import datetime

def main():
    # Configuration
    config_name = "flow_rtpo_sd3"  # or "flow_rtpo_debug" for testing
    
    # Set environment variables for multi-GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "flow_rtpo"
    
    # Create output directory
    output_dir = f"logs/flow_rtpo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Flow-RTPO Multi-GPU Training...")
    print(f"Config: {config_name}")
    print(f"Output Dir: {output_dir}")
    
    # Get the script directory and set up Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Paths to the training script and config
    train_script = os.path.join(project_root, "scripts", "train_flow_rtpo.py")
    config_path = os.path.join(project_root, "config", "flow_rtpo.py")
    
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
    print(f"Checking if flow_grpo exists: {os.path.exists(os.path.join(project_root, 'flow_grpo'))}")
    print(f"Checking if flow_grpo/stat_tracking.py exists: {os.path.exists(os.path.join(project_root, 'flow_grpo', 'stat_tracking.py'))}")
    
    # Set the config argument for the training script
    sys.argv = [
        train_script,
        f"--config={config_path}:{config_name}"
    ]
    
    print(f"Running training script with args: {sys.argv}")
    
    # Import and run the training script directly
    try:
        # Change to project root directory
        os.chdir(project_root)
        
        # Import the training script module
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_flow_rtpo", train_script)
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        print("Training script executed successfully")
        
    except Exception as e:
        print(f"Error running training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 