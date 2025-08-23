#!/usr/bin/env python3
"""
Flow-RTPO Training Script for Accelerate Launch
Multi-GPU training for Flow-RTPO hierarchical red teaming
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    # Configuration
    config_name = "flow_rtpo_sd3"  # or "flow_rtpo_debug" for testing
    
    # Set environment variables for multi-GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Eight GPUs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = "flow_rtpo"
    
    # Optional: Set master address and port for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    
    # Create output directory
    output_dir = f"logs/flow_rtpo/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Flow-RTPO Multi-GPU Training...")
    print(f"Config: {config_name}")
    print(f"Output Dir: {output_dir}")
    print(f"GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("Number of processes: 7")
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Paths to the training script and config
    train_script = os.path.join(project_root, "scripts", "train_flow_rtpo.py")
    config_path = os.path.join(project_root, "config", "flow_rtpo.py")
    
    # Build the command with proper Python path
    cmd = [
        sys.executable, train_script,
        f"--config={config_path}:{config_name}"
    ]
    
    # Set environment to include project root in Python path
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Python path: {env['PYTHONPATH']}")
    print(f"Project root: {project_root}")
    print(f"Checking if flow_grpo exists: {os.path.exists(os.path.join(project_root, 'flow_grpo'))}")
    print(f"Checking if flow_grpo/stat_tracking.py exists: {os.path.exists(os.path.join(project_root, 'flow_grpo', 'stat_tracking.py'))}")
    
    # Run the training script with accelerate launch
    try:
        # Use accelerate launch instead of direct subprocess
        accelerate_cmd = [
            "accelerate", "launch",
            "--num_processes=8",
            "--num_machines=1", 
            "--mixed_precision=no",
            "--dynamo_backend=no",
            train_script,
            f"--config={config_path}:{config_name}"
        ]
        
        print(f"Running accelerate command: {' '.join(accelerate_cmd)}")
        
        # Redirect output to both console and log file
        with open(os.path.join(output_dir, "training.log"), "w") as log_file:
            process = subprocess.Popen(
                accelerate_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env,
                cwd=project_root  # Set working directory to project root
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                log_file.flush()
            
            process.wait()
            
        if process.returncode == 0:
            print(f"\nTraining completed successfully. Logs saved to: {output_dir}")
        else:
            print(f"\nTraining failed with exit code: {process.returncode}")
            sys.exit(process.returncode)
            
    except Exception as e:
        print(f"Error running training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 