#!/usr/bin/env python3
"""
Reward worker entry script for GPU isolation.
CUDA_VISIBLE_DEVICES is set by parent process before this script runs.
"""

import os
import argparse
import time
import pickle
from multiprocessing import Queue, Event
import torch

# Set memory optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker_id", type=int, required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--vlm_model", required=True)
    parser.add_argument("--w_cvar", type=float, required=True)
    parser.add_argument("--w_quality", type=float, required=True)
    parser.add_argument("--vlm_batch_size", type=int, required=True)
    args = parser.parse_args()
    
    # CUDA_VISIBLE_DEVICES is already set by parent process
    print(f"Reward worker {args.worker_id} starting on GPU {args.gpu_id} (local device 0)")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Set device to local 0 (which is the isolated GPU)
    torch.cuda.set_device(0)
    print(f"Using local device 0 (physical GPU {args.gpu_id})")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Import reward function after GPU setup
    from flow_grpo.toxicity_rewards import toxicity_reward_function
    
    # Initialize reward function
    reward_fn = toxicity_reward_function(
        device="cuda:0",
        vlm_model=args.vlm_model,
        w_cvar=args.w_cvar,
        w_quality=args.w_quality,
        vlm_batch_size=args.vlm_batch_size
    )
    
    print(f"Reward worker {args.worker_id} initialized successfully")
    
    # Simple test to verify GPU isolation
    test_tensor = torch.randn(1, 3, 224, 224, device="cuda:0")
    print(f"Test tensor created on device: {test_tensor.device}")
    
    # Keep process alive for now (will be enhanced with queue communication)
    print(f"Reward worker {args.worker_id} ready and waiting...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"Reward worker {args.worker_id} shutting down")

if __name__ == "__main__":
    main() 