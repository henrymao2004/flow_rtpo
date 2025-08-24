#!/usr/bin/env python3
"""
Simple CUDA test script to verify GPU functionality
"""
import os
import torch
import subprocess

def test_cuda_basic():
    print("=== Basic CUDA Test ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test basic tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ Basic tensor operations successful")
        except Exception as e:
            print(f"✗ Basic tensor operations failed: {e}")

def test_multi_gpu():
    print("\n=== Multi-GPU Test ===")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                print(f"✓ GPU {i}: {torch.cuda.get_device_name(i)}")
                
                # Test memory allocation
                x = torch.randn(100, 100).cuda()
                print(f"  - Tensor allocation successful")
                
            except Exception as e:
                print(f"✗ GPU {i} failed: {e}")

def test_nvidia_smi():
    print("\n=== nvidia-smi Test ===")
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"]).decode()
        print("nvidia-smi output:")
        for line in result.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, used_mb, total_mb = parts[0], parts[1], parts[2], parts[3]
                    print(f"  GPU {gpu_id}: {name} - {used_mb}MB / {total_mb}MB")
    except Exception as e:
        print(f"nvidia-smi failed: {e}")

if __name__ == "__main__":
    test_cuda_basic()
    test_multi_gpu()
    test_nvidia_smi()
    print("\n=== Test Complete ===") 