#!/usr/bin/env python3
"""
Test script to verify Detoxify local loading from checkpoint folder.
This script tests if Detoxify can be successfully loaded from a local cache directory.
"""

import os
import sys
import time
import torch
from detoxify import Detoxify


def test_detoxify_local_loading(cache_dir="/root/detoxify-original"):
    """
    Test loading Detoxify from local checkpoint folder.
    
    Args:
        cache_dir (str): Path to the local detoxify checkpoint folder
    """
    print("=" * 60)
    print("DETOXIFY LOCAL LOADING TEST")
    print("=" * 60)
    
    # Check if cache directory exists
    print(f"[CHECK] Cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        print(f"[ERROR] Cache directory does not exist: {cache_dir}")
        return False
    
    print(f"[CHECK] Cache directory exists: ✓")
    
    # List contents of cache directory
    print(f"[CHECK] Contents of {cache_dir}:")
    try:
        contents = os.listdir(cache_dir)
        for item in contents:
            item_path = os.path.join(cache_dir, item)
            item_type = "DIR" if os.path.isdir(item_path) else "FILE"
            print(f"  - {item} ({item_type})")
    except Exception as e:
        print(f"[ERROR] Cannot list directory contents: {e}")
        return False
    
    # Check for hub subdirectory (for PyTorch Hub models)
    hub_dir = os.path.join(cache_dir, "hub")
    print(f"[CHECK] Hub directory: {hub_dir}")
    if os.path.exists(hub_dir):
        print(f"[CHECK] Hub directory exists: ✓")
        try:
            hub_contents = os.listdir(hub_dir)
            print(f"[CHECK] Hub directory contents:")
            for item in hub_contents[:10]:  # Show first 10 items
                print(f"  - {item}")
            if len(hub_contents) > 10:
                print(f"  ... and {len(hub_contents) - 10} more items")
        except Exception as e:
            print(f"[WARNING] Cannot list hub directory: {e}")
    else:
        print(f"[CHECK] Hub directory does not exist")
    
    # Check for the specific checkpoint that was being downloaded
    expected_checkpoint = "toxic_original-c1212f89.ckpt"
    checkpoint_paths = [
        os.path.join(cache_dir, expected_checkpoint),
        os.path.join(cache_dir, "hub", expected_checkpoint),
        os.path.join(cache_dir, "hub", "checkpoints", expected_checkpoint),
        os.path.join(cache_dir, "checkpoints", expected_checkpoint)
    ]
    
    print(f"[CHECK] Looking for checkpoint file: {expected_checkpoint}")
    checkpoint_found = False
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"[CHECK] Found checkpoint at: {checkpoint_path} ✓")
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            print(f"[CHECK] Checkpoint size: {file_size:.1f} MB")
            checkpoint_found = True
            break
    
    if not checkpoint_found:
        print(f"[WARNING] Checkpoint file {expected_checkpoint} not found in expected locations")
        print(f"[INFO] This means Detoxify will download it during first use")
        
        # Create the checkpoints directory if it doesn't exist
        checkpoints_dir = os.path.join(cache_dir, "hub", "checkpoints")
        if not os.path.exists(checkpoints_dir):
            print(f"[SETUP] Creating checkpoints directory: {checkpoints_dir}")
            os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Check CUDA availability
    print(f"[CHECK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CHECK] CUDA device count: {torch.cuda.device_count()}")
        print(f"[CHECK] Current CUDA device: {torch.cuda.current_device()}")
        device = "cuda"
    else:
        print(f"[CHECK] Using CPU device")
        device = "cpu"
    
    # Save original environment variables
    print(f"[SETUP] Saving original environment variables...")
    orig_torch_home = os.environ.get('TORCH_HOME')
    orig_hf_home = os.environ.get('HF_HOME')
    orig_hf_hub_cache = os.environ.get('HF_HUB_CACHE')
    
    print(f"[SETUP] Original TORCH_HOME: {orig_torch_home}")
    print(f"[SETUP] Original HF_HOME: {orig_hf_home}")
    print(f"[SETUP] Original HF_HUB_CACHE: {orig_hf_hub_cache}")
    
    try:
        # Set environment variables to use local cache
        print(f"[SETUP] Setting environment variables for local cache...")
        os.environ['TORCH_HOME'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_HUB_CACHE'] = os.path.join(cache_dir, "hub")
        
        print(f"[SETUP] New TORCH_HOME: {os.environ.get('TORCH_HOME')}")
        print(f"[SETUP] New HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"[SETUP] New HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
        
        # Test loading Detoxify
        print(f"[LOAD] Attempting to load Detoxify 'original' model...")
        start_time = time.time()
        
        try:
            detoxify_model = Detoxify('original', device=device)
            load_time = time.time() - start_time
            print(f"[LOAD] Successfully loaded Detoxify model in {load_time:.2f}s: ✓")
            
            # Test inference
            print(f"[TEST] Testing inference with sample text...")
            test_texts = [
                "This is a normal sentence.",
                "You are a wonderful person.",
                "The weather is nice today."
            ]
            
            try:
                inference_start = time.time()
                scores = detoxify_model.predict(test_texts)
                inference_time = time.time() - inference_start
                
                print(f"[TEST] Inference completed in {inference_time:.3f}s: ✓")
                print(f"[TEST] Sample results:")
                
                for i, text in enumerate(test_texts):
                    print(f"  Text {i+1}: \"{text}\"")
                    for metric in ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']:
                        if metric in scores:
                            score = scores[metric][i] if hasattr(scores[metric], '__getitem__') else scores[metric]
                            print(f"    {metric}: {score:.6f}")
                    print()
                
                print(f"[SUCCESS] All tests passed! Detoxify is working correctly.")
                return True
                
            except Exception as inference_error:
                print(f"[ERROR] Inference test failed: {inference_error}")
                print(f"[ERROR] Model loaded but cannot perform inference")
                return False
            
        except Exception as load_error:
            print(f"[ERROR] Failed to load Detoxify model: {load_error}")
            print(f"[ERROR] This could mean:")
            print(f"  1. Model files are not in the expected location")
            print(f"  2. Model files are corrupted")
            print(f"  3. Cache directory structure is incorrect")
            print(f"  4. Insufficient permissions")
            return False
    
    finally:
        # Restore original environment variables
        print(f"[CLEANUP] Restoring original environment variables...")
        if orig_torch_home is not None:
            os.environ['TORCH_HOME'] = orig_torch_home
        else:
            os.environ.pop('TORCH_HOME', None)
            
        if orig_hf_home is not None:
            os.environ['HF_HOME'] = orig_hf_home
        else:
            os.environ.pop('HF_HOME', None)
            
        if orig_hf_hub_cache is not None:
            os.environ['HF_HUB_CACHE'] = orig_hf_hub_cache
        else:
            os.environ.pop('HF_HUB_CACHE', None)
        
        print(f"[CLEANUP] Environment variables restored")


def test_fallback_loading():
    """Test fallback loading without local cache."""
    print("\n" + "=" * 60)
    print("DETOXIFY FALLBACK LOADING TEST")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FALLBACK] Testing standard Detoxify loading on {device}...")
    
    try:
        start_time = time.time()
        detoxify_model = Detoxify('original', device=device)
        load_time = time.time() - start_time
        
        print(f"[FALLBACK] Standard loading successful in {load_time:.2f}s: ✓")
        
        # Quick inference test
        test_result = detoxify_model.predict(["Hello world"])
        print(f"[FALLBACK] Quick inference test passed: ✓")
        return True
        
    except Exception as e:
        print(f"[FALLBACK] Standard loading failed: {e}")
        return False


def main():
    """Main test function."""
    print("Testing Detoxify local loading capabilities...\n")
    
    # Test 1: Local loading from specified cache directory
    cache_dir = "/root/detoxify-original"
    local_success = test_detoxify_local_loading(cache_dir)
    
    # Test 2: Fallback loading (for comparison)
    fallback_success = test_fallback_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Local loading ({cache_dir}): {'✓ PASSED' if local_success else '✗ FAILED'}")
    print(f"Fallback loading (standard): {'✓ PASSED' if fallback_success else '✗ FAILED'}")
    
    if local_success:
        print(f"\n[CONCLUSION] Local Detoxify loading is working correctly!")
        print(f"[CONCLUSION] You can use the local cache at: {cache_dir}")
    elif fallback_success:
        print(f"\n[CONCLUSION] Local loading failed, but standard loading works.")
        print(f"[CONCLUSION] Consider checking the local cache directory or using standard loading.")
    else:
        print(f"\n[CONCLUSION] Both local and standard loading failed.")
        print(f"[CONCLUSION] There may be an issue with the Detoxify installation or environment.")
    
    # Exit code
    sys.exit(0 if local_success else 1)


if __name__ == "__main__":
    main()