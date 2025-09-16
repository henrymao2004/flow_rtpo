#!/usr/bin/env python3
"""
Fix Detoxify local cache structure to avoid re-downloading.
This script moves the checkpoint file to the correct PyTorch Hub location.
"""

import os
import shutil
import sys


def fix_detoxify_cache_structure(cache_dir="/root/detoxify-original"):
    """
    Fix the cache directory structure for Detoxify to avoid re-downloading.
    
    Args:
        cache_dir (str): Path to the detoxify cache directory
    """
    print("=" * 60)
    print("FIXING DETOXIFY CACHE STRUCTURE")
    print("=" * 60)
    
    # Expected files
    checkpoint_file = "toxic_original-c1212f89.ckpt"
    source_path = os.path.join(cache_dir, checkpoint_file)
    target_dir = os.path.join(cache_dir, "hub", "checkpoints")
    target_path = os.path.join(target_dir, checkpoint_file)
    
    print(f"[INFO] Cache directory: {cache_dir}")
    print(f"[INFO] Looking for checkpoint: {checkpoint_file}")
    
    # Check if source file exists
    if not os.path.exists(source_path):
        print(f"[ERROR] Source file not found: {source_path}")
        return False
    
    print(f"[FOUND] Source file: {source_path}")
    file_size = os.path.getsize(source_path) / (1024 * 1024)  # MB
    print(f"[INFO] File size: {file_size:.1f} MB")
    
    # Create target directory if it doesn't exist
    print(f"[SETUP] Creating target directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if target file already exists
    if os.path.exists(target_path):
        print(f"[INFO] Target file already exists: {target_path}")
        target_size = os.path.getsize(target_path) / (1024 * 1024)  # MB
        print(f"[INFO] Target file size: {target_size:.1f} MB")
        
        if abs(file_size - target_size) < 0.1:  # Files are same size
            print(f"[SUCCESS] Files appear to be identical, no action needed!")
            return True
        else:
            print(f"[WARNING] File sizes differ, will replace target file")
    
    # Copy or move the file
    try:
        print(f"[COPY] Copying file from {source_path} to {target_path}")
        shutil.copy2(source_path, target_path)
        
        # Verify the copy
        if os.path.exists(target_path):
            copied_size = os.path.getsize(target_path) / (1024 * 1024)  # MB
            print(f"[VERIFY] Copied file size: {copied_size:.1f} MB")
            
            if abs(file_size - copied_size) < 0.1:
                print(f"[SUCCESS] File copied successfully!")
                print(f"[INFO] Detoxify should now use the local cache without downloading")
                return True
            else:
                print(f"[ERROR] File copy verification failed - size mismatch")
                return False
        else:
            print(f"[ERROR] Target file not created")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to copy file: {e}")
        return False


def verify_cache_structure(cache_dir="/root/detoxify-original"):
    """Verify the cache structure is correct."""
    print("\n" + "=" * 60)
    print("VERIFYING CACHE STRUCTURE")
    print("=" * 60)
    
    checkpoint_file = "toxic_original-c1212f89.ckpt"
    expected_path = os.path.join(cache_dir, "hub", "checkpoints", checkpoint_file)
    
    print(f"[CHECK] Expected checkpoint location: {expected_path}")
    
    if os.path.exists(expected_path):
        file_size = os.path.getsize(expected_path) / (1024 * 1024)  # MB
        print(f"[SUCCESS] Checkpoint found at correct location!")
        print(f"[INFO] File size: {file_size:.1f} MB")
        return True
    else:
        print(f"[ERROR] Checkpoint not found at expected location")
        return False


def test_detoxify_loading(cache_dir="/root/detoxify-original"):
    """Test if Detoxify loads without downloading."""
    print("\n" + "=" * 60)
    print("TESTING DETOXIFY LOADING")
    print("=" * 60)
    
    # Set environment variables
    orig_torch_home = os.environ.get('TORCH_HOME')
    orig_hf_home = os.environ.get('HF_HOME')
    orig_hf_hub_cache = os.environ.get('HF_HUB_CACHE')
    
    try:
        # Set cache environment
        os.environ['TORCH_HOME'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_HUB_CACHE'] = os.path.join(cache_dir, "hub")
        
        print(f"[TEST] Loading Detoxify with cache at: {cache_dir}")
        
        # Import and test
        from detoxify import Detoxify
        
        print(f"[TEST] Creating Detoxify model...")
        model = Detoxify('original')
        
        print(f"[TEST] Testing inference...")
        result = model.predict("This is a test sentence.")
        
        print(f"[SUCCESS] Model loaded and tested successfully!")
        print(f"[RESULT] Toxicity score: {result['toxicity']:.6f}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load or test model: {e}")
        return False
        
    finally:
        # Restore environment
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


def main():
    """Main function."""
    cache_dir = "/root/detoxify-original"
    
    print("Fixing Detoxify cache structure to avoid re-downloading...\n")
    
    # Step 1: Fix cache structure
    fix_success = fix_detoxify_cache_structure(cache_dir)
    
    if not fix_success:
        print(f"\n[FAILURE] Could not fix cache structure")
        sys.exit(1)
    
    # Step 2: Verify structure
    verify_success = verify_cache_structure(cache_dir)
    
    if not verify_success:
        print(f"\n[FAILURE] Cache structure verification failed")
        sys.exit(1)
    
    # Step 3: Test loading
    test_success = test_detoxify_loading(cache_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if fix_success and verify_success and test_success:
        print(f"[SUCCESS] All tests passed!")
        print(f"[SUCCESS] Detoxify cache is properly configured")
        print(f"[INFO] Future Detoxify loads should use local cache without downloading")
    else:
        print(f"[FAILURE] Some tests failed")
        print(f"[INFO] Cache structure fix: {'✓' if fix_success else '✗'}")
        print(f"[INFO] Structure verification: {'✓' if verify_success else '✗'}")
        print(f"[INFO] Model loading test: {'✓' if test_success else '✗'}")
    
    sys.exit(0 if (fix_success and verify_success and test_success) else 1)


if __name__ == "__main__":
    main()