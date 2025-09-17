#!/usr/bin/env python3
"""
Minimal SwanLab Test Script

A simple, quick test to verify SwanLab is working.

Usage:
    python test_swanlab_simple.py
"""

import swanlab
import time
import random

def simple_test():
    """Quick SwanLab test with minimal logging."""
    print("🧪 Running simple SwanLab test...")
    
    try:
        # Login
        print("📝 Logging in...")
        swanlab.login(api_key="YiUzV5i2rB0pybueoH8A8", save=True)
        
        # Initialize run
        print("🚀 Initializing run...")
        swanlab.init(
            project="swanlab_simple_test",
            experiment_name=f"simple_test_{int(time.time())}",
            config={"test": True, "simple": True}
        )
        
        # Log a few metrics
        print("📊 Logging metrics...")
        for i in range(5):
            swanlab.log({
                "metric": random.random(),
                "step": i
            }, step=i)
            time.sleep(0.1)
        
        print("✅ Simple test completed successfully!")
        print("📊 View at: https://swanlab.cn/@sevens/swanlab_simple_test")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    simple_test()