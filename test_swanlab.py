#!/usr/bin/env python3
"""
Simple SwanLab Test Script

This script tests basic SwanLab functionality including:
- Authentication/login
- Project initialization
- Metric logging
- Image logging
- Configuration logging
- Run completion

Usage:
    python test_swanlab.py
"""

import swanlab
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import os

def create_sample_image(step):
    """Create a simple sample image for testing."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create some sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x + step * 0.1) * np.exp(-x * 0.1)
    
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title(f'Sample Plot - Step {step}')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.grid(True, alpha=0.3)
    
    # Save to temporary file
    temp_path = f"temp_plot_{step}.png"
    plt.savefig(temp_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return temp_path

def test_swanlab():
    """Main test function for SwanLab."""
    
    print("🧪 Starting SwanLab Test...")
    
    try:
        # Step 1: Login (using the API key from the project)
        print("📝 Step 1: Logging in to SwanLab...")
        api_key = "YiUzV5i2rB0pybueoH8A8"  # From the project
        swanlab.login(api_key=api_key, save=True)
        print("✅ Login successful!")
        
        # Step 2: Initialize a test run
        print("🚀 Step 2: Initializing SwanLab run...")
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "model": "test_model",
            "optimizer": "adam",
            "test_parameter": 42.0
        }
        
        swanlab.init(
            project="swanlab_test",
            experiment_name=f"test_run_{int(time.time())}",
            description="Simple SwanLab functionality test",
            config=config
        )
        print("✅ Run initialized successfully!")
        
        # Step 3: Log some metrics over time
        print("📊 Step 3: Logging metrics...")
        for step in range(20):
            # Simulate training metrics
            train_loss = 2.0 * np.exp(-step * 0.1) + random.uniform(0, 0.1)
            train_accuracy = 1 - np.exp(-step * 0.15) + random.uniform(-0.05, 0.05)
            val_loss = train_loss + random.uniform(0, 0.2)
            val_accuracy = train_accuracy - random.uniform(0, 0.1)
            
            # Log metrics
            swanlab.log({
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "val/loss": val_loss,
                "val/accuracy": val_accuracy,
                "learning_rate": config["learning_rate"] * (0.95 ** step),
                "step": step
            }, step=step)
            
            # Log an image every 5 steps
            if step % 5 == 0:
                print(f"🖼️  Logging image at step {step}...")
                temp_image_path = create_sample_image(step)
                
                swanlab.log({
                    "sample_plots": swanlab.Image(temp_image_path, caption=f"Sample plot at step {step}")
                }, step=step)
                
                # Clean up temporary file
                os.remove(temp_image_path)
            
            # Small delay to simulate training time
            time.sleep(0.1)
            
            if step % 5 == 0:
                print(f"  Logged metrics for step {step}")
        
        print("✅ Metrics logged successfully!")
        
        # Step 4: Log some summary statistics
        print("📈 Step 4: Logging summary statistics...")
        summary_stats = {
            "final_train_loss": train_loss,
            "final_train_accuracy": train_accuracy,
            "final_val_loss": val_loss,
            "final_val_accuracy": val_accuracy,
            "total_steps": 20,
            "training_time": 2.0,  # Mock training time
            "best_val_accuracy": max([1 - np.exp(-i * 0.15) + random.uniform(-0.05, 0.05) for i in range(20)])
        }
        
        swanlab.log(summary_stats)
        print("✅ Summary statistics logged!")
        
        # Step 5: Test different data types
        print("🔢 Step 5: Testing different data types...")
        swanlab.log({
            "text_data": "This is a test string",
            "boolean_flag": True,
            "nested_config": {
                "sub_param1": 123,
                "sub_param2": "nested_value"
            },
            "array_data": [1, 2, 3, 4, 5],
            "numpy_array": np.array([0.1, 0.2, 0.3, 0.4, 0.5]).tolist()
        })
        print("✅ Different data types logged!")
        
        print("\n🎉 SwanLab test completed successfully!")
        print("📊 Check your SwanLab dashboard to see the logged data:")
        print("   https://swanlab.cn/@sevens/swanlab_test")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during SwanLab test: {str(e)}")
        return False
    
    finally:
        # Clean up any remaining temporary files
        for i in range(20):
            temp_file = f"temp_plot_{i}.png"
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    """Main entry point."""
    print("=" * 50)
    print("SwanLab Functionality Test")
    print("=" * 50)
    
    success = test_swanlab()
    
    if success:
        print("\n✅ All tests passed! SwanLab is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()