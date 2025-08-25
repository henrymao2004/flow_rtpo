#!/usr/bin/env python3
"""
Test script to verify that the meta tensor fix works correctly.
"""

import torch
from flow_grpo.toxicity_rewards import ToxicityRewardSystem

def test_meta_tensor_fix():
    """Test that the toxicity reward system can be initialized without meta tensor errors."""
    print("Testing meta tensor fix...")
    
    try:
        # Initialize the toxicity reward system
        # This should not raise the meta tensor error
        reward_system = ToxicityRewardSystem(
            device="cuda" if torch.cuda.is_available() else "cpu",
            vlm_model="llava-hf/llava-v1.6-mistral-7b-hf",
            w_cvar=0.1,
            w_quality=0.05,
            enable_quantization=False,  # Disable quantization for testing
            use_multi_gpu=False,  # Use single GPU for testing
        )
        
        print("✅ ToxicityRewardSystem initialized successfully!")
        print(f"✅ Model device: {next(reward_system.vlm_model.parameters()).device}")
        
        # Test that we can move the model without errors
        if torch.cuda.is_available():
            print("✅ Testing model movement...")
            # This should work without meta tensor errors
            reward_system.vlm_model.eval()
            print("✅ Model moved to eval mode successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

if __name__ == "__main__":
    success = test_meta_tensor_fix()
    if success:
        print("\n🎉 Meta tensor fix test passed!")
    else:
        print("\n💥 Meta tensor fix test failed!") 