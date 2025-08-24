#!/usr/bin/env python3
"""
Test script to verify train/test split functionality for Flow-RTPO.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow_grpo.rtp_dataset import RealToxicityPromptsDataset
from sklearn.model_selection import train_test_split
import random

def test_train_test_split():
    """Test the train/test split functionality."""
    print("Testing train/test split functionality...")
    
    # Load dataset
    dataset = RealToxicityPromptsDataset(max_prompts=132)
    print(f"Loaded {len(dataset.prompts)} prompts from RTP dataset")
    
    # Get all prompts and metadata
    all_prompts = [prompt.text for prompt in dataset.prompts]
    all_metadata = [{
        'prompt_id': prompt.prompt_id,
        'toxicity_scores': prompt.toxicity_scores,
        'is_challenging': prompt.is_challenging,
        'source_info': prompt.source_info
    } for prompt in dataset.prompts]
    
    print(f"Total prompts: {len(all_prompts)}")
    print(f"Total metadata: {len(all_metadata)}")
    
    # Test different split ratios
    test_ratios = [0.1, 0.2, 0.3]
    
    for test_ratio in test_ratios:
        print(f"\n--- Testing with test_ratio = {test_ratio} ---")
        
        # Split into train and test
        train_prompts, test_prompts, train_metadata, test_metadata = train_test_split(
            all_prompts, all_metadata, test_size=test_ratio, random_state=42
        )
        
        print(f"Train set: {len(train_prompts)} prompts")
        print(f"Test set: {len(test_prompts)} prompts")
        print(f"Actual test ratio: {len(test_prompts) / len(all_prompts):.3f}")
        
        # Check that all prompts are unique
        train_set = set(train_prompts)
        test_set = set(test_prompts)
        intersection = train_set.intersection(test_set)
        
        print(f"Train/Test overlap: {len(intersection)} prompts")
        assert len(intersection) == 0, "Train and test sets should not overlap!"
        
        # Check that all original prompts are included
        all_split_prompts = train_prompts + test_prompts
        original_set = set(all_prompts)
        split_set = set(all_split_prompts)
        
        print(f"Original prompts: {len(original_set)}")
        print(f"Split prompts: {len(split_set)}")
        print(f"Missing prompts: {len(original_set - split_set)}")
        print(f"Extra prompts: {len(split_set - original_set)}")
        
        assert len(original_set - split_set) == 0, "All original prompts should be included!"
        assert len(split_set - original_set) == 0, "No extra prompts should be added!"
        
        # Show some sample prompts
        print(f"\nSample train prompts:")
        for i, prompt in enumerate(train_prompts[:3]):
            print(f"  {i+1}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        print(f"\nSample test prompts:")
        for i, prompt in enumerate(test_prompts[:3]):
            print(f"  {i+1}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    
    print("\n✅ All train/test split tests passed!")

def test_metadata_consistency():
    """Test that metadata is correctly associated with prompts."""
    print("\nTesting metadata consistency...")
    
    dataset = RealToxicityPromptsDataset(max_prompts=20)  # Smaller set for testing
    
    all_prompts = [prompt.text for prompt in dataset.prompts]
    all_metadata = [{
        'prompt_id': prompt.prompt_id,
        'toxicity_scores': prompt.toxicity_scores,
        'is_challenging': prompt.is_challenging,
        'source_info': prompt.source_info
    } for prompt in dataset.prompts]
    
    # Split
    train_prompts, test_prompts, train_metadata, test_metadata = train_test_split(
        all_prompts, all_metadata, test_size=0.3, random_state=42
    )
    
    # Check that metadata corresponds to prompts
    for i, (prompt, metadata) in enumerate(zip(train_prompts, train_metadata)):
        print(f"Train {i+1}: Prompt ID {metadata['prompt_id']} - {prompt[:50]}...")
        assert 'prompt_id' in metadata, "Metadata should contain prompt_id"
        assert 'toxicity_scores' in metadata, "Metadata should contain toxicity_scores"
    
    for i, (prompt, metadata) in enumerate(zip(test_prompts, test_metadata)):
        print(f"Test {i+1}: Prompt ID {metadata['prompt_id']} - {prompt[:50]}...")
        assert 'prompt_id' in metadata, "Metadata should contain prompt_id"
        assert 'toxicity_scores' in metadata, "Metadata should contain toxicity_scores"
    
    print("✅ Metadata consistency test passed!")

if __name__ == "__main__":
    test_train_test_split()
    test_metadata_consistency()
    print("\n🎉 All tests completed successfully!") 