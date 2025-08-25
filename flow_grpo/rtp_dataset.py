import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class RTPPromptData:
    """Data structure for Real Toxicity Prompts dataset entries."""
    prompt_id: str
    text: str
    toxicity_scores: Dict[str, float]
    is_challenging: bool
    source_info: Dict[str, Any]


class RealToxicityPromptsDataset(Dataset):
    """
    Dataset for Real Toxicity Prompts challenge subset.
    Loads the first 132 prompts as specified for Flow-RTPO training.
    """
    
    def __init__(self, 
                 max_prompts: int = 132,
                 split: str = "train",
                 cache_dir: Optional[str] = None,
                 local_dataset_path: Optional[str] = None):
        self.max_prompts = max_prompts
        self.split = split
        self.cache_dir = cache_dir
        self.local_dataset_path = local_dataset_path
        
        # Load dataset
        if local_dataset_path and os.path.exists(local_dataset_path):
            self.prompts = self._load_from_local()
        else:
            self.prompts = self._load_from_huggingface()
        
        # Limit to specified number of prompts
        if len(self.prompts) > self.max_prompts:
            self.prompts = self.prompts[:self.max_prompts]
        
        print(f"Loaded {len(self.prompts)} prompts from Real Toxicity Prompts dataset")
    
    def _load_from_local(self) -> List[RTPPromptData]:
        """Load RTP dataset from local directory."""
        try:
            prompts = []
            
            # Check if it's a HuggingFace dataset format
            if os.path.exists(os.path.join(self.local_dataset_path, "dataset_info.json")):
                # Load as HuggingFace dataset from local path
                dataset = load_dataset(self.local_dataset_path)
                train_data = dataset["train"]
                
                for i, item in enumerate(train_data):
                    # Extract prompt data
                    prompt_data = item.get("prompt", {})
                    if not prompt_data or "text" not in prompt_data:
                        continue
                    
                    # Only include challenging prompts
                    if not item.get("challenging", False):
                        continue
                    
                    # Create RTPPromptData
                    prompt = RTPPromptData(
                        prompt_id=f"rtp_{i}",
                        text=prompt_data["text"],
                        toxicity_scores={
                            "toxicity": prompt_data.get("toxicity", 0.0),
                            "severe_toxicity": prompt_data.get("severe_toxicity", 0.0),
                            "identity_attack": prompt_data.get("identity_attack", 0.0),
                            "insult": prompt_data.get("insult", 0.0),
                            "threat": prompt_data.get("threat", 0.0),
                            "profanity": prompt_data.get("profanity", 0.0),
                            "sexually_explicit": prompt_data.get("sexually_explicit", 0.0),
                            "flirtation": prompt_data.get("flirtation", 0.0),
                        },
                        is_challenging=item.get("challenging", True),
                        source_info={
                            "filename": item.get("filename", ""),
                            "begin": item.get("begin", 0),
                            "end": item.get("end", 0),
                        }
                    )
                    prompts.append(prompt)
            
            # Check if it's a JSONL format
            elif os.path.exists(os.path.join(self.local_dataset_path, "train.jsonl")):
                with open(os.path.join(self.local_dataset_path, "train.jsonl"), 'r') as f:
                    for i, line in enumerate(f):
                        item = json.loads(line.strip())
                        prompt_data = item.get("prompt", {})
                        if not prompt_data or "text" not in prompt_data:
                            continue
                        
                        # Only include challenging prompts
                        if not item.get("challenging", False):
                            continue
                        
                        # Create RTPPromptData
                        prompt = RTPPromptData(
                            prompt_id=f"rtp_{i}",
                            text=prompt_data["text"],
                            toxicity_scores={
                                "toxicity": prompt_data.get("toxicity", 0.0),
                                "severe_toxicity": prompt_data.get("severe_toxicity", 0.0),
                                "identity_attack": prompt_data.get("identity_attack", 0.0),
                                "insult": prompt_data.get("insult", 0.0),
                                "threat": prompt_data.get("threat", 0.0),
                                "profanity": prompt_data.get("profanity", 0.0),
                                "sexually_explicit": prompt_data.get("sexually_explicit", 0.0),
                                "flirtation": prompt_data.get("flirtation", 0.0),
                            },
                            is_challenging=item.get("challenging", True),
                            source_info={
                                "filename": item.get("filename", ""),
                                "begin": item.get("begin", 0),
                                "end": item.get("end", 0),
                            }
                        )
                        prompts.append(prompt)
            
            # Check if it's a simple JSON format
            elif os.path.exists(os.path.join(self.local_dataset_path, "prompts.json")):
                with open(os.path.join(self.local_dataset_path, "prompts.json"), 'r') as f:
                    data = json.load(f)
                    for i, item in enumerate(data):
                        prompt_data = item.get("prompt", {})
                        if not prompt_data or "text" not in prompt_data:
                            continue
                        
                        # Only include challenging prompts
                        if not item.get("challenging", False):
                            continue
                        
                        # Create RTPPromptData
                        prompt = RTPPromptData(
                            prompt_id=f"rtp_{i}",
                            text=prompt_data["text"],
                            toxicity_scores={
                                "toxicity": prompt_data.get("toxicity", 0.0),
                                "severe_toxicity": prompt_data.get("severe_toxicity", 0.0),
                                "identity_attack": prompt_data.get("identity_attack", 0.0),
                                "insult": prompt_data.get("insult", 0.0),
                                "threat": prompt_data.get("threat", 0.0),
                                "profanity": prompt_data.get("profanity", 0.0),
                                "sexually_explicit": prompt_data.get("sexually_explicit", 0.0),
                                "flirtation": prompt_data.get("flirtation", 0.0),
                            },
                            is_challenging=item.get("challenging", True),
                            source_info={
                                "filename": item.get("filename", ""),
                                "begin": item.get("begin", 0),
                                "end": item.get("end", 0),
                            }
                        )
                        prompts.append(prompt)
            
            print(f"Successfully loaded {len(prompts)} prompts from local dataset: {self.local_dataset_path}")
            return prompts
            
        except Exception as e:
            print(f"Failed to load from local dataset: {e}")
            return self._create_mock_prompts()
    
    def _load_from_huggingface(self) -> List[RTPPromptData]:
        """Load RTP dataset from HuggingFace."""
        try:
            dataset = load_dataset("allenai/real-toxicity-prompts", cache_dir=self.cache_dir)
            train_data = dataset["train"]
            
            prompts = []
            for i, item in enumerate(train_data):
                # Extract prompt data
                prompt_data = item.get("prompt", {})
                if not prompt_data or "text" not in prompt_data:
                    continue
                
                # Only include challenging prompts
                if not item.get("challenging", False):
                    continue
                
                # Create RTPPromptData
                prompt = RTPPromptData(
                    prompt_id=f"rtp_{i}",
                    text=prompt_data["text"],
                    toxicity_scores={
                        "toxicity": prompt_data.get("toxicity", 0.0),
                        "severe_toxicity": prompt_data.get("severe_toxicity", 0.0),
                        "identity_attack": prompt_data.get("identity_attack", 0.0),
                        "insult": prompt_data.get("insult", 0.0),
                        "threat": prompt_data.get("threat", 0.0),
                        "profanity": prompt_data.get("profanity", 0.0),
                        "sexually_explicit": prompt_data.get("sexually_explicit", 0.0),
                        "flirtation": prompt_data.get("flirtation", 0.0),
                    },
                    is_challenging=item.get("challenging", True),
                    source_info={
                        "filename": item.get("filename", ""),
                        "begin": item.get("begin", 0),
                        "end": item.get("end", 0),
                    }
                )
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            return self._create_mock_prompts()
    
    def _create_mock_prompts(self) -> List[RTPPromptData]:
        """Create mock prompts for testing if HuggingFace loading fails."""
        mock_prompts = [
            "Write a story about",
            "The weather today is",
            "My favorite food is",
            "In the future, technology will",
            "The most important thing in life is"
        ]
        
        prompts = []
        for i, text in enumerate(mock_prompts):
            prompt = RTPPromptData(
                prompt_id=f"mock_{i}",
                text=text,
                toxicity_scores={
                    "toxicity": 0.0,
                    "severe_toxicity": 0.0,
                    "identity_attack": 0.0,
                    "insult": 0.0,
                    "threat": 0.0,
                    "profanity": 0.0,
                    "sexually_explicit": 0.0,
                    "flirtation": 0.0,
                },
                is_challenging=False,
                source_info={"filename": "mock", "begin": 0, "end": len(text)}
            )
            prompts.append(prompt)
        
        return prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        prompt_data = self.prompts[idx]
        return {
            "prompt": prompt_data.text,
            "metadata": {
                "prompt_id": prompt_data.prompt_id,
                "toxicity_scores": prompt_data.toxicity_scores,
                "is_challenging": prompt_data.is_challenging,
                "source_info": prompt_data.source_info
            }
        }
    
    @staticmethod
    def collate_fn(examples: List[Dict[str, Any]]):
        """Collate function for DataLoader."""
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded prompts."""
        if not self.prompts:
            return {}
        
        # Compute toxicity statistics (filter out None values)
        toxicity_scores = [p.toxicity_scores["toxicity"] for p in self.prompts if p.toxicity_scores["toxicity"] is not None]
        challenging_count = sum(1 for p in self.prompts if p.is_challenging)
        
        stats = {
            "total_prompts": len(self.prompts),
            "challenging_prompts": challenging_count,
            "non_challenging_prompts": len(self.prompts) - challenging_count,
            "toxicity_stats": {
                "mean": sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0,
                "min": min(toxicity_scores) if toxicity_scores else 0.0,
                "max": max(toxicity_scores) if toxicity_scores else 0.0,
            },
            "sample_prompts": [p.text[:50] + "..." if len(p.text) > 50 else p.text 
                             for p in self.prompts[:5]]
        }
        
        return stats
    
    def save_prompts_to_file(self, filepath: str):
        """Save loaded prompts to a JSON file for inspection."""
        data = []
        for prompt in self.prompts:
            data.append({
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "toxicity_scores": prompt.toxicity_scores,
                "is_challenging": prompt.is_challenging,
                "source_info": prompt.source_info
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} prompts to {filepath}")


class RTPromptsDataLoader:
    """Data loader wrapper for RTP dataset with batch processing."""
    
    def __init__(self, 
                 dataset: RealToxicityPromptsDataset,
                 batch_size: int = 8,
                 shuffle: bool = True,
                 num_workers: int = 0):
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_rtp_dataset_and_loader(max_prompts: int = 132,
                                  batch_size: int = 8,
                                  shuffle: bool = True,
                                  cache_dir: Optional[str] = None,
                                  local_dataset_path: Optional[str] = None) -> tuple:
    """
    Convenience function to create RTP dataset and dataloader.
    
    Returns:
        (dataset, dataloader) tuple
    """
    dataset = RealToxicityPromptsDataset(
        max_prompts=max_prompts,
        cache_dir=cache_dir,
        local_dataset_path=local_dataset_path
    )
    
    dataloader = RTPromptsDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Test the dataset
    print("Testing Real Toxicity Prompts Dataset...")
    
    dataset, dataloader = create_rtp_dataset_and_loader(max_prompts=10, batch_size=3)
    
    print(f"Dataset size: {len(dataset)}")
    print("Dataset stats:", dataset.get_prompt_stats())
    
    # Test iteration
    for i, (prompts, metadatas) in enumerate(dataloader):
        print(f"\nBatch {i}:")
        for j, (prompt, metadata) in enumerate(zip(prompts, metadatas)):
            print(f"  {j}: {prompt[:50]}...")
            print(f"     Toxicity: {metadata['toxicity_scores']['toxicity']:.3f}")
        
        if i >= 2:  # Only show first few batches
            break
    
    # Save sample to file
    dataset.save_prompts_to_file("rtp_sample_prompts.json")