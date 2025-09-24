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
    Dataset for Real Toxicity Prompts.
    Randomly selects up to max_prompts prompts from the dataset.
    """
    
    def __init__(self, 
                 max_prompts: int = 5000,
                 split: str = "train",
                 cache_dir: Optional[str] = None,
                 use_local: bool = False,
                 local_base_path: str = "/mnt/data/group/zhaoliangjie/ICLR-work/",
                 local_dataset_name: str = "real-toxicity-prompts",
                 hf_dataset_name: str = "allenai/real-toxicity-prompts",
                 random_seed: int = 42,
                 heldout_test_size: int = 1000,  # 20% of 5000
                 training_test_size: int = 36):
        self.max_prompts = max_prompts
        self.split = split
        self.cache_dir = cache_dir
        self.use_local = use_local
        self.local_base_path = local_base_path
        self.local_dataset_name = local_dataset_name
        self.hf_dataset_name = hf_dataset_name
        self.random_seed = random_seed
        self.heldout_test_size = heldout_test_size
        self.training_test_size = training_test_size
        
        # Load dataset
        all_prompts = self._load_dataset()
        
        # Randomly select prompts if we have more than max_prompts
        if len(all_prompts) > self.max_prompts:
            import random
            random.seed(self.random_seed)
            all_prompts = random.sample(all_prompts, self.max_prompts)
        
        # Split into heldout test, training test, and training sets
        import random
        random.seed(self.random_seed)
        shuffled_prompts = all_prompts.copy()
        random.shuffle(shuffled_prompts)
        
        # Split: heldout_test (20%) | training_test (40 prompts) | train (rest)
        self.heldout_test_prompts = shuffled_prompts[:self.heldout_test_size]
        remaining_prompts = shuffled_prompts[self.heldout_test_size:]
        self.training_test_prompts = remaining_prompts[:self.training_test_size]
        self.train_prompts = remaining_prompts[self.training_test_size:]
        
        # Set prompts based on split
        if split == "heldout_test":
            self.prompts = self.heldout_test_prompts
        elif split == "training_test":
            self.prompts = self.training_test_prompts
        else:  # train
            self.prompts = self.train_prompts
        
        print(f"Loaded {len(self.prompts)} prompts from Real Toxicity Prompts dataset (split: {split})")
        print(f"Dataset splits - Heldout test: {len(self.heldout_test_prompts)}, Training test: {len(self.training_test_prompts)}, Train: {len(self.train_prompts)}")
        print(f"Loading mode: {'Local' if self.use_local else 'HuggingFace'}")
    
    def _load_dataset(self) -> List[RTPPromptData]:
        """Load RTP dataset from either local path or HuggingFace."""
        if self.use_local:
            return self._load_from_local()
        else:
            return self._load_from_huggingface()
    
    def _load_from_local(self) -> List[RTPPromptData]:
        """Load RTP dataset from local path."""
        try:
            local_dataset_path = os.path.join(self.local_base_path, self.local_dataset_name)
            print(f"[LOCAL] Loading dataset from: {local_dataset_path}")
            
            if not os.path.exists(local_dataset_path):
                print(f"[LOCAL] Local dataset path does not exist: {local_dataset_path}")
                print("[LOCAL] Falling back to HuggingFace loading...")
                return self._load_from_huggingface()
            
            # Try to load from local HuggingFace cache format
            try:
                dataset = load_dataset(local_dataset_path, cache_dir=self.cache_dir)
                train_data = dataset["train"]
                print(f"[LOCAL] Successfully loaded local dataset with {len(train_data)} samples")
            except Exception as e:
                print(f"[LOCAL] Failed to load as HuggingFace format: {e}")
                # Try to load from JSONL file
                jsonl_path = os.path.join(local_dataset_path, "prompts.jsonl")
                if os.path.exists(jsonl_path):
                    print(f"[LOCAL] Loading from JSONL file: {jsonl_path}")
                    train_data = []
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            train_data.append(json.loads(line.strip()))
                    print(f"[LOCAL] Successfully loaded {len(train_data)} records from JSONL")
                else:
                    print(f"[LOCAL] No JSONL file found at: {jsonl_path}")
                    print("[LOCAL] Falling back to HuggingFace loading...")
                    return self._load_from_huggingface()
            
            prompts = []
            for i, item in enumerate(train_data):
                # Extract prompt data
                prompt_data = item.get("prompt", {})
                if not prompt_data or "text" not in prompt_data:
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
                    is_challenging=item.get("challenging", False),
                    source_info={
                        "filename": item.get("filename", ""),
                        "begin": item.get("begin", 0),
                        "end": item.get("end", 0),
                    }
                )
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            print(f"[LOCAL] Failed to load from local path: {e}")
            print("[LOCAL] Falling back to HuggingFace loading...")
            return self._load_from_huggingface()
    
    def _load_from_huggingface(self) -> List[RTPPromptData]:
        """Load RTP dataset from HuggingFace."""
        try:
            print(f"[HF] Loading dataset from HuggingFace: {self.hf_dataset_name}")
            dataset = load_dataset(self.hf_dataset_name, cache_dir=self.cache_dir)
            train_data = dataset["train"]
            
            prompts = []
            for i, item in enumerate(train_data):
                # Extract prompt data
                prompt_data = item.get("prompt", {})
                if not prompt_data or "text" not in prompt_data:
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
                    is_challenging=item.get("challenging", False),
                    source_info={
                        "filename": item.get("filename", ""),
                        "begin": item.get("begin", 0),
                        "end": item.get("end", 0),
                    }
                )
                prompts.append(prompt)
                
            return prompts
            
        except Exception as e:
            print(f"[HF] Failed to load from HuggingFace: {e}")
            return self._create_mock_prompts()
    
    def _create_mock_prompts(self) -> List[RTPPromptData]:
        """Create mock prompts for testing if loading fails."""
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
            "split": self.split,
            "total_prompts": len(self.prompts),
            "challenging_prompts": challenging_count,
            "non_challenging_prompts": len(self.prompts) - challenging_count,
            "dataset_splits": {
                "heldout_test": len(self.heldout_test_prompts),
                "training_test": len(self.training_test_prompts),
                "train": len(self.train_prompts)
            },
            "toxicity_stats": {
                "mean": sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0.0,
                "min": min(toxicity_scores) if toxicity_scores else 0.0,
                "max": max(toxicity_scores) if toxicity_scores else 0.0,
            },
            "sample_prompts": [p.text[:50] + "..." if len(p.text) > 50 else p.text 
                             for p in self.prompts[:5]]
        }
        
        return stats
    
    def get_all_splits(self) -> Dict[str, List[RTPPromptData]]:
        """Get all dataset splits."""
        return {
            "heldout_test": self.heldout_test_prompts,
            "training_test": self.training_test_prompts,
            "train": self.train_prompts
        }
    
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


def create_rtp_dataset_and_loader(max_prompts: int = 5000,
                                  batch_size: int = 8,
                                  shuffle: bool = True,
                                  cache_dir: Optional[str] = None,
                                  use_local: bool = False,
                                  local_base_path: str = "/mnt/data/group/zhaoliangjie/ICLR-work/",
                                  local_dataset_name: str = "real-toxicity-prompts",
                                  hf_dataset_name: str = "allenai/real-toxicity-prompts",
                                  random_seed: int = 42,
                                  split: str = "train",
                                  heldout_test_size: int = 1000,
                                  training_test_size: int = 36) -> tuple:
    """
    Convenience function to create RTP dataset and dataloader.
    
    Returns:
        (dataset, dataloader) tuple
    """
    dataset = RealToxicityPromptsDataset(
        max_prompts=max_prompts,
        split=split,
        cache_dir=cache_dir,
        use_local=use_local,
        local_base_path=local_base_path,
        local_dataset_name=local_dataset_name,
        hf_dataset_name=hf_dataset_name,
        random_seed=random_seed,
        heldout_test_size=heldout_test_size,
        training_test_size=training_test_size
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