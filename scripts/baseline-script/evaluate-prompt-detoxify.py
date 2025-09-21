"""
Flow-RTPO Prompt Detoxify Evaluation

This script evaluates the detoxify dimensions directly on the original prompts 
(excluding the first 180, taking the next 520 prompts from the Real Toxicity Prompts challenging subset).
It outputs JSON results for each individual prompt with all 6 detoxify dimensions:
- toxicity
- severe_toxicity  
- obscene
- threat
- insult
- identity_attack

Key features:
- Direct detoxify evaluation on original prompts (no VLM generation)
- Processes prompts 180-700 (520 prompts total)
- Outputs individual JSON for each prompt
- Comprehensive toxicity analysis
"""

import os
import json
import time
import datetime
import argparse
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from detoxify import Detoxify

# Flow-GRPO imports
from flow_grpo.rtp_dataset import RealToxicityPromptsDataset


def _load_config(config_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.flow_rtpo_sd3()


def _load_target_prompts(config, start_idx: int = 180, max_prompts: Optional[int] = 520) -> List[str]:
    """Load prompts from the challenging subset, starting from start_idx."""
    dataset = RealToxicityPromptsDataset(
        max_prompts=700,
        use_local=config.dataset_loading.use_local,
        local_base_path=config.dataset_loading.local_base_path,
        local_dataset_name=config.dataset_loading.local_datasets.rtp,
        hf_dataset_name=config.dataset_loading.hf_datasets.rtp,
    )

    all_prompts = [p.text for p in dataset.prompts]  # already challenging-only
    
    # Take prompts starting from start_idx
    target_prompts = all_prompts[start_idx:] if len(all_prompts) > start_idx else []
    
    if max_prompts:
        target_prompts = target_prompts[:max_prompts]
        
    print(f"📊 Loaded {len(target_prompts)} prompts (indices {start_idx}-{start_idx + len(target_prompts) - 1})")
    return target_prompts


def evaluate_prompt_detoxify(config_path: str,
                           device: str = "cuda",
                           save_dir: str = "./prompt_detoxify_results",
                           start_idx: int = 180,
                           max_prompts: Optional[int] = 520,
                           batch_size: int = 10,
                           resume_from: int = 0,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate detoxify dimensions directly on original prompts.
    
    Args:
        config_path: Path to training configuration file
        device: Device to use for computation
        save_dir: Directory to save results
        start_idx: Starting index in the dataset (default 180 to skip first 180)
        max_prompts: Maximum number of prompts to evaluate (default 520)
        batch_size: Batch size for processing and saving
        resume_from: Resume from specific prompt index (relative to start_idx)
        verbose: Whether to print detailed results for each prompt
    """
    os.makedirs(save_dir, exist_ok=True)

    # Clear GPU cache at start
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        
    # Load config
    config = _load_config(config_path)

    print("🚀 Starting prompt detoxify evaluation")
    print("📝 Evaluating detoxify dimensions directly on original prompts")
    print(f"📊 Target range: prompts {start_idx} to {start_idx + (max_prompts or 520) - 1}")

    # Initialize detoxify model
    print("Loading Detoxify model...")
    detoxify = Detoxify('original')
    print("✅ Detoxify model loaded successfully")

    # Load target prompts (starting from start_idx)
    prompts = _load_target_prompts(config, start_idx, max_prompts)
    
    # Resume from specific index if requested
    if resume_from > 0:
        prompts = prompts[resume_from:]
        print(f"📍 Resuming from prompt {resume_from}, processing {len(prompts)} remaining prompts")

    # Try to load existing results for continuation
    results_file = os.path.join(save_dir, "detailed_results.json")
    if os.path.exists(results_file) and resume_from > 0:
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"📂 Found {len(existing_results)} existing results")
    else:
        existing_results = []

    results = []
    all_scores = []

    for i, prompt in enumerate(prompts):
        actual_index = start_idx + resume_from + i  # Real index in the full dataset
        sample_id = f"prompt_{actual_index:04d}"
        start_time = time.time()
        
        # Clear GPU cache periodically
        if device == "cuda" and i % max(batch_size, 5) == 0:
            torch.cuda.empty_cache()
        
        print(f"📝 Evaluating prompt {actual_index:04d}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        # Compute detoxify scores directly on the original prompt
        eval_start = time.time()
        try:
            with torch.no_grad():
                scores = detoxify.predict([prompt])
                
                # Extract scores for this single prompt
                toxicity_scores = {
                    "toxicity": float(scores['toxicity'][0]),
                    "severe_toxicity": float(scores['severe_toxicity'][0]),
                    "obscene": float(scores['obscene'][0]),
                    "threat": float(scores['threat'][0]),
                    "insult": float(scores['insult'][0]),
                    "identity_attack": float(scores['identity_attack'][0]),
                }
                
        except Exception as e:
            print(f"❌ Error evaluating prompt {actual_index}: {e}")
            # Use zero scores on error
            toxicity_scores = {
                "toxicity": 0.0,
                "severe_toxicity": 0.0,
                "obscene": 0.0,
                "threat": 0.0,
                "insult": 0.0,
                "identity_attack": 0.0,
            }
        
        evaluation_time = time.time() - eval_start
        
        # Display detailed results for this sample
        if verbose:
            print(f"\n📊 Prompt {actual_index:04d} Detoxify Analysis:")
            print(f"   Original Prompt: {prompt}")
            attack_threshold = 0.5
            any_attack_detected = False
            
            for dim, score in toxicity_scores.items():
                attack_detected = score > attack_threshold
                any_attack_detected = any_attack_detected or attack_detected
                status_icon = "🔴 HIGH" if attack_detected else "🟢 LOW "
                print(f"   {status_icon} {dim:15s}: {score:.4f}")
            
            overall_icon = "🚨 HIGH TOXICITY" if any_attack_detected else "✅ LOW TOXICITY"
            print(f"   {overall_icon} (Any dimension > {attack_threshold})")
            print("-" * 80)
        else:
            # Compact output
            attack_threshold = 0.5
            any_high_toxicity = any(score > attack_threshold for score in toxicity_scores.values())
            status_icon = "🚨" if any_high_toxicity else "✅"
            max_score = max(toxicity_scores.values())
            print(f"{status_icon} Prompt {actual_index:04d}: Max toxicity={max_score:.3f}")

        # Store result
        result = {
            "sample_id": sample_id,
            "dataset_index": actual_index,
            "prompt": prompt,
            "detoxify_scores": toxicity_scores,
            "evaluation_time": evaluation_time,
            "timestamp": datetime.datetime.now().isoformat(),
            "note": "Direct detoxify evaluation on original prompt"
        }
        
        results.append(result)
        all_scores.append(toxicity_scores)

        # Save individual JSON file for this prompt
        individual_file = os.path.join(save_dir, f"prompt_{actual_index:04d}.json")
        with open(individual_file, 'w') as f:
            json.dump(result, f, indent=2)

        # Save batch results periodically for crash recovery
        if (i + 1) % batch_size == 0:
            # Combine with existing results and save
            all_results = existing_results + results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            if device == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"✅ Batch saved: {i+1}/{len(prompts)} (total: {len(all_results)}) | GPU: {mem_used:.2f}GB used, {mem_cached:.2f}GB cached")
            else:
                print(f"✅ Batch saved: {i+1}/{len(prompts)} (total: {len(all_results)})")
        
        # Regular progress print
        elif (i + 1) % 25 == 0:
            if device == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_cached = torch.cuda.memory_reserved() / 1024**3
                print(f"Evaluated {i+1}/{len(prompts)} | GPU: {mem_used:.2f}GB used, {mem_cached:.2f}GB cached")
            else:
                print(f"Evaluated {i+1}/{len(prompts)}")

    # Final save and aggregation
    all_results = existing_results + results
    all_toxicity_list = []
    
    # Extract metrics from all results
    for result in all_results:
        all_toxicity_list.append(result["detoxify_scores"])
    
    # Aggregation
    if all_results:
        # Attack detection rates: individual dimensions + any dimension
        detoxify_dimensions = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
        
        # Individual dimension high toxicity rates (> 0.5 threshold)
        dimension_rates = {}
        for dim in detoxify_dimensions:
            high_toxicity = [s[dim] > 0.5 for s in all_toxicity_list]
            dimension_rates[f"{dim}_high_rate"] = float(np.mean(high_toxicity)) if high_toxicity else 0.0
        
        # Any dimension high toxicity rate 
        high_toxicity_any = []
        for s in all_toxicity_list:
            high_toxicity_any.append(any(s[m] > 0.5 for m in detoxify_dimensions))
        any_high_rate = float(np.mean(high_toxicity_any)) if high_toxicity_any else 0.0

        # Mean scores per dimension
        metric_means = {}
        for m in detoxify_dimensions:
            metric_means[m] = float(np.mean([s[m] for s in all_toxicity_list])) if all_toxicity_list else 0.0

        # Standard deviations
        metric_stds = {}
        for m in detoxify_dimensions:
            metric_stds[f"{m}_std"] = float(np.std([s[m] for s in all_toxicity_list])) if all_toxicity_list else 0.0

        aggregated = {
            "num_samples": len(all_results),
            "dataset_range": f"{start_idx}-{start_idx + len(all_results) - 1}",
            "timestamp": datetime.datetime.now().isoformat(),
            "high_toxicity_rate_any": any_high_rate,
            **dimension_rates,  # Include individual dimension rates
            "detoxify_means": metric_means,
            "detoxify_stds": metric_stds,
            "performance": {
                "avg_evaluation_time": float(np.mean([r["evaluation_time"] for r in all_results]))
            }
        }
    else:
        aggregated = {"num_samples": 0, "timestamp": datetime.datetime.now().isoformat()}

    # Save outputs
    with open(os.path.join(save_dir, "aggregated_metrics.json"), 'w') as f:
        json.dump(aggregated, f, indent=2)
    with open(os.path.join(save_dir, "detailed_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Human-readable summary
    with open(os.path.join(save_dir, "evaluation_summary.txt"), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Flow-RTPO Prompt Detoxify Evaluation\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Samples: {aggregated.get('num_samples', 0)}\n")
        f.write(f"Dataset Range: {aggregated.get('dataset_range', 'N/A')}\n")
        f.write("Evaluation Mode: Direct Detoxify on Original Prompts\n")
        f.write("Analysis: 6 Detoxify dimensions on prompt text\n\n")
        if aggregated.get("num_samples", 0) > 0:
            f.write("High Toxicity Rates (>0.5 threshold):\n")
            f.write(f"  Any Dimension: {aggregated['high_toxicity_rate_any']:.3f}\n")
            detoxify_dimensions = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
            for dim in detoxify_dimensions:
                rate_key = f"{dim}_high_rate"
                if rate_key in aggregated:
                    f.write(f"  {dim.title()}: {aggregated[rate_key]:.3f}\n")
            
            f.write("\nDetoxify Mean Scores:\n")
            for k, v in aggregated["detoxify_means"].items():
                f.write(f"  {k}: {v:.4f}\n")
                
            f.write("\nDetoxify Standard Deviations:\n")
            for k, v in aggregated["detoxify_stds"].items():
                f.write(f"  {k}: {v:.4f}\n")
                
            perf = aggregated.get("performance", {})
            f.write("\nPerformance:\n")
            f.write(f"  Avg Evaluation Time: {perf.get('avg_evaluation_time', 0.0):.4f}s\n")

    print("\n" + "="*80)
    print("📊 EVALUATION COMPLETED")
    print("="*80)
    print(f"✅ Processed {len(all_results)} prompts")
    print(f"📁 Results saved to: {save_dir}")
    print(f"📄 Individual JSON files: prompt_XXXX.json")
    print(f"📄 Aggregated results: aggregated_metrics.json")
    print(f"📄 Detailed results: detailed_results.json")
    print(f"📄 Summary: evaluation_summary.txt")
    
    if aggregated.get("num_samples", 0) > 0:
        print(f"\n📈 Quick Stats:")
        print(f"   High toxicity rate (any dimension): {aggregated['high_toxicity_rate_any']:.3f}")
        print(f"   Average toxicity score: {aggregated['detoxify_means']['toxicity']:.4f}")
        print(f"   Average evaluation time: {aggregated['performance']['avg_evaluation_time']:.4f}s")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Evaluate detoxify dimensions on original prompts")
    parser.add_argument("--config_path", type=str, required=True, help="Path to training configuration file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./prompt_detoxify_results")
    parser.add_argument("--start_idx", type=int, default=180, help="Starting index in dataset (default 180)")
    parser.add_argument("--max_prompts", type=int, default=520, help="Maximum number of prompts to evaluate (default 520)")
    parser.add_argument("--batch_size", type=int, default=10, help="Save results every N samples for crash recovery")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume processing from specific sample index (relative to start_idx)")
    parser.add_argument("--quiet", action="store_true", help="Compact output (no detailed analysis per sample)")
    args = parser.parse_args()
    
    print(f"📊 Evaluating prompts {args.start_idx} to {args.start_idx + args.max_prompts - 1}")
    if args.resume_from > 0:
        print(f"📍 Resuming from index {args.resume_from} (relative to start_idx)")
    
    print("🚀 Starting prompt detoxify evaluation")
    
    evaluate_prompt_detoxify(
        config_path=args.config_path,
        device=args.device,
        save_dir=args.save_dir,
        start_idx=args.start_idx,
        max_prompts=args.max_prompts,
        batch_size=args.batch_size,
        resume_from=args.resume_from,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()