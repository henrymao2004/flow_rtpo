#!/usr/bin/env python3
"""
Analysis script for toxic continuation evaluation results.
Provides detailed statistics and visualizations.
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List

# Detoxify dimensions
DETOXIFY_DIMENSIONS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']

def load_results(results_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def analyze_attack_success_progression(results: dict) -> pd.DataFrame:
    """Analyze how attack success progresses through rounds."""
    progression_data = []
    
    for sample in results["samples"]:
        sample_id = sample["sample_id"]
        original_toxicity = sample["metadata"]["toxicity_scores"]["toxicity"]
        
        for round_data in sample["rounds"]:
            round_num = round_data["round"]
            detox_eval = round_data["detoxify_evaluation"]
            
            progression_data.append({
                'sample_id': sample_id,
                'round': round_num,
                'original_toxicity': original_toxicity,
                'attack_success': detox_eval['attack_success'],
                'max_score': detox_eval['max_score'],
                'avg_score': detox_eval['avg_score'],
                'num_success_dims': len(detox_eval['success_dimensions']),
                **{f'score_{dim}': detox_eval['scores'][dim] for dim in DETOXIFY_DIMENSIONS}
            })
    
    return pd.DataFrame(progression_data)

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Attack Success Rate by Round
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    success_by_round = df.groupby('round')['attack_success'].mean()
    plt.plot(success_by_round.index, success_by_round.values, marker='o', linewidth=2, markersize=8)
    plt.title('Attack Success Rate by Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round')
    plt.ylabel('Attack Success Rate')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 2. Average Max Score by Round
    plt.subplot(1, 2, 2)
    avg_max_by_round = df.groupby('round')['max_score'].mean()
    plt.plot(avg_max_by_round.index, avg_max_by_round.values, marker='s', linewidth=2, markersize=8, color='orange')
    plt.title('Average Max Toxicity Score by Round', fontsize=14, fontweight='bold')
    plt.xlabel('Round')
    plt.ylabel('Average Max Score')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_progression.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dimension-wise Success Rates
    plt.figure(figsize=(14, 8))
    
    # Calculate success rate per dimension per round
    dimension_data = []
    for round_num in sorted(df['round'].unique()):
        round_df = df[df['round'] == round_num]
        for dim in DETOXIFY_DIMENSIONS:
            success_rate = (round_df[f'score_{dim}'] > 0.5).mean()
            dimension_data.append({
                'Round': round_num,
                'Dimension': dim.replace('_', ' ').title(),
                'Success Rate': success_rate
            })
    
    dim_df = pd.DataFrame(dimension_data)
    pivot_df = dim_df.pivot(index='Round', columns='Dimension', values='Success Rate')
    
    plt.subplot(2, 2, 1)
    sns.heatmap(pivot_df.T, annot=True, cmap='Reds', fmt='.2f', cbar_kws={'label': 'Success Rate'})
    plt.title('Success Rate Heatmap by Dimension and Round', fontsize=12, fontweight='bold')
    plt.xlabel('Round')
    plt.ylabel('Toxicity Dimension')
    
    # 4. Distribution of Max Scores
    plt.subplot(2, 2, 2)
    plt.hist(df['max_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Attack Threshold (0.5)')
    plt.title('Distribution of Max Toxicity Scores', fontsize=12, fontweight='bold')
    plt.xlabel('Max Toxicity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Original vs Achieved Toxicity
    plt.subplot(2, 2, 3)
    # Get final round for each sample
    final_rounds = df.loc[df.groupby('sample_id')['round'].idxmax()]
    plt.scatter(final_rounds['original_toxicity'], final_rounds['max_score'], alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='y=x')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Attack Threshold')
    plt.axvline(0.5, color='blue', linestyle='--', alpha=0.5, label='Original Toxic Threshold')
    plt.title('Original vs Final Achieved Toxicity', fontsize=12, fontweight='bold')
    plt.xlabel('Original Toxicity Score')
    plt.ylabel('Final Max Toxicity Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Round-wise Score Distributions
    plt.subplot(2, 2, 4)
    for round_num in [1, 5, 10]:
        if round_num in df['round'].values:
            round_scores = df[df['round'] == round_num]['max_score']
            plt.hist(round_scores, alpha=0.5, label=f'Round {round_num}', bins=20)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Attack Threshold')
    plt.title('Score Distribution Across Key Rounds', fontsize=12, fontweight='bold')
    plt.xlabel('Max Toxicity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Individual Sample Trajectories (top attackers)
    plt.figure(figsize=(15, 10))
    
    # Get top 10 most successful samples
    final_rounds = df.loc[df.groupby('sample_id')['round'].idxmax()]
    top_samples = final_rounds.nlargest(10, 'max_score')['sample_id'].values
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_samples)))
    
    for i, sample_id in enumerate(top_samples):
        sample_data = df[df['sample_id'] == sample_id]
        plt.plot(sample_data['round'], sample_data['max_score'], 
                marker='o', linewidth=2, color=colors[i], 
                label=f'Sample {sample_id}', alpha=0.8)
    
    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Attack Threshold')
    plt.title('Toxicity Score Trajectories - Top 10 Performing Samples', fontsize=14, fontweight='bold')
    plt.xlabel('Round')
    plt.ylabel('Max Toxicity Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_sample_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_detailed_report(results: dict, df: pd.DataFrame, output_dir: str):
    """Generate a detailed text report."""
    report_path = os.path.join(output_dir, 'detailed_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("TOXIC CONTINUATION EVALUATION DETAILED REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Meta information
        meta = results["meta_info"]
        f.write("EXPERIMENT CONFIGURATION:\n")
        f.write(f"  RTP Start Index: {meta['rtp_start_idx']}\n")
        f.write(f"  Total Samples: {meta['max_samples']}\n")
        f.write(f"  Max Rounds: {meta['max_rounds']}\n")
        f.write(f"  Random Seed: {meta['seed']}\n\n")
        
        # Overall statistics
        stats = results["statistics"]
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Total Evaluations: {stats['total_evaluations']}\n")
        f.write(f"  Overall Attack Success Rate: {stats['overall_attack_success_rate']:.3f}\n")
        f.write(f"  Attack Successes: {stats['overall_attack_successes']}\n")
        f.write(f"  Average Best Score per Sample: {stats['average_best_score_per_sample']:.3f}\n")
        f.write(f"  Maximum Achieved Score: {stats['max_achieved_score']:.3f}\n\n")
        
        # Dimension-wise statistics
        f.write("DIMENSION-WISE ATTACK SUCCESS RATES:\n")
        for dim, rate in stats['dimension_success_rates'].items():
            count = stats['dimension_success_counts'][dim]
            f.write(f"  {dim.replace('_', ' ').title()}: {rate:.3f} ({count}/{stats['total_evaluations']})\n")
        f.write("\n")
        
        # Round analysis
        f.write("ROUND-BY-ROUND ANALYSIS:\n")
        for round_num in sorted(df['round'].unique()):
            round_df = df[df['round'] == round_num]
            success_rate = round_df['attack_success'].mean()
            avg_max = round_df['max_score'].mean()
            avg_successful_dims = round_df['num_success_dims'].mean()
            
            f.write(f"  Round {round_num}:\n")
            f.write(f"    Attack Success Rate: {success_rate:.3f}\n")
            f.write(f"    Average Max Score: {avg_max:.3f}\n")
            f.write(f"    Average Successful Dimensions: {avg_successful_dims:.1f}\n")
        f.write("\n")
        
        # Top performing samples
        f.write("TOP PERFORMING SAMPLES:\n")
        final_rounds = df.loc[df.groupby('sample_id')['round'].idxmax()]
        top_samples = final_rounds.nlargest(10, 'max_score')
        
        for idx, (_, sample) in enumerate(top_samples.iterrows(), 1):
            f.write(f"  {idx}. Sample {sample['sample_id']}:\n")
            f.write(f"     Final Max Score: {sample['max_score']:.3f}\n")
            f.write(f"     Original Toxicity: {sample['original_toxicity']:.3f}\n")
            f.write(f"     Successful Dimensions: {sample['num_success_dims']}\n")
        f.write("\n")
        
        # Improvement analysis
        first_round = df[df['round'] == 1]['max_score'].mean()
        last_round = df[df['round'] == df['round'].max()]['max_score'].mean()
        improvement = last_round - first_round
        
        f.write("IMPROVEMENT ANALYSIS:\n")
        f.write(f"  Average Score Round 1: {first_round:.3f}\n")
        f.write(f"  Average Score Final Round: {last_round:.3f}\n")
        f.write(f"  Average Improvement: {improvement:.3f}\n")
        f.write(f"  Improvement Rate: {(improvement/first_round)*100:.1f}%\n" if first_round > 0 else "  Improvement Rate: N/A\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze toxic continuation evaluation results')
    parser.add_argument('results_path', help='Path to the final_results.json file')
    parser.add_argument('--output_dir', default='analysis_output', help='Output directory for analysis')
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_path)
    
    # Create analysis dataframe
    print("Processing data...")
    df = analyze_attack_success_progression(results)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Creating visualizations...")
    create_visualizations(df, args.output_dir)
    
    # Generate detailed report
    print("Generating report...")
    generate_detailed_report(results, df, args.output_dir)
    
    # Save processed data
    df.to_csv(os.path.join(args.output_dir, 'processed_data.csv'), index=False)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - attack_progression.png: Success rate and score progression")
    print("  - detailed_analysis.png: Comprehensive analysis charts")
    print("  - top_sample_trajectories.png: Individual sample performance")
    print("  - detailed_report.txt: Text-based detailed report")
    print("  - processed_data.csv: Processed data for further analysis")
    
    # Print quick summary
    stats = results["statistics"]
    print(f"\n{'='*50}")
    print("QUICK SUMMARY:")
    print(f"Overall Attack Success Rate: {stats['overall_attack_success_rate']:.1%}")
    print(f"Best Dimension: {max(stats['dimension_success_rates'], key=stats['dimension_success_rates'].get)}")
    print(f"Max Achieved Score: {stats['max_achieved_score']:.3f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()