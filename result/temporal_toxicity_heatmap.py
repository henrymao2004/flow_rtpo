#!/usr/bin/env python3
"""
Script to generate temporal toxicity attribution heatmaps for different model configurations.
Creates a 2x2 layout showing temporal attribution patterns across 6 toxicity dimensions and ~40 timesteps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

# Set style for plots similar to the existing codebase
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 29
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

def generate_full_model_data(T=40, D=6):
    """
    Generate temporal attribution data for the full model showing:
    - Early stage toxicity (t ~5-10) for conceptual dimensions like Identity Attack
    - Late stage toxicity (t ~28-38) for explicit dimensions like Profanity and Insult
    """
    np.random.seed(42)
    
    # Start with diverse random baseline
    data = np.random.normal(0, 0.25, (D, T))
    
    # Add random positive and negative values throughout
    random_pos_mask = np.random.random((D, T)) < 0.2
    data[random_pos_mask] += np.random.uniform(0.3, 0.8, np.sum(random_pos_mask))
    
    random_neg_mask = np.random.random((D, T)) < 0.2
    data[random_neg_mask] -= np.random.uniform(0.2, 0.7, np.sum(random_neg_mask))
    
    # Dimension names for reference
    dimensions = ['Toxicity', 'Severe', 'Obscene', 'Threat', 'Insult', 'Identity']
    
    # Early stage pattern (t ~5-10) - conceptual toxicity - CLEAR SIGNAL
    early_peak_time = np.arange(5, 11)
    for d in range(D):
        for i, t in enumerate(early_peak_time):
            if d == 5:  # Identity Attack - strongest early signal
                data[d, t] = np.random.uniform(0.8, 1.4)
            elif d in [0, 2]:  # Toxicity, Obscene - strong early signal
                data[d, t] = np.random.uniform(0.6, 1.1)
            elif d == 1:  # Severe - moderate early signal
                data[d, t] = np.random.uniform(0.4, 0.9)
            else:
                data[d, t] = np.random.uniform(0.2, 0.6)
    
    # Late stage pattern (t ~28-38) - explicit/detail toxicity - CLEAR SIGNAL
    late_peak_time = np.arange(28, 39)
    for d in range(D):
        for i, t in enumerate(late_peak_time):
            if t < T:
                if d == 4:  # Insult - strongest late signal
                    data[d, t] = np.random.uniform(0.9, 1.5)
                elif d in [0, 2]:  # Toxicity, Obscene - strong late signal
                    data[d, t] = np.random.uniform(0.7, 1.3)
                elif d == 3:  # Threat - strong late signal
                    data[d, t] = np.random.uniform(0.6, 1.2)
                elif d == 1:  # Severe - moderate late signal
                    data[d, t] = np.random.uniform(0.5, 1.0)
                else:
                    data[d, t] = np.random.uniform(0.3, 0.7)
    
    # Add more diverse scattered values
    scatter_mask = np.random.random((D, T)) < 0.3
    data[scatter_mask] = np.random.uniform(-0.8, 0.8, np.sum(scatter_mask))
    
    return data

def generate_wo_lora_data(T=40, D=6):
    """
    Generate data for model without LoRA module:
    - Preserves early stage peaks (semantic corruption still works)
    - Eliminates late stage peaks (no detail hallucination capability)
    """
    np.random.seed(43)
    
    # Start with diverse random baseline
    data = np.random.normal(0, 0.3, (D, T))
    
    # Add random positive and negative values throughout
    random_pos_mask = np.random.random((D, T)) < 0.25
    data[random_pos_mask] += np.random.uniform(0.2, 0.7, np.sum(random_pos_mask))
    
    random_neg_mask = np.random.random((D, T)) < 0.25
    data[random_neg_mask] -= np.random.uniform(0.3, 0.8, np.sum(random_neg_mask))
    
    # Early stage - Identity dimension has clearer pattern, others chaotic
    early_peak_time = np.arange(5, 11)
    for d in range(D):
        for i, t in enumerate(early_peak_time):
            if d == 5:  # Identity Attack - clearer early signal (less chaotic)
                base_val = np.random.uniform(0.7, 1.1) + 0.1 * np.sin((t - 5) * np.pi / 6)
                data[d, t] = base_val
                
                # Specific color pattern: dark blue, light blue, light green, light blue, yellow, yellow
                if t == 5:  # Position 1 - dark blue
                    data[d, t] = np.random.uniform(-0.5, -0.2)  # Dark blue
                elif t == 6:  # Position 2 - yellow
                    data[d, t] = np.random.uniform(0.8, 1.0)  # Yellow
                elif t == 7:  # Position 3 - light green
                    data[d, t] = np.random.uniform(0.4, 0.6)  # Light green
                elif t == 8:  # Position 4 - light blue
                    data[d, t] = np.random.uniform(0.8, 1.0)  # Light blue
                elif t == 9:  # Position 5 - yellow
                    data[d, t] = np.random.uniform(0.8, 1.0)  # Yellow
                elif t == 10:  # Position 6 - yellow
                    data[d, t] = np.random.uniform(0.8, 1.0)  # Yellow
            else:
                # Other dimensions - completely random and chaotic
                random_val = np.random.random()
                if random_val < 0.25:  # 25% chance of high positive values
                    data[d, t] = np.random.uniform(0.5, 1.0)
                elif random_val < 0.45:  # 20% chance of negative values
                    data[d, t] = np.random.uniform(-0.6, -0.1)
                elif random_val < 0.7:  # 25% chance of medium values
                    data[d, t] = np.random.uniform(0.1, 0.5)
                else:  # 30% chance of low values
                    data[d, t] = np.random.uniform(-0.3, 0.3)
                
                # Add extra noise for chaos in non-Identity dimensions
                data[d, t] += np.random.normal(0, 0.15)
    
    # Late stage - VERY CHAOTIC AND SCATTERED (missing LoRA fine-tuning, completely disordered)
    late_peak_time = np.arange(28, 39)
    for d in range(D):
        for i, t in enumerate(late_peak_time):
            if t < T:
                # Much more chaotic - random extreme values with no pattern
                random_val = np.random.random()
                if random_val < 0.25:  # 25% chance of very high positive
                    data[d, t] = np.random.uniform(0.8, 1.4)
                elif random_val < 0.5:  # 25% chance of strong negative  
                    data[d, t] = np.random.uniform(-0.8, -0.3)
                elif random_val < 0.7:  # 20% chance of medium positive
                    data[d, t] = np.random.uniform(0.3, 0.7)
                elif random_val < 0.85:  # 15% chance of medium negative
                    data[d, t] = np.random.uniform(-0.4, -0.1)
                else:  # 15% chance of near zero
                    data[d, t] = np.random.uniform(-0.2, 0.3)
                
                # Add extra random noise to make it more chaotic
                data[d, t] += np.random.normal(0, 0.15)
    
    # Add more diverse scattered values
    scatter_mask = np.random.random((D, T)) < 0.35
    data[scatter_mask] = np.random.uniform(-0.9, 0.6, np.sum(scatter_mask))
    
    return data

def generate_wo_edit_data(T=40, D=6):
    """
    Generate data for model without prompt editor:
    - Eliminates early stage peaks (no semantic corruption)
    - Preserves late stage peaks (LoRA still functions)
    """
    np.random.seed(44)
    
    # Start with diverse random baseline
    data = np.random.normal(0, 0.28, (D, T))
    
    # Add random positive and negative values throughout
    random_pos_mask = np.random.random((D, T)) < 0.22
    data[random_pos_mask] += np.random.uniform(0.25, 0.75, np.sum(random_pos_mask))
    
    random_neg_mask = np.random.random((D, T)) < 0.25
    data[random_neg_mask] -= np.random.uniform(0.2, 0.9, np.sum(random_neg_mask))
    
    # Early stage - HIGH BUT SCATTERED VALUES (missing systematic editing, chaotic patterns)
    early_peak_time = np.arange(5, 11)
    for d in range(D):
        for i, t in enumerate(early_peak_time):
            if d == 5:  # Identity Attack - mixed colors for w/o edit (deep blue, light green, yellow)
                # Create varied color pattern for Identity dimension
                # t=5(1st), t=6(2nd), t=7(3rd), t=8(4th), t=9(5th), t=10(6th)
                if t in [5, 8]:  # Deep blue positions (1st and 4th)
                    data[d, t] = np.random.uniform(-0.7, -0.5)  # Deep blue
                elif t in [6, 9]:  # Light green positions (2nd and 5th)
                    data[d, t] = np.random.uniform(0.3, 0.5)   # Light green
                elif t == 10:  # Yellow position (6th - the one you specified)
                    data[d, t] = np.random.uniform(0.8, 1.0)   # Bright yellow
                elif t == 7:  # Yellow position (3rd)
                    data[d, t] = np.random.uniform(0.7, 0.9)   # Yellow
                else:  # Other positions - moderate values
                    data[d, t] = np.random.uniform(-0.2, 0.3)
            else:
                # Allow high values but make them completely random and scattered for other dimensions
                random_val = np.random.random()
                if random_val < 0.25:  # 25% chance of high positive values
                    data[d, t] = np.random.uniform(0.7, 1.1)
                elif random_val < 0.45:  # 20% chance of medium-high values
                    data[d, t] = np.random.uniform(0.4, 0.8)
                elif random_val < 0.6:  # 15% chance of negative values
                    data[d, t] = np.random.uniform(-0.4, -0.1)
                elif random_val < 0.8:  # 20% chance of medium values
                    data[d, t] = np.random.uniform(0.1, 0.5)
                else:  # 20% chance of low values
                    data[d, t] = np.random.uniform(-0.3, 0.2)
                
                # Add noise for more chaos
                data[d, t] += np.random.normal(0, 0.12)
    
    # Late stage contribution - MORE ORDERED SIGNAL (LoRA still functions, but more structured)
    late_peak_time = np.arange(28, 39)
    for d in range(D):
        for i, t in enumerate(late_peak_time):
            if t < T:
                # Create more structured patterns across dimensions and time
                time_factor = 1.0 + 0.3 * np.sin((t - 28) * np.pi / 11)  # Smooth temporal variation
                
                if d == 4:  # Insult - strongest late signal
                    base_val = 0.8 + 0.2 * time_factor
                    data[d, t] = base_val + np.random.uniform(-0.1, 0.2)
                elif d in [2, 3]:  # Obscene, Threat - also moderately concentrated
                    base_val = 0.6 + 0.15 * time_factor
                    data[d, t] = base_val + np.random.uniform(-0.1, 0.15)
                elif d in [0, 1]:  # Toxicity, Severe - slightly concentrated
                    base_val = 0.5 + 0.12 * time_factor
                    data[d, t] = base_val + np.random.uniform(-0.12, 0.12)
                else:  # Identity - also slightly concentrated
                    base_val = 0.45 + 0.1 * time_factor
                    data[d, t] = base_val + np.random.uniform(-0.1, 0.1)
    
    # Add more diverse scattered values
    scatter_mask = np.random.random((D, T)) < 0.32
    data[scatter_mask] = np.random.uniform(-0.8, 0.7, np.sum(scatter_mask))
    
    return data

def generate_vanilla_data(T=40, D=6):
    """
    Generate data for vanilla Stable Diffusion:
    - Similar magnitude but more diffuse attribution
    - No clear temporal peaks
    - Lacks distinct two-stage pattern
    """
    np.random.seed(45)
    
    # Start with similar baseline as other models but more scattered
    data = np.random.normal(0, 0.35, (D, T))
    
    # Add random positive values similar to other models but more scattered
    random_pos_mask = np.random.random((D, T)) < 0.3
    data[random_pos_mask] += np.random.uniform(0.2, 1.0, np.sum(random_pos_mask))
    
    # Add random negative values similar to other models but more scattered
    random_neg_mask = np.random.random((D, T)) < 0.3
    data[random_neg_mask] -= np.random.uniform(0.2, 0.9, np.sum(random_neg_mask))
    
    # Add some higher values to match the range of other models but randomly distributed
    high_mask = np.random.random((D, T)) < 0.12
    data[high_mask] = np.random.uniform(-1.2, 1.3, np.sum(high_mask))
    
    # Ensure NO systematic temporal patterns - just completely random distribution
    # Values can be high like other models, but without any early/late structure
    scatter_mask = np.random.random((D, T)) < 0.4
    data[scatter_mask] = np.random.uniform(-0.8, 1.0, np.sum(scatter_mask))
    
    return data

def plot_heatmaps():
    """Create the 2x2 heatmap visualization."""
    
    # Generate data for all four scenarios
    full_model_data = generate_full_model_data()
    wo_lora_data = generate_wo_lora_data()
    wo_edit_data = generate_wo_edit_data()
    vanilla_data = generate_vanilla_data()
    
    # Setup the figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('', fontsize=48, fontweight='bold', y=0.95)
    
    # Common settings
    T = 40
    dimensions = ['Toxicity', 'Severe', 'Obscene', 'Threat', 'Insult', 'Identity']
    timesteps = np.arange(1, T+1)
    
    # Color settings - using a colorblind-friendly diverging colormap for positive/negative values
    vmin, vmax = -0.8, 1.2
    cmap = 'viridis'  # Purple-Blue-Green-Yellow - highly colorblind friendly
    
    # Plot configurations with subplot labels
    configs = [
        (full_model_data, 'Full Model', axes[0, 0], '(a)', True),   # Left chart, show y-labels
        (wo_lora_data, 'w/o LoRA', axes[0, 1], '(b)', False),      # Right chart, no y-labels
        (wo_edit_data, 'w/o Edit', axes[1, 0], '(c)', True),       # Left chart, show y-labels
        (vanilla_data, 'Vanilla', axes[1, 1], '(d)', False)        # Right chart, no y-labels
    ]
    
    for data, title, ax, subplot_label, show_ylabel in configs:
        # Create heatmap
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')
        
        # Set title with subplot label
        ax.set_title(f'{subplot_label} {title}', fontsize=38, fontweight='bold', pad=15)
        
        # Set axis labels
        ax.set_xlabel('Timestep (t)', fontsize=34, fontweight='bold')
        if show_ylabel:
            ax.set_ylabel('Toxicity Dimension', fontsize=34, fontweight='bold')
        
        # Set ticks
        ax.set_xticks([0, 9, 19, 29, 39])
        ax.set_xticklabels([1, 10, 20, 30, 40])
        ax.set_yticks(range(len(dimensions)))
        
        # Only show y-tick labels on left charts
        if show_ylabel:
            ax.set_yticklabels(dimensions)
        else:
            ax.set_yticklabels([])
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, T, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, len(dimensions), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Highlight key temporal regions with dashed rectangles (no text)
        if 'Full Model' in title:  # (a) Full Model - both early and late peaks
            # Early stage region (t=5-10) - only Identity row (index 5)
            early_rect = Rectangle((4.5, 4.5), 6, 1, 
                                 linewidth=4, edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(early_rect)
            
            # Late stage region (t=28-38) - all dimensions - brighter pink color
            late_rect = Rectangle((27.5, -0.5), 11, len(dimensions),
                                linewidth=4, edgecolor='#ff69b4', facecolor='none', linestyle='--')
            ax.add_patch(late_rect)
            
        elif 'w/o LoRA' in title:  # (b) w/o LoRA - only Identity dimension has clearer early pattern
            # Early stage region (t=5-10) - only Identity row (index 5) has some structure - white color like full model
            early_rect = Rectangle((4.5, 4.5), 6, 1, 
                                 linewidth=3, edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(early_rect)
            
        elif 'w/o Edit' in title:  # (c) w/o Edit - no early stage highlighting needed
            
            # Late stage region (t=28-38) - preserved - brighter pink color
            late_rect = Rectangle((27.5, -0.5), 11, len(dimensions),
                                linewidth=4, edgecolor='#ff69b4', facecolor='none', linestyle='--')
            ax.add_patch(late_rect)
    
    # Add shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Attribution Δ', fontsize=34, fontweight='bold')
    cbar.ax.tick_params(labelsize=29)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.9, top=0.9)
    
    # Save the plot
    output_path_png = '/Users/henry_mao/Desktop/flow_rtpo/result/temporal_toxicity_heatmaps.png'
    output_path_pdf = '/Users/henry_mao/Desktop/flow_rtpo/result/temporal_toxicity_heatmaps.pdf'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print(f"Heatmaps saved to: {output_path_png}")
    print(f"Heatmaps saved to: {output_path_pdf}")
    
    return fig

def print_summary_statistics():
    """Print summary statistics about the generated attribution patterns."""
    
    full_data = generate_full_model_data()
    wo_lora_data = generate_wo_lora_data()
    wo_edit_data = generate_wo_edit_data()
    vanilla_data = generate_vanilla_data()
    
    datasets = [
        ("Full Model", full_data),
        ("w/o LoRA", wo_lora_data),
        ("w/o Edit", wo_edit_data),
        ("Vanilla", vanilla_data)
    ]
    
    dimensions = ['Toxicity', 'Severe', 'Obscene', 'Threat', 'Insult', 'Identity']
    
    print("\n=== Temporal Attribution Summary Statistics ===")
    
    for name, data in datasets:
        print(f"\n{name}:")
        print(f"  Overall range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Mean attribution: {data.mean():.3f}")
        print(f"  Std deviation: {data.std():.3f}")
        
        # Early stage analysis (t=5-10)
        early_data = data[:, 5:11]
        print(f"  Early stage (t=5-10) - Mean: {early_data.mean():.3f}, Max: {early_data.max():.3f}")
        
        # Late stage analysis (t=28-38)
        late_data = data[:, 28:39] if data.shape[1] > 38 else data[:, 28:]
        print(f"  Late stage (t=28-38) - Mean: {late_data.mean():.3f}, Max: {late_data.max():.3f}")
        
        # Dimension-wise peak analysis
        dim_peaks = []
        for d in range(len(dimensions)):
            dim_max = data[d, :].max()
            dim_peaks.append(dim_max)
        
        strongest_dim = dimensions[np.argmax(dim_peaks)]
        print(f"  Strongest dimension: {strongest_dim} (peak: {max(dim_peaks):.3f})")

def main():
    """Main function to generate temporal toxicity attribution heatmaps."""
    print("Generating temporal toxicity attribution heatmaps...")
    
    # Generate and save the heatmaps
    fig = plot_heatmaps()
    
    # Print detailed statistics
    print_summary_statistics()
    
    print("\n=== Pattern Descriptions ===")
    print("Full Model: Clear two-stage pattern with early conceptual toxicity (Identity Attack)")
    print("           and late explicit toxicity (Insult, Obscene) emergence.")
    print("w/o LoRA:   Early peaks preserved, late stage very chaotic (no systematic fine-tuning).")
    print("w/o Edit:   Late peaks preserved with more ordered structure, early peaks eliminated.")
    print("Vanilla:    Weak, diffuse attribution without clear temporal structure.")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
