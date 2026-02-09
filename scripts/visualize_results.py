import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_layer_sweep(df, output_dir):
    """Generates plots for layer sweep analysis using matplotlib."""
    # Filter for relevant columns
    df_subset = df[['layer', 'method', 'gap', 'acc@5_winnable', 'off_diag_mean']].copy()
    
    # Standardize method names for cleaner legend
    method_map = {
        'baseline': 'Base (Mean)',
        'abtt': 'PC-Remove (D=10)',
        'sif_pool': 'SIF (Weighted)',
        'sif_abtt': 'SIF + PC-Remove'
    }
    df_subset['Method Label'] = df_subset['method'].replace(method_map)
    
    unique_methods = df_subset['Method Label'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 1: Gap vs Layer
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(unique_methods):
        subset = df_subset[df_subset['Method Label'] == method]
        plt.plot(subset['layer'], subset['gap'], marker=markers[i % len(markers)], label=method)
        
    plt.title('Anisotropy Gap vs. Layer Depth')
    plt.xlabel('Layer Index')
    plt.ylabel('Gap (Same - Diff Cosine)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_gap_analysis.png'))
    plt.close()

    # Plot 2: Accuracy vs Layer
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(unique_methods):
        subset = df_subset[df_subset['Method Label'] == method]
        plt.plot(subset['layer'], subset['acc@5_winnable'], marker=markers[i % len(markers)], label=method)

    plt.title('Code Search Accuracy (Acc@5) vs. Layer Depth')
    plt.xlabel('Layer Index')
    plt.ylabel('Acc@5 (Winnable)')
    plt.ylim(0.0, 1.0) # Accuracy is 0-1
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_accuracy.png'))
    plt.close()
    
    # Plot 3: Off-Diagonal Mean (Anisotropy Proxy)
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(unique_methods):
        subset = df_subset[df_subset['Method Label'] == method]
        plt.plot(subset['layer'], subset['off_diag_mean'], marker=markers[i % len(markers)], label=method)

    plt.title('Global Anisotropy (Off-Diagonal Mean) vs. Layer Depth')
    plt.xlabel('Layer Index')
    plt.ylabel('Avg Cosine of Random Pairs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_anisotropy.png'))
    plt.close()

def plot_musts_benchmark(df, output_dir):
    """Generates bar chart for MUSTS benchmark using matplotlib."""
    # df columns: language, method, spearman
    
    plt.figure(figsize=(12, 6))
    
    # Create improved labels
    method_map = {
        'base_mean': 'Base',
        'sif_official_r1': 'SIF (Official, r=1)',
        'phase7_r10': 'Our Method (r=10)'
    }
    df['Method Label'] = df['method'].map(method_map)
    
    languages = df['language'].unique()
    methods = df['Method Label'].unique()
    
    # Set width of bar
    barWidth = 0.25
    
    # Set position of bar on X axis
    r = [i for i in range(len(languages))]
    rs = [r]
    for i in range(1, len(methods)):
        rs.append([x + barWidth for x in rs[-1]])
        
    # Make the plot
    for i, method in enumerate(methods):
        subset = df[df['Method Label'] == method]
        # Ensure correct order mapping
        values = []
        for lang in languages:
            val = subset[subset['language'] == lang]['spearman'].values
            if len(val) > 0:
                values.append(val[0])
            else:
                values.append(0)
        
        plt.bar(rs[i], values, width=barWidth, edgecolor='white', label=method)
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Language', fontweight='bold')
    plt.ylabel('Spearman Correlation')
    plt.xticks([r + barWidth for r in range(len(languages))], languages)
    
    plt.title('MUSTS Benchmark: Spearman Correlation by Language')
    plt.legend(title='Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'musts_benchmark.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Phase 7 Results")
    parser.add_argument("--layer_csv", required=True, help="Path to layer sweep CSV")
    parser.add_argument("--musts_csv", required=True, help="Path to MUSTS results CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Data
    print(f"Loading layer data from {args.layer_csv}...")
    layer_df = pd.read_csv(args.layer_csv)
    
    print(f"Loading MUSTS data from {args.musts_csv}...")
    musts_df = pd.read_csv(args.musts_csv)

    # Plot
    print("Generating Layer Sweep plots...")
    plot_layer_sweep(layer_df, args.output_dir)
    
    print("Generating MUSTS Benchmark plots...")
    plot_musts_benchmark(musts_df, args.output_dir)

    print(f"All plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
