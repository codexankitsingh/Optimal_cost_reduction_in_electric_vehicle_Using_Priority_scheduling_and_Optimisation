import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load results
def load_results():
    if not os.path.exists('comparison_results.csv'):
        print("Error: comparison_results.csv not found!")
        print("Please run run_pipeline_comparison.py first.")
        return None
    
    df = pd.read_csv('comparison_results.csv')
    return df

def plot_metric_comparison(df, metric_col, ylabel, title, filename, lower_is_better=True):
    """Plot comparison of a metric across pipelines and EV counts."""
    plt.figure(figsize=(12, 7))
    
    pipelines = df['pipeline'].unique()
    ev_counts = sorted(df['ev_count'].unique())
    
    colors = {'Priority': '#2E86AB', 'FCFS': '#A23B72', 'SJF': '#F18F01'}
    markers = {'Priority': 'o', 'FCFS': 's', 'SJF': '^'}
    
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline].sort_values('ev_count')
        values = pipeline_data[metric_col].values
        
        plt.plot(ev_counts, values, 
                marker=markers.get(pipeline, 'o'),
                linewidth=2.5,
                markersize=8,
                label=pipeline,
                color=colors.get(pipeline, 'gray'))
    
    plt.xlabel('Number of EVs', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ev_counts)
    
    # Annotate best performer
    if lower_is_better:
        best_idx = df.groupby('ev_count')[metric_col].idxmin()
    else:
        best_idx = df.groupby('ev_count')[metric_col].idxmax()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {filename}")
    plt.close()

def plot_combined_comparison(df):
    """Create a 2x2 subplot with all metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    pipelines = df['pipeline'].unique()
    ev_counts = sorted(df['ev_count'].unique())
    
    colors = {'Priority': '#2E86AB', 'FCFS': '#A23B72', 'SJF': '#F18F01'}
    markers = {'Priority': 'o', 'FCFS': 's', 'SJF': '^'}
    
    # Plot 1: Energy Cost
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline].sort_values('ev_count')
        values = pipeline_data['avg_cost_per_ev'].values
        ax1.plot(ev_counts, values, marker=markers[pipeline], linewidth=2,
                markersize=7, label=pipeline, color=colors[pipeline])
    ax1.set_xlabel('Number of EVs', fontweight='bold')
    ax1.set_ylabel('Avg Cost per EV ($)', fontweight='bold')
    ax1.set_title('Average Net Energy Cost Comparison', fontweight='bold', pad=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(ev_counts)
    
    # Plot 2: User Satisfaction
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline].sort_values('ev_count')
        values = pipeline_data['avg_user_satisfaction_all'].values
        ax2.plot(ev_counts, values, marker=markers[pipeline], linewidth=2,
                markersize=7, label=pipeline, color=colors[pipeline])
    ax2.set_xlabel('Number of EVs', fontweight='bold')
    ax2.set_ylabel('Avg User Satisfaction', fontweight='bold')
    ax2.set_title('Average User Satisfaction Comparison', fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(ev_counts)
    
    # Plot 3: Waiting Time
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline].sort_values('ev_count')
        values = pipeline_data['avg_waiting_time_all'].values
        ax3.plot(ev_counts, values, marker=markers[pipeline], linewidth=2,
                markersize=7, label=pipeline, color=colors[pipeline])
    ax3.set_xlabel('Number of EVs', fontweight='bold')
    ax3.set_ylabel('Avg Waiting Time (hours)', fontweight='bold')
    ax3.set_title('Average Waiting Time Comparison', fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(ev_counts)
    
    # Plot 4: Gini Fairness
    for pipeline in pipelines:
        pipeline_data = df[df['pipeline'] == pipeline].sort_values('ev_count')
        values = pipeline_data['gini_fairness'].values
        ax4.plot(ev_counts, values, marker=markers[pipeline], linewidth=2,
                markersize=7, label=pipeline, color=colors[pipeline])
    ax4.set_xlabel('Number of EVs', fontweight='bold')
    ax4.set_ylabel('Gini Coefficient', fontweight='bold')
    ax4.set_title('Fairness Comparison (Gini)', fontweight='bold', pad=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(ev_counts)
    
    plt.suptitle('Pipeline Comparison Across All Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('comparison_combined.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison_combined.png")
    plt.close()

def print_winners(df):
    """Print which pipeline wins for each metric and EV count."""
    print(f"\n{'='*80}")
    print("BEST PERFORMER BY METRIC")
    print(f"{'='*80}")
    
    ev_counts = sorted(df['ev_count'].unique())
    
    metrics = {
        'avg_cost_per_ev': ('Energy Cost', True),
        'avg_user_satisfaction_all': ('User Satisfaction', False),
        'avg_waiting_time_all': ('Waiting Time', True),
        'gini_fairness': ('Fairness (Gini)', True)
    }
    
    for metric_col, (metric_name, lower_is_better) in metrics.items():
        print(f"\n{metric_name}:")
        for ev_count in ev_counts:
            subset = df[df['ev_count'] == ev_count]
            if lower_is_better:
                best = subset.loc[subset[metric_col].idxmin()]
            else:
                best = subset.loc[subset[metric_col].idxmax()]
            print(f"  {ev_count} EVs: {best['pipeline']:8s} ({best[metric_col]:.4f})")

if __name__ == "__main__":
    print("="*80)
    print("GENERATING PIPELINE COMPARISON PLOTS")
    print("="*80)
    
    df = load_results()
    if df is None:
        exit(1)
    
    print(f"\nLoaded {len(df)} results")
    print(f"Pipelines: {', '.join(df['pipeline'].unique())}")
    print(f"EV Counts: {', '.join(map(str, sorted(df['ev_count'].unique())))}")
    
    # Generate individual plots
    plot_metric_comparison(df, 'avg_cost_per_ev', 
                          'Average Cost per EV ($)', 
                          'Average Net Energy Cost Comparison',
                          'comparison_energy_cost.png',
                          lower_is_better=True)
    
    plot_metric_comparison(df, 'avg_user_satisfaction_all',
                          'Average User Satisfaction',
                          'Average User Satisfaction Comparison',
                          'comparison_user_satisfaction.png',
                          lower_is_better=False)
    
    plot_metric_comparison(df, 'avg_waiting_time_all',
                          'Average Waiting Time (hours)',
                          'Average Waiting Time Comparison',
                          'comparison_waiting_time.png',
                          lower_is_better=True)
    
    plot_metric_comparison(df, 'gini_fairness',
                          'Gini Coefficient',
                          'Fairness Comparison (Lower = More Fair)',
                          'comparison_gini_fairness.png',
                          lower_is_better=True)
    
    # Generate combined plot
    plot_combined_comparison(df)
    
    # Print winners
    print_winners(df)
    
    print(f"\n{'='*80}")
    print("All plots generated successfully!")
    print(f"{'='*80}")
