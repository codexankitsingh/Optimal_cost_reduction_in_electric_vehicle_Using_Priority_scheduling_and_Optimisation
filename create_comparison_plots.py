#!/usr/bin/env python3
"""
create_comparison_plots.py

Create detailed comparison plots for Priority Scheduling vs FCFS
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
import re
from typing import Dict, List

def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """Extract metrics from pipeline output."""
    metrics = {}
    
    # Average net energy cost per EV
    match = re.search(r'Average net energy cost per EV:\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['avg_net_energy_cost'] = float(match.group(1))
    
    # Average battery degradation cost per EV
    match = re.search(r'Average battery degradation cost per EV:\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['avg_battery_degradation_cost'] = float(match.group(1))
    
    # Grid load variance
    match = re.search(r'Grid load variance \(raw F3\):\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['grid_load_variance'] = float(match.group(1))
    
    # Average user satisfaction
    match = re.search(r'Average user satisfaction:\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['avg_user_satisfaction'] = float(match.group(1))
    
    # Average waiting time (try both formats)
    match = re.search(r'Average waiting time \(admitted EVs\):\s*([\d\.\+\-eE]+)', output)
    if not match:
        match = re.search(r'Average waiting time \(hours\):\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['avg_waiting_time'] = float(match.group(1))
    
    # Fairness (Gini)
    match = re.search(r'Fairness \(Gini on delivered kWh\):\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['fairness_gini'] = float(match.group(1))
    
    # Also extract objective J
    match = re.search(r'Objective J \(normalized weighted\):\s*([\d\.\+\-eE]+)', output)
    if match:
        metrics['objective_J'] = float(match.group(1))
    
    return metrics

def run_pipeline(script_path: str, ev_file: str, chargers: int) -> Dict[str, float]:
    """Run a pipeline script and extract metrics."""
    cmd = [sys.executable, script_path, '--ev-file', ev_file, '--chargers', str(chargers)]
    print(f"\nRunning: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"Error running {script_path}: {output}")
            return {}
        
        metrics = extract_metrics_from_output(output)
        return metrics
    except subprocess.TimeoutExpired:
        print(f"Timeout running {script_path}")
        return {}
    except Exception as e:
        print(f"Exception running {script_path}: {e}")
        return {}

def run_fleet_size_experiment(script_path: str, ev_file: str, 
                               fleet_sizes: List[int]) -> List[Dict[str, float]]:
    """Run pipeline with different fleet sizes."""
    results = []
    for chargers in fleet_sizes:
        print(f"\nFleet size: {chargers}")
        metrics = run_pipeline(script_path, ev_file, chargers)
        metrics['fleet_size'] = chargers
        results.append(metrics)
    return results

def main():
    ev_file = 'evs.csv'
    output_dir = 'comparison_results'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists('pipeline.py') or not os.path.exists('pipeline_fcfs.py'):
        print("Error: pipeline files not found")
        return
    
    if not os.path.exists(ev_file):
        print(f"Error: EV file '{ev_file}' not found")
        return
    
    print("="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)
    
    # Run both pipelines with default settings
    print("\n--- Baseline Comparison ---")
    priority_metrics = run_pipeline('pipeline.py', ev_file, 30)
    fcfs_metrics = run_pipeline('pipeline_fcfs.py', ev_file, 30)
    
    # Create comparison bar chart
    categories = []
    priority_values = []
    fcfs_values = []
    
    metrics_to_compare = {
        'avg_net_energy_cost': 'Average Net Energy\nCost per EV',
        'avg_battery_degradation_cost': 'Average Battery\nDegradation Cost',
        'grid_load_variance': 'Grid Load\nVariance',
        'avg_user_satisfaction': 'Average User\nSatisfaction',
        'avg_waiting_time': 'Average Waiting\nTime (hours)',
        'fairness_gini': 'Fairness\n(Gini Coefficient)',
        'objective_J': 'Objective J\n(normalized)'
    }
    
    for metric_key, metric_label in metrics_to_compare.items():
        if metric_key in priority_metrics and metric_key in fcfs_metrics:
            categories.append(metric_label)
            priority_values.append(priority_metrics[metric_key])
            fcfs_values.append(fcfs_metrics[metric_key])
    
    # Create bar chart
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 9))
    bars1 = ax.bar(x - width/2, priority_values, width, label='Priority Scheduling', 
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, fcfs_values, width, label='FCFS Scheduling', 
                   color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Values', fontsize=14, fontweight='bold')
    ax.set_title('Comparison: Priority Scheduling vs FCFS Scheduling\n(Baseline: 30 chargers)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}' if abs(val) < 10 else f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9, rotation=90)
    
    # autolabel(bars1, priority_values)
    # autolabel(bars2, fcfs_values)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_baseline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved baseline comparison plot to: {output_path}")
    plt.close()
    
    # Run fleet size experiments
    print("\n--- Fleet Size Experiments ---")
    fleet_sizes = [10, 15, 20, 25, 30]
    priority_fleet_results = run_fleet_size_experiment('pipeline.py', ev_file, fleet_sizes)
    fcfs_fleet_results = run_fleet_size_experiment('pipeline_fcfs.py', ev_file, fleet_sizes)
    
    # Plot fleet size effects for key metrics
    metrics_for_fleet = {
        'avg_user_satisfaction': 'User Satisfaction',
        'avg_waiting_time': 'Average Waiting Time (hours)',
        'fairness_gini': 'Fairness (Gini Coefficient)',
        'avg_net_energy_cost': 'Average Net Energy Cost per EV'
    }
    
    for metric_key, metric_label in metrics_for_fleet.items():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        fleet_sizes_data = [r.get('fleet_size', i) for i, r in enumerate(priority_fleet_results)]
        priority_vals = [r.get(metric_key, 0) for r in priority_fleet_results]
        fcfs_vals = [r.get(metric_key, 0) for r in fcfs_fleet_results]
        
        ax.plot(fleet_sizes_data, priority_vals, marker='o', markersize=10, 
                label='Priority Scheduling', linewidth=3, color='#2E86AB')
        ax.plot(fleet_sizes_data, fcfs_vals, marker='s', markersize=10, 
                label='FCFS Scheduling', linewidth=3, color='#A23B72')
        
        ax.set_xlabel('Fleet Size (Number of Chargers)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=14, fontweight='bold')
        ax.set_title(f'Effect of Fleet Size on {metric_label}', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fleet_effect_{metric_key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved fleet effect plot to: {output_path}")
        plt.close()
    
    # Create summary table
    print("\n--- Creating Summary Table ---")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for metric_key, metric_label in metrics_to_compare.items():
        if metric_key in priority_metrics and metric_key in fcfs_metrics:
            p_val = priority_metrics[metric_key]
            f_val = fcfs_metrics[metric_key]
            diff = f_val - p_val
            diff_pct = (diff / abs(p_val) * 100) if p_val != 0 else 0
            table_data.append([metric_label, f'{p_val:.6f}', f'{f_val:.6f}', 
                              f'{diff:.6f}', f'{diff_pct:+.2f}%'])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Priority', 'FCFS', 'Difference', 'Change %'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.175, 0.175, 0.175, 0.175])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells based on difference
    for i, row in enumerate(table_data):
        change_pct = float(row[4].replace('%', '').replace('+', ''))
        if abs(change_pct) > 20:
            for j in range(1, 5):
                table[(i+1, j)].set_facecolor('#ffcccc' if change_pct > 0 else '#ccffcc')
    
    plt.title('Priority vs FCFS Scheduling Comparison Summary\n(Baseline: 30 chargers)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'comparison_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary table to: {output_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"Results saved to: {output_dir}/")
    print("="*80)

if __name__ == "__main__":
    main()
