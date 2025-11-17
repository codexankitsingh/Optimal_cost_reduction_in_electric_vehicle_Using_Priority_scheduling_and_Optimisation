#!/usr/bin/env python3
"""
compare_pipelines.py

Compare pipeline.py (priority scheduling) vs pipeline_fcfs.py (FCFS) across multiple metrics:
- Average net energy cost
- Battery degradation cost
- Grid load variance
- User satisfaction
- Average waiting time
- Fairness (Gini coefficient)
- Effect of weight preferences
- Effect of fleet sizes
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys
from typing import Dict, List, Tuple
import re

def extract_metrics_from_output(output: str) -> Dict[str, float]:
    """Extract metrics from pipeline output. Tolerant to small differences in printed labels."""
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
    match = re.search(r'Grid load variance\s*\(raw F3\):\s*([\d\.\+\-eE]+)', output, flags=re.I)
    if match:
        metrics['grid_load_variance'] = float(match.group(1))
    
    # Average user satisfaction
    match = re.search(r'Average user satisfaction:\s*([\d\.\+\-eE]+)', output, flags=re.I)
    if match:
        metrics['avg_user_satisfaction'] = float(match.group(1))
    
    # Average waiting time: try a few possible printed labels
    match_all = re.search(r'Average waiting time \(all EVs in pool\):\s*([\d\.\+\-eE]+)\s*hours', output, flags=re.I)
    match_adm = re.search(r'Average waiting time \(admitted EVs\):\s*([\d\.\+\-eE]+)\s*hours', output, flags=re.I)
    match_generic = re.search(r'Average waiting time.*?:\s*([\d\.\+\-eE]+)\s*hours', output, flags=re.I)
    if match_all:
        metrics['avg_waiting_time'] = float(match_all.group(1))
        metrics['avg_waiting_time_all'] = float(match_all.group(1))
    elif match_adm:
        metrics['avg_waiting_time_admitted'] = float(match_adm.group(1))
        # set generic to admitted if nothing else
        metrics['avg_waiting_time'] = float(match_adm.group(1))
    elif match_generic:
        metrics['avg_waiting_time'] = float(match_generic.group(1))
    
    # Fairness (Gini)
    match = re.search(r'Fairness\s*\(Gini on delivered kWh\):\s*([\d\.\+\-eE]+)', output, flags=re.I)
    if match:
        metrics['fairness_gini'] = float(match.group(1))
    
    # Objective J
    match = re.search(r'Objective J\s*\(normalized weighted\):\s*([\d\.\+\-eE]+)', output, flags=re.I)
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
        # annotate missing fields with NaN to avoid later KeyErrors
        for key in ['avg_net_energy_cost', 'avg_battery_degradation_cost', 'grid_load_variance',
                    'avg_user_satisfaction', 'avg_waiting_time', 'fairness_gini', 'objective_J']:
            metrics.setdefault(key, float('nan'))
        return metrics
    except subprocess.TimeoutExpired:
        print(f"Timeout running {script_path}")
        return {}
    except Exception as e:
        print(f"Exception running {script_path}: {e}")
        return {}

def run_weight_experiment(script_path: str, ev_file: str, chargers: int, 
                           weight_configs: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Run pipeline with different weight configurations."""
    results = []
    for i, config in enumerate(weight_configs):
        print(f"\nWeight config {i+1}/{len(weight_configs)}: {config}")
        # NOTE: pipelines must accept CLI args to override weights; if they don't, this runs defaults
        metrics = run_pipeline(script_path, ev_file, chargers)
        metrics['config_name'] = f"W{config.get('w1', 0.25)}-{config.get('w2', 0.25)}-{config.get('w3', 0.25)}-{config.get('w4', 0.25)}"
        results.append(metrics)
    return results

def run_fleet_size_experiment(script_path: str, ev_file: str, 
                               fleet_sizes: List[int]) -> List[Dict[str, float]]:
    """Run pipeline with different fleet sizes (chargers)."""
    results = []
    for chargers in fleet_sizes:
        print(f"\nFleet size (chargers): {chargers}")
        metrics = run_pipeline(script_path, ev_file, chargers)
        metrics['fleet_size'] = chargers
        results.append(metrics)
    return results

def plot_comparison(priority_metrics: Dict[str, float], fcfs_metrics: Dict[str, float], 
                   output_dir: str):
    """Create comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_compare = {
        'avg_net_energy_cost': 'Average Net Energy Cost per EV',
        'avg_battery_degradation_cost': 'Average Battery Degradation Cost per EV',
        'grid_load_variance': 'Grid Load Variance',
        'avg_user_satisfaction': 'Average User Satisfaction',
        'avg_waiting_time': 'Average Waiting Time (hours)',
        'fairness_gini': 'Fairness (Gini Coefficient)',
        'objective_J': 'Objective J (normalized)'
    }
    
    categories = []
    priority_values = []
    fcfs_values = []
    
    for metric_key, metric_label in metrics_to_compare.items():
        if not (np.isnan(priority_metrics.get(metric_key, np.nan)) or np.isnan(fcfs_metrics.get(metric_key, np.nan))):
            categories.append(metric_label)
            priority_values.append(priority_metrics.get(metric_key, 0.0))
            fcfs_values.append(fcfs_metrics.get(metric_key, 0.0))
    
    if not categories:
        print("No comparable metrics found")
        return
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width/2, priority_values, width, label='Priority Scheduling')
    ax.bar(x + width/2, fcfs_values, width, label='FCFS Scheduling')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Comparison: Priority Scheduling vs FCFS Scheduling', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_pipeline_vs_fcfs.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()

def plot_weight_effects(priority_results: List[Dict], fcfs_results: List[Dict], output_dir: str):
    """Plot effect of weight preferences."""
    if not priority_results or not fcfs_results:
        print("No weight experiment results to plot.")
        return
    
    metrics_to_plot = {
        'avg_user_satisfaction': 'User Satisfaction',
        'objective_J': 'Objective J'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_key, metric_label in metrics_to_plot.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(priority_results))
        priority_vals = [r.get(metric_key, np.nan) for r in priority_results]
        fcfs_vals = [r.get(metric_key, np.nan) for r in fcfs_results]
        
        ax.plot(x, priority_vals, marker='o', label='Priority', linewidth=2)
        ax.plot(x, fcfs_vals, marker='s', label='FCFS', linewidth=2)
        ax.set_xlabel('Weight Configuration Index', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'Effect of Weight Preferences on {metric_label}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'weight_effect_{metric_key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved weight effect plot to: {output_path}")
        plt.close()

def plot_fleet_size_effects(priority_results: List[Dict], fcfs_results: List[Dict], output_dir: str):
    """Plot effect of fleet sizes (number of chargers) on satisfaction, waiting time, and fairness."""
    if not priority_results or not fcfs_results:
        print("⚠️ No data to plot fleet size effects — check pipeline outputs.")
        return
    
    # If fleet_size missing, fallback to index
    for idx, r in enumerate(priority_results):
        r.setdefault('fleet_size', idx)
    for idx, r in enumerate(fcfs_results):
        r.setdefault('fleet_size', idx)
    
    # Create dictionaries mapping fleet_size -> result for both pipelines
    # This ensures we match results by fleet_size, not by index
    priority_by_fleet = {r.get('fleet_size', i): r for i, r in enumerate(priority_results)}
    fcfs_by_fleet = {r.get('fleet_size', i): r for i, r in enumerate(fcfs_results)}
    
    # Find intersection of fleet_sizes that exist in both pipelines
    common_fleet_sizes = sorted(set(priority_by_fleet.keys()) & set(fcfs_by_fleet.keys()))
    
    if not common_fleet_sizes:
        print("⚠️ No common fleet sizes found between priority and FCFS results — cannot plot.")
        return
    
    metrics_to_plot = {
        'avg_user_satisfaction': 'Average User Satisfaction vs Fleet Size',
        'avg_waiting_time': 'Average Waiting Time vs Fleet Size',
        'fairness_gini': 'Fairness (Gini Coefficient) vs Fleet Size'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_key, title in metrics_to_plot.items():
        # Extract values matched by fleet_size (not by index)
        priority_vals = []
        fcfs_vals = []
        fleet_sizes_valid = []
        
        for fs in common_fleet_sizes:
            p_val = priority_by_fleet[fs].get(metric_key, np.nan)
            f_val = fcfs_by_fleet[fs].get(metric_key, np.nan)
            
            # Only include if both values are valid (non-NaN)
            if not np.isnan(p_val) and not np.isnan(f_val):
                fleet_sizes_valid.append(fs)
                priority_vals.append(p_val)
                fcfs_vals.append(f_val)
        
        if not fleet_sizes_valid:
            print(f"⚠️ No valid data for {metric_key} — skipping plot.")
            continue
        
        # Debug print (optional - helpful when diagnosing empty/flat plots)
        # Uncomment the next line if you need to debug values:
        # print(f"[DEBUG] {metric_key} data points: {list(zip(fleet_sizes_valid, priority_vals, fcfs_vals))}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(fleet_sizes_valid, priority_vals, marker='o', linewidth=2, label='Priority Scheduling')
        ax.plot(fleet_sizes_valid, fcfs_vals, marker='s', linewidth=2, label='FCFS Scheduling')
        
        ax.set_xlabel('Fleet Size (Number of Chargers)', fontsize=12)
        ax.set_ylabel(metric_key.replace('_', ' ').title(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'fleet_effect_{metric_key}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved fixed fleet effect plot to: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare pipeline.py vs pipeline_fcfs.py")
    parser.add_argument('--ev-file', type=str, default='evs.csv', 
                       help='EV file path (default: evs.csv)')
    parser.add_argument('--chargers', type=int, default=30,
                       help='Number of chargers (default: 30)')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for plots (default: comparison_results)')
    parser.add_argument('--run-experiments', action='store_true',
                       help='Run weight and fleet size experiments')
    args = parser.parse_args()
    
    # Check scripts and EV file exist
    if not os.path.exists('pipeline.py'):
        print("Error: pipeline.py not found")
        return
    if not os.path.exists('pipeline_fcfs.py'):
        print("Error: pipeline_fcfs.py not found")
        return
    if not os.path.exists(args.ev_file):
        print(f"Error: EV file '{args.ev_file}' not found")
        return
    
    print("="*80)
    print("PIPELINE COMPARISON: Priority Scheduling vs FCFS")
    print("="*80)
    
    # Basic comparison
    print("\n--- Running Priority Scheduling Pipeline ---")
    priority_metrics = run_pipeline('pipeline.py', args.ev_file, args.chargers)
    
    print("\n--- Running FCFS Pipeline ---")
    fcfs_metrics = run_pipeline('pipeline_fcfs.py', args.ev_file, args.chargers)
    
    # Print results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print("\nPriority Scheduling:")
    for key, val in priority_metrics.items():
        print(f"  {key}: {val}")
    
    print("\nFCFS Scheduling:")
    for key, val in fcfs_metrics.items():
        print(f"  {key}: {val}")
    
    print("\nDifferences:")
    common_keys = set(priority_metrics.keys()) & set(fcfs_metrics.keys())
    for key in common_keys:
        a = priority_metrics.get(key, np.nan)
        b = fcfs_metrics.get(key, np.nan)
        if np.isnan(a) or np.isnan(b):
            continue
        diff = b - a
        pct_diff = (diff / a * 100) if a != 0 else float('nan')
        print(f"  {key}: {diff:.6f} ({pct_diff:+.2f}%)")
    
    # Create plots
    plot_comparison(priority_metrics, fcfs_metrics, args.output_dir)
    
    # Run experiments if requested
    if args.run_experiments:
        print("\n--- Running Weight Experiments ---")
        weight_configs = [
            {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25},
            {'w1': 0.30, 'w2': 0.20, 'w3': 0.20, 'w4': 0.30},
            {'w1': 0.20, 'w2': 0.20, 'w3': 0.20, 'w4': 0.40},
        ]
        priority_weight_results = run_weight_experiment('pipeline.py', args.ev_file, args.chargers, weight_configs)
        fcfs_weight_results = run_weight_experiment('pipeline_fcfs.py', args.ev_file, args.chargers, weight_configs)
        
        plot_weight_effects(priority_weight_results, fcfs_weight_results, args.output_dir)
        
        print("\n--- Running Fleet Size Experiments ---")
        fleet_sizes = [10, 20, 30, 40, 50]
        priority_fleet_results = run_fleet_size_experiment('pipeline.py', args.ev_file, fleet_sizes)
        fcfs_fleet_results = run_fleet_size_experiment('pipeline_fcfs.py', args.ev_file, fleet_sizes)
        
        plot_fleet_size_effects(priority_fleet_results, fcfs_fleet_results, args.output_dir)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
