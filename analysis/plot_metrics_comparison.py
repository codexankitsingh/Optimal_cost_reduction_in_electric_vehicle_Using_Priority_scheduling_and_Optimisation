#!/usr/bin/env python3
"""
Plot comparison of multiple metrics between pipeline.py (priority) 
and pipeline_fcfs.py (FCFS) for different EV pool sizes: 50, 100, 150, 200, 250, 300 EVs.

Metrics compared:
1. Average user satisfaction
2. Average net energy cost per EV
3. Fairness (Gini coefficient on delivered kWh)
"""

import argparse
import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics_from_output(output: str) -> dict:
    """
    Extract metrics from pipeline output.
    
    Returns:
        Dictionary with keys: 'user_satisfaction', 'net_energy_cost', 'gini_fairness'
        Values are floats or None if not found
    """
    metrics = {}
    
    # Average user satisfaction
    pattern = r'Average user satisfaction:\s*([\d\.\+\-eE]+)'
    match = re.search(pattern, output)
    if match:
        metrics['user_satisfaction'] = float(match.group(1))
    else:
        metrics['user_satisfaction'] = None
    
    # Average net energy cost per EV
    pattern = r'Average net energy cost per EV:\s*([\d\.\+\-eE]+)'
    match = re.search(pattern, output)
    if match:
        metrics['net_energy_cost'] = float(match.group(1))
    else:
        metrics['net_energy_cost'] = None
    
    # Fairness (Gini coefficient)
    pattern = r'Fairness \(Gini on delivered kWh\):\s*([\d\.\+\-eE]+)'
    match = re.search(pattern, output)
    if match:
        metrics['gini_fairness'] = float(match.group(1))
    else:
        metrics['gini_fairness'] = None
    
    return metrics

def run_pipeline_and_extract_metrics(script_path: str, ev_file: str, chargers: int, 
                                     ngen: int = 300, pop_size: int = 120, 
                                     ga_scope: str = "full") -> dict:
    """
    Run a pipeline script and extract metrics.
    
    Returns:
        Dictionary with metrics, or empty dict if failed
    """
    cmd = [sys.executable, script_path, 
           '--ev-file', ev_file,
           '--chargers', str(chargers),
           '--T', '48',
           '--delta-t', '0.25',
           '--ngen', str(ngen),
           '--pop-size', str(pop_size)]
    
    # Add GA scope if pipeline.py
    if 'pipeline.py' in script_path and ga_scope:
        cmd.extend(['--ga-schedule-scope', ga_scope])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"Error: Pipeline failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return {}
        
        metrics = extract_metrics_from_output(output)
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"Timeout: Pipeline took longer than 30 minutes")
        return {}
    except Exception as e:
        print(f"Exception: {e}")
        return {}

def extract_from_log_file(log_file: str) -> dict:
    """Extract metrics from existing log file."""
    if not os.path.exists(log_file):
        return {}
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        return extract_metrics_from_output(content)
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Compare metrics between priority and FCFS pipelines")
    parser.add_argument('--ev-sizes', type=int, nargs='+', 
                       default=[50, 100, 150, 200, 250, 300],
                       help='EV pool sizes to test (default: 50 100 150 200 250 300)')
    parser.add_argument('--chargers', type=int, default=30,
                       help='Number of chargers (default: 30)')
    parser.add_argument('--ngen', type=int, default=300,
                       help='GA generations (default: 300)')
    parser.add_argument('--pop-size', type=int, default=120,
                       help='GA population size (default: 120)')
    parser.add_argument('--use-existing-logs', action='store_true',
                       help='Try to extract from existing log files instead of running pipelines')
    parser.add_argument('--ga-scope', type=str, default='full',
                       choices=['t0', 'full'],
                       help='GA scheduling scope for pipeline.py (default: full)')
    parser.add_argument('--out-dir', type=str, default='analysis',
                       help='Output directory for plots (default: analysis)')
    args = parser.parse_args()
    
    # EV file patterns
    ev_files = {size: f'evs_{size}.csv' for size in args.ev_sizes}
    
    # Check if EV files exist
    for size, ev_file in ev_files.items():
        if not os.path.exists(ev_file):
            print(f"Warning: EV file {ev_file} not found. Will try to generate it.")
            # Try to generate if generate_ev.py exists
            if os.path.exists('generate_ev.py'):
                print(f"Generating {ev_file}...")
                try:
                    subprocess.run([sys.executable, 'generate_ev.py', '--n', str(size), 
                                  '--out', ev_file, '--seed', '42'], check=True)
                except Exception as e:
                    print(f"Failed to generate {ev_file}: {e}")
                    return
    
    # Collect metrics
    priority_metrics = {'user_satisfaction': [], 'net_energy_cost': [], 'gini_fairness': []}
    fcfs_metrics = {'user_satisfaction': [], 'net_energy_cost': [], 'gini_fairness': []}
    ev_sizes_actual = []
    
    for size in args.ev_sizes:
        ev_file = ev_files[size]
        if not os.path.exists(ev_file):
            print(f"Skipping {size} EVs: file {ev_file} not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing {size} EVs")
        print(f"{'='*80}")
        
        # Priority pipeline (pipeline.py)
        priority_data = {}
        if args.use_existing_logs:
            log_file = f'run_{size}evs.log'
            print(f"Trying to extract from log: {log_file}")
            priority_data = extract_from_log_file(log_file)
        
        if not priority_data or any(v is None for v in priority_data.values()):
            print(f"Running priority pipeline for {size} EVs...")
            priority_data = run_pipeline_and_extract_metrics(
                'pipeline.py', ev_file, args.chargers, 
                args.ngen, args.pop_size, args.ga_scope
            )
        
        # FCFS pipeline (pipeline_fcfs.py)
        fcfs_data = {}
        if args.use_existing_logs:
            log_file = f'run_{size}evs_fcfs.log'
            print(f"Trying to extract from log: {log_file}")
            fcfs_data = extract_from_log_file(log_file)
        
        if not fcfs_data or any(v is None for v in fcfs_data.values()):
            print(f"Running FCFS pipeline for {size} EVs...")
            fcfs_data = run_pipeline_and_extract_metrics(
                'pipeline_fcfs.py', ev_file, args.chargers,
                args.ngen, args.pop_size, None
            )
        
        # Check if we got all metrics
        if (priority_data.get('user_satisfaction') is not None and 
            priority_data.get('net_energy_cost') is not None and
            priority_data.get('gini_fairness') is not None and
            fcfs_data.get('user_satisfaction') is not None and 
            fcfs_data.get('net_energy_cost') is not None and
            fcfs_data.get('gini_fairness') is not None):
            
            priority_metrics['user_satisfaction'].append(priority_data['user_satisfaction'])
            priority_metrics['net_energy_cost'].append(priority_data['net_energy_cost'])
            priority_metrics['gini_fairness'].append(priority_data['gini_fairness'])
            
            fcfs_metrics['user_satisfaction'].append(fcfs_data['user_satisfaction'])
            fcfs_metrics['net_energy_cost'].append(fcfs_data['net_energy_cost'])
            fcfs_metrics['gini_fairness'].append(fcfs_data['gini_fairness'])
            
            ev_sizes_actual.append(size)
            print(f"  Priority: satisfaction={priority_data['user_satisfaction']:.4f}, "
                  f"cost={priority_data['net_energy_cost']:.4f}, "
                  f"gini={priority_data['gini_fairness']:.4f}")
            print(f"  FCFS: satisfaction={fcfs_data['user_satisfaction']:.4f}, "
                  f"cost={fcfs_data['net_energy_cost']:.4f}, "
                  f"gini={fcfs_data['gini_fairness']:.4f}")
        else:
            print(f"  Warning: Failed to get all metrics for {size} EVs")
            missing_pri = [k for k, v in priority_data.items() if v is None]
            missing_fcfs = [k for k, v in fcfs_data.items() if v is None]
            if missing_pri:
                print(f"    Priority pipeline missing: {missing_pri}")
            if missing_fcfs:
                print(f"    FCFS pipeline missing: {missing_fcfs}")
    
    if not ev_sizes_actual:
        print("Error: No valid data collected. Cannot create plots.")
        return
    
    # Create comparison plots
    print(f"\n{'='*80}")
    print("Creating comparison plots...")
    print(f"{'='*80}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Define plot configurations
    plot_configs = [
        {
            'key': 'user_satisfaction',
            'title': 'Average User Satisfaction',
            'ylabel': 'User Satisfaction',
            'filename': 'user_satisfaction_comparison.png',
            'higher_is_better': True
        },
        {
            'key': 'net_energy_cost',
            'title': 'Average Net Energy Cost per EV',
            'ylabel': 'Energy Cost ($)',
            'filename': 'net_energy_cost_comparison.png',
            'higher_is_better': False
        },
        {
            'key': 'gini_fairness',
            'title': 'Fairness (Gini Coefficient)',
            'ylabel': 'Gini Coefficient',
            'filename': 'gini_fairness_comparison.png',
            'higher_is_better': False  # Lower Gini = more fair
        }
    ]
    
    # Create individual plots
    for config in plot_configs:
        key = config['key']
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot lines
        ax.plot(ev_sizes_actual, priority_metrics[key], 
                marker='o', linewidth=2.5, markersize=8, 
                label='Priority Scheduling', color='#2E86AB', markerfacecolor='#2E86AB')
        ax.plot(ev_sizes_actual, fcfs_metrics[key], 
                marker='s', linewidth=2.5, markersize=8, 
                label='FCFS Scheduling', color='#A23B72', markerfacecolor='#A23B72')
        
        # Styling
        ax.set_xlabel('EV Pool Size', fontsize=14, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=14, fontweight='bold')
        ax.set_title(f'{config["title"]}: Priority vs FCFS Scheduling\n(30 Chargers)',
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
        
        # Add value annotations
        for i, (size, pri, fcfs) in enumerate(zip(ev_sizes_actual, 
                                                   priority_metrics[key], 
                                                   fcfs_metrics[key])):
            # Annotate priority values
            ax.annotate(f'{pri:.3f}', 
                       (size, pri), 
                       textcoords="offset points", 
                       xytext=(0, 10), 
                       ha='center', 
                       fontsize=9, 
                       color='#2E86AB',
                       fontweight='bold')
            # Annotate FCFS values
            ax.annotate(f'{fcfs:.3f}', 
                       (size, fcfs), 
                       textcoords="offset points", 
                       xytext=(0, -15), 
                       ha='center', 
                       fontsize=9, 
                       color='#A23B72',
                       fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        out_path = os.path.join(args.out_dir, config['filename'])
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
        plt.close()
    
    # Create combined subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, config in enumerate(plot_configs):
        key = config['key']
        ax = axes[idx]
        
        ax.plot(ev_sizes_actual, priority_metrics[key], 
                marker='o', linewidth=2.5, markersize=7, 
                label='Priority', color='#2E86AB', markerfacecolor='#2E86AB')
        ax.plot(ev_sizes_actual, fcfs_metrics[key], 
                marker='s', linewidth=2.5, markersize=7, 
                label='FCFS', color='#A23B72', markerfacecolor='#A23B72')
        
        ax.set_xlabel('EV Pool Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
        ax.set_title(config['title'], fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    
    plt.tight_layout()
    combined_path = os.path.join(args.out_dir, 'metrics_comparison_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {combined_path}")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLES")
    print(f"{'='*80}")
    
    for config in plot_configs:
        key = config['key']
        print(f"\n{config['title']}:")
        print(f"{'EV Size':<10} {'Priority':<15} {'FCFS':<15} {'Difference':<15} {'% Change':<15}")
        print("-" * 80)
        for size, pri, fcfs in zip(ev_sizes_actual, priority_metrics[key], fcfs_metrics[key]):
            diff = fcfs - pri
            pct_change = (diff / pri * 100) if pri != 0 else 0.0
            print(f"{size:<10} {pri:<15.4f} {fcfs:<15.4f} {diff:<15.4f} {pct_change:<15.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

