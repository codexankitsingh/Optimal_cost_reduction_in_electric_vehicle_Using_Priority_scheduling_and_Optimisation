#!/usr/bin/env python3
"""
Plot comparison of average waiting time (fleet-wide) between pipeline.py (priority) 
and pipeline_fcfs.py (FCFS) for different EV pool sizes: 50, 100, 150, 200, 250, 300 EVs.
"""

import argparse
import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np

def extract_waiting_time_from_output(output: str, metric_type: str = "all") -> float:
    """
    Extract average waiting time from pipeline output.
    
    Args:
        output: Pipeline stdout/stderr text
        metric_type: "all" for fleet-wide, "admitted" for admitted-only
    
    Returns:
        Average waiting time in hours, or None if not found
    """
    if metric_type == "all":
        # Try fleet-wide waiting time first
        pattern = r'Average waiting time \(all EVs in pool\):\s*([\d\.\+\-eE]+)'
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
        # Fallback to older format
        pattern = r'Average waiting time \(all EVs\):\s*([\d\.\+\-eE]+)'
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    else:
        # Admitted-only waiting time
        pattern = r'Average waiting time \(admitted EVs\):\s*([\d\.\+\-eE]+)'
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
        # Fallback to simple format
        pattern = r'Average waiting time \(hours\):\s*([\d\.\+\-eE]+)'
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    
    return None

def run_pipeline_and_extract_waiting_time(script_path: str, ev_file: str, chargers: int, 
                                         ngen: int = 300, pop_size: int = 120, 
                                         ga_scope: str = "full") -> float:
    """
    Run a pipeline script and extract fleet-wide average waiting time.
    
    Returns:
        Average waiting time in hours, or None if extraction failed
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
            return None
        
        waiting_time = extract_waiting_time_from_output(output, metric_type="all")
        return waiting_time
        
    except subprocess.TimeoutExpired:
        print(f"Timeout: Pipeline took longer than 30 minutes")
        return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def extract_from_log_file(log_file: str) -> float:
    """Extract waiting time from existing log file."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        return extract_waiting_time_from_output(content, metric_type="all")
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare waiting times between priority and FCFS pipelines")
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
    parser.add_argument('--out', type=str, default='analysis/waiting_time_comparison.png',
                       help='Output plot path (default: analysis/waiting_time_comparison.png)')
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
    
    # Collect waiting times
    priority_wait_times = []
    fcfs_wait_times = []
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
        priority_wait = None
        if args.use_existing_logs:
            log_file = f'run_{size}evs.log'
            print(f"Trying to extract from log: {log_file}")
            priority_wait = extract_from_log_file(log_file)
        
        if priority_wait is None:
            print(f"Running priority pipeline for {size} EVs...")
            priority_wait = run_pipeline_and_extract_waiting_time(
                'pipeline.py', ev_file, args.chargers, 
                args.ngen, args.pop_size, args.ga_scope
            )
        
        # FCFS pipeline (pipeline_fcfs.py)
        fcfs_wait = None
        if args.use_existing_logs:
            log_file = f'run_{size}evs_fcfs.log'
            print(f"Trying to extract from log: {log_file}")
            fcfs_wait = extract_from_log_file(log_file)
        
        if fcfs_wait is None:
            print(f"Running FCFS pipeline for {size} EVs...")
            fcfs_wait = run_pipeline_and_extract_waiting_time(
                'pipeline_fcfs.py', ev_file, args.chargers,
                args.ngen, args.pop_size, None
            )
        
        if priority_wait is not None and fcfs_wait is not None:
            priority_wait_times.append(priority_wait)
            fcfs_wait_times.append(fcfs_wait)
            ev_sizes_actual.append(size)
            print(f"  Priority: {priority_wait:.3f} hours")
            print(f"  FCFS: {fcfs_wait:.3f} hours")
        else:
            print(f"  Warning: Failed to get waiting times for {size} EVs")
            if priority_wait is None:
                print(f"    Priority pipeline failed")
            if fcfs_wait is None:
                print(f"    FCFS pipeline failed")
    
    if not ev_sizes_actual:
        print("Error: No valid data collected. Cannot create plot.")
        return
    
    # Create comparison plot
    print(f"\n{'='*80}")
    print("Creating comparison plot...")
    print(f"{'='*80}")
    
    plt.figure(figsize=(12, 7))
    
    # Plot lines
    plt.plot(ev_sizes_actual, priority_wait_times, 
             marker='o', linewidth=2.5, markersize=8, 
             label='Priority Scheduling', color='#2E86AB', markerfacecolor='#2E86AB')
    plt.plot(ev_sizes_actual, fcfs_wait_times, 
             marker='s', linewidth=2.5, markersize=8, 
             label='FCFS Scheduling', color='#A23B72', markerfacecolor='#A23B72')
    
    # Styling
    plt.xlabel('EV Pool Size', fontsize=14, fontweight='bold')
    plt.ylabel('Average Waiting Time (hours)', fontsize=14, fontweight='bold')
    plt.title('Average Waiting Time Comparison: Priority vs FCFS Scheduling\n(30 Chargers, Fleet-wide Average)',
              fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    
    # Add value annotations
    for i, (size, pri, fcfs) in enumerate(zip(ev_sizes_actual, priority_wait_times, fcfs_wait_times)):
        # Annotate priority values (above line)
        plt.annotate(f'{pri:.2f}', 
                    (size, pri), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center', 
                    fontsize=9, 
                    color='#2E86AB',
                    fontweight='bold')
        # Annotate FCFS values (below line)
        plt.annotate(f'{fcfs:.2f}', 
                    (size, fcfs), 
                    textcoords="offset points", 
                    xytext=(0, -15), 
                    ha='center', 
                    fontsize=9, 
                    color='#A23B72',
                    fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {args.out}")
    plt.close()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'EV Size':<10} {'Priority (h)':<15} {'FCFS (h)':<15} {'Difference (h)':<15} {'% Change':<15}")
    print("-" * 80)
    for size, pri, fcfs in zip(ev_sizes_actual, priority_wait_times, fcfs_wait_times):
        diff = fcfs - pri
        pct_change = (diff / pri * 100) if pri > 0 else 0.0
        print(f"{size:<10} {pri:<15.3f} {fcfs:<15.3f} {diff:<15.3f} {pct_change:<15.2f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

