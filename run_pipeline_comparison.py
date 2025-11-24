import os
import subprocess
import pandas as pd
import re
import json
from pathlib import Path

# Configuration
EV_COUNTS = [50, 100, 150, 200, 250, 300]
CHARGERS = 30
NGEN = 200
POP_SIZE = 100
PYTHON_PATH = "/Users/ankitkumarsingh/Desktop/MMTP/.venv/bin/python"

PIPELINES = {
    'Priority': 'pipeline.py',
    'FCFS': 'pipeline_fcfs.py',
    'SJF': 'pipeline_sjf.py'
}

def extract_metrics_from_output(output):
    """Extract key metrics from pipeline output."""
    metrics = {
        'avg_cost_per_ev': None,
        'avg_deg_per_ev': None,
        'avg_user_satisfaction_admitted': None,
        'avg_user_satisfaction_all': None,
        'avg_waiting_time_admitted': None,
        'avg_waiting_time_all': None,
        'gini_fairness': None,
        'final_J': None,
        'generations_executed': None
    }
    
    # Extract average net energy cost per EV
    match = re.search(r'Average net energy cost per EV:\s*([-\d.]+)', output)
    if match:
        metrics['avg_cost_per_ev'] = float(match.group(1))
    
    # Extract average battery degradation cost per EV
    match = re.search(r'Average battery degradation cost per EV:\s*([\d.]+)', output)
    if match:
        metrics['avg_deg_per_ev'] = float(match.group(1))
    
    # Extract user satisfaction (more flexible matching)
    match = re.search(r'Average user satisfaction \(admitted EVs\):\s*([\d.]+)', output)
    if match:
        metrics['avg_user_satisfaction_admitted'] = float(match.group(1))
    
    # Extract avg_user_satisfaction (without "all EVs in pool" qualifier)
    match = re.search(r'Average user satisfaction:\s*([\d.]+)', output)
    if match:
        metrics['avg_user_satisfaction_all'] = float(match.group(1))
    # Also try with qualifier
    match = re.search(r'Average user satisfaction \(all EVs in pool\):\s*([\d.]+)', output)
    if match:
        metrics['avg_user_satisfaction_all'] = float(match.group(1))
    
    # Extract waiting time (admitted)
    match = re.search(r'Average waiting time \(admitted EVs\):\s*([\d.]+)', output)
    if match:
        metrics['avg_waiting_time_admitted'] = float(match.group(1))
    
    # Extract waiting time (all)
    match = re.search(r'Average waiting time \(all EVs in pool\):\s*([\d.]+)', output)
    if match:
        metrics['avg_waiting_time_all'] = float(match.group(1))
    
    # Extract Gini fairness
    match = re.search(r'Fairness \(Gini on delivered kWh\):\s*([-\d.]+)', output)
    if match:
        metrics['gini_fairness'] = float(match.group(1))
    
    # Extract final objective J
    match = re.search(r'Objective J \(normalized weighted\):\s*([-\d.]+)', output)
    if match:
        metrics['final_J'] = float(match.group(1))
    
    # Extract generations executed
    match = re.search(r'Generations executed:\s*(\d+)', output)
    if match:
        metrics['generations_executed'] = int(match.group(1))
    
    return metrics

def run_pipeline(pipeline_name, pipeline_script, ev_count):
    """Run a single pipeline and extract metrics."""
    print(f"\n{'='*60}")
    print(f"Running {pipeline_name} pipeline with {ev_count} EVs")
    print(f"{'='*60}")
    
    ev_file = f"evs_{ev_count}.csv"
    if not os.path.exists(ev_file):
        print(f"Error: {ev_file} not found!")
        return None
    
    cmd = [
        PYTHON_PATH, pipeline_script,
        "--ev-file", ev_file,
        "--chargers", str(CHARGERS),
        "--ngen", str(NGEN),
        "--pop-size", str(POP_SIZE),
        "--ga-schedule-scope", "full"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        
        # Extract metrics
        metrics = extract_metrics_from_output(output)
        metrics['pipeline'] = pipeline_name
        metrics['ev_count'] = ev_count
        
        # Save raw output for debugging
        output_dir = Path("comparison_outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"{pipeline_name.lower()}_{ev_count}_output.txt"
        with open(output_file, 'w') as f:
            f.write(output)
        
        print(f"✓ Completed {pipeline_name} with {ev_count} EVs")
        # Safe printing of metrics
        cost_str = f"${metrics['avg_cost_per_ev']:.2f}" if metrics['avg_cost_per_ev'] is not None else "N/A"
        sat_str = f"{metrics['avg_user_satisfaction_all']:.3f}" if metrics['avg_user_satisfaction_all'] is not None else "N/A"
        gini_str = f"{metrics['gini_fairness']:.3f}" if metrics['gini_fairness'] is not None else "N/A"
        print(f"  Cost: {cost_str}, Satisfaction: {sat_str}, Gini: {gini_str}")
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {pipeline_name} with {ev_count} EVs: {e}")
        return None

def run_all_experiments():
    """Run all pipeline experiments."""
    all_results = []
    
    for ev_count in EV_COUNTS:
        for pipeline_name, pipeline_script in PIPELINES.items():
            metrics = run_pipeline(pipeline_name, pipeline_script, ev_count)
            if metrics:
                all_results.append(metrics)
    
    return all_results

def save_results_to_csv(results):
    """Save results to CSV files."""
    if not results:
        print("No results to save!")
        return
    
    df = pd.DataFrame(results)
    
    # Save detailed results
    df.to_csv('comparison_results.csv', index=False)
    print(f"\n✓ Saved detailed results to comparison_results.csv")
    
    # Create summary statistics
    summary = df.groupby(['pipeline', 'ev_count']).agg({
        'avg_cost_per_ev': 'mean',
        'avg_user_satisfaction_all': 'mean',
        'avg_waiting_time_all': 'mean',
        'gini_fairness': 'mean',
        'final_J': 'mean'
    }).reset_index()
    
    summary.to_csv('comparison_summary.csv', index=False)
    print(f"✓ Saved summary to comparison_summary.csv")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    print("="*80)
    print("PIPELINE COMPARISON EXPERIMENT")
    print("="*80)
    print(f"EV Counts: {EV_COUNTS}")
    print(f"Pipelines: {list(PIPELINES.keys())}")
    print(f"Generations: {NGEN}, Population: {POP_SIZE}, Chargers: {CHARGERS}")
    print("="*80)
    
    results = run_all_experiments()
    save_results_to_csv(results)
    
    print(f"\n{'='*80}")
    print("Experiment completed! Run plot_pipeline_comparison.py to generate plots.")
    print(f"{'='*80}")
