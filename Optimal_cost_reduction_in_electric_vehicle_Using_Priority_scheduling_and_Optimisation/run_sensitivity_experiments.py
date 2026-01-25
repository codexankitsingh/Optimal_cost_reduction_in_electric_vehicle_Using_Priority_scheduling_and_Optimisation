#!/usr/bin/env python3
"""
run_sensitivity_experiments.py

Runs all 4 sensitivity experiments across Priority, FCFS, and SJF pipelines:
1. EV Composition (Compact vs SUV)
2. Charger Mix (Slow/Medium/Fast)
3. Initial SoC (Low/Moderate/High)
4. Price Trends (Flat/Mild/Volatile)

Outputs: sensitivity_results.csv
"""

import os
import subprocess
import pandas as pd
import re
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PYTHON_PATH = "/Users/ankitkumarsingh/Desktop/MMTP/.venv/bin/python"
NGEN = 100  # Reduced for faster runs
POP_SIZE = 80
CHARGERS = 30

PIPELINES = {
    "Priority": SCRIPT_DIR / "pipeline.py",
    "FCFS": SCRIPT_DIR / "pipeline_fcfs.py",
    "SJF": SCRIPT_DIR / "pipeline_sjf.py",
}

# Experiment Configurations
EXPERIMENTS = {
    "composition": {
        "description": "EV Composition (Compact vs SUV)",
        "scenarios": {
            "70% Compact": "sensitivity_datasets/evs_comp_scen1.csv",
            "50% Compact": "sensitivity_datasets/evs_comp_scen2.csv",
            "30% Compact": "sensitivity_datasets/evs_comp_scen3.csv",
        },
    },
    "charger_mix": {
        "description": "Charger Power Distribution (Slow/Medium/Fast)",
        "scenarios": {
            "Slow (3-5kW)": "sensitivity_datasets/evs_charger_slow.csv",
            "Medium (5-9kW)": "sensitivity_datasets/evs_charger_medium.csv",
            "Fast (9-15kW)": "sensitivity_datasets/evs_charger_fast.csv",
        },
    },
    "soc_sensitivity": {
        "description": "Initial SoC Distribution",
        "scenarios": {
            "Low SoC": "sensitivity_datasets/evs_soc_low.csv",
            "Moderate SoC": "sensitivity_datasets/evs_soc_mod.csv",
            "High SoC": "sensitivity_datasets/evs_soc_high.csv",
        },
    },
}


def extract_metrics_from_output(output):
    """Extract key metrics from pipeline output."""
    metrics = {
        "avg_cost_per_ev": None,
        "avg_deg_per_ev": None,
        "avg_user_satisfaction": None,
        "avg_waiting_time": None,
        "gini_fairness": None,
        "load_variance": None,
    }

    patterns = {
        "avg_cost_per_ev": r"Average net energy cost per EV:\s*([-\d.]+)",
        "avg_deg_per_ev": r"Average battery degradation cost per EV:\s*([\d.]+)",
        "avg_user_satisfaction": r"Average user satisfaction.*?:\s*([\d.]+)",
        "avg_waiting_time": r"Average waiting time \(all EVs in pool\):\s*([\d.]+)",
        "gini_fairness": r"Fairness \(Gini.*?\):\s*([-\d.]+)",
        "load_variance": r"Grid load variance.*?:\s*([\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))

    return metrics


def run_pipeline(pipeline_name, pipeline_script, ev_file, charger_config=None, price_config=None):
    """Run a single pipeline and extract metrics."""
    cmd = [
        PYTHON_PATH,
        str(pipeline_script),
        "--ev-file",
        str(SCRIPT_DIR / ev_file),
        "--chargers",
        str(CHARGERS),
        "--ngen",
        str(NGEN),
        "--pop-size",
        str(POP_SIZE),
        "--ga-schedule-scope",
        "full",
    ]

    # Add charger config if specified (future extension)
    # Add price config if specified (future extension)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(SCRIPT_DIR), timeout=600)
        output = result.stdout + result.stderr
        metrics = extract_metrics_from_output(output)
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {pipeline_name} failed: {e}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {pipeline_name} exceeded 10 minutes")
        return None



def append_result_to_csv(result):
    """Append a single result row to CSV."""
    df = pd.DataFrame([result])
    output_path = SCRIPT_DIR / "sensitivity_results.csv"
    # Append if exists, else write header
    write_header = not output_path.exists()
    df.to_csv(output_path, mode='a', header=write_header, index=False)


def run_experiment(experiment_name, config):
    """Run a single experiment across all scenarios and pipelines."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config['description']}")
    print(f"{'='*70}")

    results = []


    for scenario_name, scenario_config in config["scenarios"].items():
        print(f"\n--- Scenario: {scenario_name} ---")

        # All experiments now use file paths directly
        ev_file = scenario_config

        for pipeline_name, pipeline_script in PIPELINES.items():
            print(f"  Running {pipeline_name}...")
            metrics = run_pipeline(pipeline_name, pipeline_script, ev_file)

            if metrics:
                result = {
                    "experiment": experiment_name,
                    "scenario": scenario_name,
                    "pipeline": pipeline_name,
                    **metrics,
                }
                results.append(result)
                append_result_to_csv(result)  # Save immediately
                
                cost_val = metrics.get('avg_cost_per_ev')
                sat_val = metrics.get('avg_user_satisfaction')
                cost_str = f"${cost_val:.2f}" if cost_val is not None else "N/A"
                sat_str = f"{sat_val:.3f}" if sat_val is not None else "N/A"
                print(f"    Cost: {cost_str}, Satisfaction: {sat_str}")

    return results


def main():
    print("=" * 70)
    print("SENSITIVITY EXPERIMENTS")
    print("=" * 70)
    print(f"Pipelines: {list(PIPELINES.keys())}")
    print(f"Generations: {NGEN}, Population: {POP_SIZE}, Chargers: {CHARGERS}")
    
    # Remove old results file to start fresh
    output_path = SCRIPT_DIR / "sensitivity_results.csv"
    if output_path.exists():
        output_path.unlink()
        print(f"Removed previous results file: {output_path}")

    all_results = []

    for exp_name, exp_config in EXPERIMENTS.items():
        results = run_experiment(exp_name, exp_config)
        all_results.extend(results)

    print(f"\n{'='*70}")
    print(f"All experiments completed. Results saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
