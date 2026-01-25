import os
import csv
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Configuration
EV_COUNTS = [50, 100, 150, 200, 250, 300]
CHARGERS = 30
NGEN = 500
POP_SIZE = 100
EVS_SOURCE = SCRIPT_DIR / 'evs.csv'
PIPELINE_SCRIPT = SCRIPT_DIR / 'pipeline.py'
OUTPUT_PLOT = SCRIPT_DIR / 'convergence_plot_ev_variations.png'

def load_source_evs(filepath):
    evs = []
    with open(str(filepath), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evs.append(row)
    return evs

def generate_ev_file(target_n, source_evs, filename):
    # Create N EVs by cycling through source and updating IDs
    generated = []
    source_len = len(source_evs)
    for i in range(target_n):
        src = source_evs[i % source_len].copy()
        src['id'] = i + 1
        # Add small noise to arrival/departure to avoid identical clones if cycling
        if i >= source_len:
            # perturb arrival slightly
            t_arr = int(src['T_arr_idx'])
            noise = random.randint(-2, 2)
            src['T_arr_idx'] = max(0, min(47, t_arr + noise))
        generated.append(src)
    
    keys = source_evs[0].keys()
    # Save to file
    output_path = SCRIPT_DIR / filename
    with open(str(output_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(generated)
    print(f"Generated {filename} with {target_n} EVs.")

def run_experiment():
    if not os.path.exists(EVS_SOURCE):
        print(f"Error: {EVS_SOURCE} not found.")
        return

    source_evs = load_source_evs(EVS_SOURCE)
    results = {}

    for n in EV_COUNTS:
        # Force re-run to get fresh results with new data
        history_file_specific = SCRIPT_DIR / f"ga_history_{n}.csv"
        if history_file_specific.exists():
            os.remove(history_file_specific)

        print(f"\n--- Running experiment for {n} EVs ---")
        # Use the pre-generated files
        ev_file = SCRIPT_DIR / f"evs_{n}.csv"
        if not ev_file.exists():
            print(f"Error: {ev_file} not found. Run generate_ev_datasets.py first.")
            continue

        # Use consistent generation count for all EV sizes
        current_ngen = NGEN

        # Run pipeline.py
        # Using --ga-schedule-scope full to ensure GA sees all EVs (problem size scales with N)
        cmd = [
            "/Users/ankitkumarsingh/Desktop/MMTP/.venv/bin/python", str(PIPELINE_SCRIPT),
            "--ev-file", str(ev_file),
            "--chargers", str(CHARGERS),
            "--ngen", str(current_ngen),
            "--pop-size", str(POP_SIZE),
            "--ga-schedule-scope", "full" 
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR), check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline for {n} EVs: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            continue

        # Read history
        # pipeline.py outputs ga_history.csv in its current working directory (SCRIPT_DIR)
        pipeline_output_history_file = SCRIPT_DIR / "ga_history.csv"
        if pipeline_output_history_file.exists():
            df = pd.read_csv(pipeline_output_history_file)
            results[n] = df['best_J'].tolist()
            # Rename history file to keep it
            os.rename(pipeline_output_history_file, history_file_specific)
        else:
            print(f"Warning: ga_history.csv not found for {n} EVs.")

        # Cleanup temp file - SKIPPED (keeping evs_N.csv)
        # if os.path.exists(ev_file):
        #    os.remove(ev_file)

    # Plotting
    plt.figure(figsize=(10, 6))
    for n, history in results.items():
        plt.plot(history, label=f'{n} EVs')
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (J)')
    plt.title(f'GA Convergence for Varying EV Counts (Chargers={CHARGERS})')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nPlot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    run_experiment()
