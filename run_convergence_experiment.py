import os
import csv
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
EV_COUNTS = [50, 100, 150, 200, 250, 300]
CHARGERS = 30
NGEN = 200
POP_SIZE = 100
EVS_SOURCE = 'evs.csv'
PIPELINE_SCRIPT = 'pipeline.py'
OUTPUT_PLOT = 'convergence_plot_ev_variations.png'

def load_source_evs(filepath):
    evs = []
    with open(filepath, 'r') as f:
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
    with open(filename, 'w', newline='') as f:
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
        history_file = f"ga_history_{n}.csv"
        if os.path.exists(history_file):
            os.remove(history_file)

        print(f"\n--- Running experiment for {n} EVs ---")
        # Use the pre-generated files
        ev_file = f"evs_{n}.csv"
        if not os.path.exists(ev_file):
            print(f"Error: {ev_file} not found. Run generate_ev_datasets.py first.")
            continue

        # Use consistent generation count for all EV sizes
        current_ngen = NGEN

        # Run pipeline.py
        # Using --ga-schedule-scope full to ensure GA sees all EVs (problem size scales with N)
        cmd = [
            "/Users/ankitkumarsingh/Desktop/MMTP/.venv/bin/python", PIPELINE_SCRIPT,
            "--ev-file", ev_file,
            "--chargers", str(CHARGERS),
            "--ngen", str(current_ngen),
            "--pop-size", str(POP_SIZE),
            "--ga-schedule-scope", "full" 
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running pipeline for {n} EVs: {e}")
            continue

        # Read history
        if os.path.exists("ga_history.csv"):
            df = pd.read_csv("ga_history.csv")
            results[n] = df['best_J'].tolist()
            # Rename history file to keep it
            os.rename("ga_history.csv", f"ga_history_{n}.csv")
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
