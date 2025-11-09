#!/usr/bin/env python3
"""
Plot GA convergence from saved ga_history CSV files created by pipeline.py.
Example filenames expected: run_50evs_30ch_ga_history.csv or ga_history.csv (single run).
This script will look for files matching a pattern and plot best_J vs generation for each run.
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', type=str, default='run_*_ga_history.csv')
    p.add_argument('--out', type=str, default='analysis/convergence_multiple_sizes.png')
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print("No ga_history CSV files found with pattern:", args.pattern)
        return

    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(range(len(files)))
    
    for idx, f in enumerate(files):
        try:
            df = pd.read_csv(f)
            label = os.path.basename(f).replace('_ga_history.csv','').replace('run_', '')
            if 'gen' in df.columns and 'best_J' in df.columns:
                plt.plot(df['gen'], df['best_J'], label=label, linewidth=2, 
                        color=colors[idx], marker='o', markersize=3, markevery=max(1, len(df)//20))
            else:
                print("Skipping", f, "missing cols")
        except Exception as e:
            print("Error reading", f, e)

    plt.xlabel('Generation', fontsize=12, fontweight='bold')
    plt.ylabel('Best J (objective)', fontsize=12, fontweight='bold')
    plt.title('GA Convergence: Multiple EV Pool Sizes (30 Chargers)', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to: {args.out}")
    plt.close()

if __name__ == "__main__":
    main()

