#!/usr/bin/env python3
"""
compare_three_pipelines.py

Generate comparison plots (Priority+GA, FCFS+GA, SJF+GA) for fleet sizes
50,100,150,200,250,300.

The script will try to parse existing run log files (e.g. `run_50evs.log`) to
extract derived metrics. If `--run` is passed the script will execute the
corresponding pipeline scripts for each fleet size (use `--debug` to enable
short GA runs).

Outputs (PNG plots) are saved to `comparison_three_results/` by default.
"""

import argparse
import os
import re
import subprocess
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def extract_metrics_from_text(text: str) -> Dict[str, float]:
    m = {}
    # Average net energy cost per EV
    mm = re.search(r'Average net energy cost per EV:\s*([\d\.Ee\+\-]+)', text)
    if mm:
        m['avg_net_energy_cost'] = float(mm.group(1))
    # Average user satisfaction (admitted EVs)
    mm = re.search(r'Average user satisfaction \(admitted EVs\):\s*([\d\.Ee\+\-]+)', text)
    if mm:
        m['avg_user_satisfaction_admitted'] = float(mm.group(1))
    # Average user satisfaction (all EVs in pool)
    mm = re.search(r'Average user satisfaction \(all EVs in pool\):\s*([\d\.Ee\+\-]+)', text)
    if mm:
        m['avg_user_satisfaction'] = float(mm.group(1))
    # Fairness (Gini)
    mm = re.search(r'Fairness \(Gini .*?\):\s*([\d\.Ee\+\-]+)', text)
    if mm:
        m['gini'] = float(mm.group(1))
    # Average waiting time (all EVs in pool) preferred, else admitted
    mm = re.search(r'Average waiting time \(all EVs in pool\):\s*([\d\.Ee\+\-]+)\s*hours', text)
    if mm:
        m['avg_waiting_time'] = float(mm.group(1))
    else:
        mm = re.search(r'Average waiting time \(admitted EVs\):\s*([\d\.Ee\+\-]+)\s*hours', text)
        if mm:
            m['avg_waiting_time'] = float(mm.group(1))

    # If any missing, set NaN
    for key in ['avg_net_energy_cost', 'avg_user_satisfaction', 'gini', 'avg_waiting_time']:
        m.setdefault(key, float('nan'))
    return m


def read_log_if_exists(path: str) -> str:
    if os.path.exists(path):
        with open(path, 'r') as fh:
            return fh.read()
    return ''


def run_pipeline_and_capture(script: str, ev_file: str, chargers: int, debug: bool = False) -> str:
    cmd = [sys.executable, script, '--ev-file', ev_file, '--chargers', str(chargers)]
    if debug:
        cmd.append('--debug-short-run')
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + '\n' + proc.stderr
    if proc.returncode != 0:
        print(f"Warning: script {script} exited with {proc.returncode}")
    return out


def gather_metrics(pipelines: Dict[str, str], fleet_sizes: List[int], ev_prefix: str, run_scripts: bool, debug: bool, out_dir: str = None):
    # results[pipeline_name] = list of metrics in order of fleet_sizes
    results = {name: [] for name in pipelines}

    for n in fleet_sizes:
        ev_file = f"{ev_prefix}_{n}.csv" if ev_prefix and not ev_prefix.endswith(str(n)) else f"evs_{n}.csv"
        # be flexible: if ev file doesn't exist, try `evs_{n}.csv` in repo root
        if not os.path.exists(ev_file):
            ev_file = os.path.join(os.path.dirname(__file__), '..', f'evs_{n}.csv')
            ev_file = os.path.normpath(ev_file)
        for name, script in pipelines.items():
            # try to read log first: patterns: run_{n}evs_{name}.log, run_{n}evs.log
            log_candidates = [
                os.path.join(out_dir, f'run_{n}evs_{name}.log'),
                os.path.join(out_dir, f'run_{n}evs.log')
            ]
            log_text = ''
            for lc in log_candidates:
                if os.path.exists(lc):
                    log_text = read_log_if_exists(lc)
                    break
            if not log_text and run_scripts:
                # execute the pipeline script and capture output
                log_text = run_pipeline_and_capture(script, ev_file, 30, debug=debug)
                # if an output directory is provided, save the captured log for future parsing
                if out_dir:
                    out_path = None  # Initialize out_path to avoid unbound error
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f'run_{n}evs_{name}.log')
                        with open(out_path, 'w', encoding='utf-8') as fh:
                            fh.write(log_text)
                    except Exception as e:
                        print(f"Warning: failed to write log to {out_path}: {e}")
            if not log_text:
                print(f"No data for pipeline={name}, fleet={n} (tried logs and running). Setting NaNs.")
                metrics = {k: float('nan') for k in ['avg_net_energy_cost','avg_user_satisfaction','gini','avg_waiting_time']}
            else:
                metrics = extract_metrics_from_text(log_text)
            results[name].append(metrics)

    return results


def plot_results(results: Dict[str, List[Dict[str, float]]], fleet_sizes: List[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    methods = list(results.keys())

    # four plots requested:
    # 1 waiting time comparison
    # 2 user satisfaction comparison
    # 3 gini fairness comparison
    # 4 average net energy cost per ev comparison

    metrics_keys = [
        ('avg_waiting_time', 'Average Waiting Time (hours)'),
        ('avg_user_satisfaction', 'Average User Satisfaction'),
        ('gini', 'Fairness (Gini Coefficient)'),
        ('avg_net_energy_cost', 'Average Net Energy Cost per EV')
    ]

    for key, label in metrics_keys:
        plt.figure(figsize=(8, 5))
        for method in methods:
            vals = [m.get(key, float('nan')) for m in results[method]]
            vals = np.array(vals, dtype=float)
            plt.plot(fleet_sizes, vals, marker='o', label=method)
        plt.xlabel('Number of EVs')
        plt.ylabel(label)
        plt.title(f'{label} — Priority vs FCFS vs SJF')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'{key}_comparison.png')
        plt.savefig(out_path, dpi=200)
        print(f'Saved: {out_path}')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare Priority+GA, FCFS+GA, SJF+GA across fleet sizes')
    parser.add_argument('--ev-prefix', type=str, default='evs', help='Prefix for EV files (default: evs)')
    parser.add_argument('--sizes', type=int, nargs='+', default=[50,100,150,200,250,300],
                        help='Fleet sizes (number of EVs).')
    parser.add_argument('--output-dir', type=str, default='comparison_three_results_final', help='Output dir for plots')
    parser.add_argument('--run', action='store_true', help='Actually run the pipeline scripts if logs missing (may be slow)')
    parser.add_argument('--debug', action='store_true', help='Use debug short runs when executing pipelines')
    args = parser.parse_args()

    pipelines = {
        'priority+ga': os.path.join('..', 'pipeline.py'),
        'fcfs+ga': os.path.join('..', 'pipeline_fcfs.py'),
        'sjf+ga': os.path.join('..', 'pipeline_sjf.py')
    }

    # Normalize script paths relative to this script
    for k in list(pipelines.keys()):
        pipelines[k] = os.path.normpath(os.path.join(os.path.dirname(__file__), pipelines[k]))

    print('Pipelines:', pipelines)

    results = gather_metrics(pipelines, args.sizes, args.ev_prefix, run_scripts=args.run, debug=args.debug, out_dir=args.output_dir)

    plot_results(results, args.sizes, args.output_dir)


if __name__ == '__main__':
    main()
