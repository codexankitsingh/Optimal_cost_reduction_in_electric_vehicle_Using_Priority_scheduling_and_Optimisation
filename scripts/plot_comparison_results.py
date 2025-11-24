#!/usr/bin/env python3
"""
Parse pipeline run logs and plot comparison metrics for multiple fleet sizes.

Usage examples:
  python3 scripts/plot_comparison_results.py --log-dir . --output-dir comparison_plot_outputs \
      --fleets 50 100 150 200 250 300

The script looks recursively for files named like `run_{N}evs*.log` and extracts
the following metrics from each pipeline run (expected printed lines):
  - Average net energy cost per EV: <value>
  - Average user satisfaction: <value>
  - Fairness (Gini on delivered kWh): <value>
  - Average waiting time (all EVs in pool): <value> hours

If a metric is missing for a (fleet, pipeline) pair it will be recorded as NaN.
Output:
  - `comparison_metrics.csv` in `--output-dir`
  - Four PNGs, one per metric, saved to `--output-dir`
"""
import argparse
import glob
import os
import re
import math
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


METRIC_PATTERNS = {
    'avg_net_energy_cost': re.compile(r'Average net energy cost per EV:\s*([\d\.eE\-+]+)', flags=re.I),
    'avg_user_satisfaction': re.compile(r'Average user satisfaction(?:\s*\([^)]+\))?:\s*([\d\.eE\-+]+)', flags=re.I),
    'gini': re.compile(r'Fairness \(Gini .*?\):\s*([\d\.eE\-+]+)', flags=re.I),
    'avg_waiting_time': re.compile(r'Average waiting time .*?:\s*([\d\.eE\-+]+)\s*hours', flags=re.I),
}


def extract_metrics_from_text(text):
    out = {}
    for k, pat in METRIC_PATTERNS.items():
        # For user satisfaction prefer the 'all EVs in pool' variant if present
        if k == 'avg_user_satisfaction':
            m_all = re.search(r'Average user satisfaction \(all EVs in pool\):\s*([\d\.eE\-+]+)', text, flags=re.I)
            if m_all:
                try:
                    out[k] = float(m_all.group(1))
                    continue
                except Exception:
                    out[k] = float('nan')
            # fall back to any variant
        m = pat.search(text)
        if m:
            try:
                out[k] = float(m.group(1))
            except Exception:
                out[k] = float('nan')
        else:
            out[k] = float('nan')
    return out


def detect_pipeline_from_text(text):
    t = text.lower()
    # check for explicit GA history file mentions (we saved these with pipeline suffixes)
    if 'ga_history_sjf' in t or 'ga_history_sjf.csv' in t:
        return 'sjf+ga'
    if 'ga_history_fcfs' in t or 'ga_history_fcfs.csv' in t:
        return 'fcfs+ga'
    if 'ga_history' in t and 'fcfs' in t:
        return 'fcfs+ga'
    if 'ga_history' in t and 'sjf' in t:
        return 'sjf+ga'

    if 'sjf' in t or 'shortest-job-first' in t or 'shortest job' in t:
        return 'sjf+ga'
    if 'fcfs' in t or 'first-come-first-served' in t or 'first come first served' in t or 'first come' in t:
        return 'fcfs+ga'
    if 'priority' in t or 'priority admission' in t:
        return 'priority+ga'
    return None


def find_logs_for_fleet(log_dir, fleet):
    pattern = os.path.join(log_dir, '**', f'run_{fleet}evs*.log')
    files = glob.glob(pattern, recursive=True)
    return files


def gather_metrics(log_dir, fleets, pipelines):
    rows = []
    for fleet in fleets:
        files = find_logs_for_fleet(log_dir, fleet)
        # Initialize assigned dictionary with None values
        assigned = {p: None for p in pipelines}
        # filename-based
        for f in files:
            name = os.path.basename(f).lower()
            clean_name = re.sub(r'[^a-z0-9]', '', name)
            for p in pipelines:
                slug = re.sub(r'[^a-z0-9]', '', p.lower())
                if slug in clean_name:
                    assigned[p] = f
        # content-based detection for any unassigned
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
            except Exception:
                continue
            detected = detect_pipeline_from_text(txt)
            if detected and detected in assigned and assigned[detected] is None:
                assigned[detected] = f
        # As last resort, parse generic run_{N}evs.log and try to infer pipeline from headers
        generic_files = [f for f in files if os.path.basename(f).lower() == f'run_{fleet}evs.log']
        for f in generic_files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
            except Exception:
                continue
            detected = detect_pipeline_from_text(txt)
            if detected and detected in assigned and assigned[detected] is None:
                assigned[detected] = f

        # If no file was directly assigned to any pipeline, try a content-first pass
        content_assigned = {p: None for p in pipelines}
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    txt = fh.read()
            except Exception:
                continue
            detected = detect_pipeline_from_text(txt)
            if detected and detected in content_assigned and content_assigned[detected] is None:
                content_assigned[detected] = f

        # Merge filename-based assigned and content-based assigned (prefer filename mapping)
        for p in pipelines:
            fpath = assigned.get(p) or content_assigned.get(p)
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                        txt = fh.read()
                except Exception:
                    txt = ''
                metrics = extract_metrics_from_text(txt)
                row = {'fleet': fleet, 'pipeline': p}
                row.update(metrics)
                rows.append(row)
            else:
                # last resort: if there's only one file and only one pipeline missing, assign heuristically
                if len(files) == 1 and len(pipelines) == 1:
                    fpath = files[0]
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as fh:
                            txt = fh.read()
                    except Exception:
                        txt = ''
                    metrics = extract_metrics_from_text(txt)
                    row = {'fleet': fleet, 'pipeline': p}
                    row.update(metrics)
                    rows.append(row)
                else:
                    row = {'fleet': fleet, 'pipeline': p}
                    for k in METRIC_PATTERNS.keys():
                        row[k] = float('nan')
                    rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(df, metric, fleets, pipelines, outdir):
    pivot = df.pivot(index='fleet', columns='pipeline', values=metric)
    pivot = pivot.reindex(fleets)

    plt.figure(figsize=(10,6))
    markers = ['o', 's', 'D', '^', 'v']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, p in enumerate(pipelines):
        if p in pivot.columns:
            y = pivot[p].values
            # handle NaN gracefully: matplotlib will break lines at NaN
            plt.plot(fleets, y, marker=markers[i % len(markers)], color=colors[i % len(colors)], label=p)
    plt.xlabel('Number of EVs')
    if metric == 'avg_user_satisfaction':
        plt.ylabel('Average User Satisfaction')
        title = 'Average User Satisfaction — Priority vs FCFS vs SJF'
    elif metric == 'avg_waiting_time':
        plt.ylabel('Average Waiting Time (hours)')
        title = 'Average Waiting Time — Priority vs FCFS vs SJF'
    elif metric == 'gini':
        plt.ylabel('Gini Coefficient')
        title = 'Fairness (Gini) — Priority vs FCFS vs SJF'
    else:
        plt.ylabel('Average Net Energy Cost per EV')
        title = 'Average Net Energy Cost per EV — Priority vs FCFS vs SJF'
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(fleets)
    plt.legend()
    fname = os.path.join(outdir, f'{metric}_comparison.png')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', default='.', help='Directory to search for run logs (recursive)')
    parser.add_argument('--output-dir', default='comparison_plot_outputs', help='Directory to save CSV and PNGs')
    parser.add_argument('--fleets', nargs='+', type=int, default=[50,100,150,200,250,300], help='Fleet sizes to include')
    parser.add_argument('--pipelines', nargs='+', default=['priority+ga','fcfs+ga','sjf+ga'], help='Pipeline names to look for')
    parser.add_argument('--pipeline-dirs', default=None,
                        help='Optional comma-separated mapping pipeline=dir. Example: "priority+ga:/path/to/priority_logs,fcfs+ga:/path/to/fcfs_logs"')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If pipeline_dirs mapping provided, prefer those directories per-pipeline
    if args.pipeline_dirs:
        mapping = {}
        for part in args.pipeline_dirs.split(','):
            if '=' in part:
                k, v = part.split('=', 1)
            elif ':' in part:
                # allow either k=path or k:path
                k, v = part.split(':', 1)
            else:
                continue
            mapping[k.strip()] = v.strip()
        # create consolidated temporary log-dir listing where each pipeline's files will be found
        # gather_metrics will still search recursively, but we'll point it to each provided dir in turn
        frames = []
        for p in args.pipelines:
            d = mapping.get(p)
            if not d:
                # fallback to main log-dir
                df_p = gather_metrics(args.log_dir, args.fleets, [p])
            else:
                df_p = gather_metrics(d, args.fleets, [p])
            frames.append(df_p)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = gather_metrics(args.log_dir, args.fleets, args.pipelines)
    summary_csv = os.path.join(args.output_dir, 'comparison_metrics.csv')
    df.to_csv(summary_csv, index=False)
    print('Saved CSV summary to:', summary_csv)

    # Create plots for each metric
    for metric in ['avg_waiting_time', 'avg_user_satisfaction', 'gini', 'avg_net_energy_cost']:
        plot_metric(df, metric, args.fleets, args.pipelines, args.output_dir)
        print('Saved plot for', metric)


if __name__ == '__main__':
    main()
