#!/usr/bin/env python3
"""
Plot results from GA runs:
- Best & mean J per generation
- Occupancy time-series
- Waiting time histogram
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def plot_ga_convergence(history_file='ga_history.csv', output='best_J.png'):
    """Plot best J and mean J vs generation"""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found")
        return
    
    df = pd.read_csv(history_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['gen'], df['best_J'], label='Best J', linewidth=2)
    ax.plot(df['gen'], df['mean_J'], label='Mean J', linewidth=1.5, alpha=0.7)
    ax.fill_between(df['gen'], df['mean_J'] - df['std_J'], 
                     df['mean_J'] + df['std_J'], alpha=0.2, label='±1 std')
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Objective J (normalized)', fontsize=12)
    ax.set_title('GA Convergence: Best and Mean Objective', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved convergence plot to: {output}")
    plt.close()

def plot_occupancy(schedule_file='best_schedule_normalized.csv', 
                   output='occupancy.png', M=30):
    """Plot occupancy (number of active EVs) over time"""
    if not os.path.exists(schedule_file):
        print(f"Warning: {schedule_file} not found")
        return
    
    df = pd.read_csv(schedule_file, index_col=0)
    schedule = df.values
    T = schedule.shape[1]
    
    # Count active EVs per slot
    occupancy = []
    for t in range(T):
        active = int(np.sum(np.abs(schedule[:, t]) > 1e-6))
        occupancy.append(active)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(T), occupancy, linewidth=2, label='Active EVs')
    ax.axhline(y=M, color='r', linestyle='--', label=f'Capacity (M={M})')
    ax.fill_between(range(T), 0, M, alpha=0.1, color='green', label='Available capacity')
    
    ax.set_xlabel('Time Slot', fontsize=12)
    ax.set_ylabel('Number of Active EVs', fontsize=12)
    ax.set_title('Occupancy Over Time', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(M + 2, max(occupancy) + 2))
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved occupancy plot to: {output}")
    plt.close()

def plot_waiting_time_histogram(waiting_times_dict=None, output='waiting_hist.png'):
    """Plot waiting time histogram"""
    if waiting_times_dict is None:
        print("Warning: No waiting times provided")
        return
    
    waiting_times = list(waiting_times_dict.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(waiting_times, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(waiting_times), color='r', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(waiting_times):.2f}h')
    ax.axvline(x=np.median(waiting_times), color='g', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(waiting_times):.2f}h')
    
    ax.set_xlabel('Waiting Time (hours)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Waiting Time Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved waiting time histogram to: {output}")
    plt.close()

def plot_soc_final(schedule_file='best_schedule_normalized.csv',
                   ev_file='evs.csv', output='soc_final.png'):
    """Plot final SoC distribution"""
    if not os.path.exists(schedule_file):
        print(f"Warning: {schedule_file} not found")
        return
    
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline import load_ev_pool, compute_Si_from_schedule
    
    df = pd.read_csv(schedule_file, index_col=0)
    schedule = df.values
    ev_ids = [int(idx) for idx in df.index]
    
    # Load EV data
    if os.path.exists(ev_file):
        ev_pool = load_ev_pool(ev_file)
        ev_dict = {ev['id']: ev for ev in ev_pool}
        evs = []
        for ev_id in ev_ids:
            if ev_id in ev_dict:
                ev = ev_dict[ev_id].copy()
                ev['T_arr_idx'] = ev.get('T_arr_idx', 0)
                ev['T_dep_idx'] = ev.get('T_dep_idx', schedule.shape[1])
                evs.append(ev)
            else:
                evs.append({
                    'id': ev_id,
                    'Ecap': 40.0,
                    'SoC_init': 0.2,
                    'SoC_max': 0.8,
                    'SoC_min': 0.0,
                    'T_arr_idx': 0,
                    'T_dep_idx': schedule.shape[1]
                })
    else:
        evs = [{'id': ev_id, 'Ecap': 40.0, 'SoC_init': 0.2, 'SoC_max': 0.8, 
                'SoC_min': 0.0, 'T_arr_idx': 0, 'T_dep_idx': schedule.shape[1]} 
               for ev_id in ev_ids]
    
    # Compute final SoC for each EV
    final_soc = []
    delta_t = 0.5
    for i, ev in enumerate(evs):
        SoC = ev['SoC_init']
        Ecap = ev['Ecap']
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', schedule.shape[1])
        
        for t in range(Tarr, min(Tdep, schedule.shape[1])):
            if Ecap > 0:
                SoC += (schedule[i, t] * delta_t) / Ecap
        final_soc.append(SoC)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_soc, bins=20, edgecolor='black', alpha=0.7)
    
    # Mark SoC bounds
    if evs:
        soc_min_avg = np.mean([ev.get('SoC_min', 0.0) for ev in evs])
        soc_max_avg = np.mean([ev.get('SoC_max', 0.8) for ev in evs])
        ax.axvline(x=soc_min_avg, color='r', linestyle='--', linewidth=2, label='Avg SoC_min')
        ax.axvline(x=soc_max_avg, color='g', linestyle='--', linewidth=2, label='Avg SoC_max')
    
    ax.set_xlabel('Final SoC', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Final State of Charge Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved final SoC distribution to: {output}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot GA results')
    parser.add_argument('--history', type=str, default='ga_history.csv',
                        help='GA history CSV file')
    parser.add_argument('--schedule', type=str, default='best_schedule_normalized.csv',
                        help='Schedule CSV file')
    parser.add_argument('--ev-file', type=str, default='evs.csv',
                        help='EV data file')
    parser.add_argument('--chargers', type=int, default=30,
                        help='Number of chargers M')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating plots...")
    plot_ga_convergence(
        os.path.join(args.output_dir, args.history),
        os.path.join(args.output_dir, 'best_J.png')
    )
    plot_occupancy(
        os.path.join(args.output_dir, args.schedule),
        os.path.join(args.output_dir, 'occupancy.png'),
        args.chargers
    )
    plot_soc_final(
        os.path.join(args.output_dir, args.schedule),
        args.ev_file,
        os.path.join(args.output_dir, 'soc_final.png')
    )
    
    print("\nAll plots generated!")

