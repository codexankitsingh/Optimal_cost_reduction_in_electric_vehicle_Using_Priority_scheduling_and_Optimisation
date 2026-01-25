#!/usr/bin/env python3
"""
plot_sensitivity_results.py

Generates plots for the 3 sensitivity experiments:
1. EV Composition
2. Charger Mix (Power Distribution)
3. Initial SoC

Each plot shows 3 pipelines (Priority, FCFS, SJF) across scenarios.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_FILE = SCRIPT_DIR / "sensitivity_results.csv"
OUTPUT_DIR = SCRIPT_DIR / "sensitivity_plots"

COLORS = {"Priority": "#2E86AB", "FCFS": "#A23B72", "SJF": "#F18F01"}
METRICS = {
    "avg_cost_per_ev": ("Avg Cost per EV ($)", True),
    "avg_deg_per_ev": ("Avg Degradation Cost ($)", True),
    "avg_user_satisfaction": ("User Satisfaction", False),
    "avg_waiting_time": ("Avg Waiting Time (h)", True),
    "gini_fairness": ("Fairness (Gini)", True),
    "load_variance": ("Grid Load Variance", True),
}


def plot_experiment(df, experiment_name, title):
    """Create a 2x3 subplot for one experiment."""
    exp_data = df[df["experiment"] == experiment_name]
    if exp_data.empty:
        print(f"No data for experiment: {experiment_name}")
        return

    scenarios = exp_data["scenario"].unique()
    pipelines = ["Priority", "FCFS", "SJF"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric, (ylabel, lower_better)) in enumerate(METRICS.items()):
        ax = axes[idx]
        x = np.arange(len(scenarios))
        width = 0.25

        for i, pipeline in enumerate(pipelines):
            pipe_data = exp_data[exp_data["pipeline"] == pipeline]
            values = []
            for s in scenarios:
                matched = pipe_data[pipe_data["scenario"] == s][metric]
                if len(matched) > 0:
                    val = matched.values[0]
                    # Clamp extreme negative Gini values for visualization
                    if metric == "gini_fairness" and val < -1:
                        val = -1
                    values.append(val)
                else:
                    values.append(0)
            bars = ax.bar(x + i * width, values, width, label=pipeline, color=COLORS[pipeline])
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.01:  # Only label non-negligible values
                    label = f'{height:.2f}'
                    if abs(height) < 1000:
                        ax.annotate(label,
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3 if height > 0 else -10),
                                    textcoords="offset points",
                                    ha='center', va='bottom' if height > 0 else 'top',
                                    fontsize=6, rotation=90)

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ylabel, fontweight="bold", pad=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"sensitivity_{experiment_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_comparison(df):
    """Create a summary plot showing Cost comparison across all 3 experiments."""
    experiments = df["experiment"].unique()
    n_exp = len(experiments)

    fig, axes = plt.subplots(1, n_exp, figsize=(6 * n_exp, 5))
    if n_exp == 1:
        axes = [axes]

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        exp_data = df[df["experiment"] == exp]
        scenarios = exp_data["scenario"].unique()

        x = np.arange(len(scenarios))
        width = 0.25

        for i, pipeline in enumerate(["Priority", "FCFS", "SJF"]):
            pipe_data = exp_data[exp_data["pipeline"] == pipeline]
            values = [pipe_data[pipe_data["scenario"] == s]["avg_cost_per_ev"].values[0] if len(pipe_data[pipe_data["scenario"] == s]) > 0 else 0 for s in scenarios]
            ax.bar(x + i * width, values, width, label=pipeline, color=COLORS[pipeline])

        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_ylabel("Avg Cost per EV ($)", fontweight="bold")
        ax.set_title(exp.replace("_", " ").title(), fontweight="bold", pad=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenarios, rotation=15, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Cost Comparison Across All Experiments", fontsize=16, fontweight="bold")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "sensitivity_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING SENSITIVITY EXPERIMENT PLOTS")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print(f"Error: {RESULTS_FILE} not found!")
        print("Please run run_sensitivity_experiments.py first.")
        return

    df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(df)} rows from {RESULTS_FILE}")
    
    # Remove duplicates if any (keep last)
    df = df.drop_duplicates(subset=["experiment", "scenario", "pipeline"], keep="last")
    print(f"Rows after removing duplicates: {len(df)}")

    # Generate individual experiment plots
    plot_experiment(df, "composition", "EV Composition Sensitivity")
    plot_experiment(df, "charger_mix", "Charger Power Distribution Sensitivity")
    plot_experiment(df, "soc_sensitivity", "Initial SoC Sensitivity")

    # Generate summary plot
    plot_summary_comparison(df)

    print("\n" + "=" * 60)
    print("All plots generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

