import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
EV_COUNTS = [50, 100, 150, 200, 250, 300]
CHARGERS = 30
OUTPUT_PLOT = 'convergence_plot_ev_variations.png'

def plot_convergence():
    results = {}
    
    for n in EV_COUNTS:
        history_file = f"ga_history_{n}.csv"
        if os.path.exists(history_file):
            print(f"Loading {history_file}...")
            df = pd.read_csv(history_file)
            # Ensure we have the best_J column
            if 'best_J' in df.columns:
                results[n] = df['best_J'].tolist()
                print(f"  Loaded {len(df)} generations for {n} EVs")
                print(f"  Initial J: {df['best_J'].iloc[0]:.6f}, Final J: {df['best_J'].iloc[-1]:.6f}")
            else:
                print(f"  Warning: 'best_J' column not found in {history_file}")
        else:
            print(f"Warning: {history_file} not found. Skipping {n} EVs.")
    
    if not results:
        print("Error: No history files found. Please run experiments first.")
        return
    
    # Plotting
    plt.figure(figsize=(12, 7))
    for n, history in sorted(results.items()):
        plt.plot(history, label=f'{n} EVs', linewidth=2, marker='o', markevery=max(1, len(history)//20))
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness (J)', fontsize=12)
    plt.title(f'GA Convergence for Varying EV Counts (Chargers={CHARGERS})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"\nPlot saved to {OUTPUT_PLOT}")
    plt.close()

if __name__ == "__main__":
    plot_convergence()
