import argparse
import csv
import random
import numpy as np
import math

# Distributions from EXPERIMENT_SETUP.md
# Ecap: Bimodal (70% compact: 35-55 kWh, 30% SUV: 70-100 kWh)
# SoC_init: Beta(2, 5) scaled to [0.05, 0.70)
# SoC_max: SoC_init + uniform(0.20, 0.45), capped at 0.95
# T_stay: Triangular(1, 12, mode=4) hours
# Arrival: Bimodal Gaussian (mu=8h, sigma=2h) and (mu=18h, sigma=2h)
# Charger Rating: Correlated with Ecap (larger packs -> higher power)

def generate_evs(n, seed=42, T=48, delta_t=0.25):
    random.seed(seed)
    np.random.seed(seed)
    
    evs = []
    for i in range(n):
        # Ecap: Bimodal
        if random.random() < 0.7:
            # Compact
            Ecap = random.uniform(35, 55)
            # Charger rating for compact
            R_i = random.choice([3.6, 7.0])
        else:
            # SUV
            Ecap = random.uniform(70, 100)
            # Charger rating for SUV
            R_i = random.choice([11.0, 22.0])
            
        # SoC_init: Beta(2, 5) scaled
        soc_init_raw = np.random.beta(2, 5)
        SoC_init = 0.05 + soc_init_raw * (0.70 - 0.05)
        
        # SoC_max
        soc_add = random.uniform(0.20, 0.45)
        SoC_max = min(0.95, SoC_init + soc_add)
        
        # SoC_min
        SoC_min = 0.05
        
        # T_stay: Triangular
        T_stay = random.triangular(1, 12, 4)
        
        # Arrival: Bimodal Gaussian (hours)
        if random.random() < 0.5:
            # Morning peak
            arr_h = np.random.normal(8, 2)
        else:
            # Evening peak
            arr_h = np.random.normal(18, 2)
            
        # Clamp arrival to [0, 24)
        arr_h = max(0.0, min(23.99, arr_h))
        
        # Convert to slots
        T_arr_idx = int(arr_h / delta_t)
        stay_slots = int(math.ceil(T_stay / delta_t))
        T_dep_idx = T_arr_idx + stay_slots
        
        # Feasibility check: ensure enough time to charge
        # (SoC_max - SoC_init) * Ecap <= 0.95 * R_i * T_stay
        req_energy = (SoC_max - SoC_init) * Ecap
        max_possible = 0.95 * R_i * T_stay
        if req_energy > max_possible:
            # Adjust SoC_max to be feasible
            feasible_add = (max_possible / Ecap)
            SoC_max = SoC_init + feasible_add
        
        ev = {
            'id': i + 1,
            'Ecap': round(Ecap, 2),
            'SoC_init': round(SoC_init, 3),
            'SoC_max': round(SoC_max, 3),
            'SoC_min': SoC_min,
            'R_i': R_i,
            'P_ref': R_i,
            'P_dis_min': -R_i, # Bidirectional
            'T_stay': round(T_stay, 2),
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': round(np.random.lognormal(mean=np.log(0.02), sigma=0.5), 4) # Log-normal cdeg
        }
        evs.append(ev)
        
    return evs

def save_evs(evs, filename):
    keys = ['id', 'Ecap', 'SoC_init', 'SoC_max', 'SoC_min', 'R_i', 'P_ref', 'P_dis_min', 'T_stay', 'T_arr_idx', 'T_dep_idx', 'cdeg']
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(evs)
    print(f"Saved {len(evs)} EVs to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--counts', type=int, nargs='+', default=[50, 100, 150, 200, 250, 300], help='List of EV counts to generate')
    args = parser.parse_args()
    
    for n in args.counts:
        evs = generate_evs(n, seed=42+n) # Different seed for each size to ensure variety
        filename = f"evs_{n}.csv"
        save_evs(evs, filename)
