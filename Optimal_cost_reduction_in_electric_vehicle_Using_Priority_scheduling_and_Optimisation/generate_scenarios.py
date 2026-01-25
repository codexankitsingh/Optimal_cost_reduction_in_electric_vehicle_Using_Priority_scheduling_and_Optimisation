#!/usr/bin/env python3
"""
generate_scenarios.py

Generates EV datasets for sensitivity experiments:
1. EV Composition (Compact vs SUV)
2. Initial SoC distributions (Low/Moderate/High)
3. Charger Mix (simulated via P_ref distribution)
4. Price Trend (use standard dataset - price handled at runtime)

Usage:
    python generate_scenarios.py
"""

import csv
import random
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "sensitivity_datasets"
N_EVS = 100  # Number of EVs per scenario
T_SLOTS = 48  # Time slots (half-hour for 24h)

# EV Type Definitions
EV_TYPES = {
    "compact": {"Ecap_range": (35, 55), "P_ref_range": (3.6, 7.0)},
    "suv": {"Ecap_range": (70, 100), "P_ref_range": (7.0, 11.0)},
}

# Scenario Definitions
COMPOSITION_SCENARIOS = {
    "comp_scen1": {"compact": 0.70, "suv": 0.30},
    "comp_scen2": {"compact": 0.50, "suv": 0.50},
    "comp_scen3": {"compact": 0.30, "suv": 0.70},
}

SOC_SCENARIOS = {
    "soc_low": (0.05, 0.30),
    "soc_mod": (0.30, 0.60),
    "soc_high": (0.60, 0.80),
}

# Charger Mix Scenarios (simulated via P_ref distribution)
CHARGER_MIX_SCENARIOS = {
    "charger_slow": {"P_ref_range": (3.0, 5.0)},     # 60% Slow chargers -> low power
    "charger_medium": {"P_ref_range": (5.0, 9.0)},   # 30% Slow -> medium mix
    "charger_fast": {"P_ref_range": (9.0, 15.0)},    # 20% Slow -> fast chargers
}


def generate_ev(ev_id, ev_type, soc_range, p_ref_override=None):
    """Generate a single EV record."""
    type_def = EV_TYPES[ev_type]
    Ecap = random.uniform(*type_def["Ecap_range"])
    if p_ref_override:
        P_ref = random.uniform(*p_ref_override)
    else:
        P_ref = random.uniform(*type_def["P_ref_range"])
    SoC_init = random.uniform(*soc_range)
    SoC_max = min(1.0, SoC_init + random.uniform(0.3, 0.5))  # Target SoC
    T_arr_idx = random.randint(0, 30)  # Arrival within first 15 hours
    T_stay = random.uniform(2.0, 8.0)  # Stay duration in hours

    return {
        "id": ev_id,
        "Ecap": round(Ecap, 2),
        "SoC_init": round(SoC_init, 3),
        "SoC_max": round(SoC_max, 3),
        "SoC_min": 0.0,
        "P_ref": round(P_ref, 2),
        "P_dis_min": round(-P_ref, 2),
        "T_stay": round(T_stay, 2),
        "T_arr_idx": T_arr_idx,
        "T_dep_idx": "",  # Let pipeline calculate from T_stay
        "cdeg": 0.02,
    }


def generate_composition_dataset(scenario_name, composition, soc_range=(0.20, 0.60)):
    """Generate dataset with specific EV type composition."""
    evs = []
    ev_id = 1
    for ev_type, fraction in composition.items():
        count = int(N_EVS * fraction)
        for _ in range(count):
            evs.append(generate_ev(ev_id, ev_type, soc_range))
            ev_id += 1

    # Fill remaining EVs if rounding caused shortfall
    while len(evs) < N_EVS:
        evs.append(generate_ev(ev_id, "compact", soc_range))
        ev_id += 1

    random.shuffle(evs)  # Randomize arrival order
    return evs


def generate_soc_dataset(scenario_name, soc_range, composition=None):
    """Generate dataset with specific SoC distribution."""
    if composition is None:
        composition = {"compact": 0.50, "suv": 0.50}
    return generate_composition_dataset(scenario_name, composition, soc_range)


def generate_charger_mix_dataset(scenario_name, p_ref_range):
    """Generate dataset simulating a specific charger power distribution."""
    evs = []
    soc_range = (0.20, 0.60)
    for ev_id in range(1, N_EVS + 1):
        ev_type = "compact" if random.random() < 0.5 else "suv"
        evs.append(generate_ev(ev_id, ev_type, soc_range, p_ref_override=p_ref_range))
    random.shuffle(evs)
    return evs


def save_dataset(evs, filename):
    """Save EV list to CSV."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / filename
    fieldnames = list(evs[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evs)
    print(f"Generated: {filepath} ({len(evs)} EVs)")


def main():
    random.seed(42)  # Reproducibility

    print("=" * 60)
    print("GENERATING SENSITIVITY EXPERIMENT DATASETS")
    print("=" * 60)

    # 1. Composition Scenarios
    print("\n--- EV Composition Scenarios ---")
    for name, comp in COMPOSITION_SCENARIOS.items():
        evs = generate_composition_dataset(name, comp)
        save_dataset(evs, f"evs_{name}.csv")

    # 2. SoC Scenarios
    print("\n--- Initial SoC Scenarios ---")
    for name, soc_range in SOC_SCENARIOS.items():
        evs = generate_soc_dataset(name, soc_range)
        save_dataset(evs, f"evs_{name}.csv")

    # 3. Charger Mix Scenarios
    print("\n--- Charger Mix Scenarios ---")
    for name, config in CHARGER_MIX_SCENARIOS.items():
        evs = generate_charger_mix_dataset(name, config["P_ref_range"])
        save_dataset(evs, f"evs_{name}.csv")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

