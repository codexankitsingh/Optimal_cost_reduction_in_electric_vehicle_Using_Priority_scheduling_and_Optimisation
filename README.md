
```md
# Optimal Cost Reduction in Electric Vehicle Charge-Discharge Scheduling (implementation)

This repository contains an implementation of the two-stage EV charge/discharge scheduling framework (priority scheduling + GA optimization) described in the paper “Optimal Cost Reduction in Electric Vehicle’s Charge-Discharge Scheduling while Maximizing User Satisfaction” by Ankit kumar singh, Sapna Kushwah, Avadh Kishor and Pramod Kumar Singh. 

## Overview

A workplace parking EV scheduling system that:

* **Stage-I:** assigns limited chargers to EVs using an urgency / degradation / grid stress / price priority score.
* **Stage-II:** optimizes per-EV charging/discharging power across time (encoded as real-valued chromosomes) using a Genetic Algorithm to minimize combined cost (energy cost, battery degradation, grid variance) and maximize user satisfaction.

This repo implements the above two-stage framework and experimental scripts to reproduce the results in the paper. See the cited paper for formal problem formulation, objective functions and evaluation details. 

## Key features

* Priority-based charger admission (Stage-I).
* GA-based power scheduling (Stage-II) with:
    * Real-valued encoding, SBX crossover, polynomial mutation.
    * Constraint handling via penalty functions (SoC violation, charger occupancy, grid capacity).
* Cost components implemented: energy cost (time-of-use), battery degradation cost, grid load variance, and user satisfaction metric.
* Scripts to run experiments, change weights for the multi-objective weighted sum, and visualize schedules / load profiles.

## Repository structure (suggested)

If your file names differ, update paths/commands below to match your repo.

```

.
├─ data/                      \# sample EV profiles, price profiles, grid settings
├─ src/
│  ├─ priority\_scheduler.py   \# Stage-I implementation
│  ├─ ga\_scheduler.py         \# Stage-II GA implementation
│  ├─ models.py               \# EV, Charger, and utility classes
│  └─ utils.py                \# common helpers (SoC update, cost calc, etc.)
├─ experiments/
│  ├─ run\_experiment.py       \# high-level script to run Stage I + II and save results
│  └─ configs/                \# JSON/YAML experiment configuration files
├─ results/                   \# outputs: schedules, logs, plots
├─ notebooks/                 \# analysis / visualization notebooks (optional)
└─ README.md                  \# \<-- you are editing this

````

## Requirements & install

A generic Python environment (if your implementation is in another language, adapt accordingly).

```bash
# create virtual env (python3)
python3 -m venv venv
source venv/bin/activate

# install (example)
pip install -r requirements.txt

# If you don't have requirements.txt, common packages:
pip install numpy scipy matplotlib pandas deap tqdm
````

## Configuration

Experiments are driven by a configuration file (JSON/YAML). Typical configurable fields:

  * `N`, `M`, `T`, `delta_t`
  * EV parameters per EV: `Ecap`, `SoC_init`, `SoC_max`, `Pref`, `T_arr`, `T_dep`, `cdeg`
  * Grid: `Pmax`, `base_load` profile
  * Price profiles: `pi_buy[t]`, `pi_rev[t]`
  * GA hyperparameters: `pop_size`, `generations`, `pc`, `pm`, `eta_c`, mutation params
  * Objective weights: `w1`,`w2`,`w3`,`w4`

Place these files in `experiments/configs/`.

## How to run (example)

**Stage-I: priority admission for each slot**

```bash
python src/priority_scheduler.py --config experiments/configs/config1.json --out results/stage1_assignment.json
```

**Stage-II: GA optimization for admitted vehicles (reads stage1 output)**

```bash
python src/ga_scheduler.py --config experiments/configs/config1.json \
    --assignment results/stage1_assignment.json \
    --out results/final_schedule.json
```

**One-line script to run full experiment (if provided)**

```bash
python experiments/run_experiment.py --config experiments/configs/config1.json
```

*(If your code is in C++ or mixed, replace `python` commands with compiled binaries or wrappers. Add sample compile/run commands in Makefile.)*

## Input data format

  * **EV file (CSV/JSON):** each EV row should include:
    `id`, `Ecap (kWh)`, `SoC_init (0–1)`, `SoC_max (0–1)`, `Pref (kW)`, `Tarr (slot)`, `Tdep (slot)`, `cdeg`
  * **Price file:** time series arrays `pi_buy[t]`, `pi_rev[t]` for `t=1..T`.
  * **Grid file:** `Pmax`, `base_load[t]` (optional).

Provide a `data/sample/` folder with small example files so users can run quick tests.

## Outputs

  * `results/final_schedule.json` — per-EV per-slot `p_i,t (kW)`, charger assignment `z_m,i,t`, SoC time series.
  * `results/metrics.json` — computed F1..F4, weighted objective J, penalties.
  * `results/plots/` — load profile, SoC trajectories, GA convergence plot.

## Reproducibility tips

  * Fix random seed when comparing GA runs.
  * Keep GA hyperparameters and objective weights logged with results.
  * Save population snapshots if you want to analyze intermediate solutions.

## Recommended experiments (to reproduce paper tables/plots)

  * Vary objective weights `w1..w4` and measure tradeoffs between cost and user satisfaction.
  * Compare scheduling with vs without V2G (disable discharging in config).
  * Sensitivity to number of chargers `M` and grid capacity `Pmax`.
  * Ablation: priority scheduling only vs priority + GA.

## Contributing

  * Update `experiments/configs/` with new scenarios.
  * Add unit tests for SoC propagation, cost functions and constraint handling.
  * Open issues for bugs / feature requests and send PRs with clear descriptions.

<!-- end list -->

```
```
