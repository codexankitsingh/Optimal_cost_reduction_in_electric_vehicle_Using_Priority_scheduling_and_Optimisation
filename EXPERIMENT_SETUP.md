# Experiment and Simulation Setup Documentation

## 1. System Overview

This project implements a **two-stage EV charging/discharging scheduling system** that optimizes energy management for electric vehicle fleets using priority-based admission and genetic algorithm (GA) optimization. The system supports bidirectional charging (V2G - Vehicle-to-Grid) and compares two admission strategies: **Priority Scheduling** and **First-Come-First-Served (FCFS)**.

---

## 2. System Architecture

### 2.1 Two-Stage Pipeline

#### **Stage-I: Admission Control**
- **Priority Scheduling (`pipeline.py`)**: 
  - Computes priority scores (λ_i) for each EV based on urgency, energy requirements, and arrival time
  - Performs time-series simulation (t=0 to T-1) to admit EVs dynamically
  - Tracks waiting times and admission timeline
  - Releases chargers when EVs depart
  
- **FCFS Scheduling (`pipeline_fcfs.py`)**:
  - Admits first M EVs based on earliest arrival time (T_arr_idx)
  - Simple one-shot admission at t=0
  - No dynamic re-admission

#### **Stage-II: Genetic Algorithm Optimization**
- Optimizes power schedules for admitted EVs
- Minimizes multi-objective cost function J
- Enforces constraints (SoC limits, occupancy, grid capacity)
- Supports bidirectional power flow (charging/discharging)

---

## 3. System Parameters

### 3.1 Time Discretization
- **Time Horizon (T)**: 48 slots (default)
- **Slot Duration (δt)**: 0.25 hours (15 minutes)
- **Total Horizon**: 12 hours (48 × 0.25h)
- **Time Unit**: Discrete time slots indexed t ∈ {0, 1, ..., T-1}

### 3.2 Infrastructure Parameters
- **Number of Chargers (M)**: 30 (default, configurable)
- **Maximum Grid Power (P_max)**: 100.0 kW
- **Average Grid Power (P_avg)**: 40.0 kW

### 3.3 Pricing Parameters
- **Buying Price (π_buy)**: 0.25 $/kWh (constant, can be time-varying)
- **Revenue Price (π_rev)**: 0.18 $/kWh (for V2G discharge)
- **Price Range**:
  - π_buy: [0.10, 0.50] $/kWh
  - π_rev: [0.05, 0.30] $/kWh

### 3.4 Battery Parameters
- **Degradation Cost (cdeg)**: 0.02 $/kWh (per-EV, heterogeneous)
- **Default Charger Ratings (R_i)**: [3.6, 7.0, 11.0, 22.0] kW
- **Charger Distribution**: [30%, 40%, 22%, 8%] (compact to fast charging)

---

## 4. EV Data Model

### 4.1 EV Attributes (from CSV/JSON)

Each EV is characterized by:

| Attribute | Description | Default/Range |
|-----------|------------|---------------|
| `id` | Unique EV identifier | Integer |
| `Ecap` | Battery capacity (kWh) | 35-100 kWh (bimodal) |
| `SoC_init` | Initial State of Charge | 0.05-0.70 (beta distribution) |
| `SoC_max` | Maximum SoC target | SoC_init + [0.20, 0.45], capped at 0.95 |
| `SoC_min` | Minimum SoC limit | 0.05-0.20 |
| `R_i` or `P_ref` | Charger rating (kW) | 3.6, 7.0, 11.0, or 22.0 |
| `P_dis_min` | Minimum discharge power (kW) | -R_i (default, negative) |
| `T_stay` | Stay duration (hours) | 1-12 hours (triangular, mode=4) |
| `T_arr_idx` | Arrival time slot | 0-47 (bimodal: 8:00, 18:00) |
| `T_dep_idx` | Departure time slot | T_arr_idx + ceil(T_stay/δt) |
| `cdeg` | Battery degradation cost ($/kWh) | 0.005-0.05 (log-normal, μ=0.02) |

### 4.2 EV Generation (`generate_ev.py`)

**Distributions:**
- **Ecap**: Bimodal (70% compact: 35-55 kWh, 30% SUV: 70-100 kWh)
- **SoC_init**: Beta(2, 5) scaled to [0.05, 0.70)
- **SoC_max**: SoC_init + uniform(0.20, 0.45), capped at 0.95
- **T_stay**: Triangular(1, 12, mode=4) hours
- **Arrival**: Bimodal Gaussian (μ=8h, σ=2h) and (μ=18h, σ=2h)
- **Charger Rating**: Correlated with Ecap (larger packs → higher power)
- **Feasibility**: Ensures (SoC_max - SoC_init) × Ecap ≤ 0.95 × R_i × T_stay

**Example Command:**
```bash
python generate_ev.py --n 100 --out evs_100.csv --seed 42
```

---

## 5. Objective Function

### 5.1 Multi-Objective Cost Function (J)

The GA minimizes the normalized weighted objective:

```
J = w₁·F₁_norm + w₂·F₂_norm + w₃·F₃_norm - w₄·F₄_norm + Ω_norm
```

Where:

#### **F₁: Net Energy Cost**
```
F₁ = Σ_t Σ_i [π_buy(t)·p_i(t)·δt  if p_i(t) > 0]
     + Σ_t Σ_i [π_rev(t)·p_i(t)·δt if p_i(t) < 0]
```
- Cost of buying energy (charging) + revenue from selling (discharging)
- Normalized by: `(max(π_buy) - min(π_rev)) × max_energy_over_horizon`

#### **F₂: Battery Degradation Cost**
```
F₂ = Σ_i Σ_t cdeg_i · |p_i(t)| · δt
```
- Linear degradation model based on absolute power throughput
- Normalized by: `max(cdeg) × max_energy_over_horizon`

#### **F₃: Grid Load Variance**
```
F₃ = Σ_t (L_t - L̄)²
where L_t = Σ_i p_i(t),  L̄ = (1/T) Σ_t L_t
```
- Penalizes load fluctuations (grid stability)
- Normalized by: `max_power_sum² × T`

#### **F₄: User Satisfaction (maximized, so negated)**
```
F₄ = Σ_i S_i
where S_i = 1 - φ_i · δ_i

φ_i = min(1.0, p_req_i / p_ref_i)  (urgency factor)
δ_i = (SoC_max - SoC_T) / (SoC_max - SoC_init)  (unmet need)
```
- S_i ∈ [0, 1]: 1 = fully satisfied, 0 = not satisfied
- Normalized by: `M` (number of EVs)

#### **Ω: Constraint Violation Penalties**
```
Ω = α₁·V_SoC + α₂·V_occ + α₃·V_grid

V_SoC = Σ violations in kWh (SoC < SoC_min or SoC > SoC_max)
V_occ = Σ excess EVs per slot (active > M)
V_grid = Σ excess power per slot in kWh (L_t > P_max)
```
- Penalty weights: α₁ = α₂ = α₃ = 50.0
- Normalized by: `max_energy_over_horizon`

### 5.2 Objective Weights
- **w₁ = w₂ = w₃ = w₄ = 0.25** (equal weighting)

---

## 6. Constraints

### 6.1 State of Charge (SoC) Constraints
```
SoC_min ≤ SoC_i(t) ≤ SoC_max  ∀i, ∀t ∈ [T_arr, T_dep)
```
- Enforced via penalty in Ω
- SoC evolution: `SoC_i(t+1) = SoC_i(t) + (p_i(t) · δt) / Ecap_i`

### 6.2 Occupancy Constraints
```
|{i : |p_i(t)| > ε}| ≤ M  ∀t
```
- Maximum M EVs can be active (charging/discharging) simultaneously
- Enforced via repair operator and penalty

### 6.3 Grid Power Constraints
```
|L_t| = |Σ_i p_i(t)| ≤ P_max  ∀t
```
- Total grid load cannot exceed P_max
- Enforced via penalty

### 6.4 Power Bounds
```
P_dis_min ≤ p_i(t) ≤ P_ref  ∀i, ∀t ∈ [T_arr, T_dep)
p_i(t) = 0  ∀i, ∀t ∉ [T_arr, T_dep)
```
- Power limits per EV (bidirectional)
- Zero power outside arrival/departure window

---

## 7. Genetic Algorithm Configuration

### 7.1 GA Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `pop_size` | 120 (default) | Population size |
| `ngen` | 300 (default) | Number of generations |
| `cxpb` | 0.9 | Crossover probability |
| `mutpb` | 0.4 | Mutation probability |
| `eta_c` | 20.0 | SBX crossover distribution index |
| `eta_m` | 20.0 | Polynomial mutation distribution index |
| `tournament_size` | 3 | Tournament selection size |
| `stagnation_generations` | max(100, 0.4×ngen) | Early stopping threshold |
| `elitism_k` | max(1, min(5, 0.02×pop_size)) | Number of elites preserved |
| `seed_count` | max(1, 0.05×pop_size) | Number of seeded individuals |

### 7.2 GA Operators

#### **Selection**
- Tournament selection (size=3)

#### **Crossover**
- Simulated Binary Bounded (SBX) crossover
- Bounded by per-gene lower/upper bounds
- Preserves fixed genes (outside EV windows)

#### **Mutation**
- Polynomial Bounded mutation
- Per-gene mutation probability: `indpb = min(0.05, 1/T)`
- For T=48: indpb ≈ 0.0208

#### **Repair Operator**
1. **Bounds Clamping**: Clamp genes to [lower, upper]
2. **Energy Repair**: Scale/redistribute power to meet ΔE_i targets
3. **Occupancy Enforcement**: Remove excess EVs per slot using priority (priority) or FCFS (earliest arrival)
4. **SoC Constraint**: Handled via penalty (not hard constraint)

#### **Seeding Strategy**
- **Greedy Seed**: Fill cheapest slots first until ΔE satisfied
- **FCFS Seeds**: Allocate power in earliest available slots (with fraction variation)
- **Noise Perturbation**: Add 1% random noise to seeds for diversity

### 7.3 Fitness Invalidation
- All offspring fitness values invalidated after crossover/mutation
- Forces re-evaluation to prevent stale fitness

### 7.4 Convergence Criteria
- **Stagnation**: Stop if best fitness doesn't improve for `stagnation_generations`
- **Max Generations**: Stop after `ngen` generations
- **Diversity Tracking**: Monitors population diversity (pairwise distance)

---

## 8. Priority Score Calculation (Stage-I)

### 8.1 Priority Formula (λ_i)

```
λ_i = α·φ_i + β·(ΔE_i / Ecap_i) + γ·(1 / (T_arr_i + 1))
```

Where:
- **φ_i = min(1.0, p_req_i / p_ref_i)**: Urgency factor
- **ΔE_i = (SoC_max - SoC_init) × Ecap_i**: Energy requirement
- **T_arr_i**: Arrival time slot
- **α, β, γ**: Weighting factors (default: equal weights)

### 8.2 Time-Series Admission

1. **Initialize**: All EVs in waiting queue
2. **For each slot t = 0 to T-1**:
   - Release chargers from departing EVs (T_dep = t)
   - Compute priority scores for waiting EVs
   - Admit top M EVs (or available chargers)
   - Track `first_admit_slot` for each EV
3. **Compute waiting times**: `wait_i = (first_admit_slot - T_arr_i) × δt`

---

## 9. Experimental Setup

### 9.1 Comparison Methodology

**Two Pipelines:**
1. **Priority Pipeline (`pipeline.py`)**: Priority-based admission + GA
2. **FCFS Pipeline (`pipeline_fcfs.py`)**: First-come-first-served admission + GA

**Metrics Compared:**
- Average waiting time (fleet-wide)
- Average user satisfaction (fleet-wide)
- Average net energy cost per EV
- Gini fairness coefficient (on delivered kWh)

### 9.2 Experimental Scenarios

**EV Pool Sizes**: 50, 100, 150, 200, 250, 300 EVs
**Chargers**: 30 (fixed)
**GA Scope**: 
- `t0`: Schedule only EVs admitted at t=0 (default)
- `full`: Schedule entire EV pool (GA chooses who to serve)

### 9.3 Execution Commands

#### **Priority Pipeline:**
```bash
python pipeline.py \
  --ev-file evs_100.csv \
  --chargers 30 \
  --T 48 \
  --delta-t 0.25 \
  --ngen 300 \
  --pop-size 120 \
  --ga-schedule-scope full
```

#### **FCFS Pipeline:**
```bash
python pipeline_fcfs.py \
  --ev-file evs_100.csv \
  --chargers 30 \
  --T 48 \
  --delta-t 0.25 \
  --ngen 300 \
  --pop-size 120
```

#### **Comparison Plots:**
```bash
# Waiting time comparison
python analysis/plot_waiting_time_comparison.py \
  --ngen 150 --pop-size 80 --ga-scope full

# Metrics comparison (satisfaction, cost, fairness)
python analysis/plot_metrics_comparison.py \
  --ngen 150 --pop-size 80 --ga-scope full

# GA convergence plot
python analysis/plot_ga_convergence.py
```

---

## 10. Output Files

### 10.1 Schedule Files
- `best_schedule_normalized.csv`: Priority pipeline schedule (M×T matrix)
- `best_schedule_fcfs.csv`: FCFS pipeline schedule (M×T matrix)

### 10.2 GA History Files
- `ga_history.csv`: Priority pipeline GA convergence (generations × metrics)
- `ga_history_fcfs.csv`: FCFS pipeline GA convergence
- `run_*evs_*ch_ga_history.csv`: Per-experiment GA histories

### 10.3 Summary Files
- `run_summary_*.json`: Per-run metrics summary (JSON format)

### 10.4 Plot Files
- `analysis/waiting_time_comparison.png`: Waiting time comparison
- `analysis/user_satisfaction_comparison.png`: Satisfaction comparison
- `analysis/net_energy_cost_comparison.png`: Cost comparison
- `analysis/gini_fairness_comparison.png`: Fairness comparison
- `analysis/metrics_comparison_combined.png`: Combined metrics
- `analysis/convergence_multiple_sizes.png`: GA convergence plot

---

## 11. Key Implementation Details

### 11.1 Bidirectional Charging (V2G)
- Negative power (p_i(t) < 0) represents discharging
- Revenue from discharging: `π_rev × |p_i(t)| × δt`
- Default `P_dis_min = -P_ref` (symmetric bidirectional)

### 11.2 Normalization Strategy
- All objective components normalized to [0, 1] range
- Prevents dominance of large-magnitude terms
- Uses realistic upper bounds (not theoretical maxima)

### 11.3 Waiting Time Calculation
- **Priority**: From Stage-I time-series simulation
- **FCFS**: 
  - Admitted EVs: 0 (admitted at t=0)
  - Non-admitted EVs: Full stay duration (never served)

### 11.4 User Satisfaction Calculation
- **Admitted EVs**: Computed from actual schedule (SoC_T)
- **Non-admitted EVs**: SoC_T = SoC_init (no charging received)
- **Fleet-wide**: Average over all EVs in pool

### 11.5 Gini Fairness
- Measures inequality in delivered energy (kWh)
- Formula: `G = (2×Σ(i×x_i)) / (n×Σx_i) - (n+1)/n`
- Range: [0, 1] (0 = perfect equality, 1 = maximum inequality)

---

## 12. Validation and Constraints

### 12.1 Constraint Checking
- SoC bounds checked per slot per EV
- Occupancy checked per slot (active EVs ≤ M)
- Grid power checked per slot (|L_t| ≤ P_max)
- Violations penalized in objective function

### 12.2 Feasibility
- EV generation ensures: `ΔE ≤ 0.95 × R_i × T_stay`
- Repair operator enforces hard constraints (occupancy, bounds)
- SoC constraints enforced via penalties (soft)

---

## 13. Performance Considerations

### 13.1 Computational Complexity
- **GA**: O(pop_size × ngen × M × T × eval_cost)
- **Stage-I**: O(T × N × log(N)) (priority sorting)
- **Evaluation**: O(M × T) per individual

### 13.2 Scalability
- Tested up to 300 EVs, 30 chargers, 48 time slots
- Population size: 60-120
- Generations: 80-500

### 13.3 Convergence
- Stagnation detection: std(J) → 0 or no improvement
- Early stopping: `stagnation_generations = max(100, 0.4×ngen)`
- Diversity tracking for monitoring

---

## 14. Reproducibility

### 14.1 Random Seeds
- **GA**: `random.seed(42)` (fixed)
- **EV Generation**: `--seed 42` (configurable)
- Ensures reproducible results

### 14.2 Data Files
- EV datasets: `evs_50.csv`, `evs_100.csv`, ..., `evs_300.csv`
- Generated with fixed seed for consistency

---

## 15. Dependencies

### 15.1 Python Packages
- `numpy`: Numerical operations
- `pandas`: Data handling
- `deap`: Genetic algorithm framework
- `matplotlib`: Plotting
- `scipy`: (optional) Advanced statistics

### 15.2 Python Version
- Python 3.7+ recommended

---

## 16. Future Extensions

### 16.1 Potential Enhancements
- Time-varying electricity prices
- Multiple charging stations
- EV routing integration
- Real-time re-optimization
- Machine learning for priority prediction
- Multi-objective Pareto optimization

---

## References

- DEAP Framework: https://github.com/DEAP/deap
- EV Charging Optimization Literature
- V2G (Vehicle-to-Grid) Research

---

**Last Updated**: Based on current codebase state
**Version**: 1.0

