# Changelog: Pipeline Fixes for EV Charging-Discharging Scheduling

## Overview
This document summarizes the fixes applied to `pipeline.py` to address convergence issues, constraint violations, and implementation gaps relative to the paper requirements.

## Key Fixes

### A. Stage-I Time-Slot Simulation ✓
**Problem**: `run_stage1()` performed one-shot admission at t=0, causing zero waiting times when N > M.

**Solution**: Implemented `simulate_stage1_timeseries()` that:
- Simulates per-slot admission (t=0 to T-1)
- Recomputes priority scores λ_i(t) per slot for waiting EVs
- Tracks EV states (assigned/released) based on arrival/departure
- Releases chargers when EVs depart
- Computes accurate waiting times for all EVs

**Impact**: Waiting times now correctly reflect queue behavior when N > M.

**Files Modified**:
- `pipeline.py`: Added `simulate_stage1_timeseries()` function (lines 201-323)
- `pipeline.py`: Updated `main()` to use time-series simulation (lines 868-877)

---

### B. Bidirectional Operation ✓
**Problem**: Lower bounds were fixed at 0.0, preventing discharging (V2G).

**Solution**:
- Modified `build_bounds_and_mask()` to set `lower = -pmax` by default (or use `P_dis_min` if provided)
- Updated `load_ev_pool()` to parse `P_dis_min` field
- Default `P_dis_min = -P_ref` if not specified

**Impact**: GA can now optimize for both charging and discharging operations.

**Files Modified**:
- `pipeline.py`: `build_bounds_and_mask()` (lines 465-492)
- `pipeline.py`: `load_ev_pool()` (lines 51, 70)
- `pipeline.py`: `main()` (lines 894-897)

---

### C. Charger Assignment Constraints ✓
**Problem**: Repair function removed smallest contributors, not considering priority.

**Solution**: Updated `repair_individual()` to:
- Keep EVs with largest absolute power when occupancy limit is reached (priority-based)
- Strictly enforce occupancy ≤ M per slot

**Impact**: Better fairness and alignment with Stage-I priority logic.

**Files Modified**:
- `pipeline.py`: `repair_individual()` (lines 569-584)

---

### D. SoC Dynamics & Penalties ✓
**Problem**: Penalties used SoC units directly, not kWh; grid violations in kW not kWh.

**Solution**:
- Convert SoC violations to kWh: `violation_kwh = violation_soc * Ecap`
- Convert grid violations to kWh: `(Lt - P_max) * delta_t`
- Ensure `SoC_min` is parsed and used

**Impact**: Penalty units consistent with objective function normalization.

**Files Modified**:
- `pipeline.py`: `compute_penalty()` (lines 387-421)
- `pipeline.py`: `load_ev_pool()` (lines 48, 67)

---

### E. Normalization & Objective Scaling ✓
**Problem**: Denominator calculation for F1 used max price, not price difference.

**Solution**:
- F1 denominator: `(max_pi_buy - min_pi_rev) * max_energy` (reflects cost-revenue difference)
- Added epsilon (1e-9) consistently
- Improved Omega normalization

**Impact**: Better gradient signals, improved convergence.

**Files Modified**:
- `pipeline.py`: `make_fitness_function()` (lines 427-449)

---

### F. GA Logging & Convergence Tracking ✓
**Problem**: No per-generation statistics to diagnose convergence.

**Solution**:
- Added `ga_history` list tracking: gen, best_J, mean_J, std_J, V_SoC, V_occ, V_grid
- Saved to `ga_history.csv` after GA completes
- Improved console output (every 5 generations)

**Impact**: Easier diagnosis of convergence patterns.

**Files Modified**:
- `pipeline.py`: `run_ga()` (lines 710-783, 792-797)

---

### G. CLI Arguments & Hyperparameters ✓
**Problem**: Hyperparameters hardcoded.

**Solution**: Added CLI arguments:
- `--ngen`: Number of generations (default 300)
- `--pop-size`: Population size (default 120)
- `--debug-short-run`: Quick test mode (ngen=80, pop_size=60)

**Impact**: Easier experimentation and debugging.

**Files Modified**:
- `pipeline.py`: `main()` (lines 823-833, 921-934)

---

## New Files Created

### Test Scripts
1. **`tests/test_small_scenario.py`**: 
   - Tests Stage-I time-series (non-zero waiting times)
   - Tests occupancy constraints (≤ M)
   - Tests SoC bounds
   - Tests GA convergence

2. **`tests/test_bidirectional.py`**:
   - Verifies negative power bounds allowed
   - Tests GA can produce discharging schedules

3. **`tests/assert_constraints.py`**:
   - Post-processes schedule CSV to check C1-C6
   - Reports violations with details

### Plotting Script
4. **`scripts/plot_results.py`**:
   - Plots: best/mean J vs generation
   - Occupancy over time
   - Final SoC distribution
   - Waiting time histogram

---

## Usage

### Basic Run
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --T 48 --delta-t 0.5
```

### Debug Mode (Quick Test)
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --debug-short-run
```

### Custom Hyperparameters
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --ngen 200 --pop-size 100
```

### Run Tests
```bash
# Run all tests
python tests/test_small_scenario.py
python tests/test_bidirectional.py

# Check constraints on schedule
python tests/assert_constraints.py --schedule best_schedule_normalized.csv
```

### Generate Plots
```bash
python scripts/plot_results.py --history ga_history.csv --schedule best_schedule_normalized.csv
```

---

## Output Files

1. **`best_schedule_normalized.csv`**: Best schedule matrix (EVs × time slots)
2. **`ga_history.csv`**: Per-generation statistics (gen, best_J, mean_J, std_J, V_SoC, V_occ, V_grid)

---

## Expected Improvements

1. **Convergence**: Best J should decrease over first 20-50 generations, then plateau
2. **Waiting Times**: Non-zero when N > M (average waiting time > 0)
3. **Constraints**: Occupancy ≤ M, SoC within bounds (violations < 1.0 kWh typically)
4. **Bidirectional**: Negative power values appear when beneficial (price-driven)

---

## Notes

- Random seed set to 42 for reproducibility
- Penalty scaling (alpha1, alpha2, alpha3 = 50.0) may need tuning for specific scenarios
- Stage-II GA runs on EVs admitted at t=0; full per-epoch simulation can be added later if needed

