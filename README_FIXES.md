# Pipeline Fixes - Quick Start Guide

## What Was Fixed

1. **Stage-I Time-Series Simulation**: Now simulates per-slot admission, producing accurate waiting times
2. **Bidirectional Operation**: Supports discharging (negative power) for V2G
3. **Constraint Enforcement**: Improved charger assignment and SoC bounds checking
4. **GA Convergence**: Fixed normalization, added logging, improved repair logic
5. **Testing & Validation**: Added test scripts and plotting tools

## Quick Test

```bash
# Generate test EVs (if needed)
python generate_ev.py --n 40 --out evs_test.csv

# Run pipeline in debug mode
python pipeline.py --ev-file evs_test.csv --chargers 30 --debug-short-run

# Run tests
python tests/test_small_scenario.py

# Check constraints on generated schedule
python tests/assert_constraints.py --schedule best_schedule_normalized.csv

# Generate plots
python scripts/plot_results.py
```

## Expected Results

After running the pipeline:
- `best_schedule_normalized.csv`: Power schedule (EVs × time slots)
- `ga_history.csv`: GA convergence statistics

### Key Metrics to Check:
- **Average waiting time (all EVs)**: Should be > 0 when N > M
- **Best J convergence**: Should decrease in first 20-50 generations
- **Occupancy violations**: Should be 0 (V_occ = 0)
- **SoC violations**: Should be < 1.0 kWh (V_SoC < 1.0)

## Full Example Command

```bash
python pipeline.py \
  --ev-file evs.csv \
  --chargers 30 \
  --T 48 \
  --delta-t 0.5 \
  --ngen 300 \
  --pop-size 120
```

For details on all fixes, see `CHANGELOG.md`.

