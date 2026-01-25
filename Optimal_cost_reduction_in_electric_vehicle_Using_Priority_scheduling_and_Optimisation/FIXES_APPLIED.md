# Pipeline Fixes Applied - Summary

## All Fixes Successfully Applied ✓

### 1. CLI Flag `--ga-schedule-scope` ✓
- Added argument with choices `['t0', 'full']`
- Default: `'t0'` for backward compatibility
- Controls whether GA schedules only t=0 admitted EVs or entire pool

### 2. Mutation `indpb` Fixed ✓
- Now computed from number of **variable** (non-fixed) genes
- Formula: `indpb = 1.0 / max(1, num_variable_genes)`
- Ensures mutation actually touches free genes

### 3. Improved Seeding ✓
- Added `greedy_seed_individual()`: deterministic greedy baseline
- First seed: greedy (deterministic)
- Remaining seeds: diverse Stage-I seeds with varied fractions (0.1-0.9)
- Reduces seed diversity collapse

### 4. Priority-Based Occupancy Repair ✓
- Repair now uses **λ (priority scores)** instead of absolute power
- Calls `calc_priority_scores()` per slot for active EVs
- Keeps top `num_chargers` EVs by λ value
- Updated signature: `repair_individual(..., pi_rev=None, system_params=None)`

### 5. Full Scope GA Support ✓
- When `--ga-schedule-scope full`: GA schedules entire EV pool
- GA chooses which EVs to serve each slot (natural waiting times)
- Backward compatible: default `t0` scope unchanged

### 6. Scheduling Order Printing ✓
- Prints "First 30 admitted EV IDs" (admitted at t=0)
- Prints "Scheduling order (by first admit slot)" for first 200 EVs
- Shows `EV{id}: first admitted at slot t={slot}` or `NOT ADMITTED`

### 7. Updated All Repair Calls ✓
- All `repair_individual()` calls now pass `pi_rev` and `system_params`
- Updated in: initial population evaluation, offspring repair

### 8. System Params Passed to GA ✓
- `run_ga()` signature updated to accept `system_params`
- Passed through to repair function for priority computation

---

## Usage Examples

### Quick Debug Test (t0 scope)
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --debug-short-run --ga-schedule-scope t0
```

### Full Pool GA (slower, more comprehensive)
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --T 48 --delta-t 0.5 --ngen 200 --pop-size 200 --ga-schedule-scope full
```

### Custom Parameters
```bash
python pipeline.py --ev-file evs.csv --chargers 30 --ngen 150 --pop-size 100 --ga-schedule-scope t0
```

---

## Expected Output Format

```
--- First admitted EVs (t=0) ---
Count at t=0: 18
First 30 admitted EV IDs: EV5, EV12, EV24, EV33, EV35, EV36, EV42, EV47, EV48, EV60, EV68, EV72, EV78, EV80, EV83, EV84, EV95, EV96

--- Scheduling order (by first admit slot) ---
EV5: first admitted at slot t=0
EV12: first admitted at slot t=0
...
EV2: first admitted at slot t=3
...
EV77: NOT ADMITTED
```

---

## Key Improvements

1. **Better Convergence**: Mutation now targets variable genes, seeds are more diverse
2. **Fairer Repair**: Priority-based repair aligns with Stage-I logic
3. **Flexible Scope**: Can schedule full pool or just t=0 admitted EVs
4. **Clear Reporting**: Scheduling order and admission info printed clearly

---

## Acceptance Criteria Checklist

- [x] CLI flag `--ga-schedule-scope` added
- [x] Mutation `indpb` computed from variable genes
- [x] Greedy seed + diverse Stage-I seeds implemented
- [x] Priority-based occupancy repair (uses λ)
- [x] Full scope GA support
- [x] Scheduling order and first 30 EVs printed
- [x] All repair calls updated with pi_rev/system_params
- [x] System params passed to run_ga

---

## Files Modified

- `pipeline.py`: All fixes applied
  - Added `greedy_seed_individual()` function
  - Updated `repair_individual()` signature and logic
  - Updated `run_ga()` signature and seeding
  - Updated `main()` for full scope support and printing
  - Fixed mutation `indpb` calculation

---

## Next Steps (Testing)

Run the test commands to verify:
1. Convergence: `ga_history.csv` shows `std_J > 0` for multiple generations
2. Best J improvement: decreases in first 20 gens (debug mode)
3. Scheduling order prints correctly
4. Full scope works when `--ga-schedule-scope full`

