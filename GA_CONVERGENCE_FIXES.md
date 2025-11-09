# GA Convergence Fixes - Complete Summary

## Critical Bug Fixed: Stale Fitness Values ✓

### Problem Diagnosed
The GA appeared to "converge" instantly because **offspring fitness was never invalidated** after crossover/mutation. Offspring kept parent fitness values, so DEAP never re-evaluated them, causing:
- Zero std_J after a few generations
- Population appearing converged despite different genomes
- No real search progress

### Fix A: Invalidate Offspring Fitness ✓
**Location**: `pipeline.py` lines 867-873

```python
# FIX A: Invalidate fitness for all offspring so they get re-evaluated
for ind in offspring:
    try:
        del ind.fitness.values
    except Exception:
        ind.fitness.valid = False
```

**Effect**: All offspring are now properly re-evaluated after genetic operators.

---

## Additional Fixes Applied

### Fix B: Increased Mutation Probability ✓
**Location**: `pipeline.py` lines 781-784

**Before**: `indpb = 1.0 / max(1, num_variable_genes)` → ~0.0007 for large genomes (negligible mutation)

**After**: `indpb = min(0.05, 1.0 / max(1, T))` → ~0.0208 for T=48 (meaningful per-gene mutation)

**Also**: Increased `mutpb` from 0.3 to 0.4 for better exploration

### Fix C: Reduced and Diversified Seeding ✓
**Location**: `pipeline.py` lines 797-814

- Reduced seed count to 5% of population (was 20%)
- Added 1% noise perturbation to seeds to increase diversity
- Limits deterministic greedy seed dominance

### Fix D: Added Debug Logging ✓
**Location**: `pipeline.py` lines 887-889

- Logs `invalid_inds_count` each generation
- Verifies that offspring are actually being re-evaluated
- Helps diagnose if stale fitness issue persists

### Fix E: Match Paper Time Discretization ✓
**Location**: `pipeline.py` line 957

**Before**: `default=0.5` (30-minute slots)

**After**: `default=0.25` (15-minute slots, matches paper)

**Impact**: Energy calculations (F1, F2) now match paper's 15-minute slot discretization

### Fix F: Optimized Elitism ✓
**Location**: `pipeline.py` line 1136

**Before**: `elitism_k = max(2, int(0.02 * args.pop_size))`

**After**: `elitism_k = max(1, min(5, int(0.02 * args.pop_size)))`

**Effect**: Keeps elitism to small absolute number (1-5) regardless of pop_size

---

## New Metrics Added

### Population Diversity Tracking ✓
**Location**: `pipeline.py` lines 902-921

- Computes average pairwise Euclidean distance in population
- Tracked in `ga_history.csv` as `diversity` column
- Helps monitor population convergence

---

## Expected Behavior After Fixes

### Before Fixes:
- `std_J` → 0 after ~10 generations
- `mean_J` = `best_J` (all identical)
- No improvement after initial seed

### After Fixes:
- `std_J` > 0 for many generations (exploration)
- `mean_J` decreases steadily
- `best_J` improves over generations (with plateaus)
- `diversity` decreases gradually as population converges
- `invalid_inds_count` = `pop_size` every generation (confirming re-evaluation)

---

## Recommended Test Command

```bash
python pipeline.py \
  --ev-file evs.csv \
  --chargers 30 \
  --T 48 \
  --delta-t 0.25 \
  --ngen 200 \
  --pop-size 120 \
  --ga-schedule-scope t0
```

For more exploration:
```bash
python pipeline.py \
  --ev-file evs.csv \
  --chargers 30 \
  --T 48 \
  --delta-t 0.25 \
  --ngen 300 \
  --pop-size 150 \
  --ga-schedule-scope t0
```

---

## Verification Checklist

After running, check:

- [ ] `DEBUG Gen X: invalid_inds_count=120` (should equal pop_size every gen)
- [ ] `std_J > 0` for at least 20-50 generations
- [ ] `best_J` improves at least once in first 20 generations
- [ ] `diversity` starts high and decreases gradually
- [ ] `mean_J` decreases over time
- [ ] `ga_history.csv` shows `diversity` column populated

---

## Files Modified

- `pipeline.py`: All fixes applied
  - Lines 781-784: Mutation probability fix
  - Lines 797-814: Seeding improvements
  - Lines 832-837: Debug logging
  - Lines 854-873: Fitness invalidation (CRITICAL)
  - Lines 887-889: Invalid inds logging
  - Lines 902-921: Diversity computation
  - Line 957: Default delta_t changed to 0.25
  - Line 1131: mutpb increased to 0.4
  - Line 1136: Elitism optimization

---

## Impact

These fixes address the core convergence issue. The GA should now:
1. Properly explore the search space
2. Show genuine improvement over generations
3. Maintain population diversity during exploration
4. Match paper's time discretization for fair comparison

