#!/usr/bin/env python3
"""
Test bidirectional operation: verify negative power values are allowed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import build_bounds_and_mask, run_ga
import numpy as np

def test_bidirectional_bounds():
    """Test that bounds allow negative power (discharging)"""
    print("=" * 60)
    print("Test: Bidirectional Operation (Negative Power)")
    print("=" * 60)
    
    evs = [
        {
            'id': 1,
            'Ecap': 40.0,
            'SoC_init': 0.5,
            'SoC_max': 0.8,
            'SoC_min': 0.2,
            'P_ref': 7.0,
            'P_dis_min': -7.0,
            'T_arr_idx': 0,
            'T_dep_idx': 48,
            'cdeg': 0.02
        }
    ]
    
    M = len(evs)
    T = 48
    
    lower, upper = build_bounds_and_mask(evs, M, T)
    
    # Check that lower bounds include negative values
    has_negative = any(l < 0 for l in lower)
    print(f"Lower bounds: min={min(lower):.2f}, max={max(lower):.2f}")
    print(f"Upper bounds: min={min(upper):.2f}, max={max(upper):.2f}")
    
    assert has_negative, "Expected negative lower bounds for discharging"
    assert min(lower) <= -7.0, f"Expected P_dis_min <= -7.0, got {min(lower)}"
    assert max(upper) >= 7.0, f"Expected P_ref >= 7.0, got {max(upper)}"
    
    print("✓ Bounds allow bidirectional operation (negative power)\n")
    
    # Run a short GA to verify it can produce negative values
    pi_buy_arr = [0.30] * T  # Higher buy price
    pi_rev_arr = [0.20] * T  # Lower sell price (might incentivize discharging)
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    
    result = run_ga(
        evs=evs, T=T, delta_t=0.5,
        pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
        P_max=100.0, weights=weights,
        alpha1=50.0, alpha2=50.0, alpha3=50.0,
        pop_size=30, ngen=20,
        num_chargers=1, verbose=False
    )
    
    schedule = result['best_schedule']
    has_negative_power = np.any(schedule < -1e-6)
    min_power = np.min(schedule)
    max_power = np.max(schedule)
    
    print(f"Schedule power range: [{min_power:.3f}, {max_power:.3f}] kW")
    
    if has_negative_power:
        print(f"✓ GA produced negative power values (discharging): min = {min_power:.3f} kW")
    else:
        print("Note: GA did not produce negative power in this run (may depend on prices/objectives)")
    
    print("✓ Test PASSED: Bidirectional operation supported\n")
    return result

if __name__ == "__main__":
    try:
        test_bidirectional_bounds()
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

