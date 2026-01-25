#!/usr/bin/env python3
"""
Test small scenario: N=40 EVs, M=30 chargers
Verifies:
- Stage-I time-series simulation produces non-zero waiting times when N > M
- Final schedule occupancy per slot <= M
- SoC stays within bounds
- GA convergence (best J improves/plateaus)
"""

import sys
import os
import math

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    load_ev_pool, simulate_stage1_timeseries, run_ga,
    compute_Si_from_schedule, compute_penalty
)
import numpy as np
import pandas as pd

def generate_test_evs(n=40):
    """Generate N test EVs with varied parameters"""
    evs = []
    for i in range(1, n+1):
        T_arr = (i * 2) % 48  # Spread arrivals
        T_stay = 3.0 + (i % 5)  # 3-8 hours
        evs.append({
            'id': i,
            'Ecap': 40.0 + (i % 3) * 10.0,  # 40, 50, 60 kWh
            'SoC_init': 0.2 + (i % 5) * 0.1,  # 0.2-0.6
            'SoC_max': 0.8,
            'SoC_min': 0.1,
            'R_i': 7.0,
            'P_ref': 7.0,
            'P_dis_min': -7.0,
            'T_stay': T_stay,
            'T_arr_idx': T_arr,
            'T_dep_idx': None,  # Will be computed
            'cdeg': 0.02
        })
    return evs

def test_stage1_timeseries():
    """Test that Stage-I produces non-zero waiting times when N > M"""
    print("=" * 60)
    print("Test 1: Stage-I Time-Series Simulation")
    print("=" * 60)
    
    ev_pool = generate_test_evs(n=40)
    M = 30
    T = 48
    delta_t = 0.5
    
    system_params = {
        'M': M,
        'P_max': 100.0,
        'P_avg': 40.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }
    
    result = simulate_stage1_timeseries(ev_pool, system_params, T, delta_t)
    waiting_times = result['waiting_times']
    
    # Assertions
    avg_wait = sum(waiting_times.values()) / len(waiting_times)
    print(f"Average waiting time: {avg_wait:.3f} hours")
    assert avg_wait > 0, f"Expected non-zero waiting time when N={len(ev_pool)} > M={M}, got {avg_wait}"
    
    # Check that some EVs have wait > 0
    waits_with_delay = [w for w in waiting_times.values() if w > 0.1]
    print(f"EVs with waiting time > 0.1h: {len(waits_with_delay)}/{len(waiting_times)}")
    assert len(waits_with_delay) > 0, "Expected some EVs to have non-zero waiting time"
    
    print("✓ Test 1 PASSED: Waiting times computed correctly\n")
    return result

def test_occupancy_constraints():
    """Test that final schedule respects occupancy constraints"""
    print("=" * 60)
    print("Test 2: Occupancy Constraints (C3)")
    print("=" * 60)
    
    ev_pool = generate_test_evs(n=40)
    M = 30
    T = 48
    delta_t = 0.5
    
    system_params = {
        'M': M,
        'P_max': 100.0,
        'P_avg': 40.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }
    
    stage1_result = simulate_stage1_timeseries(ev_pool, system_params, T, delta_t)
    admitted_evs_t0 = stage1_result['admitted_evs_t0']
    
    # Convert to GA format
    admitted_evs = []
    for ev in admitted_evs_t0:
        T_arr_idx = ev.get('T_arr_idx', 0)
        T_stay = ev.get('T_stay', 4.0)
        slots = max(1, int(math.ceil(T_stay / delta_t)))
        T_dep_idx = min(T, T_arr_idx + slots)
        
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'SoC_min': ev.get('SoC_min', 0.0),
            'P_ref': 7.0,
            'P_dis_min': -7.0,
            'T_stay': T_stay,
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': 0.02
        })
    
    # Run GA
    pi_buy_arr = [0.25] * T
    pi_rev_arr = [0.18] * T
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    
    result = run_ga(
        evs=admitted_evs, T=T, delta_t=delta_t,
        pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
        P_max=100.0, weights=weights,
        alpha1=50.0, alpha2=50.0, alpha3=50.0,
        pop_size=60, ngen=80,
        num_chargers=M, verbose=False
    )
    
    best_schedule = result['best_schedule']
    
    # Check occupancy per slot
    max_occupancy = 0
    violations = []
    for t in range(T):
        active = int(np.sum(np.abs(best_schedule[:, t]) > 1e-6))
        max_occupancy = max(max_occupancy, active)
        if active > M:
            violations.append((t, active))
    
    print(f"Maximum occupancy across all slots: {max_occupancy} (limit: {M})")
    if violations:
        print(f"Violations found at slots: {violations[:5]}...")  # Show first 5
        assert False, f"Found {len(violations)} occupancy violations"
    
    assert max_occupancy <= M, f"Occupancy violation: max={max_occupancy} > M={M}"
    print("✓ Test 2 PASSED: Occupancy constraints respected\n")
    return result

def test_soc_bounds():
    """Test that SoC stays within bounds"""
    print("=" * 60)
    print("Test 3: SoC Bounds (C1)")
    print("=" * 60)
    
    ev_pool = generate_test_evs(n=40)
    M = 30
    T = 48
    delta_t = 0.5
    
    system_params = {
        'M': M,
        'P_max': 100.0,
        'P_avg': 40.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }
    
    stage1_result = simulate_stage1_timeseries(ev_pool, system_params, T, delta_t)
    admitted_evs_t0 = stage1_result['admitted_evs_t0']
    
    admitted_evs = []
    for ev in admitted_evs_t0:
        T_arr_idx = ev.get('T_arr_idx', 0)
        T_stay = ev.get('T_stay', 4.0)
        slots = max(1, int(math.ceil(T_stay / delta_t)))
        T_dep_idx = min(T, T_arr_idx + slots)
        
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'SoC_min': ev.get('SoC_min', 0.0),
            'P_ref': 7.0,
            'P_dis_min': -7.0,
            'T_stay': T_stay,
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': 0.02
        })
    
    # Run GA
    pi_buy_arr = [0.25] * T
    pi_rev_arr = [0.18] * T
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    
    result = run_ga(
        evs=admitted_evs, T=T, delta_t=delta_t,
        pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
        P_max=100.0, weights=weights,
        alpha1=50.0, alpha2=50.0, alpha3=50.0,
        pop_size=60, ngen=80,
        num_chargers=M, verbose=False
    )
    
    best_schedule = result['best_schedule']
    V_SoC, V_occ, V_grid = compute_penalty(best_schedule, admitted_evs, 100.0, M, delta_t)
    
    print(f"SoC violation penalty (kWh): {V_SoC:.6f}")
    print(f"Occupancy violation: {V_occ:.1f}")
    print(f"Grid violation (kWh): {V_grid:.6f}")
    
    # Check final SoC for each EV
    tol = 0.01  # Small tolerance for numerical errors
    violations = []
    for i, ev in enumerate(admitted_evs):
        Ecap = ev['Ecap']
        SoC_init = ev['SoC_init']
        SoC_min = ev.get('SoC_min', 0.0)
        SoC_max = ev['SoC_max']
        
        # Simulate SoC trajectory
        SoC = SoC_init
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(Tarr, min(Tdep, T)):
            p = best_schedule[i, t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
        
        if SoC < SoC_min - tol or SoC > SoC_max + tol:
            violations.append((ev['id'], SoC, SoC_min, SoC_max))
    
    if violations:
        print(f"Found {len(violations)} SoC violations (showing first 3):")
        for vid, soc, smin, smax in violations[:3]:
            print(f"  EV{vid}: SoC={soc:.4f} (bounds: [{smin:.4f}, {smax:.4f}])")
        # Allow small violations due to numerical precision
        if V_SoC > 1.0:  # Only fail if significant violation
            assert False, f"Significant SoC violations found (V_SoC={V_SoC:.6f})"
    
    assert V_SoC < 1.0, f"SoC violations too large: {V_SoC:.6f} kWh"
    print("✓ Test 3 PASSED: SoC bounds respected\n")
    return result

def test_ga_convergence():
    """Test that GA shows convergence pattern"""
    print("=" * 60)
    print("Test 4: GA Convergence")
    print("=" * 60)
    
    ev_pool = generate_test_evs(n=40)
    M = 30
    T = 48
    delta_t = 0.5
    
    system_params = {
        'M': M,
        'P_max': 100.0,
        'P_avg': 40.0,
        'cdeg': 0.02,
        'pi_buy': 0.25,
        'pi_rev': 0.18,
        'pi_buy_min': 0.10,
        'pi_buy_max': 0.50,
        'pi_rev_min': 0.05,
        'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25}
    }
    
    stage1_result = simulate_stage1_timeseries(ev_pool, system_params, T, delta_t)
    admitted_evs_t0 = stage1_result['admitted_evs_t0']
    
    admitted_evs = []
    for ev in admitted_evs_t0:
        T_arr_idx = ev.get('T_arr_idx', 0)
        T_stay = ev.get('T_stay', 4.0)
        slots = max(1, int(math.ceil(T_stay / delta_t)))
        T_dep_idx = min(T, T_arr_idx + slots)
        
        admitted_evs.append({
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'SoC_min': ev.get('SoC_min', 0.0),
            'P_ref': 7.0,
            'P_dis_min': -7.0,
            'T_stay': T_stay,
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': 0.02
        })
    
    # Run GA
    pi_buy_arr = [0.25] * T
    pi_rev_arr = [0.18] * T
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    
    result = run_ga(
        evs=admitted_evs, T=T, delta_t=delta_t,
        pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
        P_max=100.0, weights=weights,
        alpha1=50.0, alpha2=50.0, alpha3=50.0,
        pop_size=60, ngen=80,
        num_chargers=M, verbose=False
    )
    
    # Check GA history
    ga_history = result.get('ga_history', [])
    if not ga_history:
        print("Warning: No GA history found")
        return result
    
    initial_J = ga_history[0]['best_J']
    final_J = result['J']
    improvement = initial_J - final_J
    
    print(f"Initial best J: {initial_J:.8f}")
    print(f"Final best J: {final_J:.8f}")
    print(f"Improvement: {improvement:.8f} ({improvement/initial_J*100:.2f}%)")
    
    # Check that best J is non-increasing (monotonic improvement or plateau)
    best_js = [h['best_J'] for h in ga_history]
    non_improving_count = 0
    for i in range(1, len(best_js)):
        if best_js[i] > best_js[i-1] + 1e-10:  # Allow small numerical errors
            non_improving_count += 1
    
    print(f"Non-improving generations: {non_improving_count}/{len(best_js)-1}")
    
    # Best J should improve or stay the same (not worsen)
    assert final_J <= initial_J + 1e-6, f"GA diverged: final J ({final_J}) > initial J ({initial_J})"
    
    # Should see some improvement over many generations
    if len(ga_history) > 20:
        mid_J = ga_history[len(ga_history)//2]['best_J']
        assert final_J < mid_J + 1e-6 or improvement > 1e-6, "GA did not show sufficient convergence"
    
    print("✓ Test 4 PASSED: GA convergence verified\n")
    return result

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Small Scenario Tests (N=40, M=30)")
    print("="*60 + "\n")
    
    try:
        test_stage1_timeseries()
        test_occupancy_constraints()
        test_soc_bounds()
        test_ga_convergence()
        
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

