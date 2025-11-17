#!/usr/bin/env python3
"""
pipeline_updated.py

Full updated pipeline:
- Stage-I priority scheduling (general n EVs from CSV/JSON)
- Stage-II GA (DEAP) with normalization, seeding, repair, improved bounds
- CLI options: --ev-file, --chargers, with sensible defaults
"""

import argparse
import csv
import json
import math
import os
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from deap import base, creator, tools

# -------------------------
# Utility: load EVs (CSV or JSON)
# -------------------------
def load_ev_pool(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    evs = []
    if ext in ('.csv',):
        with open(path, 'r', newline='') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # convert numeric fields robustly; fall back to sensible defaults
                def f(key, default=None, typ=float):
                    v = row.get(key, None)
                    if v is None or v == '':
                        return default
                    try:
                        return typ(v)
                    except:
                        return default

                ev = {
                    'id': int(f('id', default=0, typ=int)),
                    'Ecap': f('Ecap', default=40.0),
                    'SoC_init': f('SoC_init', default=0.2),
                    'SoC_max': f('SoC_max', default=0.8),
                    'SoC_min': f('SoC_min', default=0.0),
                    'R_i': f('R_i', default=None),
                    'P_ref': f('P_ref', default=None),
                    'P_dis_min': f('P_dis_min', default=None),
                    'T_stay': f('T_stay', default=4.0),
                    'T_arr_idx': int(f('T_arr_idx', default=0, typ=int)),
                    'T_dep_idx': int(f('T_dep_idx', default=-1, typ=int)) if f('T_dep_idx', default=-1) >= 0 else None,
                    'cdeg': f('cdeg', default=0.02),
                }
                evs.append(ev)
    elif ext in ('.json', '.jsn'):
        with open(path, 'r') as fh:
            data = json.load(fh)
            for row in data:
                ev = {
                    'id': int(row.get('id', 0)),
                    'Ecap': float(row.get('Ecap', 40.0)),
                    'SoC_init': float(row.get('SoC_init', 0.2)),
                    'SoC_max': float(row.get('SoC_max', 0.8)),
                    'SoC_min': float(row.get('SoC_min', 0.0)),
                    'R_i': row.get('R_i', None),
                    'P_ref': row.get('P_ref', None),
                    'P_dis_min': row.get('P_dis_min', None),
                    'T_stay': float(row.get('T_stay', 4.0)),
                    'T_arr_idx': int(row.get('T_arr_idx', 0)),
                    'T_dep_idx': int(row.get('T_dep_idx')) if row.get('T_dep_idx') is not None else None,
                    'cdeg': float(row.get('cdeg', 0.02)),
                }
                evs.append(ev)
    else:
        raise ValueError("Unsupported EV file type: must be CSV or JSON")
    return evs


# -------------------------
# Stage-I helpers
# -------------------------
def calc_delta_E(ev: Dict) -> float:
    return max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])

def calc_p_req(ev: Dict) -> float:
    deltaE = calc_delta_E(ev)
    T_stay = ev.get('T_stay', 0.0)
    if T_stay <= 0:
        return float('inf') if deltaE > 0 else 0.0
    raw = deltaE / T_stay
    R_i = ev.get('R_i', ev.get('P_ref', None))
    if R_i is None or R_i <= 0:
        # fallback to a realistic default (7 kW)
        R_i = 7.0
    return min(raw, R_i)

def calc_phi(ev: Dict) -> float:
    p_req = calc_p_req(ev)
    pref = ev.get('R_i', ev.get('P_ref', None))
    if pref is None or pref <= 0:
        pref = 7.0
    return min(1.0, p_req / pref)

def calc_degradation_factor(ev_list: List[Dict], cdeg: float) -> Dict[int, float]:
    vals = [ (ev.get('cdeg', cdeg) * calc_delta_E(ev)) for ev in ev_list ]
    vmin, vmax = min(vals), max(vals)
    denom = (vmax - vmin) if vmax != vmin else 1.0
    return { ev['id']: (ev.get('cdeg', cdeg) * calc_delta_E(ev) - vmin) / denom for ev in ev_list }

def calc_grid_stress_factor(ev_list: List[Dict], P_avg: float, P_max: float) -> float:
    p_hat = sum(calc_p_req(ev) for ev in ev_list)
    denom = (P_max - P_avg) if (P_max - P_avg) != 0 else 1.0
    return max(0.0, (p_hat - P_avg)/denom)

def calc_price_factor(pi_buy: float, pi_rev: float,
                      pi_buy_min: float, pi_buy_max: float,
                      pi_rev_min: float, pi_rev_max: float) -> float:
    buy_denom = (pi_buy_max - pi_buy_min) if (pi_buy_max - pi_buy_min) != 0 else 1.0
    rev_denom = (pi_rev_max - pi_rev_min) if (pi_rev_max - pi_rev_min) != 0 else 1.0
    P_buy = (pi_buy - pi_buy_min) / buy_denom
    P_rev = (pi_rev - pi_rev_min) / rev_denom
    return P_buy - P_rev

def calc_priority_scores(ev_list: List[Dict], weights: Dict[str, float],
                         cdeg: float, P_avg: float, P_max: float,
                         pi_buy: float, pi_rev: float,
                         pi_buy_min: float, pi_buy_max: float,
                         pi_rev_min: float, pi_rev_max: float):
    phi_map = {ev['id']: calc_phi(ev) for ev in ev_list}
    D_map = calc_degradation_factor(ev_list, cdeg)
    G_factor = calc_grid_stress_factor(ev_list, P_avg, P_max)
    P_factor = calc_price_factor(pi_buy, pi_rev, pi_buy_min, pi_buy_max, pi_rev_min, pi_rev_max)
    lambda_scores = {}
    details = {}
    for ev in ev_list:
        eid = ev['id']
        phi = phi_map[eid]
        D = D_map[eid]
        lam = (weights['w_s'] * phi - weights['w_d'] * D - weights['w_g'] * G_factor - weights['w_p'] * P_factor)
        lambda_scores[eid] = lam
        details[eid] = {
            'phi': phi,
            'Dfactor': D,
            'Gfactor': G_factor,
            'Pfactor': P_factor,
            'lambda': lam,
            'p_req': calc_p_req(ev),
            'DeltaE': calc_delta_E(ev)
        }
    return lambda_scores, details

def assign_chargers(lambda_scores: Dict[int, float], M: int) -> List[int]:
    sorted_evs = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    return [eid for eid,_ in sorted_evs[:M]]

def run_stage1(ev_list: List[Dict], system_params: Dict) -> Tuple[List[Dict], Dict]:
    lambda_scores, details = calc_priority_scores(
        ev_list,
        system_params['weights'],
        system_params['cdeg'],
        system_params['P_avg'],
        system_params['P_max'],
        system_params['pi_buy'],
        system_params['pi_rev'],
        system_params['pi_buy_min'],
        system_params['pi_buy_max'],
        system_params['pi_rev_min'],
        system_params['pi_rev_max']
    )
    admitted_ids = assign_chargers(lambda_scores, system_params['M'])
    admitted = [ev for ev in ev_list if ev['id'] in admitted_ids]

    print("=== Stage-I Priority Scheduling ===\n")
    print("Step 1 — Urgency (DeltaE, p_req, phi) (Eq. (7),(9),(10)):")
    for ev in ev_list:
        d = details[ev['id']]
        print(f"EV{ev['id']}: DeltaE = {d['DeltaE']:.3f} kWh, p_req = {d['p_req']:.3f} kW, phi = {d['phi']:.3f}")
    print("\nStep 2 — Degradation (cdeg * DeltaE then normalized) (Eq. (23)):")
    for ev in ev_list:
        raw_deg = system_params['cdeg'] * details[ev['id']]['DeltaE']
        print(f"EV{ev['id']}: cdeg*DeltaE = {raw_deg:.3f}, Dfactor = {details[ev['id']]['Dfactor']:.3f}")
    print(f"\nStep 3 — Grid stress factor (Gfactor) (Eq. (24)):\nGfactor = {next(iter(details.values()))['Gfactor']:.4f}")
    print(f"\nStep 4 — Price factor (Pfactor) (Eq. (25),(26)):\nPfactor = {next(iter(details.values()))['Pfactor']:.3f}\n")
    print("Step 5 — Priority scores (lambda_i) (Eq. (22)):")
    for ev in ev_list:
        lam = details[ev['id']]['lambda']
        print(f"lambda_{ev['id']} = {lam:.5f}")
    print()
    sorted_by_lambda = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
    rank_str = " > ".join([f"EV{eid}" for eid,_ in sorted_by_lambda])
    print("Step 6 — Ranking and Admission:")
    print(f"Ranking (desc lambda): {rank_str}")
    print(f"With M={system_params['M']} chargers, admitted EVs: {', '.join('EV'+str(x) for x in admitted_ids)}")
    waiting = [ev['id'] for ev in ev_list if ev['id'] not in admitted_ids]
    print(f"Waiting EVs: {', '.join('EV'+str(x) for x in waiting)}\n")
    return admitted, details

def simulate_stage1_timeseries(ev_pool: List[Dict], system_params: Dict, T: int, delta_t: float) -> Dict:
    """
    Simulate Stage-I admission per time slot.
    Returns admission timeline with first_admit_slot for each EV and per-slot admitted sets.
    """
    M = system_params['M']
    admission_timeline = {}  # slot t -> list of admitted EV ids
    ev_state = {}  # ev_id -> {assigned: bool, first_admit_slot: int or None, assigned_slots: set}
    
    # Initialize EV states
    for ev in ev_pool:
        ev_id = ev['id']
        T_arr = ev.get('T_arr_idx', 0)
        T_dep = ev.get('T_dep_idx', None)
        if T_dep is None:
            T_stay = ev.get('T_stay', 0.0)
            slots_stay = max(1, int(math.ceil(T_stay / delta_t)))
            T_dep = min(T, T_arr + slots_stay)
        
        ev_state[ev_id] = {
            'assigned': False,
            'first_admit_slot': None,
            'assigned_slots': set(),
            'T_arr': T_arr,
            'T_dep': T_dep,
            'ev': ev
        }
    
    # Simulate each time slot
    for t in range(T):
        admission_timeline[t] = []
        # Find EVs present at this slot
        present_evs = []
        for ev_id, state in ev_state.items():
            if state['T_arr'] <= t < state['T_dep']:
                present_evs.append((ev_id, state))
        
        # Release EVs that have departed or reached target SoC (simplified: release on departure)
        currently_assigned = []
        for ev_id, state in ev_state.items():
            if state['assigned'] and t >= state['T_dep']:
                state['assigned'] = False
                state['assigned_slots'].discard(t)
            if state['assigned']:
                currently_assigned.append(ev_id)
        
        # Count available chargers
        available = M - len(currently_assigned)
        
        if available > 0:
            # Compute priority scores for waiting EVs (those present but not assigned)
            waiting_evs_list = [s['ev'] for ev_id, s in present_evs if not s['assigned']]
            
            # If t=0 and not enough EVs present, include early arrivals (next 2 slots) to fill chargers
            if t == 0 and len(waiting_evs_list) < available:
                early_arrivals = []
                for ev_id, state in ev_state.items():
                    # Include EVs arriving in next 2 slots (t+1, t+2)
                    if (state['T_arr'] > t and state['T_arr'] <= t + 2 and 
                        state['T_arr'] < state['T_dep'] and not state['assigned']):
                        early_arrivals.append((ev_id, state['ev']))
                # Add early arrivals to waiting list
                for ev_id, ev in early_arrivals:
                    if ev not in waiting_evs_list:
                        waiting_evs_list.append(ev)
            
            if waiting_evs_list:
                # Get time-varying prices if available, else use constant
                pi_buy_t = system_params.get('pi_buy', 0.25)
                if isinstance(pi_buy_t, (list, np.ndarray)) and len(pi_buy_t) > t:
                    pi_buy_val = pi_buy_t[t]
                else:
                    pi_buy_val = pi_buy_t if not isinstance(pi_buy_t, (list, np.ndarray)) else pi_buy_t[0] if len(pi_buy_t) > 0 else 0.25
                
                pi_rev_t = system_params.get('pi_rev', 0.18)
                if isinstance(pi_rev_t, (list, np.ndarray)) and len(pi_rev_t) > t:
                    pi_rev_val = pi_rev_t[t]
                else:
                    pi_rev_val = pi_rev_t if not isinstance(pi_rev_t, (list, np.ndarray)) else pi_rev_t[0] if len(pi_rev_t) > 0 else 0.18
                
                lambda_scores, _ = calc_priority_scores(
                    waiting_evs_list,
                    system_params['weights'],
                    system_params['cdeg'],
                    system_params['P_avg'],
                    system_params['P_max'],
                    pi_buy_val,
                    pi_rev_val,
                    system_params['pi_buy_min'],
                    system_params['pi_buy_max'],
                    system_params['pi_rev_min'],
                    system_params['pi_rev_max']
                )
                
                # Rank and assign up to available chargers
                sorted_waiting = sorted(lambda_scores.items(), key=lambda item: (-item[1], item[0]))
                to_admit = min(available, len(sorted_waiting))
                
                for idx in range(to_admit):
                    ev_id = sorted_waiting[idx][0]
                    ev_state[ev_id]['assigned'] = True
                    ev_state[ev_id]['assigned_slots'].add(t)
                    if ev_state[ev_id]['first_admit_slot'] is None:
                        ev_state[ev_id]['first_admit_slot'] = t
                    admission_timeline[t].append(ev_id)
                    
                    # If this EV arrives later, mark it for future slots too
                    if ev_state[ev_id]['T_arr'] > t:
                        # Pre-admit: mark slots from arrival to departure
                        for future_t in range(ev_state[ev_id]['T_arr'], min(ev_state[ev_id]['T_dep'], T)):
                            if future_t not in admission_timeline:
                                admission_timeline[future_t] = []
                            admission_timeline[future_t].append(ev_id)
        
        # Track all currently assigned at this slot
        for ev_id, state in ev_state.items():
            if state['assigned'] and t < state['T_dep']:
                admission_timeline[t].append(ev_id)
        
        # Remove duplicates while preserving order
        admission_timeline[t] = list(dict.fromkeys(admission_timeline[t]))
    
    # Compute waiting times
    waiting_times = {}
    for ev_id, state in ev_state.items():
        first_admit = state['first_admit_slot']
        T_arr = state['T_arr']
        if first_admit is None:
            # Never admitted - wait entire stay duration
            wait_slots = state['T_dep'] - T_arr
        else:
            wait_slots = first_admit - T_arr
        waiting_times[ev_id] = max(0, wait_slots) * delta_t
    
    # Get admitted at t=0 for backward compatibility
    admitted_at_t0 = admission_timeline.get(0, [])
    admitted_evs_t0 = [ev for ev in ev_pool if ev['id'] in admitted_at_t0]
    
    return {
        'admission_timeline': admission_timeline,
        'waiting_times': waiting_times,
        'ev_state': ev_state,
        'admitted_at_t0': admitted_at_t0,
        'admitted_evs_t0': admitted_evs_t0
    }

# -------------------------
# Stage-II (GA) helpers
# -------------------------
def flatten_index(i, t, T): return i * T + t
def unflatten(individual, M, T): return np.array(individual, dtype=float).reshape((M, T))

def compute_F1(net_schedule, pi_buy, pi_rev, delta_t):
    M, T = net_schedule.shape
    cost = 0.0
    for t in range(T):
        for i in range(M):
            p = net_schedule[i,t]
            if p > 0:
                cost += pi_buy[t] * p * delta_t
            elif p < 0:
                cost += pi_rev[t] * p * delta_t
    return cost

def compute_F2(net_schedule, cdeg_arr, delta_t):
    M, T = net_schedule.shape
    return sum(cdeg_arr[i] * abs(net_schedule[i,t]) * delta_t for i in range(M) for t in range(T))

def compute_F3(net_schedule):
    L = np.sum(net_schedule, axis=0)
    Lbar = np.mean(L)
    return np.sum((L - Lbar)**2)

def compute_Si_from_schedule(net_schedule, evs, delta_t):
    M, T = net_schedule.shape
    Si_list = [0.0] * M
    for idx in range(M):
        ev = evs[idx]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        SoC_max = ev['SoC_max']
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(Tarr, Tdep):
            p = net_schedule[idx, t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
        SoC_T = SoC
        Ereq = max(0.0, (SoC_max - ev['SoC_init']) * Ecap)
        Tstay = max(ev.get('T_stay', 1e-9), 1e-9)
        preq = (Ereq / Tstay) if Tstay > 0 else (float('inf') if Ereq > 0 else 0.0)
        pref = ev.get('P_ref', ev.get('R_i', 7.0))
        if pref is None or pref <= 0:
            pref = 7.0
        phi_i = min(1.0, preq / pref)
        if SoC_T >= SoC_max:
            delta_i = 0.0
        else:
            denom = (SoC_max - ev['SoC_init'])
            delta_i = (SoC_max - SoC_T) / denom if denom > 0 else 1.0
            delta_i = max(0.0, delta_i)
        Si_list[idx] = max(0.0, 1.0 - phi_i * delta_i)
    return Si_list

def compute_F4(net_schedule, evs, delta_t):
    Si = compute_Si_from_schedule(net_schedule, evs, delta_t)
    return sum(Si), Si

def compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t):
    M, T = net_schedule.shape
    V_SoC = 0.0
    eps = 1e-9
    for i in range(M):
        ev = evs[i]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        SoC_min = ev.get('SoC_min', 0.0)
        SoC_max = ev['SoC_max']
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(Tarr, min(Tdep, T)):
            p = net_schedule[i,t]
            if Ecap > 0:
                SoC += (p * delta_t) / Ecap
            # Convert SoC violations to kWh (penalty units)
            if SoC < SoC_min - eps:
                violation_soc = SoC_min - SoC  # SoC units
                violation_kwh = violation_soc * Ecap  # Convert to kWh
                V_SoC += violation_kwh
            elif SoC > SoC_max + eps:
                violation_soc = SoC - SoC_max  # SoC units
                violation_kwh = violation_soc * Ecap  # Convert to kWh
                V_SoC += violation_kwh
    V_grid = 0.0
    V_occ = 0.0
    for t in range(T):
        Lt = float(np.sum(net_schedule[:,t]))
        if Lt > P_max:
            V_grid += (Lt - P_max) * delta_t  # Convert to energy (kWh)
        active = int(np.sum(np.abs(net_schedule[:,t]) > 1e-6))
        if active > num_chargers:
            V_occ += float(active - num_chargers)  # Unit: count of excess EVs
    return V_SoC, V_occ, V_grid

# -------------------------
# GA: fitness factory (with normalization)
# -------------------------
def make_fitness_function(evs, M, T, delta_t,
                          pi_buy, pi_rev,
                          cdeg_arr,
                          P_max, num_chargers,
                          w1, w2, w3, w4,
                          alpha1, alpha2, alpha3):
    # Improved normalization denominators using realistic upper bounds
    max_power_sum = sum(ev.get('P_ref', ev.get('R_i', 7.0)) for ev in evs) if evs else 1.0
    max_energy_over_horizon = max_power_sum * delta_t * T
    
    # F1: use difference between max buy and min rev prices for better normalization
    pi_buy_arr = pi_buy if hasattr(pi_buy, '__len__') else [pi_buy]
    pi_rev_arr = pi_rev if hasattr(pi_rev, '__len__') else [pi_rev]
    max_pi_buy = max(pi_buy_arr)
    min_pi_rev = min(pi_rev_arr)
    max_price_diff = max(1e-9, max_pi_buy - min_pi_rev)  # ensure positive
    denom_F1 = (max_price_diff * max_energy_over_horizon + 1e-9)
    
    max_cdeg = max(cdeg_arr) if len(cdeg_arr) > 0 else 1.0
    denom_F2 = (max_cdeg * max_energy_over_horizon + 1e-9)
    
    # F3: variance normalization - use max possible variance
    max_power_sum_sq = max_power_sum ** 2
    denom_F3 = (max_power_sum_sq * T + 1e-9)
    
    denom_F4 = (M + 1e-9)
    
    # Omega: penalty normalization - scale relative to typical energy
    denom_Omega = (max_energy_over_horizon + 1e-9)
    def evaluate(individual):
        net_schedule = unflatten(individual, M, T)
        F1 = compute_F1(net_schedule, pi_buy, pi_rev, delta_t)
        F2 = compute_F2(net_schedule, cdeg_arr, delta_t)
        F3 = compute_F3(net_schedule)
        F4_sum, Si_list = compute_F4(net_schedule, evs, delta_t)
        V_SoC, V_occ, V_grid = compute_penalty(net_schedule, evs, P_max, num_chargers, delta_t)
        Omega_raw = alpha1 * V_SoC + alpha2 * V_occ + alpha3 * V_grid
        F1_norm = F1 / denom_F1
        F2_norm = F2 / denom_F2
        F3_norm = F3 / denom_F3
        F4_norm = F4_sum / denom_F4
        Omega_norm = Omega_raw / denom_Omega
        J = w1 * F1_norm + w2 * F2_norm + w3 * F3_norm - w4 * F4_norm + Omega_norm
        individual._cached = {
            'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4_sum,
            'F1_norm': F1_norm, 'F2_norm': F2_norm, 'F3_norm': F3_norm, 'F4_norm': F4_norm,
            'Si_list': Si_list,
            'V_SoC': V_SoC, 'V_occ': V_occ, 'V_grid': V_grid,
            'Omega_raw': Omega_raw, 'Omega_norm': Omega_norm,
            'J': J
        }
        return (J,)
    return evaluate

# -------------------------
# Bounds, seeding, repair
# -------------------------
def build_bounds_and_mask(evs, M, T):
    lower = [0.0] * (M * T)
    upper = [0.0] * (M * T)
    for i in range(M):
        ev = evs[i]
        pmax = ev.get('P_ref', ev.get('R_i', 7.0))
        if pmax is None or pmax <= 0:
            pmax = 7.0
        # Allow bidirectional: P_dis_min (negative) for discharging
        pmin = ev.get('P_dis_min', None)
        if pmin is None:
            # Default: allow discharging up to same magnitude as charging
            pmin = -pmax
        else:
            # Ensure pmin is negative and within bounds
            pmin = max(-pmax, min(0.0, pmin))
        
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(T):
            idx = flatten_index(i, t, T)
            if Tarr <= t < Tdep:
                lower[idx] = pmin
                upper[idx] = pmax
            else:
                lower[idx] = 0.0
                upper[idx] = 0.0
    return lower, upper

def greedy_seed_individual(evs, M, T, delta_t, pi_buy):
    """Greedy seed: for each EV fill cheapest earliest slots until DeltaE satisfied (deterministic)."""
    chrom = [0.0] * (M * T)
    price = np.array(pi_buy)
    for i in range(M):
        ev = evs[i]
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        if DeltaE <= 1e-9:
            continue
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        slots = list(range(Tarr, Tdep))
        # sort slots by price then by earliest
        slots_sorted = sorted(slots, key=lambda t: (price[t], t))
        remaining_energy = DeltaE
        pmax = ev.get('P_ref', ev.get('R_i', 7.0))
        for t in slots_sorted:
            if remaining_energy <= 1e-9:
                break
            add_kwh = min(pmax * delta_t, remaining_energy)
            chrom[flatten_index(i, t, T)] = add_kwh / delta_t
            remaining_energy -= add_kwh
    return chrom

def seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=0.25):
    chrom = [0.0] * (M * T)
    price_arr = np.array(pi_buy)
    for i in range(M):
        ev = evs[i]
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        Tstay = max(ev.get('T_stay', 0.0), 1e-9)
        raw_preq = DeltaE / Tstay
        preq = min(raw_preq, ev.get('P_ref', ev.get('R_i', 7.0)))
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            continue
        stay_slots = list(range(Tarr, Tdep))
        prices_in_stay = price_arr[stay_slots]
        k = max(1, int(math.ceil(seed_fraction * len(stay_slots))))
        chosen_idx_rel = list(np.argsort(prices_in_stay)[:k])
        # distribute energy evenly across chosen slots
        for rel in chosen_idx_rel:
            t = stay_slots[rel]
            chrom[flatten_index(i, t, T)] = preq
    return chrom

def repair_individual(individual, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers, pi_rev=None, system_params=None):
    """Repair:
     - zero outside windows
     - scale/redistribute per-EV energy to try meet DeltaE (greedy to cheapest slots)
     - enforce per-slot active <= num_chargers by zeroing smallest contributors
    """
    mat = unflatten(individual, M, T)
    price = np.array(pi_buy)
    # Zero outside bounds and clamp
    for idx in range(M*T):
        if abs(upper_arr[idx] - lower_arr[idx]) < 1e-12:
            mat.flat[idx] = lower_arr[idx]
        else:
            mat.flat[idx] = max(lower_arr[idx], min(upper_arr[idx], mat.flat[idx]))

    # Per-EV energy repair
    for i in range(M):
        ev = evs[i]
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        if Tdep <= Tarr:
            mat[i, :] = 0.0
            continue
        slots = list(range(Tarr, Tdep))
        current_energy = float(np.sum(mat[i, slots]) * delta_t)
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        target_energy = min(DeltaE, ev.get('P_ref', ev.get('R_i', 7.0)) * len(slots) * delta_t)
        if abs(current_energy - target_energy) < 1e-6:
            continue
        # if current > target: scale down uniformly across active slots
        if current_energy > 0 and current_energy > target_energy:
            factor = target_energy / current_energy if current_energy > 0 else 0.0
            for t in slots:
                mat[i,t] *= factor
        elif current_energy < target_energy:
            # greedily fill cheapest slots (lowest price)
            need = target_energy - current_energy
            slot_order = sorted(slots, key=lambda t: price[t])
            for t in slot_order:
                avail = upper_arr[flatten_index(i, t, T)] - mat[i,t]
                if avail <= 1e-9:
                    continue
                add = min(avail * delta_t, need) / delta_t
                # add in kW such that energy added = add * delta_t
                mat[i,t] += add
                need -= add * delta_t
                if need <= 1e-9:
                    break
        # clamp per-slot
        for t in slots:
            mat[i,t] = max(lower_arr[flatten_index(i,t,T)], min(upper_arr[flatten_index(i,t,T)], mat[i,t]))

    # Enforce occupancy per slot using priority lambda_i(t)
    for t in range(T):
        col = mat[:, t]
        active_idx = np.where(np.abs(col) > 1e-6)[0].tolist()
        if len(active_idx) <= num_chargers:
            continue
        
        # Build a lightweight ev list for lambda computation
        present_evs = []
        for idx in active_idx:
            ev = evs[idx]
            ev_copy = {
                'id': ev['id'],
                'Ecap': ev['Ecap'],
                'SoC_init': ev['SoC_init'],
                'SoC_max': ev['SoC_max'],
                'R_i': ev.get('R_i', ev.get('P_ref', None)),
                'P_ref': ev.get('P_ref', ev.get('R_i', None)),
                'T_stay': ev.get('T_stay', 0.0),
                'T_arr_idx': ev.get('T_arr_idx', 0),
                'T_dep_idx': ev.get('T_dep_idx', T),
                'cdeg': ev.get('cdeg', 0.02)
            }
            present_evs.append(ev_copy)
        
        # determine pi_buy_t and pi_rev_t if arrays
        if hasattr(pi_buy, '__len__') and len(pi_buy) > t:
            pi_buy_t = pi_buy[t]
        else:
            pi_buy_t = pi_buy[0] if hasattr(pi_buy, '__len__') else pi_buy
        
        if pi_rev is not None:
            if hasattr(pi_rev, '__len__') and len(pi_rev) > t:
                pi_rev_t = pi_rev[t]
            else:
                pi_rev_t = pi_rev[0] if hasattr(pi_rev, '__len__') else pi_rev
        else:
            pi_rev_t = system_params.get('pi_rev', 0.18) if system_params else 0.18
        
        # compute lambda scores for present_evs
        if system_params and present_evs:
            lambda_map, _ = calc_priority_scores(
                present_evs,
                system_params.get('weights', {'w_s':0.25,'w_d':0.25,'w_g':0.25,'w_p':0.25}),
                system_params.get('cdeg', 0.02),
                system_params.get('P_avg', 40.0),
                system_params.get('P_max', 100.0),
                pi_buy_t, pi_rev_t,
                system_params.get('pi_buy_min', 0.10),
                system_params.get('pi_buy_max', 0.50),
                system_params.get('pi_rev_min', 0.05),
                system_params.get('pi_rev_max', 0.30)
            )
        else:
            # Fallback: use absolute power as proxy
            lambda_map = {evs[idx]['id']: abs(col[idx]) for idx in active_idx}
        
        # build (idx, lambda) list and keep top num_chargers indices
        active_with_lambda = [(idx, lambda_map.get(evs[idx]['id'], 0.0)) for idx in active_idx]
        active_with_lambda_sorted = sorted(active_with_lambda, key=lambda x: -x[1])  # highest lambda first
        keep_indices = {pair[0] for pair in active_with_lambda_sorted[:num_chargers]}
        
        for idx in active_idx:
            if idx not in keep_indices:
                mat[idx, t] = 0.0

    # write back into individual
    flat = mat.flatten().tolist()
    individual[:] = flat
    return individual

# -------------------------
# GA orchestration
# -------------------------
def run_ga(evs, T, delta_t,
           pi_buy, pi_rev,
           P_max,
           weights,
           alpha1, alpha2, alpha3,
           pop_size=120, ngen=300,
           cxpb=0.9, mutpb=0.3, eta_c=20.0, eta_m=20.0,
           tournament_size=3,
           stagnation_generations=40,
           seed_count=10,
           elitism_k=2,
           num_chargers=10,
           system_params=None,
           verbose=True):
    random.seed(42)
    M = len(evs)
    if M == 0:
        raise ValueError("No admitted EVs passed to run_ga")
    cdeg_arr = [ev.get('cdeg', 0.02) for ev in evs]
    num_chargers = int(num_chargers)
    w1 = weights.get('w1', 0.25)
    w2 = weights.get('w2', 0.25)
    w3 = weights.get('w3', 0.25)
    w4 = weights.get('w4', 0.25)

    lower_arr, upper_arr = build_bounds_and_mask(evs, M, T)

    # DEAP creator
    try:
        creator.FitnessMin
    except Exception:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except Exception:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    def attr_float_at_index(idx):
        lo = lower_arr[idx]; up = upper_arr[idx]
        if abs(up - lo) < 1e-12:
            return lo
        return random.uniform(lo, up)

    def generate_individual():
        indiv = [0.0] * (M * T)
        for idx in range(M * T):
            indiv[idx] = attr_float_at_index(idx)
        return creator.Individual(indiv)

    toolbox.register("individual", generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    eval_fn = make_fitness_function(evs, M, T, delta_t, pi_buy, pi_rev, cdeg_arr, P_max, num_chargers,
                                    w1, w2, w3, w4, alpha1, alpha2, alpha3)
    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)

    # safe bounded operators
    safe_low = list(lower_arr); safe_up = list(upper_arr)
    fixed_mask = [False] * (M*T)
    for idx in range(M*T):
        if abs(lower_arr[idx] - upper_arr[idx]) < 1e-12:
            fixed_mask[idx] = True
            safe_up[idx] = safe_low[idx] + 1e-9

    def safe_mate(ind1, ind2):
        tools.cxSimulatedBinaryBounded(ind1, ind2, low=safe_low, up=safe_up, eta=eta_c)
        for idx in range(M*T):
            if fixed_mask[idx]:
                ind1[idx] = lower_arr[idx]; ind2[idx] = lower_arr[idx]
        return ind1, ind2

    # FIX B: Per-gene mutation probability. Use per-slot or small-percent baseline instead of 1/(huge genome).
    # Aim: mutate ~1 gene per EV per generation on average or a small percent of genes.
    num_variable_genes = sum(0 if fixed_mask[idx] else 1 for idx in range(M*T))
    indpb = min(0.05, 1.0 / max(1, T))   # e.g., for T=48 -> ~0.0208

    def safe_mutate(ind):
        tools.mutPolynomialBounded(ind, low=safe_low, up=safe_up, eta=eta_m, indpb=indpb)
        for idx in range(M*T):
            if fixed_mask[idx]:
                ind[idx] = lower_arr[idx]
        return ind,

    toolbox.register("mate", safe_mate)
    toolbox.register("mutate", safe_mutate)

    pop = toolbox.population(n=pop_size)
    # FIX C: seeding: deterministic greedy seed + diverse stage1 seeds (with perturbation)
    # Reduce seed count to avoid overwhelming random initialization
    seed_count = max(1, int(0.05 * pop_size))  # Limit to 5% of population
    num_seed = min(seed_count, pop_size)
    if num_seed > 0:
        # greedy deterministic baseline
        pop[0][:] = greedy_seed_individual(evs, M, T, delta_t, pi_buy)
    for s in range(1, num_seed):
        # make seeds more diverse in fraction
        frac = 0.1 + 0.8 * random.random()
        seed_chrom = seed_individual_using_stage1(evs, M, T, delta_t, pi_buy, seed_fraction=frac)
        # Add small noise to seed to diversify
        noise_level = 0.01  # 1% of per-slot bounds
        for idx in range(len(seed_chrom)):
            if not fixed_mask[idx]:
                span = safe_up[idx] - safe_low[idx]
                seed_chrom[idx] += random.uniform(-noise_level*span, noise_level*span)
        pop[s][:] = seed_chrom

    # evaluate initial population
    for ind in pop:
        # repair before evaluating
        repair_individual(ind, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers, pi_rev=pi_rev, system_params=system_params)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    best = tools.selBest(pop, 1)[0]
    best_J = best.fitness.values[0]
    no_improve = 0
    gen = 0
    
    # GA history logging
    ga_history = []
    
    # Debug: check initial population fitness validity
    if verbose:
        valid_count = sum(1 for ind in pop if ind.fitness.valid)
        print(f"DEBUG: initial population fitness validity counts: {valid_count}/{len(pop)}")
        print(f"GA start: pop_size={pop_size}, ngen={ngen}, M={M}, T={T}, indpb={indpb:.6f}")
        print("Initial best J =", round(best_J, 8))
    
    # Log initial generation
    mean_J = np.mean([ind.fitness.values[0] for ind in pop])
    std_J = np.std([ind.fitness.values[0] for ind in pop])
    best_cached = getattr(best, '_cached', {})
    ga_history.append({
        'gen': 0,
        'best_J': best_J,
        'mean_J': mean_J,
        'std_J': std_J,
            'V_SoC': best_cached.get('V_SoC', 0.0),
            'V_occ': best_cached.get('V_occ', 0.0),
            'V_grid': best_cached.get('V_grid', 0.0),
            'diversity': 0.0  # Initial diversity
        })

    while gen < ngen and no_improve < stagnation_generations:
        gen += 1
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # crossover
        for i in range(0, len(offspring), 2):
            if i+1 >= len(offspring): break
            if random.random() <= cxpb:
                toolbox.mate(offspring[i], offspring[i+1])
        # mutation
        for i in range(len(offspring)):
            if random.random() <= mutpb:
                toolbox.mutate(offspring[i])
        
        # FIX A: Invalidate fitness for all offspring so they get re-evaluated
        for ind in offspring:
            # safe deletion (some clones may not have fitness values yet)
            try:
                del ind.fitness.values
            except Exception:
                ind.fitness.valid = False
        
        # Now repair & evaluate invalid individuals
        for ind in offspring:
            repair_individual(ind, evs, M, T, delta_t, lower_arr, upper_arr, pi_buy, num_chargers, pi_rev=pi_rev, system_params=system_params)
        
        # Evaluate those that were invalid (will now be all offspring)
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_inds:
            fitnesses = list(map(toolbox.evaluate, invalid_inds))
            for ind, fit in zip(invalid_inds, fitnesses):
                ind.fitness.values = fit
        
        # FIX D: Log how many offspring were actually re-evaluated (debug)
        if verbose and (gen % 10 == 0 or gen <= 5):
            print(f"DEBUG Gen {gen}: invalid_inds_count={len(invalid_inds)}")
        # elitism
        elites = tools.selBest(pop, elitism_k)
        combined = offspring + list(map(toolbox.clone, elites))
        combined.sort(key=lambda ind: ind.fitness.values[0])
        pop = combined[:pop_size]
        current_best = tools.selBest(pop, 1)[0]
        current_best_J = current_best.fitness.values[0]
        
        # Compute statistics for this generation
        mean_J = np.mean([ind.fitness.values[0] for ind in pop])
        std_J = np.std([ind.fitness.values[0] for ind in pop])
        best_cached = getattr(current_best, '_cached', {})
        
        # Compute population diversity (optional metric)
        try:
            mats = [np.array(ind, dtype=float) for ind in pop]
            n = len(mats)
            if n >= 2:
                # Sample-based pairwise distance for large populations
                sample_size = min(20, n)
                sample_indices = random.sample(range(n), sample_size)
                diversity_sum = 0.0
                diversity_count = 0
                for i in sample_indices:
                    for j in sample_indices:
                        if i < j:
                            diversity_sum += np.linalg.norm(mats[i] - mats[j])
                            diversity_count += 1
                avg_pairwise_distance = diversity_sum / diversity_count if diversity_count > 0 else 0.0
            else:
                avg_pairwise_distance = 0.0
        except Exception:
            avg_pairwise_distance = 0.0
        
        if current_best_J + 1e-12 < best_J:
            best = toolbox.clone(current_best)
            best_J = current_best_J
            no_improve = 0
            if verbose:
                print(f"Gen {gen} improved best J -> {best_J:.8f} (mean: {mean_J:.8f}, std: {std_J:.8f}, diversity: {avg_pairwise_distance:.4f})")
        else:
            no_improve += 1
            if verbose and (gen % 5 == 0 or gen == 1):
                print(f"Gen {gen} best J {best_J:.8f} (mean: {mean_J:.8f}, std: {std_J:.8f}, diversity: {avg_pairwise_distance:.4f}, no improve: {no_improve})")
        
        # Log this generation
        ga_history.append({
            'gen': gen,
            'best_J': best_J,
            'mean_J': mean_J,
            'std_J': std_J,
            'V_SoC': best_cached.get('V_SoC', 0.0),
            'V_occ': best_cached.get('V_occ', 0.0),
            'V_grid': best_cached.get('V_grid', 0.0),
            'diversity': avg_pairwise_distance
        })

    # finalize
    best_schedule = unflatten(best, M, T)
    breakdown = getattr(best, "_cached", None)
    if breakdown is None:
        best.fitness.values = toolbox.evaluate(best)
        breakdown = best._cached

    # Save GA history to CSV
    history_df = pd.DataFrame(ga_history)
    history_path = os.path.abspath('ga_history.csv')
    history_df.to_csv(history_path, index=False)
    if verbose:
        print(f"Saved GA history to: {history_path}")

    result = {
        'best_schedule': best_schedule,
        'J': breakdown['J'],
        'F1': breakdown['F1'], 'F2': breakdown['F2'], 'F3': breakdown['F3'], 'F4': breakdown['F4'],
        'F1_norm': breakdown.get('F1_norm'), 'F2_norm': breakdown.get('F2_norm'),
        'F3_norm': breakdown.get('F3_norm'), 'F4_norm': breakdown.get('F4_norm'),
        'Si_list': breakdown['Si_list'],
        'V_SoC': breakdown['V_SoC'], 'V_occ': breakdown['V_occ'], 'V_grid': breakdown['V_grid'],
        'Omega_raw': breakdown['Omega_raw'], 'Omega_norm': breakdown['Omega_norm'],
        'generations_executed': gen,
        'ga_history': ga_history
    }
    return result

# -------------------------
# Orchestrator main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev-file', type=str, default=None,
                        help='EV file (CSV or JSON). If omitted, looks for evs.csv / evs.json in cwd.')
    parser.add_argument('--chargers', type=int, default=30, help='Number of chargers (M). Default 30.')
    parser.add_argument('--T', type=int, default=48, help='Number of time slots (e.g. 48 for 24h half-hour slots).')
    parser.add_argument('--delta-t', type=float, default=0.25, help='Slot duration in hours (default 0.25 for 15-min slots to match paper).')
    parser.add_argument('--ngen', type=int, default=300, help='Number of GA generations. Default 300.')
    parser.add_argument('--pop-size', type=int, default=120, help='GA population size. Default 120.')
    parser.add_argument('--debug-short-run', action='store_true',
                        help='Use short run parameters for debugging (ngen=80, pop_size=60).')
    parser.add_argument('--ga-schedule-scope', type=str, default='t0',
                        choices=['t0', 'full'],
                        help="GA scheduling scope: 't0' schedule only EVs admitted at t=0 (default), 'full' schedule entire EV pool so GA chooses who to serve.")
    args = parser.parse_args()
    
    # Override with debug params if requested
    if args.debug_short_run:
        args.ngen = 80
        args.pop_size = 60
        print("Debug short-run mode: ngen=80, pop_size=60")

    # find default ev file if omitted
    ev_file = args.ev_file
    if ev_file is None:
        if os.path.exists('evs.csv'):
            ev_file = 'evs.csv'
            print("No --ev-file provided; found 'evs.csv' in current directory and will use it.")
        elif os.path.exists('evs.json'):
            ev_file = 'evs.json'
            print("No --ev-file provided; found 'evs.json' in current directory and will use it.")
        else:
            print("No EV file found. Exiting.")
            return

    ev_pool = load_ev_pool(ev_file)
    print(f"Loaded {len(ev_pool)} EVs from {ev_file}")

    # system params (tune if needed)
    T = args.T
    delta_t = args.delta_t
    system_params = {
        'M': args.chargers,
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

    # Stage-I: Time-series simulation
    print("\n=== Stage-I Time-Series Simulation ===")
    stage1_result = simulate_stage1_timeseries(ev_pool, system_params, T, delta_t)
    waiting_times = stage1_result['waiting_times']
    admitted_evs_t0 = stage1_result['admitted_evs_t0']
    admission_timeline = stage1_result['admission_timeline']
    ev_state = stage1_result['ev_state']
    
    print(f"Admitted at t=0: {len(admitted_evs_t0)} EVs")
    print(f"Waiting times computed for all {len(ev_pool)} EVs")
    avg_wait_all = sum(waiting_times.values()) / len(waiting_times) if waiting_times else 0.0
    print(f"Average waiting time (all EVs): {avg_wait_all:.3f} hours")
    
    # Print first 30 admitted EVs (admitted at t=0 order preserved)
    first_admitted = stage1_result['admitted_at_t0']
    print("\n--- First admitted EVs (t=0) ---")
    to_show = first_admitted[:30]
    if to_show:
        print(f"Count at t=0: {len(first_admitted)}")
        print("First 30 admitted EV IDs:", ", ".join(f"EV{eid}" for eid in to_show))
    else:
        print("No EVs admitted at t=0")
    
    # Print scheduling order (first_admit_slot ascending)
    schedule_order = []
    for ev_id, st in ev_state.items():
        fas = st['first_admit_slot']
        if fas is None:
            # mark as not admitted
            schedule_order.append((ev_id, float('inf')))
        else:
            schedule_order.append((ev_id, fas))
    
    schedule_order_sorted = sorted(schedule_order, key=lambda x: (x[1], x[0]))
    print("\n--- Scheduling order (by first admit slot) ---")
    for ev_id, slot in schedule_order_sorted[:200]:  # print first 200 entries to keep it readable
        if slot == float('inf'):
            print(f"EV{ev_id}: NOT ADMITTED")
        else:
            print(f"EV{ev_id}: first admitted at slot t={slot}")

    # Stage-II: convert EVs for GA based on scope
    if args.ga_schedule_scope == 'full':
        # Schedule entire EV pool - GA chooses who to serve
        print("\n--- GA Scheduling Scope: FULL (entire EV pool) ---")
        ga_evs = []
        for ev in ev_pool:
            T_arr_idx = int(ev.get('T_arr_idx', 0))
            T_dep_idx = ev.get('T_dep_idx', None)
            if T_dep_idx is None:
                slots = max(1, int(math.ceil(ev.get('T_stay', 0.0) / delta_t)))
                T_dep_idx = min(T, T_arr_idx + slots)
            else:
                T_dep_idx = min(T, T_dep_idx)
            
            p_ref = ev.get('R_i', ev.get('P_ref', None))
            if p_ref is None or p_ref <= 0:
                p_ref = 7.0
            
            ga_evs.append({
                'id': ev['id'],
                'Ecap': ev['Ecap'],
                'SoC_init': ev['SoC_init'],
                'SoC_max': ev['SoC_max'],
                'SoC_min': ev.get('SoC_min', 0.0),
                'P_ref': p_ref,
                'P_dis_min': ev.get('P_dis_min', -p_ref),
                'T_arr_idx': T_arr_idx,
                'T_dep_idx': T_dep_idx,
                'cdeg': ev.get('cdeg', system_params['cdeg'])
            })
        ga_evs_for_run = ga_evs
        print(f"GA will schedule all {len(ga_evs_for_run)} EVs in the pool")
    else:
        # Original behavior: only EVs admitted at t=0
        print("\n--- GA Scheduling Scope: t0 (only EVs admitted at t=0) ---")
        admitted_evs = []
        for ev in admitted_evs_t0:
            T_arr_idx = int(ev.get('T_arr_idx', 0))
            T_dep_idx = ev.get('T_dep_idx', None)
            if T_dep_idx is None:
                slots = max(1, int(math.ceil(ev.get('T_stay', 0.0) / delta_t)))
                T_dep_idx = min(T, T_arr_idx + slots)
            else:
                T_dep_idx = min(T, T_dep_idx)
            
            p_ref = ev.get('R_i', ev.get('P_ref', None))
            if p_ref is None or p_ref <= 0:
                p_ref = 7.0
            
            # Set P_dis_min if not present (default to -p_ref for bidirectional)
            p_dis_min = ev.get('P_dis_min', None)
            if p_dis_min is None:
                p_dis_min = -p_ref
            
            admitted_evs.append({
                'id': ev['id'],
                'Ecap': ev['Ecap'],
                'SoC_init': ev['SoC_init'],
                'SoC_max': ev['SoC_max'],
                'SoC_min': ev.get('SoC_min', 0.0),
                'P_ref': p_ref,
                'P_dis_min': p_dis_min,
                'T_stay': ev.get('T_stay', 0.0),
                'T_arr_idx': T_arr_idx,
                'T_dep_idx': T_dep_idx,
                'cdeg': ev.get('cdeg', system_params['cdeg'])
            })
        ga_evs_for_run = admitted_evs
        print(f"GA will schedule {len(ga_evs_for_run)} EVs admitted at t=0")

    # GA hyperparams & inputs
    pi_buy_arr = [system_params['pi_buy']] * T
    pi_rev_arr = [system_params['pi_rev']] * T
    P_max = system_params['P_max']
    weights = {'w1': 0.25, 'w2': 0.25, 'w3': 0.25, 'w4': 0.25}
    alpha1, alpha2, alpha3 = 50.0, 50.0, 50.0

    # Run GA with user-specified or default parameters
    result = run_ga(evs=ga_evs_for_run, T=T, delta_t=delta_t,
                    pi_buy=pi_buy_arr, pi_rev=pi_rev_arr,
                    P_max=P_max,
                    weights=weights,
                    alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
                    pop_size=args.pop_size, ngen=args.ngen,
                    cxpb=0.9, mutpb=0.4,  # Increased mutpb for better exploration
                    eta_c=20.0, eta_m=20.0,
                    tournament_size=3,
                    stagnation_generations=max(100, int(args.ngen * 0.4)),  # Allow more generations for convergence
                    seed_count=min(20, max(1, int(0.2 * args.pop_size))),  # This will be overridden in run_ga (Fix C)
                    elitism_k=max(1, min(5, int(0.02 * args.pop_size))),  # FIX F: Keep elitism small absolute (1-5)
                    num_chargers=system_params['M'],
                    system_params=system_params,
                    verbose=True)

    # Print results
    print("\n=== GA Result Summary ===")
    print("Objective J (normalized weighted):", result['J'])
    print("F1 (cost, raw) :", result['F1'], "F1_norm:", result['F1_norm'])
    print("F2 (deg, raw):", result['F2'], "F2_norm:", result['F2_norm'])
    print("F3 (var, raw):", result['F3'], "F3_norm:", result['F3_norm'])
    print("F4 (satisfaction, raw):", result['F4'], "F4_norm:", result['F4_norm'])
    print("Penalties (raw):", result['Omega_raw'], "Penalties_norm:", result['Omega_norm'])
    print("Violations (SoC, occ, grid):", result['V_SoC'], result['V_occ'], result['V_grid'])
    print("Generations executed:", result['generations_executed'])

    # Additional metrics: averages, waiting time, fairness (Gini)
    M = len(ga_evs_for_run)
    avg_cost_per_ev = (result['F1'] / M) if M > 0 else float('nan')
    avg_deg_per_ev = (result['F2'] / M) if M > 0 else float('nan')
    avg_user_satisfaction = (sum(result['Si_list']) / M) if M > 0 else float('nan')

    # Average waiting time and delivered energy
    # Use waiting times from Stage-I simulation
    wait_hours_admitted = [waiting_times.get(ev['id'], 0.0) for ev in ga_evs_for_run]
    avg_wait_hours = (sum(wait_hours_admitted) / len(wait_hours_admitted)) if wait_hours_admitted else 0.0
    
    # Also compute delivered energy for scheduled EVs
    best = result['best_schedule']
    delivered_kwh = []
    for i, ev in enumerate(ga_evs_for_run):
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        energy = 0.0
        for t in range(Tarr, min(Tdep, T)):
            p = best[i][t]
            energy += p * delta_t  # include both charging (positive) and discharging (negative)
        delivered_kwh.append(energy)

    # Gini coefficient
    def gini(values):
        n = len(values)
        if n == 0:
            return float('nan')
        sorted_vals = sorted(values)
        cum = 0.0
        for idx, val in enumerate(sorted_vals, start=1):
            cum += idx * val
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        return (2 * cum) / (n * total) - (n + 1) / n

    gini_energy = gini(delivered_kwh)

    print("\n--- Derived Metrics ---")
    print("Average net energy cost per EV:", avg_cost_per_ev)
    print("Average battery degradation cost per EV:", avg_deg_per_ev)
    print("Grid load variance (raw F3):", result['F3'])
    print("Average user satisfaction:", avg_user_satisfaction)
    print(f"Average waiting time (admitted EVs): {avg_wait_hours:.3f} hours")
    print(f"Average waiting time (all EVs in pool): {avg_wait_all:.3f} hours")
    print("Fairness (Gini on delivered kWh):", gini_energy)

    # Save best schedule
    df = pd.DataFrame(best, index=[ev['id'] for ev in ga_evs_for_run])
    csv_path = os.path.abspath('best_schedule_normalized.csv')
    df.to_csv(csv_path, index=True)
    print(f"Saved best_schedule_normalized.csv to: {csv_path}")

    # Per-EV diagnostics
    print("\n--- Per-EV Diagnostics (scheduled) ---")
    for i, ev in enumerate(ga_evs_for_run):
        delivered = float(np.sum(best[i, ev['T_arr_idx']:ev['T_dep_idx']]) * delta_t)
        SoC_T = ev['SoC_init'] + (delivered / ev['Ecap'] if ev['Ecap'] > 0 else 0.0)
        Si = compute_Si_from_schedule(best, ga_evs_for_run, delta_t)[i]
        print(f"EV{ev['id']}: delivered={delivered:.2f} kWh, SoC_T={SoC_T:.3f}, S_i={Si:.3f}")

if __name__ == "__main__":
    main()
