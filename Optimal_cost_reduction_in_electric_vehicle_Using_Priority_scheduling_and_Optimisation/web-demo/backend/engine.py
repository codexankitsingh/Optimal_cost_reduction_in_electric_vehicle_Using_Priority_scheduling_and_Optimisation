"""
engine.py — Algorithm wrapper module.

Imports from parent project's pipeline modules and exposes clean functions
for the web API. This avoids code duplication and keeps algorithms in sync.
"""
import sys
import os
import math
import random
import numpy as np

# Add parent project to path so we can import pipeline modules
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import from parent project
from pipeline import (
    load_ev_pool as _load_ev_pool,
    calc_delta_E, calc_p_req, calc_phi,
    calc_degradation_factor, calc_grid_stress_factor, calc_price_factor,
    calc_priority_scores, assign_chargers, run_stage1,
    compute_F1, compute_F2, compute_F3, compute_F4, compute_Si_from_schedule,
    compute_penalty, make_fitness_function,
    build_bounds_and_mask, greedy_seed_individual, seed_individual_using_stage1,
    repair_individual_lightweight, repair_individual,
    run_ga as _run_ga_priority,
    flatten_index, unflatten,
)
from pipeline_fcfs import admit_fcfs, run_ga as _run_ga_fcfs
from pipeline_sjf import admit_sjf, run_ga as _run_ga_sjf
from generate_ev_datasets import generate_evs as _generate_evs


def ev_dict_from_model(ev_data) -> dict:
    """Convert Pydantic EVData model to the dict format used by pipelines."""
    d = {
        'id': ev_data.id,
        'Ecap': ev_data.Ecap,
        'SoC_init': ev_data.SoC_init,
        'SoC_max': ev_data.SoC_max,
        'SoC_min': ev_data.SoC_min,
        'R_i': ev_data.R_i if ev_data.R_i is not None else 7.0,
        'P_ref': ev_data.P_ref if ev_data.P_ref is not None else ev_data.R_i if ev_data.R_i is not None else 7.0,
        'P_dis_min': ev_data.P_dis_min,
        'T_stay': ev_data.T_stay,
        'T_arr_idx': ev_data.T_arr_idx,
        'T_dep_idx': ev_data.T_dep_idx,
        'cdeg': ev_data.cdeg,
    }
    return d


def generate_fleet(n: int, seed: int = 42, T: int = 48, delta_t: float = 0.25) -> list:
    """Generate synthetic EV fleet data."""
    evs = _generate_evs(n, seed=seed, T=T, delta_t=delta_t)
    return evs


def generate_tou_prices(T: int, delta_t: float = 0.25):
    """Generate realistic Time-of-Use pricing for T slots."""
    hours = np.arange(T) * delta_t
    pi_buy = np.zeros(T)
    pi_rev = np.zeros(T)
    for t in range(T):
        h = hours[t] % 24
        if 6 <= h < 10:      # Morning peak
            pi_buy[t] = 0.35 + 0.10 * np.sin((h - 6) / 4 * np.pi)
        elif 10 <= h < 16:   # Midday (solar)
            pi_buy[t] = 0.15 + 0.05 * np.sin((h - 10) / 6 * np.pi)
        elif 16 <= h < 21:   # Evening peak
            pi_buy[t] = 0.40 + 0.10 * np.sin((h - 16) / 5 * np.pi)
        elif 21 <= h < 24:   # Night
            pi_buy[t] = 0.12
        else:                # Early morning
            pi_buy[t] = 0.10
        pi_rev[t] = pi_buy[t] * 0.7  # Revenue is ~70% of buy price
    return pi_buy.tolist(), pi_rev.tolist()


def run_priority_admission(ev_list: list, system_params: dict) -> dict:
    """Run Stage-I priority-based admission. Returns admission results with details."""
    lambda_scores, details = calc_priority_scores(
        ev_list,
        system_params['weights'],
        system_params.get('cdeg', 0.02),
        system_params.get('P_avg', 40.0),
        system_params.get('P_max', 100.0),
        system_params.get('pi_buy', 0.25),
        system_params.get('pi_rev', 0.18),
        system_params.get('pi_buy_min', 0.10),
        system_params.get('pi_buy_max', 0.50),
        system_params.get('pi_rev_min', 0.05),
        system_params.get('pi_rev_max', 0.30),
    )
    M = system_params.get('M', 30)
    admitted_ids = assign_chargers(lambda_scores, M)
    waiting_ids = [ev['id'] for ev in ev_list if ev['id'] not in admitted_ids]

    # Build detail records keyed by string ID for JSON
    detail_records = {}
    for ev in ev_list:
        eid = ev['id']
        d = details[eid]
        detail_records[str(eid)] = {
            'phi': round(d['phi'], 6),
            'Dfactor': round(d['Dfactor'], 6),
            'Gfactor': round(d['Gfactor'], 6),
            'Pfactor': round(d['Pfactor'], 6),
            'lambda': round(d['lambda'], 6),
            'p_req': round(d['p_req'], 4),
            'DeltaE': round(d['DeltaE'], 4),
        }

    return {
        'admitted_ids': admitted_ids,
        'waiting_ids': waiting_ids,
        'details': detail_records,
        'strategy': 'priority',
        'lambda_scores': {str(k): round(v, 6) for k, v in lambda_scores.items()},
    }


def run_fcfs_admission(ev_list: list, M: int) -> dict:
    """Run FCFS admission. Returns admitted/waiting IDs."""
    # Sort by arrival time, then by id
    sorted_evs = sorted(ev_list, key=lambda ev: (ev.get('T_arr_idx', 0), ev.get('id', 0)))
    admitted = sorted_evs[:M]
    waiting = sorted_evs[M:]
    return {
        'admitted_ids': [ev['id'] for ev in admitted],
        'waiting_ids': [ev['id'] for ev in waiting],
        'details': {},
        'strategy': 'fcfs',
    }


def run_sjf_admission(ev_list: list, M: int) -> dict:
    """Run SJF admission. Returns admitted/waiting IDs."""
    jobs = []
    for ev in ev_list:
        DeltaE = max(0.0, (ev['SoC_max'] - ev['SoC_init']) * ev['Ecap'])
        p_ref = ev.get('R_i', ev.get('P_ref', 7.0))
        if p_ref is None or p_ref <= 0:
            job_len = ev.get('T_stay', 1e9)
        else:
            job_len = 0.0 if DeltaE <= 1e-9 else DeltaE / p_ref
        jobs.append((ev, job_len))
    sorted_jobs = sorted(jobs, key=lambda x: (x[1], x[0].get('T_arr_idx', 0), x[0].get('id', 0)))
    admitted = [pair[0] for pair in sorted_jobs[:M]]
    waiting = [pair[0] for pair in sorted_jobs[M:]]

    details = {}
    for ev, jl in sorted_jobs:
        details[str(ev['id'])] = {'job_length_hours': round(jl, 4)}

    return {
        'admitted_ids': [ev['id'] for ev in admitted],
        'waiting_ids': [ev['id'] for ev in waiting],
        'details': details,
        'strategy': 'sjf',
    }


def prepare_evs_for_ga(admitted_evs: list, T: int, delta_t: float) -> list:
    """Prepare admitted EV dicts for GA (ensure T_dep_idx, P_ref are set)."""
    ga_evs = []
    for ev in admitted_evs:
        T_arr_idx = int(ev.get('T_arr_idx', 0))
        T_dep_idx = ev.get('T_dep_idx', None)
        if T_dep_idx is None:
            slots = max(1, int(math.ceil(ev.get('T_stay', 0.0) / delta_t)))
            T_dep_idx = min(T, T_arr_idx + slots)
        else:
            T_dep_idx = min(T, int(T_dep_idx))
        p_ref = ev.get('R_i', ev.get('P_ref', 7.0))
        if p_ref is None or p_ref <= 0:
            p_ref = 7.0
        ga_ev = {
            'id': ev['id'],
            'Ecap': ev['Ecap'],
            'SoC_init': ev['SoC_init'],
            'SoC_max': ev['SoC_max'],
            'SoC_min': ev.get('SoC_min', 0.0),
            'R_i': p_ref,
            'P_ref': p_ref,
            'P_dis_min': ev.get('P_dis_min', -p_ref),
            'T_stay': ev.get('T_stay', (T_dep_idx - T_arr_idx) * delta_t),
            'T_arr_idx': T_arr_idx,
            'T_dep_idx': T_dep_idx,
            'cdeg': ev.get('cdeg', 0.02),
        }
        ga_evs.append(ga_ev)
    return ga_evs


def compute_soc_evolution(schedule: np.ndarray, evs: list, T: int, delta_t: float) -> list:
    """Compute SoC evolution for each EV across all time slots."""
    M = len(evs)
    soc_matrix = []
    for i in range(M):
        ev = evs[i]
        Ecap = ev['Ecap']
        SoC = ev['SoC_init']
        soc_row = [SoC]
        Tarr = ev.get('T_arr_idx', 0)
        Tdep = ev.get('T_dep_idx', T)
        for t in range(T):
            if Tarr <= t < Tdep and Ecap > 0:
                p = schedule[i, t] if i < schedule.shape[0] else 0.0
                SoC += (p * delta_t) / Ecap
            soc_row.append(round(SoC, 6))
        soc_matrix.append(soc_row)
    return soc_matrix


def run_full_simulation(ev_list: list, system_params: dict, ga_params: dict, strategy: str = "priority") -> dict:
    """Run full Stage-I + Stage-II simulation."""
    M = system_params.get('M', 30)
    T = system_params.get('T', 48)
    delta_t = system_params.get('delta_t', 0.25)

    # Generate TOU prices
    pi_buy, pi_rev = generate_tou_prices(T, delta_t)

    # Stage-I: Admission
    if strategy == 'priority':
        admission = run_priority_admission(ev_list, system_params)
    elif strategy == 'fcfs':
        admission = run_fcfs_admission(ev_list, M)
    elif strategy == 'sjf':
        admission = run_sjf_admission(ev_list, M)
    else:
        admission = run_priority_admission(ev_list, system_params)

    admitted_ids = admission['admitted_ids']
    admitted_evs_raw = [ev for ev in ev_list if ev['id'] in admitted_ids]

    # Prepare for GA
    ga_evs = prepare_evs_for_ga(admitted_evs_raw, T, delta_t)

    if len(ga_evs) == 0:
        return {
            'admission': admission,
            'schedule': [],
            'metrics': {},
            'ga_history': [],
            'soc_evolution': [],
            'grid_load': [0.0] * T,
            'pi_buy': pi_buy,
            'pi_rev': pi_rev,
        }

    # Stage-II: GA Optimization
    weights_ga = {
        'w1': ga_params.get('w1', 0.25),
        'w2': ga_params.get('w2', 0.25),
        'w3': ga_params.get('w3', 0.25),
        'w4': ga_params.get('w4', 0.25),
    }

    random.seed(42)
    np.random.seed(42)

    result = _run_ga_priority(
        evs=ga_evs,
        T=T,
        delta_t=delta_t,
        pi_buy=np.array(pi_buy),
        pi_rev=np.array(pi_rev),
        P_max=system_params.get('P_max', 100.0),
        weights=weights_ga,
        alpha1=ga_params.get('alpha1', 50.0),
        alpha2=ga_params.get('alpha2', 50.0),
        alpha3=ga_params.get('alpha3', 50.0),
        pop_size=ga_params.get('pop_size', 60),
        ngen=ga_params.get('ngen', 50),
        cxpb=ga_params.get('cxpb', 0.9),
        mutpb=ga_params.get('mutpb', 0.3),
        eta_c=ga_params.get('eta_c', 20.0),
        eta_m=ga_params.get('eta_m', 20.0),
        tournament_size=ga_params.get('tournament_size', 3),
        stagnation_generations=ga_params.get('stagnation_generations', 40),
        seed_count=ga_params.get('seed_count', 10),
        elitism_k=ga_params.get('elitism_k', 2),
        num_chargers=M,
        system_params=system_params,
        verbose=False,
    )

    schedule = result['best_schedule']
    M_actual = schedule.shape[0]

    # Compute SoC evolution
    soc_evolution = compute_soc_evolution(schedule, ga_evs, T, delta_t)

    # Compute grid load
    grid_load = np.sum(schedule, axis=0).tolist()

    # Build EV schedule details
    ev_schedules = []
    for i in range(M_actual):
        ev = ga_evs[i]
        row = schedule[i, :].tolist()
        states = []
        for t in range(T):
            p = row[t]
            if abs(p) < 1e-6:
                states.append('idle')
            elif p > 0:
                states.append('charging')
            else:
                states.append('discharging')
        ev_schedules.append({
            'ev_id': ev['id'],
            'power': [round(x, 4) for x in row],
            'states': states,
            'soc': soc_evolution[i],
        })

    metrics = {
        'J': round(result.get('J', 0), 6),
        'F1': round(result.get('F1', 0), 4),
        'F2': round(result.get('F2', 0), 4),
        'F3': round(result.get('F3', 0), 4),
        'F4': round(result.get('F4', 0), 4),
        'F1_norm': round(result.get('F1_norm', 0) or 0, 6),
        'F2_norm': round(result.get('F2_norm', 0) or 0, 6),
        'F3_norm': round(result.get('F3_norm', 0) or 0, 6),
        'F4_norm': round(result.get('F4_norm', 0) or 0, 6),
        'V_SoC': round(result.get('V_SoC', 0), 4),
        'V_occ': round(result.get('V_occ', 0), 4),
        'V_grid': round(result.get('V_grid', 0), 4),
        'Omega_raw': round(result.get('Omega_raw', 0), 4),
        'generations_executed': result.get('generations_executed', 0),
        'avg_satisfaction': round(
            np.mean(result.get('Si_list', [0])) if result.get('Si_list') else 0, 4
        ),
    }

    return {
        'admission': admission,
        'ev_schedules': ev_schedules,
        'metrics': metrics,
        'ga_history': result.get('ga_history', []),
        'grid_load': [round(x, 4) for x in grid_load],
        'pi_buy': [round(x, 4) for x in pi_buy],
        'pi_rev': [round(x, 4) for x in pi_rev],
    }


def get_example_dataset() -> dict:
    """Return the paper's numerical example data."""
    example_evs = [
        {'id': 1, 'Ecap': 30, 'SoC_init': 0.2, 'SoC_max': 0.8, 'R_i': 7.0, 'P_ref': 7.0, 'P_dis_min': -7.0,
         'T_stay': 4.0, 'T_arr_idx': 0, 'T_dep_idx': 16, 'SoC_min': 0.0, 'cdeg': 0.02},
        {'id': 2, 'Ecap': 50, 'SoC_init': 0.3, 'SoC_max': 0.9, 'R_i': 11.0, 'P_ref': 11.0, 'P_dis_min': -11.0,
         'T_stay': 6.0, 'T_arr_idx': 4, 'T_dep_idx': 28, 'SoC_min': 0.0, 'cdeg': 0.02},
        {'id': 3, 'Ecap': 40, 'SoC_init': 0.15, 'SoC_max': 0.85, 'R_i': 7.0, 'P_ref': 7.0, 'P_dis_min': -7.0,
         'T_stay': 3.0, 'T_arr_idx': 8, 'T_dep_idx': 20, 'SoC_min': 0.0, 'cdeg': 0.02},
        {'id': 4, 'Ecap': 60, 'SoC_init': 0.5, 'SoC_max': 0.95, 'R_i': 22.0, 'P_ref': 22.0, 'P_dis_min': -22.0,
         'T_stay': 2.0, 'T_arr_idx': 0, 'T_dep_idx': 8, 'SoC_min': 0.0, 'cdeg': 0.015},
        {'id': 5, 'Ecap': 45, 'SoC_init': 0.25, 'SoC_max': 0.75, 'R_i': 7.0, 'P_ref': 7.0, 'P_dis_min': -7.0,
         'T_stay': 5.0, 'T_arr_idx': 12, 'T_dep_idx': 32, 'SoC_min': 0.0, 'cdeg': 0.025},
    ]
    system_params = {
        'M': 3, 'T': 48, 'delta_t': 0.25, 'P_max': 50.0, 'P_avg': 20.0,
        'cdeg': 0.02, 'pi_buy': 0.25, 'pi_rev': 0.18,
        'pi_buy_min': 0.10, 'pi_buy_max': 0.50,
        'pi_rev_min': 0.05, 'pi_rev_max': 0.30,
        'weights': {'w_s': 0.25, 'w_d': 0.25, 'w_g': 0.25, 'w_p': 0.25},
    }
    return {'evs': example_evs, 'system_params': system_params}
