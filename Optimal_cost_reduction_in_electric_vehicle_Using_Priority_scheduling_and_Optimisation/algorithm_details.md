# Algorithm Implementation Details: Standard Genetic Algorithm

## 1. Algorithm Identification
The algorithm implemented in `pipeline.py` is a **Standard Genetic Algorithm (SGA)** utilising a **Weighted Sum Approach** for multi-objective optimization.

**It is NOT NSGA-II.**

### Why it is not NSGA-II:
- **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** is a multi-objective optimization algorithm that:
  1. Maintains a **Pareto Front** of non-dominated solutions.
  2. Uses **Fast Non-Dominated Sorting** to rank individuals.
  3. Uses **Crowding Distance** to maintain diversity.
  4. Returns a set of trade-off solutions (Pareto set) rather than a single best solution.

- **The Implemented Algorithm**:
  1. Combines multiple objectives (F1, F2, F3, F4) into a **single scalar fitness value** (J) using weighted summation.
  2. Uses standard **Tournament Selection** on this single fitness value.
  3. Returns a **single best solution** that minimizes the weighted aggregate cost function.

---

## 2. Step-by-Step Implementation Guide

### Step 1: Input & Initialization
**Input**:
- List of EVs with parameters (Capacity, Initial SoC, Max SoC, Arrival Time, Departure Time, Max Power).
- System constraints (Max Grid Power, Number of Chargers M).
- Price signals (Buy Price, Sell Price).

**Genome Representation**:
- A real-valued vector of size M x T.
- `individual[i * T + t]` represents the charging power (kW) of EV `i` at time slot `t`.

**Population Initialization**:
- **Random Initialization**: Most individuals are initialized with random power values within bounds.
- **Seeding (Hybrid)**: A fraction of the population is "seeded" with good heuristic solutions to speed up convergence.
  - *Greedy Seed*: Allocates power to the cheapest time slots first until energy demand is met.
  - *Stage-I Seed*: Uses the priority scheduling result as a starting baseline.

### Step 2: Fitness Function (Evaluation)
The core of the algorithm is the single-objective fitness function `J`.

**Formula**:
```
J = w1 * F1_norm + w2 * F2_norm + w3 * F3_norm - w4 * F4_norm + Omega_norm
```

Where `_norm` indicates normalized values.

**Objectives Details**:

1. **Energy Cost (F1)**:
   Measures the net cost of electricity (buying at high price vs selling at low price).
   ```
   F1 = Sum(t=1 to T) Sum(i=1 to M) Cost(P_i_t)
   ```

2. **Battery Degradation (F2)**:
   Penalizes heavy battery usage to extend life.
   ```
   F2 = Sum(t=1 to T) Sum(i=1 to M) [ C_deg_i * abs(P_i_t) * delta_t ]
   ```

3. **Grid Load Variance (F3)**:
   Smoothens the grid load curve (Peak Shaving).
   ```
   F3 = Sum(t=1 to T) (L_t - L_avg)^2
   ```

4. **User Satisfaction (F4)**:
   Maximizes the ratio of delivered energy to requested energy.
   ```
   F4 = Sum(i=1 to M) S_i
   ```

**Constraints & Penalties (Omega)**:
```
Omega = alpha1 * V_SoC + alpha2 * V_occ + alpha3 * V_grid
```
- **V_SoC**: Penalty for violating min/max SoC limits.
- **V_occ**: Penalty if more than `M` chargers are used at a time.
- **V_grid**: Penalty if total load exceeds transformer limit `P_max`.

### Step 3: Selection
**Mechanism**: **Tournament Selection**
- Randomly select `k` individuals (default k=3) from the population.
- Choose the one with the lowest fitness value `J`.
- Repeat until the mating pool is full.

### Step 4: Crossover (Recombination)
**Operator**: **Simulated Binary Crossover (SBX)**
- **Probability**: 0.9 (90%)
- **Process**: Two parent vectors are blended to create two offspring.
- **Behavior**: Creates offspring near the parents (exploitation) but can expand the search space (exploration).

### Step 5: Mutation
**Operator**: **Polynomial Mutation**
- **Probability**: 0.4 (40%) per individual.
- **Process**: Adds polynomial noise to a gene.
- **Per-Gene Probability**: 1/T (ensures approx. 1 gene per EV is mutated on average).
- **Purpose**: Introduces small random variations to prevent getting stuck in local optima.

### Step 6: Repair Mechanism
Before evaluation, a **Repair Function** is applied to ensure feasibility:
1.  **Clamping**: Forces power values to be within [P_min, P_max].
2.  **Availability**: Forces power to 0 if the EV is not physically present at the station.
3.  **Charger Constraint**: If > M EVs are active at time t, the algorithm:
    - Calculates priority scores (from Stage-I logic).
    - Keeps the top M priority EVs.
    - Forces the others to 0 power for that slot.

### Step 7: Elitism and Replacement
- **Elitism**: The top `k` (e.g., 2-5) individuals from the previous generation are **always** copied to the next generation.
- **Replacement**: The rest of the new population is formed by the offspring from crossover and mutation.

## 3. Summary of Parameters Used
| Parameter | Value | Description |
|-----------|-------|-------------|
| Population Size | 120 | Number of solutions in each generation |
| Generations | 300 | Maximum iterations |
| Crossover Prob | 0.9 | High crossover for mixing traits |
| Mutation Prob | 0.4 | Moderate mutation for diversity |
| Weights | 0.25 | Equal importance to all objectives |
| Penalty Alphas | 50.0 | High penalty to enforce value constraints |

## 4. Execution Flow
1.  **Start**: Initialize population (Random + Seeds).
2.  **Repair**: Apply bound and occupancy constraints to all individuals.
3.  **Evaluate**: Calculate `J` for all individuals.
4.  **Log**: Record best and average fitness.
5.  **Check Stop Condition**: (Max Gen or Stagnation).
    - **Yes**: Return Best Solution.
    - **No**: Continue.
6.  **Select**: Tournament selection.
7.  **Mate**: Apply SBX crossover.
8.  **Mutate**: Apply Polynomial mutation.
9.  **Elitism**: Inject best individuals from previous generation.
10. **Loop**: Go to Step 2.

---

## 5. Illustrative Example (Tiny Simulation)

To understand the process, let's run a **Tiny Simulation** with simple numbers.

### Scenario Setup
*   **Time (T)**: 4 slots (15 mins each) -> Total 1 Hour.
*   **Chargers (M)**: 2 Chargers.
*   **EV Pool**: 3 EVs (A, B, C).
*   **Grid Limit**: 20 kW.

**EV Data**:
| EV | Arrival (Slot) | Depart (Slot) | Stay | Energy Req (kWh) | P_max (kW) |
|----|---------------|--------------|------|------------------|------------|
| **A** | 0 | 4 | 4 | 10 kWh | 7.0 |
| **B** | 0 | 3 | 3 | 5 kWh | 7.0 |
| **C** | 1 | 3 | 2 | 10 kWh | 22.0 (Fast) |

**Required Power Calculation**:
*   `P_req` = Energy / (Stay * 0.25h)
*   **EV A**: 10 / (4 * 0.25) = 10 kW. (But P_max is 7, so constrained to **7 kW**).
*   **EV B**: 5 / (3 * 0.25) = 6.67 kW.
*   **EV C**: 10 / (2 * 0.25) = 20 kW. (Urgent!)

---

### Phase 1: Stage-I Priority Scheduling
The system decides *who gets to plug in* at each slot.

#### Time Slot t=0:
*   **Present**: EVs A, B.
*   **Chargers**: 2 available.
*   **Action**: Both A and B are admitted. Assumed charging at full/requested power.
    *   Charger 1 -> EV A
    *   Charger 2 -> EV B

#### Time Slot t=1:
*   **Present**: EVs A, B, C.
*   **Chargers**: 2 available. **Contention!** (3 EVs > 2 Chargers).
*   **Priority Calculation** (simplified):
    *   **EV C**: Needs 20 kW. Extremely urgent. Priority **High**.
    *   **EV A**: Needs ~7 kW. Priority **Medium**.
    *   **EV B**: Needs ~6 kW. Priority **Low**.
*   **Decision**:
    *   **EV C** takes a spot (High Priority).
    *   **EV A** keeps a spot (Medium Priority).
    *   **EV B** is **disconnected** (Low Priority) -> Penalty for switching/waiting.
*   **Result**: Chargers: [C, A]. Waiting: [B].

#### Time Slot t=2:
*   **Present**: EVs A, B, C.
*   **Situation**: Same as t=1.
*   **Result**: Chargers: [C, A]. Waiting: [B].

#### Time Slot t=3:
*   **Present**: EV A (B and C departed).
*   **Result**: Charger 1 -> EV A.

---

### Phase 2: Stage-II Genetic Algorithm
Now the GA optimizes the **exact power levels** (kW) for the admitted EVs to minimize cost and grid stress.

**Genome Representation**:
A matrix of size [M=2, T=4].
Let's see a random **Individual X** generated by the GA:

| Slot | Charger 1 (EV A / C) | Charger 2 (EV B / A) |
|------|-----------------------|----------------------|
| t=0 | 5.0 kW (A) | 6.0 kW (B) |
| t=1 | 18.0 kW (C)* | 2.0 kW (A) |
| t=2 | 15.0 kW (C)* | 3.0 kW (A) |
| t=3 | 6.0 kW (A) | 0.0 kW |

*(Note: The GA maps "Charger ID" to "EV ID" based on Stage-I assignment)*

**Evaluation of Individual X**:

1.  **Check Constraints**:
    *   **Total Load at t=1**: 18 + 2 = 20 kW. (<= Limit 20kW). **OK**.
    *   **EV C Energy**: (18 + 15) * 0.25 = **8.25 kWh**.
        *   Target: 10 kWh.
        *   **Deficit**: 1.75 kWh.
        *   **Penalty**: High penalty added to fitness `J`.

2.  **Calculate Cost (F1)**:
    *   Sum of (Power * Price) for all slots.
    *   If `t=1` is expensive (Peak Hour), the 18kW load hurts the score.

3.  **Result**:
    *   This individual has a **bad fitness** (High `J`) because EV C didn't get enough energy.

**Evolution (Next Generation)**:
*   **Mutation**: Might change `t=1` power for C from 18.0 -> 21.0 (closer to target).
*   but 21.0 + 2.0 > 20.0 (Grid Limit).
*   **Repair**: The repair function clamps it down or reduces A's power to fit C.
    *   New: C=20.0, A=0.0. (Total 20).
*   **Re-Evaluate**: Now EV C meets target (20+20)*0.25 = 10kWh. Penalty drops. Fitness improves!

**Final Output**:
After 300 generations, the best plan minimizes Cost + Degradation while satisfying the Energy Targets (penalties -> 0).
