"""
Pydantic models for API request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EVData(BaseModel):
    id: int
    Ecap: float = 40.0
    SoC_init: float = 0.2
    SoC_max: float = 0.8
    SoC_min: float = 0.0
    R_i: Optional[float] = 7.0
    P_ref: Optional[float] = None
    P_dis_min: Optional[float] = None
    T_stay: float = 4.0
    T_arr_idx: int = 0
    T_dep_idx: Optional[int] = None
    cdeg: float = 0.02


class SystemParams(BaseModel):
    M: int = Field(default=30, description="Number of chargers")
    T: int = Field(default=48, description="Number of time slots")
    delta_t: float = Field(default=0.25, description="Slot duration in hours")
    P_max: float = Field(default=100.0, description="Max grid power (kW)")
    P_avg: float = Field(default=40.0, description="Average grid power (kW)")
    cdeg: float = Field(default=0.02, description="Degradation coefficient")
    pi_buy: float = Field(default=0.25, description="Buy price ($/kWh)")
    pi_rev: float = Field(default=0.18, description="Revenue/sell price ($/kWh)")
    pi_buy_min: float = Field(default=0.10)
    pi_buy_max: float = Field(default=0.50)
    pi_rev_min: float = Field(default=0.05)
    pi_rev_max: float = Field(default=0.30)
    weights: Dict[str, float] = Field(
        default={"w_s": 0.25, "w_d": 0.25, "w_g": 0.25, "w_p": 0.25}
    )


class GAParams(BaseModel):
    pop_size: int = Field(default=60, description="GA population size")
    ngen: int = Field(default=50, description="Number of GA generations")
    cxpb: float = Field(default=0.9, description="Crossover probability")
    mutpb: float = Field(default=0.3, description="Mutation probability")
    eta_c: float = Field(default=20.0, description="SBX distribution index")
    eta_m: float = Field(default=20.0, description="Polynomial mutation distribution index")
    tournament_size: int = Field(default=3)
    stagnation_generations: int = Field(default=40)
    seed_count: int = Field(default=10)
    elitism_k: int = Field(default=2)
    w1: float = Field(default=0.25, description="Energy cost weight")
    w2: float = Field(default=0.25, description="Degradation weight")
    w3: float = Field(default=0.25, description="Grid variance weight")
    w4: float = Field(default=0.25, description="User satisfaction weight")
    alpha1: float = Field(default=50.0, description="SoC penalty coefficient")
    alpha2: float = Field(default=50.0, description="Occupancy penalty coefficient")
    alpha3: float = Field(default=50.0, description="Grid penalty coefficient")


class SimulationRequest(BaseModel):
    evs: List[EVData]
    system_params: SystemParams = SystemParams()
    strategy: str = Field(default="priority", description="Admission strategy: priority, fcfs, sjf")


class GARequest(BaseModel):
    evs: List[EVData]
    system_params: SystemParams = SystemParams()
    ga_params: GAParams = GAParams()
    strategy: str = Field(default="priority", description="Admission strategy")


class FleetGenerationRequest(BaseModel):
    n: int = Field(default=50, description="Number of EVs to generate")
    seed: int = Field(default=42)
    T: int = Field(default=48)
    delta_t: float = Field(default=0.25)


class AdmissionResult(BaseModel):
    admitted_ids: List[int]
    waiting_ids: List[int]
    details: Dict[str, Any] = {}
    strategy: str


class GAResult(BaseModel):
    schedule: List[List[float]]  # M x T matrix
    metrics: Dict[str, Any]
    ga_history: List[Dict[str, Any]]
    soc_evolution: List[List[float]]  # M x T SoC values
    grid_load: List[float]  # T aggregate load values
    admitted_ids: List[int]
    strategy: str
