"""
FastAPI server for the EV Scheduling Web Demo.
Provides REST endpoints for simulation, GA optimization, and data generation.
"""
import sys
import os

# Add parent project to path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    SimulationRequest, GARequest, FleetGenerationRequest,
    EVData, SystemParams, GAParams,
)
from engine import (
    ev_dict_from_model, generate_fleet, run_priority_admission,
    run_fcfs_admission, run_sjf_admission, run_full_simulation,
    get_example_dataset, generate_tou_prices,
)

app = FastAPI(
    title="EV Scheduling Framework API",
    description="Two-stage EV charge-discharge scheduling: Priority Scheduling + GA Optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "ev-scheduling-api"}


@app.get("/api/example")
async def get_example():
    """Returns the paper's numerical example dataset."""
    return get_example_dataset()


@app.post("/api/generate-fleet")
async def generate_fleet_endpoint(req: FleetGenerationRequest):
    """Generate a synthetic EV fleet."""
    try:
        evs = generate_fleet(req.n, seed=req.seed, T=req.T, delta_t=req.delta_t)
        return {"evs": evs, "count": len(evs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate/admission")
async def simulate_admission(req: SimulationRequest):
    """Run Stage-I admission using the specified strategy."""
    try:
        ev_list = [ev_dict_from_model(ev) for ev in req.evs]
        sp = req.system_params.model_dump()

        if req.strategy == 'priority':
            result = run_priority_admission(ev_list, sp)
        elif req.strategy == 'fcfs':
            result = run_fcfs_admission(ev_list, sp['M'])
        elif req.strategy == 'sjf':
            result = run_sjf_admission(ev_list, sp['M'])
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate/ga")
async def simulate_ga(req: GARequest):
    """Run full Stage-I + Stage-II (GA optimization) pipeline."""
    try:
        ev_list = [ev_dict_from_model(ev) for ev in req.evs]
        sp = req.system_params.model_dump()
        gp = req.ga_params.model_dump()

        result = run_full_simulation(ev_list, sp, gp, strategy=req.strategy)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/prices")
async def get_prices(T: int = 48, delta_t: float = 0.25):
    """Get Time-of-Use price schedule."""
    pi_buy, pi_rev = generate_tou_prices(T, delta_t)
    hours = [round(t * delta_t, 2) for t in range(T)]
    return {"pi_buy": pi_buy, "pi_rev": pi_rev, "hours": hours, "T": T}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
