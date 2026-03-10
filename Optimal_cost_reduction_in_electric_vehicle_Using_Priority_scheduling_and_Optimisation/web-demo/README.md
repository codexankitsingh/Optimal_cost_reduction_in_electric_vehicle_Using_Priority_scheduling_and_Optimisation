# EV Scheduling Web Demo

Interactive web application demonstrating the **Two-Stage EV Charge-Discharge Scheduling Framework** — combining Priority Scheduling with Genetic Algorithm optimization.

## 🚀 Quick Start (Local Development)

### Prerequisites
- **Python 3.9+** (with pip)
- **Node.js 18+** (with npm)

### 1️⃣ Start the Backend

```bash
cd web-demo/backend
pip install -r requirements.txt
python server.py
```
Backend runs at **http://localhost:8000** — API docs at **/docs**

### 2️⃣ Start the Frontend (separate terminal)

```bash
cd web-demo/frontend
npm install
npm run dev
```
Frontend runs at **http://localhost:5173**

### 🐳 Docker Alternative

```bash
cd web-demo
docker-compose up --build
```

## 📂 Architecture

```
web-demo/
├── backend/
│   ├── server.py         # FastAPI REST API
│   ├── engine.py         # Algorithm wrapper (imports from parent project)
│   ├── models.py         # Pydantic request/response models
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx          # Project overview
│   │   │   ├── AlgorithmsPage.jsx    # Algorithm pseudocode & equations
│   │   │   ├── SimulationStudio.jsx  # Interactive simulation tool
│   │   │   └── DeveloperPage.jsx     # API documentation
│   │   ├── App.jsx       # React Router setup
│   │   └── index.css     # Design system (dark theme)
│   └── package.json
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/example` | Paper's numerical example dataset |
| POST | `/api/generate-fleet` | Generate synthetic EV fleet |
| POST | `/api/simulate/admission` | Run Stage-I admission only |
| POST | `/api/simulate/ga` | Run full Stage-I + Stage-II pipeline |
| GET | `/api/prices` | Get TOU pricing schedule |

## ✨ Features

- **Simulation Studio**: Configure fleet, strategy, GA params; visualize with 6 interactive charts
- **Three Strategies**: Priority, FCFS, SJF admission comparison
- **Real-time Plots**: Schedule heatmap, SoC evolution, grid load, cost breakdown, GA convergence
- **Algorithm Documentation**: Interactive pseudocode with equation breakdowns
- **V2G Support**: Bidirectional charging visualization

## 🔬 Algorithms (from Parent Project)

The backend imports directly from the parent project's Python modules:
- `pipeline.py` — Priority scheduling + GA
- `pipeline_fcfs.py` — FCFS admission + GA
- `pipeline_sjf.py` — SJF admission + GA
- `generate_ev_datasets.py` — Realistic EV fleet generator

## 📝 Assumptions & Notes

- GA defaults for web: `pop_size=60, ngen=50` (configurable in UI)
- Time-of-Use prices are auto-generated with realistic bimodal pattern
- Frontend uses Plotly.js for interactive, exportable charts
