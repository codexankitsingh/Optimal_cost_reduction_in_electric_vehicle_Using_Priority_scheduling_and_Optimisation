export default function DeveloperPage() {
    const endpoints = [
        {
            method: 'GET',
            path: '/api/health',
            desc: 'Health check',
            response: '{ "status": "ok", "service": "ev-scheduling-api" }',
        },
        {
            method: 'GET',
            path: '/api/example',
            desc: 'Returns the paper\'s numerical example dataset (5 EVs)',
            response: '{ "evs": [...], "system_params": {...} }',
        },
        {
            method: 'POST',
            path: '/api/generate-fleet',
            desc: 'Generate synthetic EV fleet with realistic distributions',
            body: '{ "n": 50, "seed": 42, "T": 48, "delta_t": 0.25 }',
            response: '{ "evs": [...], "count": 50 }',
        },
        {
            method: 'POST',
            path: '/api/simulate/admission',
            desc: 'Run Stage-I admission only (priority/FCFS/SJF)',
            body: '{ "evs": [...], "system_params": {...}, "strategy": "priority" }',
            response: '{ "admitted_ids": [1,2,...], "waiting_ids": [...], "details": {...} }',
        },
        {
            method: 'POST',
            path: '/api/simulate/ga',
            desc: 'Run full pipeline: Stage-I admission + Stage-II GA optimization',
            body: '{ "evs": [...], "system_params": {...}, "ga_params": {...}, "strategy": "priority" }',
            response: '{ "admission": {...}, "ev_schedules": [...], "metrics": {...}, "ga_history": [...], "grid_load": [...] }',
        },
        {
            method: 'GET',
            path: '/api/prices?T=48&delta_t=0.25',
            desc: 'Get Time-of-Use pricing schedule',
            response: '{ "pi_buy": [...], "pi_rev": [...], "hours": [...], "T": 48 }',
        },
    ]

    return (
        <div>
            <div className="section-header" style={{ marginBottom: 'var(--space-xl)' }}>
                <h1>API Documentation</h1>
                <p>REST API reference for the EV Scheduling backend. Base URL: <code style={{ color: 'var(--color-accent-light)' }}>http://localhost:8000</code></p>
            </div>

            {/* Quick Start */}
            <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                <div className="card-header"><h3>🚀 Quick Start</h3></div>
                <div className="code-block">
                    <span className="code-comment"># 1. Start the backend</span>{'\n'}
                    <span className="code-keyword">cd</span> web-demo/backend{'\n'}
                    pip install -r requirements.txt{'\n'}
                    python server.py{'\n\n'}
                    <span className="code-comment"># 2. Start the frontend (separate terminal)</span>{'\n'}
                    <span className="code-keyword">cd</span> web-demo/frontend{'\n'}
                    npm install{'\n'}
                    npm run dev{'\n\n'}
                    <span className="code-comment"># 3. Or use Docker</span>{'\n'}
                    <span className="code-keyword">cd</span> web-demo{'\n'}
                    docker-compose up
                </div>
            </div>

            {/* Endpoints */}
            <div className="section">
                <h2 style={{ marginBottom: 'var(--space-lg)' }}>Endpoints</h2>
                {endpoints.map((ep, i) => (
                    <div key={i} className="card" style={{ marginBottom: 'var(--space-md)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)', marginBottom: 'var(--space-md)' }}>
                            <span className={`badge ${ep.method === 'GET' ? 'badge-emerald' : 'badge-blue'}`}>{ep.method}</span>
                            <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--color-text-primary)', fontSize: '0.9375rem' }}>{ep.path}</code>
                        </div>
                        <p style={{ marginBottom: 'var(--space-md)', fontSize: '0.875rem' }}>{ep.desc}</p>

                        {ep.body && (
                            <div style={{ marginBottom: 'var(--space-sm)' }}>
                                <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Request Body</span>
                                <div className="code-block" style={{ marginTop: '4px', fontSize: '0.75rem' }}>{ep.body}</div>
                            </div>
                        )}
                        <div>
                            <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: 'var(--color-text-muted)', textTransform: 'uppercase' }}>Response</span>
                            <div className="code-block" style={{ marginTop: '4px', fontSize: '0.75rem' }}>{ep.response}</div>
                        </div>
                    </div>
                ))}
            </div>

            {/* Data Models */}
            <div className="section">
                <h2 style={{ marginBottom: 'var(--space-lg)' }}>Data Models</h2>
                <div className="card">
                    <div className="card-header"><h3>EVData</h3></div>
                    <table className="data-table">
                        <thead>
                            <tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr>
                        </thead>
                        <tbody>
                            {[
                                ['id', 'int', '-', 'Unique EV identifier'],
                                ['Ecap', 'float', '40.0', 'Battery capacity (kWh)'],
                                ['SoC_init', 'float', '0.2', 'Initial state-of-charge (0–1)'],
                                ['SoC_max', 'float', '0.8', 'Target max SoC'],
                                ['SoC_min', 'float', '0.0', 'Minimum allowed SoC'],
                                ['R_i', 'float', '7.0', 'Charger power rating (kW)'],
                                ['T_stay', 'float', '4.0', 'Duration of stay (hours)'],
                                ['T_arr_idx', 'int', '0', 'Arrival time slot index'],
                                ['T_dep_idx', 'int', 'null', 'Departure time slot index'],
                                ['cdeg', 'float', '0.02', 'Battery degradation coefficient'],
                            ].map(([f, t, d, desc], i) => (
                                <tr key={i}>
                                    <td style={{ color: 'var(--color-accent-light)' }}>{f}</td>
                                    <td><span className="badge badge-teal">{t}</span></td>
                                    <td>{d}</td>
                                    <td style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem' }}>{desc}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Architecture */}
            <div className="section">
                <h2 style={{ marginBottom: 'var(--space-lg)' }}>Architecture</h2>
                <div className="card">
                    <div className="code-block">
                        ┌─────────────────────────────────────────────────────────┐{'\n'}
                        │                    Frontend (React + Vite)              │{'\n'}
                        │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │{'\n'}
                        │  │   Home   │ │Algorithm │ │ Studio   │ │ API Docs │  │{'\n'}
                        │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │{'\n'}
                        │        │              │           │                     │{'\n'}
                        │        └──────────────┴───────────┘                    │{'\n'}
                        │                      │ HTTP REST                       │{'\n'}
                        └───────────────────── │ ────────────────────────────────┘{'\n'}
                        {'                       ▼'}{'\n'}
                        ┌─────────────────────────────────────────────────────────┐{'\n'}
                        │                  Backend (FastAPI)                       │{'\n'}
                        │  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐   │{'\n'}
                        │  │  server.py   │ │  engine.py   │ │  models.py    │   │{'\n'}
                        │  │  (REST API)  │ │  (wrapper)   │ │  (Pydantic)   │   │{'\n'}
                        │  └──────────────┘ └──────────────┘ └───────────────┘   │{'\n'}
                        │                        │ imports                        │{'\n'}
                        └──────────────────── │ ──────────────────────────────────┘{'\n'}
                        {'                       ▼'}{'\n'}
                        ┌─────────────────────────────────────────────────────────┐{'\n'}
                        │            Core Algorithms (Parent Project)              │{'\n'}
                        │  ┌────────────┐ ┌──────────────┐ ┌──────────────┐      │{'\n'}
                        │  │ pipeline.py│ │pipeline_fcfs │ │ pipeline_sjf │      │{'\n'}
                        │  │ (Priority) │ │   (FCFS)     │ │   (SJF)      │      │{'\n'}
                        │  └────────────┘ └──────────────┘ └──────────────┘      │{'\n'}
                        └─────────────────────────────────────────────────────────┘
                    </div>
                </div>
            </div>
        </div>
    )
}
