import { useState, useCallback } from 'react'
import Plot from 'react-plotly.js'

const API_BASE = 'http://localhost:8000'

const DEFAULT_PARAMS = {
    n_evs: 10,
    chargers: 5,
    T: 96,
    delta_t: 0.25,
    P_max: 100,
    strategy: 'priority',
    pop_size: 60,
    ngen: 50,
    w1: 0.25,
    w2: 0.25,
    w3: 0.25,
    w4: 0.25,
}

const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(10,14,26,0.6)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 11 },
    margin: { l: 60, r: 30, t: 50, b: 60 },
    xaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.15)', automargin: true },
    yaxis: { gridcolor: 'rgba(148,163,184,0.08)', zerolinecolor: 'rgba(148,163,184,0.15)', automargin: true },
}

const PLOTLY_CONFIG = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }

export default function SimulationStudio() {
    const [params, setParams] = useState(DEFAULT_PARAMS)
    const [loading, setLoading] = useState(false)
    const [status, setStatus] = useState('')
    const [result, setResult] = useState(null)
    const [fleetData, setFleetData] = useState(null)
    const [activeTab, setActiveTab] = useState('timeline')

    const updateParam = useCallback((key, value) => {
        setParams(prev => ({ ...prev, [key]: value }))
    }, [])

    const handleFileUpload = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (evt) => {
            try {
                const text = evt.target.result;
                const lines = text.split('\n').filter(l => l.trim().length > 0);
                if (lines.length < 2) throw new Error("CSV must have headers and data");
                const headers = lines[0].split(',').map(h => h.trim());
                const evs = [];
                for (let i = 1; i < lines.length; i++) {
                    const vals = lines[i].split(',').map(v => v.trim());
                    if (vals.length !== headers.length) continue;
                    const ev = {};
                    headers.forEach((h, idx) => {
                        ev[h] = parseFloat(vals[idx]);
                    });
                    evs.push(ev);
                }
                setFleetData(evs);
                setStatus(`Loaded ${evs.length} EVs from CSV. Please adjust Chargers and System Parameters appropriately.`);
            } catch (err) {
                setStatus(`❌ Error reading CSV: ${err.message}`);
            }
        };
        reader.readAsText(file);
    };

    const downloadCSV = useCallback(() => {
        if (!fleetData || fleetData.length === 0) return;
        const headers = ["id", "Ecap", "SoC_init", "SoC_max", "SoC_min", "R_i", "P_ref", "P_dis_min", "T_stay", "T_arr_idx", "T_dep_idx", "cdeg"];
        const rows = fleetData.map(ev => {
            return headers.map(h => {
                let val = ev[h];
                if (val === undefined || val === null) {
                    if (h === 'SoC_min') val = 0.05;
                    else if (h === 'R_i' || h === 'P_ref') val = 7.0;
                    else if (h === 'P_dis_min') val = -7.0;
                    else if (h === 'cdeg') val = 0.02;
                    else if (h === 'T_dep_idx') val = (ev.T_arr_idx || 0) + Math.ceil((ev.T_stay || 4.0) / params.delta_t);
                    else val = '';
                }
                return val;
            }).join(',');
        });
        const csvContent = "data:text/csv;charset=utf-8," + headers.join(',') + "\n" + rows.join('\n');
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `ev_fleet_${fleetData.length}_${new Date().getTime()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }, [fleetData, params.delta_t]);

    const generateFleet = useCallback(async () => {
        setStatus('Generating EV fleet...')
        try {
            const res = await fetch(`${API_BASE}/api/generate-fleet`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n: params.n_evs, seed: 42, T: params.T, delta_t: params.delta_t }),
            })
            const data = await res.json()
            setFleetData(data.evs)
            setStatus(`Fleet generated: ${data.count} EVs`)
        } catch (e) {
            setStatus(`Error: ${e.message}. Is the backend running?`)
        }
    }, [params])

    const runSimulation = useCallback(async () => {
        if (!fleetData) {
            await generateFleet()
        }

        const evs = fleetData || []
        if (evs.length === 0) {
            setStatus('No EVs loaded. Generate a fleet first.')
            return
        }

        setLoading(true)
        setStatus('Running simulation (Stage I + GA optimization)...')

        try {
            const reqBody = {
                evs: evs.map(ev => ({
                    id: ev.id,
                    Ecap: ev.Ecap,
                    SoC_init: ev.SoC_init,
                    SoC_max: ev.SoC_max,
                    SoC_min: ev.SoC_min !== undefined ? ev.SoC_min : 0.05,
                    R_i: ev.R_i || ev.P_ref || 7.0,
                    P_ref: ev.P_ref || ev.R_i || 7.0,
                    P_dis_min: ev.P_dis_min !== undefined ? ev.P_dis_min : -(ev.R_i || ev.P_ref || 7.0),
                    T_stay: ev.T_stay,
                    T_arr_idx: ev.T_arr_idx,
                    T_dep_idx: ev.T_dep_idx || (ev.T_arr_idx + Math.ceil((ev.T_stay || 4.0) / params.delta_t)),
                    cdeg: ev.cdeg !== undefined ? ev.cdeg : 0.02,
                })),
                system_params: {
                    M: params.chargers,
                    T: params.T,
                    delta_t: params.delta_t,
                    P_max: params.P_max,
                    P_avg: params.P_max * 0.4,
                    cdeg: 0.02,
                    pi_buy: 0.25,
                    pi_rev: 0.18,
                    pi_buy_min: 0.10,
                    pi_buy_max: 0.50,
                    pi_rev_min: 0.05,
                    pi_rev_max: 0.30,
                    weights: { w_s: 0.25, w_d: 0.25, w_g: 0.25, w_p: 0.25 },
                },
                ga_params: {
                    pop_size: params.pop_size,
                    ngen: params.ngen,
                    w1: params.w1,
                    w2: params.w2,
                    w3: params.w3,
                    w4: params.w4,
                    cxpb: 0.9,
                    mutpb: 0.3,
                    eta_c: 20.0,
                    eta_m: 20.0,
                    tournament_size: 3,
                    stagnation_generations: 40,
                    seed_count: 10,
                    elitism_k: 2,
                    alpha1: 50.0,
                    alpha2: 50.0,
                    alpha3: 50.0,
                },
                strategy: params.strategy,
            }

            const res = await fetch(`${API_BASE}/api/simulate/ga`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody),
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.detail || 'Simulation failed')
            }

            const data = await res.json()
            setResult(data)
            setStatus(`✅ Simulation complete — ${data.metrics?.generations_executed || 0} generations, J = ${data.metrics?.J?.toFixed(6) || 'N/A'}`)
        } catch (e) {
            setStatus(`❌ Error: ${e.message}`)
        } finally {
            setLoading(false)
        }
    }, [fleetData, params, generateFleet])

    // ── Chart Builders ──
    const buildTimelineData = () => {
        if (!result?.ev_schedules) return null
        const schedules = result.ev_schedules
        const T = params.T
        const hours = Array.from({ length: T }, (_, t) => (t * params.delta_t).toFixed(2))

        const zData = schedules.map(s => s.power)
        const yLabels = schedules.map(s => `EV ${s.ev_id}`)

        return {
            data: [{
                z: zData,
                x: hours,
                y: yLabels,
                type: 'heatmap',
                colorscale: [
                    [0, '#f43f5e'],
                    [0.35, '#1e293b'],
                    [0.5, '#1e293b'],
                    [0.65, '#0ea5e9'],
                    [1, '#10b981']
                ],
                zmid: 0,
                colorbar: { title: 'Power (kW)', titlefont: { color: '#94a3b8' }, tickfont: { color: '#94a3b8' } },
                hovertemplate: '%{y}<br>Hour: %{x}h<br>Power: %{z:.2f} kW<extra></extra>',
            }],
            layout: {
                ...PLOTLY_LAYOUT_BASE,
                title: { text: 'EV Charging Schedule (Heatmap)', font: { color: '#f1f5f9', size: 14 } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 'Time (hours)', font: { size: 12, color: '#e2e8f0' } } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'Electric Vehicles (EVs)', font: { size: 12, color: '#e2e8f0' } } },
                height: Math.max(300, schedules.length * 28 + 100),
            },
        }
    }

    const buildSoCPlot = () => {
        if (!result?.ev_schedules) return null
        const traces = result.ev_schedules.map((s, idx) => {
            const hours = s.soc.map((_, t) => (t * params.delta_t).toFixed(2))
            const colors = ['#3b82f6', '#14b8a6', '#f59e0b', '#f43f5e', '#a855f7', '#10b981', '#0ea5e9', '#ec4899', '#6366f1', '#84cc16']
            return {
                x: hours,
                y: s.soc,
                name: `EV ${s.ev_id}`,
                type: 'scatter',
                mode: 'lines',
                line: { color: colors[idx % colors.length], width: 2 },
                hovertemplate: `EV ${s.ev_id}<br>Hour: %{x}h<br>SoC: %{y:.3f}<extra></extra>`,
            }
        })
        return {
            data: traces,
            layout: {
                ...PLOTLY_LAYOUT_BASE,
                title: { text: 'State of Charge Evolution', font: { color: '#f1f5f9', size: 14 } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 'Time (hours)', font: { size: 12, color: '#e2e8f0' } } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'State of Charge (SoC)', font: { size: 12, color: '#e2e8f0' } }, range: [0, 1] },
                height: 400,
                showlegend: true,
                legend: { font: { color: '#94a3b8', size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
            },
        }
    }

    const buildGridLoadPlot = () => {
        if (!result?.grid_load) return null
        const hours = result.grid_load.map((_, t) => (t * params.delta_t).toFixed(2))
        return {
            data: [
                {
                    x: hours,
                    y: result.grid_load,
                    type: 'scatter',
                    mode: 'lines',
                    fill: 'tozeroy',
                    fillcolor: 'rgba(59, 130, 246, 0.15)',
                    line: { color: '#3b82f6', width: 2 },
                    name: 'Aggregate Load',
                    hovertemplate: 'Hour: %{x}h<br>Load: %{y:.2f} kW<extra></extra>',
                },
                {
                    x: hours,
                    y: Array(hours.length).fill(params.P_max),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#f43f5e', width: 2, dash: 'dash' },
                    name: `P_max (${params.P_max} kW)`,
                },
                ...(result.pi_buy ? [{
                    x: hours,
                    y: result.pi_buy,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#f59e0b', width: 2, dash: 'dot' },
                    name: 'Price Signal ($/kWh)',
                    yaxis: 'y2',
                    opacity: 0.8,
                }] : []),
            ],
            layout: {
                ...PLOTLY_LAYOUT_BASE,
                title: { text: 'Aggregate Grid Load', font: { color: '#f1f5f9', size: 14 } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 'Time (hours)', font: { size: 12, color: '#e2e8f0' } } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'Power Load (kW)', font: { size: 12, color: '#e2e8f0' } } },
                yaxis2: { overlaying: 'y', side: 'right', showgrid: false, title: { text: 'Price ($/kWh)', font: { color: '#f59e0b' } }, tickfont: { color: '#f59e0b' } },
                height: 380,
                showlegend: true,
                legend: { font: { color: '#94a3b8', size: 10 }, bgcolor: 'rgba(0,0,0,0)', orientation: 'h', y: -0.15 },
            },
        }
    }

    const buildCostBreakdown = () => {
        if (!result?.metrics) return null
        const m = result.metrics
        return {
            data: [{
                x: ['Cost (w₁·F₁\')', 'Deg. (w₂·F₂\')', 'Grid (w₃·F₃\')', 'Sat. (w₄·F₄\')'],
                y: [
                    (m.F1_norm || 0) * params.w1,
                    (m.F2_norm || 0) * params.w2,
                    (m.F3_norm || 0) * params.w3,
                    (m.F4_norm || 0) * params.w4
                ],
                text: [
                    (m.F1_norm || 0).toFixed(4),
                    (m.F2_norm || 0).toFixed(4),
                    (m.F3_norm || 0).toFixed(4),
                    (m.F4_norm || 0).toFixed(4)
                ],
                textposition: 'auto',
                type: 'bar',
                marker: {
                    color: ['#3b82f6', '#f59e0b', '#f43f5e', '#10b981'],
                    line: { color: 'rgba(255,255,255,0.1)', width: 1 },
                },
                hovertemplate: '%{x}<br>Weighted Norm Value: %{y:.4f}<extra></extra>',
            }],
            layout: {
                ...PLOTLY_LAYOUT_BASE,
                title: { text: 'Weighted Normalized Cost Breakdown', font: { color: '#f1f5f9', size: 14 } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 'Objective Functions', font: { size: 12, color: '#e2e8f0' } } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'Weighted Normalized Contribution', font: { size: 12, color: '#e2e8f0' } } },
                height: 350,
            },
        }
    }

    const buildGAConvergence = () => {
        if (!result?.ga_history?.length) return null
        const gens = result.ga_history.map(h => h.gen)
        const bestJ = result.ga_history.map(h => h.best_J)
        const meanJ = result.ga_history.map(h => h.mean_J)
        return {
            data: [
                {
                    x: gens, y: bestJ, name: 'Best J', type: 'scatter', mode: 'lines',
                    line: { color: '#10b981', width: 2.5 },
                },
                {
                    x: gens, y: meanJ, name: 'Mean J', type: 'scatter', mode: 'lines',
                    line: { color: '#3b82f6', width: 1.5, dash: 'dot' },
                },
            ],
            layout: {
                ...PLOTLY_LAYOUT_BASE,
                title: { text: 'GA Convergence', font: { color: '#f1f5f9', size: 14 } },
                xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 'Generation Number', font: { size: 12, color: '#e2e8f0' } } },
                yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'Fitness Value (J)', font: { size: 12, color: '#e2e8f0' } } },
                height: 350,
                showlegend: true,
                legend: { font: { color: '#94a3b8' }, bgcolor: 'rgba(0,0,0,0)' },
            },
        }
    }

    const chartTabs = [
        { id: 'timeline', label: '📊 Schedule' },
        { id: 'soc', label: '🔋 SoC' },
        { id: 'grid', label: '⚡ Grid Load' },
        { id: 'cost', label: '💰 Cost' },
        { id: 'convergence', label: '🧬 GA Convergence' },
        { id: 'priority', label: '📋 Admission' },
    ]

    return (
        <div>
            <div className="section-header" style={{ marginBottom: 'var(--space-lg)' }}>
                <h1>Simulation Studio</h1>
                <p>Configure, simulate, and visualize EV charging schedule optimization.</p>
            </div>

            <div className="studio-layout">
                {/* ── Sidebar ── */}
                <div className="studio-sidebar">
                    <div className="card">
                        {/* Fleet Config */}
                        <div className="param-section">
                            <div className="param-section-title">Fleet Configuration</div>
                            <div className="param-row">
                                <div className="input-group">
                                    <label>Number of EVs</label>
                                    <input type="number" min="2" max="100" value={params.n_evs}
                                        onChange={e => updateParam('n_evs', parseInt(e.target.value) || 10)} />
                                </div>
                                <div className="input-group">
                                    <label>Chargers (M)</label>
                                    <input type="number" min="1" max="50" value={params.chargers}
                                        onChange={e => updateParam('chargers', parseInt(e.target.value) || 5)} />
                                </div>
                            </div>
                        </div>

                        {/* System */}
                        <div className="param-section">
                            <div className="param-section-title">System Parameters</div>
                            <div className="param-row">
                                <div className="input-group">
                                    <label>Time Slots (T)</label>
                                    <input type="number" min="8" max="96" value={params.T}
                                        onChange={e => updateParam('T', parseInt(e.target.value) || 48)} />
                                </div>
                                <div className="input-group">
                                    <label>Grid P_max (kW)</label>
                                    <input type="number" min="10" value={params.P_max}
                                        onChange={e => updateParam('P_max', parseFloat(e.target.value) || 100)} />
                                </div>
                            </div>
                        </div>

                        {/* Strategy */}
                        <div className="param-section">
                            <div className="param-section-title">Admission Strategy</div>
                            <div className="input-group">
                                <label>Strategy</label>
                                <select value={params.strategy} onChange={e => updateParam('strategy', e.target.value)}>
                                    <option value="priority">Priority Scheduling</option>
                                    <option value="fcfs">FCFS (First-Come-First-Served)</option>
                                    <option value="sjf">SJF (Shortest-Job-First)</option>
                                </select>
                            </div>
                        </div>

                        {/* GA */}
                        <div className="param-section">
                            <div className="param-section-title">GA Hyperparameters</div>
                            <div className="param-row">
                                <div className="input-group">
                                    <label>Population</label>
                                    <input type="number" min="20" max="300" value={params.pop_size}
                                        onChange={e => updateParam('pop_size', parseInt(e.target.value) || 60)} />
                                </div>
                                <div className="input-group">
                                    <label>Generations</label>
                                    <input type="number" min="10" max="500" value={params.ngen}
                                        onChange={e => updateParam('ngen', parseInt(e.target.value) || 50)} />
                                </div>
                            </div>
                        </div>

                        {/* Weights */}
                        <div className="param-section">
                            <div className="param-section-title">Objective Weights</div>
                            <div className="param-row">
                                <div className="input-group">
                                    <label>w₁ (Cost)</label>
                                    <input type="number" min="0" max="1" step="0.05" value={params.w1}
                                        onChange={e => updateParam('w1', parseFloat(e.target.value) || 0.25)} />
                                </div>
                                <div className="input-group">
                                    <label>w₂ (Degrad.)</label>
                                    <input type="number" min="0" max="1" step="0.05" value={params.w2}
                                        onChange={e => updateParam('w2', parseFloat(e.target.value) || 0.25)} />
                                </div>
                            </div>
                            <div className="param-row" style={{ marginTop: 'var(--space-sm)' }}>
                                <div className="input-group">
                                    <label>w₃ (Variance)</label>
                                    <input type="number" min="0" max="1" step="0.05" value={params.w3}
                                        onChange={e => updateParam('w3', parseFloat(e.target.value) || 0.25)} />
                                </div>
                                <div className="input-group">
                                    <label>w₄ (Satisf.)</label>
                                    <input type="number" min="0" max="1" step="0.05" value={params.w4}
                                        onChange={e => updateParam('w4', parseFloat(e.target.value) || 0.25)} />
                                </div>
                            </div>
                        </div>

                        {/* Actions */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-sm)', marginTop: 'var(--space-sm)' }}>
                            <div style={{ display: 'flex', gap: 'var(--space-sm)' }}>
                                <button className="btn btn-secondary" onClick={generateFleet} disabled={loading} title="Generate random fleet" style={{ flex: 1, padding: '0.5rem', fontSize: '0.8rem' }}>
                                    🔄 Generate
                                </button>
                                <label className="btn btn-secondary" title="Import EV parameters from CSV" style={{ flex: 1, textAlign: 'center', cursor: 'pointer', margin: 0, padding: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem' }}>
                                    📂 Import
                                    <input type="file" accept=".csv" onChange={handleFileUpload} style={{ display: 'none' }} />
                                </label>
                                <button className="btn btn-secondary" onClick={downloadCSV} disabled={!fleetData || fleetData.length === 0} title="Download current fleet as CSV" style={{ flex: 1, padding: '0.5rem', fontSize: '0.8rem' }}>
                                    📥 Export
                                </button>
                            </div>
                            <button className="btn btn-primary btn-lg" onClick={runSimulation} disabled={loading}>
                                {loading ? (
                                    <><span className="spinner" style={{ width: 18, height: 18 }}></span> Running...</>
                                ) : '▶ Run Simulation'}
                            </button>
                        </div>

                        {status && (
                            <p style={{ fontSize: '0.75rem', marginTop: 'var(--space-sm)', color: status.includes('Error') || status.includes('❌') ? 'var(--color-rose)' : 'var(--color-teal)' }}>
                                {status}
                            </p>
                        )}
                    </div>

                    {/* Fleet Preview */}
                    {fleetData && (
                        <div className="card" style={{ padding: 'var(--space-md)' }}>
                            <h4 style={{ marginBottom: 'var(--space-sm)', fontSize: '0.8125rem' }}>
                                Fleet Preview ({fleetData.length} EVs)
                            </h4>
                            <div style={{ overflowX: 'auto', paddingBottom: '4px' }}>
                                <table className="data-table" style={{ fontSize: '0.6875rem', minWidth: '400px' }}>
                                    <thead>
                                        <tr>
                                            <th>ID</th><th>Ecap</th><th>SoC₀</th><th>SoC_max</th><th>SoC_min</th>
                                            <th>R_i</th><th>P_ref</th><th>P_dis</th><th>T_stay</th><th>Arr</th><th>Dep</th><th>cdeg</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {fleetData.slice(0, 15).map(ev => (
                                            <tr key={ev.id}>
                                                <td>{ev.id}</td>
                                                <td>{ev.Ecap}</td>
                                                <td>{ev.SoC_init?.toFixed(2)}</td>
                                                <td>{ev.SoC_max?.toFixed(2)}</td>
                                                <td>{ev.SoC_min !== undefined ? ev.SoC_min?.toFixed(2) : '0.05'}</td>
                                                <td>{(ev.R_i || ev.P_ref || 7.0).toFixed(1)}</td>
                                                <td>{(ev.P_ref || ev.R_i || 7.0).toFixed(1)}</td>
                                                <td>{ev.P_dis_min !== undefined ? ev.P_dis_min?.toFixed(1) : -(ev.R_i || ev.P_ref || 7.0).toFixed(1)}</td>
                                                <td>{ev.T_stay?.toFixed(1)}</td>
                                                <td>{ev.T_arr_idx}</td>
                                                <td>{ev.T_dep_idx !== undefined ? ev.T_dep_idx : ev.T_arr_idx + Math.ceil((ev.T_stay || 0) / params.delta_t)}</td>
                                                <td>{ev.cdeg !== undefined ? ev.cdeg?.toFixed(4) : '0.0200'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}
                </div>

                {/* ── Main Area ── */}
                <div className="studio-main">
                    {!result ? (
                        <div className="card" style={{ textAlign: 'center', padding: 'var(--space-3xl)' }}>
                            <span style={{ fontSize: '3rem', display: 'block', marginBottom: 'var(--space-lg)' }}>⚡</span>
                            <h3 style={{ marginBottom: 'var(--space-md)' }}>Ready to Simulate</h3>
                            <p>Configure parameters on the left and click <strong>"Run Simulation"</strong> to launch the
                                two-stage scheduling pipeline. The GA will optimize charging power across all {params.T} time slots.</p>
                        </div>
                    ) : (
                        <>
                            {/* Metrics Row */}
                            <div className="grid-4">
                                <div className="stat-card">
                                    <div className="stat-value">{result.metrics?.J?.toFixed(4)}</div>
                                    <div className="stat-label">Fitness (J)</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{result.metrics?.F1?.toFixed(2)}</div>
                                    <div className="stat-label">Energy Cost (F₁)</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{result.metrics?.avg_satisfaction?.toFixed(3)}</div>
                                    <div className="stat-label">Avg Satisfaction</div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-value">{result.metrics?.generations_executed}</div>
                                    <div className="stat-label">Generations</div>
                                </div>
                            </div>

                            {/* Charts Tabs */}
                            <div className="tabs" style={{ marginBottom: 'var(--space-md)' }}>
                                {chartTabs.map(t => (
                                    <button key={t.id} className={`tab ${activeTab === t.id ? 'active' : ''}`}
                                        onClick={() => setActiveTab(t.id)}>{t.label}</button>
                                ))}
                            </div>

                            {/* Chart Area */}
                            <div className="chart-container">
                                {activeTab === 'timeline' && buildTimelineData() && (
                                    <Plot {...buildTimelineData()} config={PLOTLY_CONFIG} style={{ width: '100%' }} />
                                )}
                                {activeTab === 'soc' && buildSoCPlot() && (
                                    <Plot {...buildSoCPlot()} config={PLOTLY_CONFIG} style={{ width: '100%' }} />
                                )}
                                {activeTab === 'grid' && buildGridLoadPlot() && (
                                    <Plot {...buildGridLoadPlot()} config={PLOTLY_CONFIG} style={{ width: '100%' }} />
                                )}
                                {activeTab === 'cost' && buildCostBreakdown() && (
                                    <Plot {...buildCostBreakdown()} config={PLOTLY_CONFIG} style={{ width: '100%' }} />
                                )}
                                {activeTab === 'convergence' && buildGAConvergence() && (
                                    <Plot {...buildGAConvergence()} config={PLOTLY_CONFIG} style={{ width: '100%' }} />
                                )}
                                {activeTab === 'priority' && result?.admission && (
                                    <div>
                                        <div className="grid-2" style={{ marginBottom: 'var(--space-lg)' }}>
                                            <div className="stat-card">
                                                <div className="stat-value">{result.admission.admitted_ids?.length || 0}</div>
                                                <div className="stat-label">Admitted EVs</div>
                                            </div>
                                            <div className="stat-card">
                                                <div className="stat-value">{result.admission.waiting_ids?.length || 0}</div>
                                                <div className="stat-label">Waiting EVs</div>
                                            </div>
                                        </div>
                                        <h4 style={{ marginBottom: 'var(--space-sm)' }}>
                                            Strategy: <span className="badge badge-blue">{result.admission.strategy?.toUpperCase()}</span>
                                        </h4>
                                        {result.admission.details && Object.keys(result.admission.details).length > 0 && (
                                            <div className="priority-panel" style={{ marginTop: 'var(--space-md)' }}>
                                                <table className="data-table">
                                                    <thead>
                                                        <tr>
                                                            <th>EV ID</th>
                                                            <th>λ Score</th>
                                                            <th>φ (Urgency)</th>
                                                            <th>D Factor</th>
                                                            <th>ΔE (kWh)</th>
                                                            <th>Status</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {Object.entries(result.admission.details)
                                                            .sort((a, b) => (b[1].lambda || 0) - (a[1].lambda || 0))
                                                            .map(([id, d]) => (
                                                                <tr key={id}>
                                                                    <td>EV {id}</td>
                                                                    <td>{d.lambda?.toFixed(5)}</td>
                                                                    <td>{d.phi?.toFixed(4)}</td>
                                                                    <td>{d.Dfactor?.toFixed(4)}</td>
                                                                    <td>{d.DeltaE?.toFixed(2)}</td>
                                                                    <td>
                                                                        {result.admission.admitted_ids?.includes(parseInt(id)) ? (
                                                                            <span className="badge badge-emerald">Admitted</span>
                                                                        ) : (
                                                                            <span className="badge badge-rose">Waiting</span>
                                                                        )}
                                                                    </td>
                                                                </tr>
                                                            ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        )}
                                        {result.admission.strategy !== 'priority' && (
                                            <p style={{ marginTop: 'var(--space-md)', fontSize: '0.8125rem' }}>
                                                Admitted IDs: {result.admission.admitted_ids?.join(', ')}
                                            </p>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Extra Metrics */}
                            <div className="card">
                                <div className="card-header"><h3>📋 Detailed Metrics</h3></div>
                                <div className="grid-3">
                                    {[
                                        ['F₁ (Energy Cost)', result.metrics?.F1?.toFixed(4), '$/period'],
                                        ['F₂ (Degradation)', result.metrics?.F2?.toFixed(4), 'cost units'],
                                        ['F₃ (Grid Variance)', result.metrics?.F3?.toFixed(4), 'kW²'],
                                        ['F₄ (Satisfaction)', result.metrics?.F4?.toFixed(4), 'Σ Sᵢ'],
                                        ['V_SoC (Penalty)', result.metrics?.V_SoC?.toFixed(4), 'kWh'],
                                        ['V_grid (Penalty)', result.metrics?.V_grid?.toFixed(4), 'kWh'],
                                        ['V_occ (Penalty)', result.metrics?.V_occ?.toFixed(4), 'count'],
                                        ['Ω (Total Penalty)', result.metrics?.Omega_raw?.toFixed(4), 'weighted'],
                                        ['Fitness J', result.metrics?.J?.toFixed(6), 'normalized'],
                                    ].map(([label, value, unit], i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--color-border)' }}>
                                            <span style={{ fontSize: '0.8125rem', color: 'var(--color-text-secondary)' }}>{label}</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', color: 'var(--color-text-primary)' }}>
                                                {value} <span style={{ fontSize: '0.6875rem', color: 'var(--color-text-muted)' }}>{unit}</span>
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
