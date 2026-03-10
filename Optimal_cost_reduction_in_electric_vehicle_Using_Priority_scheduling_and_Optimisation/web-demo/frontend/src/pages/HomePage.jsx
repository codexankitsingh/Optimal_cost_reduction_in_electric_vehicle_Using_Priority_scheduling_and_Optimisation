import { Link } from 'react-router-dom'

const features = [
    {
        icon: '🔋',
        title: 'Priority Scheduling',
        desc: 'Stage-I admission control using urgency, degradation, grid stress, and price factors.',
        badge: 'Stage I',
        badgeClass: 'badge-blue',
    },
    {
        icon: '🧬',
        title: 'Genetic Algorithm',
        desc: 'Stage-II power optimization with SBX crossover, polynomial mutation, and constraint repair.',
        badge: 'Stage II',
        badgeClass: 'badge-teal',
    },
    {
        icon: '⚡',
        title: 'V2G Support',
        desc: 'Vehicle-to-Grid bidirectional charging allows EVs to sell power during peak price hours.',
        badge: 'Feature',
        badgeClass: 'badge-amber',
    },
    {
        icon: '📊',
        title: 'Multi-Objective',
        desc: 'Minimizes energy cost, battery degradation, grid variance while maximizing user satisfaction.',
        badge: '4 Objectives',
        badgeClass: 'badge-emerald',
    },
]

const objectives = [
    { label: 'F₁', title: 'Energy Cost', formula: 'Σ πᵢₜ · Pᵢₜ · Δt', color: 'var(--color-accent)' },
    { label: 'F₂', title: 'Battery Degradation', formula: 'Σ cᵈᵉᵍ · |Pᵢₜ| · Δt', color: 'var(--color-amber)' },
    { label: 'F₃', title: 'Grid Load Variance', formula: 'Σ (Lₜ - L̄)²', color: 'var(--color-rose)' },
    { label: 'F₄', title: 'User Satisfaction', formula: 'Σ Sᵢ (maximize)', color: 'var(--color-emerald)' },
]

export default function HomePage() {
    return (
        <div>
            {/* Hero */}
            <section className="hero animate-in">
                <h1>EV Charge-Discharge Scheduling</h1>
                <p className="subtitle">
                    An interactive two-stage framework combining Priority Scheduling with Genetic Algorithm optimization
                    to minimize costs, reduce battery degradation, and maximize user satisfaction in workplace EV charging stations.
                </p>
                <div className="hero-actions" style={{ marginBottom: 'var(--space-2xl)' }}>
                    <Link to="/studio" className="btn btn-primary btn-lg glow-pulse">
                        ▶ Launch Simulation Studio
                    </Link>
                    <Link to="/algorithms" className="btn btn-secondary btn-lg">
                        📖 View Algorithms
                    </Link>
                </div>

                <div className="hero-image-container" style={{
                    position: 'relative',
                    borderRadius: 'var(--radius-lg)',
                    overflow: 'hidden',
                    border: '1px solid rgba(255,255,255,0.05)',
                    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
                    maxWidth: '800px',
                    margin: '0 auto'
                }}>
                    <img
                        src="/hero_ev_aesthetic.png"
                        alt="High-tech EV Charging Station"
                        style={{
                            width: '100%',
                            height: 'auto',
                            display: 'block'
                        }}
                    />
                </div>
            </section>

            {/* Features Grid */}
            <section className="section">
                <div className="section-header">
                    <h2>Framework Features</h2>
                    <p>A complete scheduling pipeline for electric vehicle charging infrastructure.</p>
                </div>
                <div className="grid-auto">
                    {features.map((f, i) => (
                        <div key={i} className={`card animate-in delay-${i + 1}`}>
                            <div className="card-header">
                                <span style={{ fontSize: '1.5rem' }}>{f.icon}</span>
                                <h3>{f.title}</h3>
                                <span className={`badge ${f.badgeClass}`} style={{ marginLeft: 'auto' }}>{f.badge}</span>
                            </div>
                            <p>{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Objectives */}
            <section className="section">
                <div className="section-header">
                    <h2>Optimization Objectives</h2>
                    <p>The weighted-sum fitness function J balances four competing objectives.</p>
                </div>
                <div className="equation" style={{ marginBottom: 'var(--space-xl)' }}>
                    J = w₁·F₁ + w₂·F₂ + w₃·F₃ − w₄·F₄ + Ω(X)
                </div>
                <div className="grid-4">
                    {objectives.map((obj, i) => (
                        <div key={i} className="stat-card animate-in" style={{ borderLeft: `3px solid ${obj.color}` }}>
                            <div className="stat-value" style={{ fontSize: '1.25rem' }}>{obj.label}</div>
                            <div className="stat-label" style={{ marginTop: '4px' }}>{obj.title}</div>
                            <p style={{ fontSize: '0.75rem', marginTop: '8px', fontFamily: 'var(--font-mono)' }}>{obj.formula}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Strategies Comparison */}
            <section className="section">
                <div className="section-header">
                    <h2>Admission Strategies</h2>
                    <p>Three Stage-I strategies are compared across multiple fleet sizes.</p>
                </div>
                <div className="card">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Method</th>
                                <th>Best For</th>
                                <th>Energy Cost</th>
                                <th>Satisfaction</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span className="badge badge-blue">Priority</span></td>
                                <td>λᵢ(t) scoring (urgency + price + grid stress)</td>
                                <td>Balanced performance</td>
                                <td>Moderate</td>
                                <td>High</td>
                            </tr>
                            <tr>
                                <td><span className="badge badge-teal">FCFS</span></td>
                                <td>First-Come-First-Served (earliest arrival)</td>
                                <td>Fairness & satisfaction</td>
                                <td>Higher</td>
                                <td>Highest</td>
                            </tr>
                            <tr>
                                <td><span className="badge badge-amber">SJF</span></td>
                                <td>Shortest-Job-First (smallest ΔE/P_ref)</td>
                                <td>Throughput & cost</td>
                                <td>Lowest</td>
                                <td>Lower</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            {/* CTA */}
            <section className="section" style={{ textAlign: 'center' }}>
                <div className="card" style={{ display: 'inline-block', maxWidth: '600px', textAlign: 'center' }}>
                    <h3 style={{ marginBottom: 'var(--space-md)' }}>Ready to simulate?</h3>
                    <p style={{ marginBottom: 'var(--space-lg)' }}>
                        Configure your EV fleet, choose an admission strategy, and watch the GA optimize the charging schedule in real time.
                    </p>
                    <Link to="/studio" className="btn btn-teal btn-lg">Open Simulation Studio →</Link>
                </div>
            </section>
        </div>
    )
}
