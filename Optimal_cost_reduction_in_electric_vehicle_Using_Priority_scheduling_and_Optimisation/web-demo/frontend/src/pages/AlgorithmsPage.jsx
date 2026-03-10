import { useState } from 'react'

const sections = [
    {
        id: 'stage1',
        title: 'Stage I: Priority Scheduling',
        badge: 'Admission Control',
        badgeClass: 'badge-blue',
    },
    {
        id: 'stage2',
        title: 'Stage II: Genetic Algorithm',
        badge: 'Power Optimization',
        badgeClass: 'badge-teal',
    },
    {
        id: 'constraints',
        title: 'Constraints & Penalties',
        badge: 'Feasibility',
        badgeClass: 'badge-rose',
    },
    {
        id: 'hyperparams',
        title: 'Hyperparameters',
        badge: 'Tuning',
        badgeClass: 'badge-amber',
    },
]

export default function AlgorithmsPage() {
    const [activeSection, setActiveSection] = useState('stage1')

    return (
        <div>
            <section className="section-header" style={{ marginBottom: 'var(--space-xl)' }}>
                <h1>Algorithms</h1>
                <p>Complete technical breakdown of the two-stage EV charge-discharge scheduling framework.</p>
            </section>

            {/* Tabs */}
            <div className="tabs">
                {sections.map(s => (
                    <button
                        key={s.id}
                        className={`tab ${activeSection === s.id ? 'active' : ''}`}
                        onClick={() => setActiveSection(s.id)}
                    >
                        {s.title}
                    </button>
                ))}
            </div>

            {/* Stage I */}
            {activeSection === 'stage1' && (
                <div className="animate-in">
                    <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                        <div className="card-header">
                            <span style={{ fontSize: '1.5rem' }}>🔋</span>
                            <h3>Priority-Based Admission Control</h3>
                            <span className="badge badge-blue">Stage I</span>
                        </div>
                        <p style={{ marginBottom: 'var(--space-lg)' }}>
                            When the number of EVs (N) exceeds available chargers (M), the system ranks EVs using a
                            composite priority score λᵢ(t) and admits the top-M.
                        </p>

                        <h4 style={{ marginBottom: 'var(--space-md)', color: 'var(--color-text-primary)' }}>Priority Score Equation</h4>
                        <div className="equation">
                            λᵢ(t) = w_s · φᵢ − w_d · D_factor(i) − w_g · G_factor(t) − w_p · P_factor(t)
                        </div>

                        <div className="grid-2" style={{ marginTop: 'var(--space-xl)', gap: 'var(--space-md)' }}>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-accent)' }}>
                                <h4 style={{ color: 'var(--color-accent-light)', marginBottom: '8px' }}>φᵢ — Urgency Factor</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>φᵢ = min(1, p_req / P_ref)</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Where p_req = ΔEᵢ / T_stay. Measures how urgently the EV needs charging relative to its charger rating.
                                    Higher urgency → higher priority.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-amber)' }}>
                                <h4 style={{ color: 'var(--color-amber)', marginBottom: '8px' }}>D_factor — Degradation</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>D(i) = normalize(cᵈᵉᵍ · ΔEᵢ)</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Normalized battery degradation cost. EVs with high energy demand and degradation coefficient
                                    get a penalty (lower priority) to protect battery life.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-rose)' }}>
                                <h4 style={{ color: 'var(--color-rose)', marginBottom: '8px' }}>G_factor — Grid Stress</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>G(t) = max(0, (P̂ − P_avg) / (P_max − P_avg))</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Measures current grid load relative to capacity.
                                    When load is high, new charging requests get lower priority.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-emerald)' }}>
                                <h4 style={{ color: 'var(--color-emerald)', marginBottom: '8px' }}>P_factor — Price</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>P(t) = P_buy_norm − P_rev_norm</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Normalized electricity price difference.
                                    When buy price is high relative to sell price, charging priority decreases.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Pseudocode */}
                    <div className="card">
                        <div className="card-header">
                            <span style={{ fontSize: '1.2rem' }}>📝</span>
                            <h3>Stage-I Pseudocode</h3>
                        </div>
                        <div className="code-block">
                            <span className="code-keyword">function</span> <span className="code-function">PriorityAdmission</span>(ev_pool, M, t):{'\n'}
                            {'  '}<span className="code-comment">// Step 1: Compute factors for all present EVs</span>{'\n'}
                            {'  '}<span className="code-keyword">for each</span> ev <span className="code-keyword">in</span> ev_pool <span className="code-keyword">where</span> T_arr ≤ t {'<'} T_dep:{'\n'}
                            {'    '}φ[ev] = min(<span className="code-number">1.0</span>, (ΔE / T_stay) / P_ref){'\n'}
                            {'    '}D[ev] = normalize(c_deg · ΔE){'\n'}
                            {'  '}{'\n'}
                            {'  '}<span className="code-comment">// Step 2: Compute common factors</span>{'\n'}
                            {'  '}G = max(<span className="code-number">0</span>, (Σ p_req − P_avg) / (P_max − P_avg)){'\n'}
                            {'  '}P = (π_buy − π_buy_min)/(π_buy_max − π_buy_min) − (π_rev − π_rev_min)/(π_rev_max − π_rev_min){'\n'}
                            {'  '}{'\n'}
                            {'  '}<span className="code-comment">// Step 3: Compute priority scores</span>{'\n'}
                            {'  '}<span className="code-keyword">for each</span> ev:{'\n'}
                            {'    '}λ[ev] = w_s · φ[ev] − w_d · D[ev] − w_g · G − w_p · P{'\n'}
                            {'  '}{'\n'}
                            {'  '}<span className="code-comment">// Step 4: Rank and admit top M</span>{'\n'}
                            {'  '}admitted = top_M(sort(λ, descending)){'\n'}
                            {'  '}<span className="code-keyword">return</span> admitted
                        </div>
                    </div>
                </div>
            )}

            {/* Stage II */}
            {activeSection === 'stage2' && (
                <div className="animate-in">
                    <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                        <div className="card-header">
                            <span style={{ fontSize: '1.5rem' }}>🧬</span>
                            <h3>Genetic Algorithm Power Scheduling</h3>
                            <span className="badge badge-teal">Stage II</span>
                        </div>
                        <p style={{ marginBottom: 'var(--space-lg)' }}>
                            Once M EVs are admitted, the GA optimizes their charging/discharging
                            power profile across T time slots. The genome is a real-valued matrix P[M × T].
                        </p>

                        <h4 style={{ marginBottom: 'var(--space-md)', color: 'var(--color-text-primary)' }}>Fitness Function</h4>
                        <div className="equation">
                            J(X) = w₁·F₁(X)/‖F₁‖ + w₂·F₂(X)/‖F₂‖ + w₃·F₃(X)/‖F₃‖ − w₄·F₄(X)/‖F₄‖ + Ω(X)/‖Ω‖
                        </div>

                        <div className="grid-2" style={{ marginTop: 'var(--space-md)', gap: 'var(--space-md)' }}>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-accent)' }}>
                                <h4 style={{ color: 'var(--color-accent-light)', marginBottom: '8px' }}>F₁ — Total Energy Cost ($)</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>Σ_t Σ_i [π_buy(t)·P⁺ᵢ(t) − π_rev(t)·P⁻ᵢ(t)]</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Total grid energy cost over time T. EV charging (P⁺) pays buy price, discharging/V2G (P⁻) earns revenue.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-amber)' }}>
                                <h4 style={{ color: 'var(--color-amber)', marginBottom: '8px' }}>F₂ — Battery Degradation</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>Σ_t Σ_i cᵈᵉᵍᵢ · |Pᵢ(t)| · Δt</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Cost of battery aging. Both charging and discharging stress the battery and accrue degradation cost.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-rose)' }}>
                                <h4 style={{ color: 'var(--color-rose)', marginBottom: '8px' }}>F₃ — Grid Load Variance</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>Σ_t (L(t) − L_avg)²</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Variance of total aggregate grid load L(t). Minimizing this fills "valleys" and avoids "peaks" in overall power consumption.
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left', borderLeft: '3px solid var(--color-emerald)' }}>
                                <h4 style={{ color: 'var(--color-emerald)', marginBottom: '8px' }}>F₄ — EV Owner Satisfaction</h4>
                                <div className="equation" style={{ fontSize: '0.8rem', margin: '8px 0' }}>Σ_i Sᵢ ; Sᵢ = 1 − f(SoC_dep, SoC_target)</div>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Measures how close EV batteries are to their target SoC at departure. Note we maximizing this term (− w₄ in Eq).
                                </p>
                            </div>
                        </div>

                        <div className="grid-2" style={{ marginTop: 'var(--space-xl)' }}>
                            <div className="stat-card" style={{ textAlign: 'left' }}>
                                <h4 style={{ color: 'var(--color-accent-light)', marginBottom: '8px' }}>Genome Encoding</h4>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Real-valued flat vector of length M×T. Gene[i·T + t] = power (kW) of EV i at slot t.
                                    Positive = charging, Negative = discharging (V2G).
                                </p>
                            </div>
                            <div className="stat-card" style={{ textAlign: 'left' }}>
                                <h4 style={{ color: 'var(--color-teal)', marginBottom: '8px' }}>Seeding Strategy</h4>
                                <p style={{ fontSize: '0.8125rem' }}>
                                    Hybrid initialization: greedy seed fills cheapest slots first; Stage-I seed
                                    uses priority results as baseline. Rest are random within bounds.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* GA Operators */}
                    <div className="grid-3">
                        <div className="card">
                            <div className="card-header"><h3>🔀 Selection</h3></div>
                            <p style={{ fontSize: '0.8125rem', marginBottom: '12px' }}>
                                <strong>Tournament Selection</strong> (k=3): Randomly sample 3 individuals,
                                keep the one with lowest J. Repeat to fill mating pool.
                            </p>
                            <div className="code-block" style={{ fontSize: '0.75rem' }}>
                                <span className="code-keyword">for</span> i <span className="code-keyword">in</span> range(pop_size):{'\n'}
                                {'  '}candidates = random.sample(pop, k=<span className="code-number">3</span>){'\n'}
                                {'  '}selected[i] = min(candidates, key=J)
                            </div>
                        </div>
                        <div className="card">
                            <div className="card-header"><h3>🧬 Crossover</h3></div>
                            <p style={{ fontSize: '0.8125rem', marginBottom: '12px' }}>
                                <strong>SBX</strong> (Simulated Binary Crossover): probability = 0.9,
                                η_c = 20. Creates offspring near parents while exploring the space.
                            </p>
                            <div className="code-block" style={{ fontSize: '0.75rem' }}>
                                <span className="code-keyword">if</span> random() {'<'} <span className="code-number">0.9</span>:{'\n'}
                                {'  '}child1, child2 = SBX(parent1, parent2,{'\n'}
                                {'    '}eta=<span className="code-number">20</span>, bounds=[lo, hi])
                            </div>
                        </div>
                        <div className="card">
                            <div className="card-header"><h3>🎲 Mutation</h3></div>
                            <p style={{ fontSize: '0.8125rem', marginBottom: '12px' }}>
                                <strong>Polynomial Mutation</strong>: probability = 0.3 per individual,
                                per-gene probability = 1/T. η_m = 20.
                            </p>
                            <div className="code-block" style={{ fontSize: '0.75rem' }}>
                                <span className="code-keyword">if</span> random() {'<'} <span className="code-number">0.3</span>:{'\n'}
                                {'  '}<span className="code-keyword">for each</span> gene <span className="code-keyword">with</span> prob <span className="code-number">1/T</span>:{'\n'}
                                {'    '}gene += polynomial_noise(eta=<span className="code-number">20</span>)
                            </div>
                        </div>
                    </div>

                    {/* GA Flow */}
                    <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                        <div className="card-header"><h3>🔄 GA Execution Flow</h3></div>
                        <div className="code-block">
                            <span className="code-number">1.</span> Initialize population (random + seeded individuals){'\n'}
                            <span className="code-number">2.</span> Repair all individuals (enforce bounds, windows, occupancy){'\n'}
                            <span className="code-number">3.</span> Evaluate fitness J for all individuals{'\n'}
                            <span className="code-number">4.</span> <span className="code-keyword">WHILE</span> gen {'<'} max_gen <span className="code-keyword">AND</span> no_improve {'<'} stagnation_limit:{'\n'}
                            {'  '}<span className="code-number">a.</span> Select parents via tournament selection{'\n'}
                            {'  '}<span className="code-number">b.</span> Apply SBX crossover (probability = 0.9){'\n'}
                            {'  '}<span className="code-number">c.</span> Apply polynomial mutation (probability = 0.3){'\n'}
                            {'  '}<span className="code-number">d.</span> Repair offspring{'\n'}
                            {'  '}<span className="code-number">e.</span> Evaluate offspring fitness{'\n'}
                            {'  '}<span className="code-number">f.</span> Inject elite individuals from previous generation{'\n'}
                            {'  '}<span className="code-number">g.</span> Select next generation (μ + λ){'\n'}
                            {'  '}<span className="code-number">h.</span> Update best solution{'\n'}
                            <span className="code-number">5.</span> <span className="code-keyword">RETURN</span> best solution (schedule matrix P[M×T])
                        </div>
                    </div>
                </div>
            )}

            {/* Constraints */}
            {activeSection === 'constraints' && (
                <div className="animate-in">
                    <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
                        <div className="card-header">
                            <span style={{ fontSize: '1.5rem' }}>🔒</span>
                            <h3>Physical & Operational Constraints</h3>
                        </div>
                        <p style={{ marginBottom: 'var(--space-xl)' }}>
                            The following constraints are enforced via both repair operators and soft penalties
                            added to the objective function.
                        </p>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
                            {[
                                { id: 'C1', title: 'Power Bounds', eq: 'P_min ≤ Pᵢₜ ≤ P_max', desc: 'Charger physical limits. Negative P_min for V2G.' },
                                { id: 'C2', title: 'SoC Limits', eq: 'SoC_min ≤ SoCᵢₜ ≤ SoC_max', desc: 'Battery safety bounds to prevent over/under-charging.' },
                                { id: 'C3', title: 'Time Windows', eq: 'Pᵢₜ = 0 if t < T_arr or t ≥ T_dep', desc: 'EVs can only charge while physically present.' },
                                { id: 'C4', title: 'Charger Occupancy', eq: 'Σᵢ zᵢₜ ≤ M for all t', desc: 'Number of active EVs per slot cannot exceed available chargers.' },
                                { id: 'C5', title: 'Grid Capacity', eq: 'Σᵢ Pᵢₜ ≤ P_max_grid', desc: 'Total aggregate load cannot exceed transformer/grid limit.' },
                                { id: 'C6', title: 'Energy Balance', eq: 'SoCᵢ(T_dep) ≥ SoC_target', desc: 'EVs should reach their target SoC by departure.' },
                            ].map(c => (
                                <div key={c.id} className="stat-card" style={{ textAlign: 'left', display: 'flex', gap: 'var(--space-lg)', alignItems: 'flex-start' }}>
                                    <span className="badge badge-rose" style={{ minWidth: '32px', textAlign: 'center' }}>{c.id}</span>
                                    <div>
                                        <h4 style={{ color: 'var(--color-text-primary)', marginBottom: '4px' }}>{c.title}</h4>
                                        <div className="equation" style={{ margin: '8px 0', fontSize: '0.8rem' }}>{c.eq}</div>
                                        <p style={{ fontSize: '0.8125rem' }}>{c.desc}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header"><h3>Penalty Function Ω(X)</h3></div>
                        <div className="equation" style={{ marginBottom: 'var(--space-lg)' }}>
                            Ω(X) = α₁ · V_SoC + α₂ · V_occ + α₃ · V_grid
                        </div>
                        <div className="grid-3">
                            <div className="stat-card">
                                <div className="stat-value" style={{ fontSize: '1rem' }}>V_SoC</div>
                                <div className="stat-label">SoC violations (kWh)</div>
                                <p style={{ fontSize: '0.75rem', marginTop: '4px' }}>Penalty for exceeding min/max SoC bounds at any slot</p>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value" style={{ fontSize: '1rem' }}>V_occ</div>
                                <div className="stat-label">Occupancy violations</div>
                                <p style={{ fontSize: '0.75rem', marginTop: '4px' }}>Penalty when more than M EVs are active in a slot</p>
                            </div>
                            <div className="stat-card">
                                <div className="stat-value" style={{ fontSize: '1rem' }}>V_grid</div>
                                <div className="stat-label">Grid violations (kWh)</div>
                                <p style={{ fontSize: '0.75rem', marginTop: '4px' }}>Penalty when aggregate load exceeds P_max_grid</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Hyperparameters */}
            {activeSection === 'hyperparams' && (
                <div className="animate-in">
                    <div className="card">
                        <div className="card-header">
                            <span style={{ fontSize: '1.5rem' }}>⚙️</span>
                            <h3>GA Hyperparameters</h3>
                        </div>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Parameter</th>
                                    <th>Symbol</th>
                                    <th>Default</th>
                                    <th>Range</th>
                                    <th>Effect</th>
                                </tr>
                            </thead>
                            <tbody>
                                {[
                                    ['Population Size', 'N_pop', '120', '50–300', 'Larger → better exploration, slower per generation'],
                                    ['Generations', 'N_gen', '300', '50–1000', 'More generations → better convergence (diminishing returns)'],
                                    ['Crossover Prob.', 'p_c', '0.9', '0.6–1.0', 'High → strong trait mixing between parents'],
                                    ['Mutation Prob.', 'p_m', '0.3', '0.1–0.5', 'Higher → more exploration, risk of losing good solutions'],
                                    ['SBX η_c', 'η_c', '20', '2–100', 'Higher → children closer to parents (exploitation)'],
                                    ['Mutation η_m', 'η_m', '20', '2–100', 'Higher → smaller mutations (fine-tuning)'],
                                    ['Tournament Size', 'k', '3', '2–7', 'Larger → stronger selection pressure'],
                                    ['Stagnation Limit', '-', '40', '20–100', 'Generations without improvement before early stop'],
                                    ['Elite Count', 'k_elite', '2', '1–10', 'Top individuals always survive to next generation'],
                                    ['Seed Count', '-', '10', '5–30', 'Number of heuristic-seeded individuals in initial population'],
                                ].map(([param, sym, def_, range_, effect], i) => (
                                    <tr key={i}>
                                        <td style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-sans)' }}>{param}</td>
                                        <td>{sym}</td>
                                        <td><span className="badge badge-blue">{def_}</span></td>
                                        <td>{range_}</td>
                                        <td style={{ fontFamily: 'var(--font-sans)', fontSize: '0.75rem' }}>{effect}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="card" style={{ marginTop: 'var(--space-xl)' }}>
                        <div className="card-header"><h3>Objective Weights</h3></div>
                        <div className="grid-4">
                            {[
                                ['w₁', '0.25', 'Energy Cost', 'var(--color-accent)'],
                                ['w₂', '0.25', 'Degradation', 'var(--color-amber)'],
                                ['w₃', '0.25', 'Grid Variance', 'var(--color-rose)'],
                                ['w₄', '0.25', 'Satisfaction', 'var(--color-emerald)'],
                            ].map(([sym, val, label, color], i) => (
                                <div key={i} className="stat-card" style={{ borderTop: `3px solid ${color}` }}>
                                    <div className="stat-value" style={{ fontSize: '1.25rem' }}>{val}</div>
                                    <div className="stat-label">{sym} — {label}</div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
