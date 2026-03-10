import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import HomePage from './pages/HomePage'
import AlgorithmsPage from './pages/AlgorithmsPage'
import SimulationStudio from './pages/SimulationStudio'
import DeveloperPage from './pages/DeveloperPage'
import './App.css'

function App() {
  return (
    <Router>
      <nav className="nav">
        <div className="nav-inner">
          <NavLink to="/" className="nav-logo">
            <span className="logo-icon">⚡</span>
            EV Scheduler
          </NavLink>
          <ul className="nav-links">
            <li><NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>Home</NavLink></li>
            <li><NavLink to="/algorithms" className={({ isActive }) => isActive ? 'active' : ''}>Algorithms</NavLink></li>
            <li><NavLink to="/studio" className={({ isActive }) => isActive ? 'active' : ''}>Simulation Studio</NavLink></li>
            <li><NavLink to="/developer" className={({ isActive }) => isActive ? 'active' : ''}>API Docs</NavLink></li>
          </ul>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/algorithms" element={<AlgorithmsPage />} />
          <Route path="/studio" element={<SimulationStudio />} />
          <Route path="/developer" element={<DeveloperPage />} />
        </Routes>
      </main>

      <footer className="footer">
        <p>EV Charge-Discharge Scheduling Framework · Priority Scheduling + Genetic Algorithm Optimization</p>
      </footer>
    </Router>
  )
}

export default App
