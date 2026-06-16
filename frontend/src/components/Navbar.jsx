// src/components/Navbar.jsx — Sidebar navigation

import { NavLink } from 'react-router-dom';
import { useHealth } from '../hooks/useApi';

const NAV_ITEMS = [
  { to: '/',           icon: '📊', label: 'Dashboard'     },
  { to: '/upload',     icon: '📤', label: 'Batch Upload'  },
  { to: '/investigate',icon: '🔍', label: 'Investigate'   },
  { to: '/alerts',     icon: '🚨', label: 'Alerts'        },
];

export default function Navbar() {
  const { data: health } = useHealth();
  const isOnline = health?.model_loaded ?? false;

  return (
    <aside className="sidebar animate-slide">
      <div className="sidebar-logo">
        <h1>🛡️ MuleShield AI</h1>
        <span>Fraud Detection Platform</span>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
          >
            <span className="nav-icon">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="model-status">
          <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
          <span>{isOnline ? `CatBoost — ${health?.n_features ?? '?'} features` : 'Model offline'}</span>
        </div>
      </div>
    </aside>
  );
}
