// src/pages/Home.jsx — Main dashboard page

import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid
} from 'recharts';
import { useDashboardStats, useModelMetrics } from '../hooks/useApi';

function RiskBar({ score }) {
  const color =
    score >= 90 ? '#ff4757' :
    score >= 70 ? '#ffa502' :
    score >= 40 ? '#1e90ff' : '#2ed573';

  return (
    <div className="risk-bar-wrap">
      <div className="risk-bar">
        <div
          className="risk-bar-fill"
          style={{ width: `${score}%`, background: color }}
        />
      </div>
      <span className="risk-num" style={{ color }}>{score}</span>
    </div>
  );
}

function LabelBadge({ label }) {
  return (
    <span className={`badge badge-${label?.toLowerCase()}`}>
      {label === 'SUSPICIOUS' ? '⚠️' : '✓'} {label}
    </span>
  );
}

function AlertBadge({ status }) {
  if (!status) return <span className="text-muted">—</span>;
  return <span className={`badge badge-${status?.toLowerCase()}`}>{status}</span>;
}

function KpiCard({ icon, value, label, color, prefix = '', suffix = '' }) {
  return (
    <div className="kpi-card" style={{ '--accent-color': color }}>
      <div className="kpi-icon">{icon}</div>
      <div className="kpi-value" style={{ color }}>
        {prefix}{typeof value === 'number' ? value.toLocaleString() : value ?? '—'}{suffix}
      </div>
      <div className="kpi-label">{label}</div>
    </div>
  );
}

function SkeletonRow() {
  return (
    <tr>
      {[1,2,3,4,5,6].map(i => (
        <td key={i}><div className="skeleton" style={{ height: 16, borderRadius: 4 }} /></td>
      ))}
    </tr>
  );
}

// Dummy chart data when no real data is available
const EMPTY_CHART = Array.from({ length: 12 }, (_, i) => ({
  month: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i],
  fraudRate: 0,
}));

export default function Home() {
  const { data: stats, loading: statsLoading } = useDashboardStats();
  const { data: metrics } = useModelMetrics();

  const chartData = EMPTY_CHART;

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2 className="page-title">🛡️ Fraud Investigation Dashboard</h2>
        <p className="page-subtitle">
          Real-time mule account detection powered by CatBoost · SHAP explainability
        </p>
      </div>

      {/* ── KPI Cards ── */}
      <div className="kpi-grid">
        <KpiCard
          icon="👥"
          value={statsLoading ? '...' : stats?.total_accounts}
          label="Total Accounts Analyzed"
          color="var(--blue)"
        />
        <KpiCard
          icon="⚠️"
          value={statsLoading ? '...' : stats?.flagged_accounts}
          label="Flagged Suspicious"
          color="var(--red)"
        />
        <KpiCard
          icon="📈"
          value={statsLoading ? '...' : stats?.fraud_rate}
          label="Fraud Rate"
          color="var(--amber)"
          suffix="%"
        />
        <KpiCard
          icon="🔔"
          value={statsLoading ? '...' : stats?.open_alerts}
          label="Open Alerts"
          color="var(--purple)"
        />
      </div>

      {/* ── Second row: model metrics + mini chart ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.4fr', gap: 20, marginBottom: 28 }}>
        {/* Model Performance Card */}
        <div className="card">
          <div className="section-title" style={{ fontSize: 15 }}>🤖 Model Performance</div>
          {metrics ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {[
                { label: 'ROC-AUC',   value: metrics.roc_auc,    color: '#2ed573' },
                { label: 'PR-AUC',    value: metrics.pr_auc,     color: '#1e90ff' },
                { label: 'Precision', value: metrics.precision,  color: '#ffa502' },
                { label: 'Recall',    value: metrics.recall,     color: '#a855f7' },
                { label: 'F1 Score',  value: metrics.f1_score,   color: '#ff6b9d' },
              ].map(({ label, value, color }) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{label}</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>
                      {(value * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="risk-bar">
                    <div className="risk-bar-fill" style={{ width: `${value * 100}%`, background: color }} />
                  </div>
                </div>
              ))}
              <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-muted)', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
                Threshold: <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-accent)' }}>
                  {metrics.optimal_threshold?.toFixed(4)}
                </span>
                &nbsp;·&nbsp;
                {metrics.n_features} features
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {[1,2,3,4,5].map(i => (
                <div key={i} className="skeleton" style={{ height: 36, borderRadius: 8 }} />
              ))}
            </div>
          )}
        </div>

        {/* Stats overview */}
        <div className="card">
          <div className="section-title" style={{ fontSize: 15 }}>📊 System Overview</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {[
              { label: 'Total Predictions', value: stats?.total_predictions, icon: '🎯' },
              { label: 'Avg Risk Score',    value: stats?.avg_risk_score ? `${stats.avg_risk_score}/100` : '—', icon: '📉' },
              { label: 'Critical Alerts',   value: stats?.critical_alerts,   icon: '🔴' },
              { label: 'Model Features',    value: metrics?.n_features ?? 50, icon: '🧬' },
            ].map(({ label, value, icon }) => (
              <div key={label} className="card-glass" style={{ padding: '16px' }}>
                <div style={{ fontSize: 22, marginBottom: 8 }}>{icon}</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                  {statsLoading ? '...' : (value ?? '—')}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 20 }}>
            <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>Quick Actions</div>
            <div style={{ display: 'flex', gap: 10 }}>
              <Link to="/upload" className="btn btn-primary btn-sm">📤 Upload CSV</Link>
              <Link to="/investigate" className="btn btn-ghost btn-sm">🔍 Investigate</Link>
              <Link to="/alerts" className="btn btn-ghost btn-sm">🚨 View Alerts</Link>
            </div>
          </div>
        </div>
      </div>

      {/* ── Recent Predictions Table ── */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div className="section-title" style={{ margin: 0, fontSize: 15 }}>📋 Recent Predictions</div>
          <Link to="/upload" className="btn btn-ghost btn-sm">View All</Link>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Account Ref</th>
                <th>Risk Score</th>
                <th>Label</th>
                <th>Probability</th>
                <th>Alert</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {statsLoading ? (
                Array.from({ length: 5 }, (_, i) => <SkeletonRow key={i} />)
              ) : stats?.recent_predictions?.length > 0 ? (
                stats.recent_predictions.map((acc) => (
                  <tr key={acc.id}>
                    <td>
                      <Link
                        to={`/investigate?id=${acc.id}`}
                        style={{ color: 'var(--text-accent)', textDecoration: 'none' }}
                      >
                        {acc.account_ref ?? acc.id.slice(0, 8) + '...'}
                      </Link>
                    </td>
                    <td><RiskBar score={acc.latest_risk_score ?? 0} /></td>
                    <td><LabelBadge label={acc.latest_label} /></td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13 }}>
                        {acc.latest_probability != null ? (acc.latest_probability * 100).toFixed(2) + '%' : '—'}
                      </span>
                    </td>
                    <td><AlertBadge status={acc.alert_status} /></td>
                    <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                      {new Date(acc.uploaded_at).toLocaleString()}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6}>
                    <div className="empty-state" style={{ padding: '32px' }}>
                      <div className="empty-state-icon">🎯</div>
                      <div className="empty-state-title">No predictions yet</div>
                      <p style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 8 }}>
                        Upload a CSV or run a single investigation to get started.
                      </p>
                      <Link to="/upload" className="btn btn-primary" style={{ marginTop: 16 }}>
                        Upload CSV
                      </Link>
                    </div>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
