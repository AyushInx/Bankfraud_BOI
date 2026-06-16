// src/pages/Alerts.jsx — Alert management page for analysts

import { useState } from 'react';
import { useAlerts, updateAlertStatus } from '../hooks/useApi';

const SEVERITY_ORDER = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };

function AlertRow({ alert, onStatusChange }) {
  const [updating, setUpdating] = useState(false);
  const [notes, setNotes] = useState('');
  const [expanded, setExpanded] = useState(false);

  const handleStatus = async (newStatus) => {
    setUpdating(true);
    try {
      await updateAlertStatus(alert.id, newStatus, notes || null);
      onStatusChange();
    } catch (e) {
      console.error(e);
    } finally {
      setUpdating(false);
    }
  };

  return (
    <>
      <tr
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
      >
        <td>
          <span className={`badge badge-${alert.severity.toLowerCase()}`}>
            {alert.severity === 'CRITICAL' ? '🔴' :
             alert.severity === 'HIGH'     ? '🟠' :
             alert.severity === 'MEDIUM'   ? '🔵' : '🟢'}&nbsp;
            {alert.severity}
          </span>
        </td>
        <td>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>
            {alert.account_ref ?? String(alert.account_id ?? '—').slice(0, 10) + '…'}
          </span>
        </td>
        <td>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="risk-bar" style={{ width: 80 }}>
              <div className="risk-bar-fill" style={{
                width: `${alert.risk_score ?? 0}%`,
                background: (alert.risk_score ?? 0) >= 90 ? '#ff4757' :
                            (alert.risk_score ?? 0) >= 70 ? '#ffa502' :
                            (alert.risk_score ?? 0) >= 40 ? '#1e90ff' : '#2ed573'
              }} />
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-primary)' }}>
              {alert.risk_score ?? '—'}
            </span>
          </div>
        </td>
        <td>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            {alert.probability != null ? (alert.probability * 100).toFixed(2) + '%' : '—'}
          </span>
        </td>
        <td>
          <span className={`badge badge-${alert.status.toLowerCase()}`}>{alert.status}</span>
        </td>
        <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
          {new Date(alert.created_at).toLocaleString()}
        </td>
        <td>
          <span style={{ color: 'var(--text-muted)', fontSize: 18 }}>{expanded ? '▲' : '▼'}</span>
        </td>
      </tr>

      {expanded && (
        <tr>
          <td colSpan={7} style={{ padding: 0 }}>
            <div style={{
              padding: '16px 20px',
              background: 'rgba(255,255,255,0.02)',
              borderBottom: '1px solid var(--border)',
            }}>
              <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                {/* Notes */}
                <div style={{ flex: 1, minWidth: 240 }}>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>Analyst Notes</div>
                  <textarea
                    rows={2}
                    className="input-field"
                    placeholder="Add investigation notes..."
                    value={notes}
                    onChange={e => setNotes(e.target.value)}
                    style={{ width: '100%', resize: 'vertical', fontSize: 13 }}
                  />
                  {alert.notes && (
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 6 }}>
                      Previous: <i>{alert.notes}</i>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>Update Status</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {alert.status !== 'REVIEWED' && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ borderColor: 'var(--amber)', color: 'var(--amber)' }}
                        onClick={() => handleStatus('REVIEWED')}
                        disabled={updating}
                      >
                        {updating ? '⏳' : '👁️'} Mark Reviewed
                      </button>
                    )}
                    {alert.status !== 'CLOSED' && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ borderColor: 'var(--text-muted)', color: 'var(--text-muted)' }}
                        onClick={() => handleStatus('CLOSED')}
                        disabled={updating}
                      >
                        {updating ? '⏳' : '✓'} Close Alert
                      </button>
                    )}
                    {alert.status === 'CLOSED' && (
                      <button
                        className="btn btn-ghost btn-sm"
                        style={{ borderColor: 'var(--red)', color: 'var(--red)' }}
                        onClick={() => handleStatus('OPEN')}
                        disabled={updating}
                      >
                        Reopen
                      </button>
                    )}
                  </div>
                </div>

                {/* Prediction ID */}
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  <div style={{ marginBottom: 4 }}>Alert ID</div>
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                    {String(alert.id).slice(0, 16)}…
                  </span>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Alerts() {
  const [statusFilter, setStatusFilter] = useState('');
  const [severityFilter, setSeverityFilter] = useState('');

  const { data: alerts, loading, error, refetch } = useAlerts(
    statusFilter || null,
    severityFilter || null,
  );

  const sorted = (alerts ?? []).sort(
    (a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]
  );

  const counts = {
    CRITICAL: sorted.filter(a => a.severity === 'CRITICAL').length,
    HIGH:     sorted.filter(a => a.severity === 'HIGH').length,
    MEDIUM:   sorted.filter(a => a.severity === 'MEDIUM').length,
    OPEN:     sorted.filter(a => a.status === 'OPEN').length,
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2 className="page-title">🚨 Alert Management</h2>
        <p className="page-subtitle">
          Review and action fraud alerts generated by the detection engine.
        </p>
      </div>

      {/* Summary bar */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Critical', count: counts.CRITICAL, color: 'var(--red)',    icon: '🔴' },
          { label: 'High',     count: counts.HIGH,     color: 'var(--amber)',  icon: '🟠' },
          { label: 'Medium',   count: counts.MEDIUM,   color: 'var(--blue)',   icon: '🔵' },
          { label: 'Open',     count: counts.OPEN,     color: 'var(--purple)', icon: '🔔' },
        ].map(({ label, count, color, icon }) => (
          <div key={label} className="card" style={{ padding: 16, textAlign: 'center' }}>
            <div style={{ fontSize: 24 }}>{icon}</div>
            <div style={{ fontSize: 28, fontWeight: 800, color, fontFamily: 'var(--font-mono)' }}>{count}</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
        <select
          className="input-field"
          value={statusFilter}
          onChange={e => setStatusFilter(e.target.value)}
          style={{ width: 'auto' }}
        >
          <option value="">All Statuses</option>
          <option value="OPEN">Open</option>
          <option value="REVIEWED">Reviewed</option>
          <option value="CLOSED">Closed</option>
        </select>

        <select
          className="input-field"
          value={severityFilter}
          onChange={e => setSeverityFilter(e.target.value)}
          style={{ width: 'auto' }}
        >
          <option value="">All Severities</option>
          <option value="CRITICAL">Critical</option>
          <option value="HIGH">High</option>
          <option value="MEDIUM">Medium</option>
          <option value="LOW">Low</option>
        </select>

        <button className="btn btn-ghost btn-sm" onClick={refetch}>🔄 Refresh</button>

        <span style={{ marginLeft: 'auto', fontSize: 13, color: 'var(--text-muted)', alignSelf: 'center' }}>
          {sorted.length} alerts
        </span>
      </div>

      {/* Alerts table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: 40, textAlign: 'center' }}>
            <div className="spinner" style={{ margin: '0 auto 16px' }} />
            <div style={{ color: 'var(--text-muted)' }}>Loading alerts…</div>
          </div>
        ) : error ? (
          <div className="empty-state">
            <div className="empty-state-icon">❌</div>
            <div className="empty-state-title">Failed to load alerts</div>
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>{error}</p>
          </div>
        ) : sorted.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🎉</div>
            <div className="empty-state-title">No alerts found</div>
            <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
              {statusFilter || severityFilter ? 'Try removing filters.' : 'No fraud alerts have been generated yet.'}
            </p>
          </div>
        ) : (
          <div className="table-container" style={{ border: 'none' }}>
            <table>
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Account</th>
                  <th>Risk Score</th>
                  <th>Probability</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(alert => (
                  <AlertRow key={alert.id} alert={alert} onStatusChange={refetch} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
