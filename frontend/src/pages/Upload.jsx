// src/pages/Upload.jsx — Batch CSV upload page with real-time progress

import { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { uploadBatch, getBatchStatus } from '../hooks/useApi';
import { useFeatureImportance } from '../hooks/useApi';

function RiskBadge({ score }) {
  const color = score >= 90 ? '#ff4757' : score >= 70 ? '#ffa502' : score >= 40 ? '#1e90ff' : '#2ed573';
  const tier  = score >= 90 ? 'CRITICAL' : score >= 70 ? 'HIGH' : score >= 40 ? 'MEDIUM' : 'LOW';
  return (
    <span className={`badge badge-${tier.toLowerCase()}`}>{score} · {tier}</span>
  );
}

function StatusBanner({ job }) {
  const pct = job.total_records > 0
    ? Math.round((job.processed_count / job.total_records) * 100)
    : 0;

  const statusColor = {
    PENDING:    'var(--blue)',
    PROCESSING: 'var(--amber)',
    COMPLETED:  'var(--green)',
    FAILED:     'var(--red)',
  }[job.status] || 'var(--text-muted)';

  return (
    <div className="card" style={{ marginBottom: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div>
          <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{job.filename}</span>
          <span style={{ marginLeft: 12, color: 'var(--text-muted)', fontSize: 12 }}>
            Job ID: <span style={{ fontFamily: 'var(--font-mono)' }}>{String(job.job_id).slice(0,8)}…</span>
          </span>
        </div>
        <span style={{ color: statusColor, fontWeight: 700, fontSize: 14 }}>
          {job.status === 'PROCESSING' && <span className="spinner" style={{ display: 'inline-block', marginRight: 8, verticalAlign: 'middle' }} />}
          {job.status}
        </span>
      </div>

      <div className="progress-bar" style={{ marginBottom: 12 }}>
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>

      <div style={{ display: 'flex', gap: 24, fontSize: 13, color: 'var(--text-secondary)' }}>
        <span>📋 Total: <b style={{ color: 'var(--text-primary)' }}>{job.total_records}</b></span>
        <span>✅ Processed: <b style={{ color: 'var(--text-primary)' }}>{job.processed_count}</b></span>
        <span>⚠️ Flagged: <b style={{ color: 'var(--red)' }}>{job.flagged_count}</b></span>
        <span>{pct}%</span>
      </div>

      {job.error_message && (
        <div style={{ marginTop: 12, padding: '10px 14px', background: 'var(--red-dim)', border: '1px solid var(--border-red)', borderRadius: 8, color: 'var(--red)', fontSize: 13 }}>
          ❌ {job.error_message}
        </div>
      )}
    </div>
  );
}

export default function Upload() {
  const [job, setJob] = useState(null);
  const [results, setResults] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const pollingRef = useRef(null);
  const { data: features } = useFeatureImportance();

  const pollJob = useCallback(async (jobId) => {
    try {
      const status = await getBatchStatus(jobId);
      setJob(status);
      if (status.status === 'COMPLETED' || status.status === 'FAILED') {
        clearInterval(pollingRef.current);
      }
    } catch (e) {
      clearInterval(pollingRef.current);
    }
  }, []);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setError(null);
    setResults([]);
    setJob(null);
    setUploading(true);

    try {
      const data = await uploadBatch(file);
      setJob(data);
      // Start polling
      pollingRef.current = setInterval(() => pollJob(data.job_id), 2000);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setUploading(false);
    }
  }, [pollJob]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
    disabled: uploading,
  });

  const downloadTemplate = () => {
    if (!features) return;
    const headers = features.map(f => f.feature).join(',');
    const blob = new Blob([headers + '\n'], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'muleshield_template.csv';
    a.click();
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2 className="page-title">📤 Batch CSV Upload</h2>
        <p className="page-subtitle">
          Upload a CSV file with account feature data for bulk fraud analysis.
        </p>
      </div>

      {/* Info card */}
      <div className="card" style={{ marginBottom: 24, borderColor: 'var(--border-accent)', background: 'rgba(30,144,255,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontWeight: 700, marginBottom: 8, color: 'var(--text-accent)' }}>
              ℹ️ CSV Format Requirements
            </div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
              Your CSV must contain the <b style={{ color: 'var(--text-primary)' }}>50 selected features</b> as column headers.
              Optionally include an <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(255,255,255,0.06)', padding: '1px 6px', borderRadius: 4 }}>account_id</code> column as an identifier.
              Missing feature values will be filled with <b>0.0</b>.
            </div>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={downloadTemplate} disabled={!features}>
            ⬇️ Download Template
          </button>
        </div>
      </div>

      {/* Dropzone */}
      <div {...getRootProps()} className={`dropzone ${isDragActive ? 'drag-over' : ''}`} style={{ marginBottom: 24 }}>
        <input {...getInputProps()} />
        <div className="dropzone-icon">
          {uploading ? '⏳' : isDragActive ? '📂' : '📁'}
        </div>
        <div className="dropzone-text">
          {uploading
            ? 'Uploading…'
            : isDragActive
            ? 'Drop the CSV here!'
            : 'Drag & drop CSV file here'}
        </div>
        <div className="dropzone-sub">
          or <span style={{ color: 'var(--blue)', textDecoration: 'underline' }}>browse files</span> · CSV only
        </div>
      </div>

      {error && (
        <div style={{ padding: '12px 16px', background: 'var(--red-dim)', border: '1px solid var(--border-red)', borderRadius: 10, color: 'var(--red)', marginBottom: 20, fontSize: 14 }}>
          ❌ {error}
        </div>
      )}

      {/* Job status */}
      {job && <StatusBanner job={job} />}

      {/* Results summary */}
      {job?.status === 'COMPLETED' && (
        <div className="card animate-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div className="section-title" style={{ margin: 0 }}>✅ Batch Results</div>
            <div style={{ display: 'flex', gap: 10 }}>
              <div style={{ textAlign: 'center', padding: '10px 20px', background: 'var(--red-dim)', borderRadius: 10 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--red)' }}>{job.flagged_count}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>FLAGGED</div>
              </div>
              <div style={{ textAlign: 'center', padding: '10px 20px', background: 'var(--green-dim)', borderRadius: 10 }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: 'var(--green)' }}>
                  {job.total_records - job.flagged_count}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>LEGITIMATE</div>
              </div>
            </div>
          </div>

          <div style={{
            padding: '16px',
            background: 'var(--bg-glass)',
            borderRadius: 10,
            border: '1px solid var(--border)',
            fontSize: 14,
            color: 'var(--text-secondary)',
          }}>
            🎯 Batch job <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-accent)' }}>
              {String(job.job_id).slice(0,8)}…
            </span> completed.{' '}
            <b style={{ color: 'var(--red)' }}>{job.flagged_count}</b> of{' '}
            <b style={{ color: 'var(--text-primary)' }}>{job.total_records}</b> accounts flagged as suspicious
            ({job.total_records > 0 ? ((job.flagged_count / job.total_records) * 100).toFixed(1) : 0}% fraud rate).
            Alerts have been created for HIGH and CRITICAL risk accounts.
            <br /><br />
            <a href="/alerts" style={{ color: 'var(--blue)', textDecoration: 'none', fontWeight: 600 }}>
              → View Alerts Dashboard
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
