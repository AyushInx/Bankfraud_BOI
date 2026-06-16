// src/pages/Investigate.jsx — Single account investigation with SHAP

import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import RiskGauge from '../components/RiskGauge';
import ShapChart from '../components/ShapChart';
import { predictSingle, useAccount, useFeatureImportance } from '../hooks/useApi';

// Default features from selected_features_list.json
const FEATURE_NAMES = [
  "F3898","F3805","F1815","F3799","F3811","F3813","F3807","F2489","F1921","F3806",
  "F2137","F1825","F1166","F1058","F3800","F2029","F1705","F3801","F1813","F1165",
  "F3812","F1819","F270","F3640","F3532","F2390","F1489","F1599","F3748","F1490",
  "F1598","F1177","F1814","F1922","F1821","F999","F2486","F2149","F1827","F3484",
  "F1597","F949","F1167","F2385","F1172","F1431","F1923","F1273","F950","F1057"
];

function ResultPanel({ result }) {
  if (!result) return null;

  const isFlag = result.label === 'SUSPICIOUS';

  return (
    <div className="animate-in">
      {/* Header */}
      <div className="card" style={{
        marginBottom: 20,
        borderColor: isFlag ? 'var(--border-red)' : 'rgba(46,213,115,0.3)',
        background: isFlag ? 'rgba(255,71,87,0.04)' : 'rgba(46,213,115,0.04)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <RiskGauge score={result.risk_score} size={180} />
            <div>
              <div style={{ fontSize: 24, fontWeight: 800, marginBottom: 8 }}>
                {isFlag ? '⚠️' : '✓'}&nbsp;
                <span style={{ color: isFlag ? 'var(--red)' : 'var(--green)' }}>
                  {result.label}
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                  Fraud Probability:&nbsp;
                  <span style={{ fontFamily: 'var(--font-mono)', color: isFlag ? 'var(--red)' : 'var(--green)', fontWeight: 700 }}>
                    {(result.probability * 100).toFixed(2)}%
                  </span>
                </div>
                <div style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                  Decision Threshold:&nbsp;
                  <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-accent)' }}>
                    {(result.threshold_used * 100).toFixed(1)}%
                  </span>
                </div>
                {result.alert_id && (
                  <div style={{ padding: '6px 12px', background: 'var(--red-dim)', borderRadius: 8, fontSize: 13, color: 'var(--red)' }}>
                    🚨 Alert created — ID: <span style={{ fontFamily: 'var(--font-mono)' }}>{String(result.alert_id).slice(0,8)}…</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Quick stats */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            {[
              { label: 'Risk Score',  value: result.risk_score + '/100',  color: isFlag ? 'var(--red)' : 'var(--green)' },
              { label: 'Risk Tier',   value: result.risk_tier,             color: 'var(--amber)' },
              { label: 'Prediction',  value: result.label,                 color: isFlag ? 'var(--red)' : 'var(--green)' },
              { label: 'Account ID',  value: String(result.account_id).slice(0,10) + '…', color: 'var(--text-muted)' },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ padding: '10px 14px', background: 'var(--bg-glass)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* SHAP Chart */}
      <div className="card">
        <div className="section-title" style={{ fontSize: 15 }}>
          🔬 SHAP Feature Contributions
        </div>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20 }}>
          Shows which features contributed most to this prediction.
          Red bars push toward fraud; green bars push toward legitimate.
        </p>
        <ShapChart features={result.top_features} maxFeatures={10} />
      </div>
    </div>
  );
}

export default function Investigate() {
  const [searchParams] = useSearchParams();
  const accountId = searchParams.get('id');

  const [features, setFeatures] = useState({});
  const [accountRef, setAccountRef] = useState('');
  const [jsonInput, setJsonInput] = useState('');
  const [useJson, setUseJson] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // If navigated with account ID, load that account
  const { data: existingAccount } = useAccount(accountId);

  useEffect(() => {
    if (existingAccount && existingAccount.predictions?.length > 0) {
      setResult(existingAccount.predictions[0]);
    }
  }, [existingAccount]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let parsedFeatures = {};

      if (useJson) {
        parsedFeatures = JSON.parse(jsonInput);
      } else {
        // Use form values, default to 0
        FEATURE_NAMES.forEach(f => {
          parsedFeatures[f] = parseFloat(features[f] ?? 0);
        });
      }

      const data = await predictSingle(parsedFeatures, accountRef || null);
      setResult(data);
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    } catch (e) {
      if (e instanceof SyntaxError) {
        setError('Invalid JSON format. Please check your input.');
      } else {
        setError(e.response?.data?.detail || e.message || 'Prediction failed');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-in">
      <div className="page-header">
        <h2 className="page-title">🔍 Account Investigation</h2>
        <p className="page-subtitle">
          Run real-time fraud scoring on a single account with SHAP explainability.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1.3fr' : '1fr', gap: 24, alignItems: 'start' }}>
        {/* Input form */}
        <div>
          <div className="card" style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <div className="section-title" style={{ margin: 0, fontSize: 15 }}>Account Data Input</div>
              <div style={{ display: 'flex', gap: 0, background: 'var(--bg-elevated)', borderRadius: 8, padding: 4 }}>
                <button
                  className="btn btn-sm"
                  style={!useJson ? { background: 'var(--bg-glass-hover)', color: 'var(--text-primary)' } : { background: 'transparent', color: 'var(--text-muted)' }}
                  onClick={() => setUseJson(false)}
                >Form</button>
                <button
                  className="btn btn-sm"
                  style={useJson ? { background: 'var(--bg-glass-hover)', color: 'var(--text-primary)' } : { background: 'transparent', color: 'var(--text-muted)' }}
                  onClick={() => setUseJson(true)}
                >JSON</button>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              {/* Account ref */}
              <div className="input-group">
                <label className="input-label">Account Reference (optional)</label>
                <input
                  className="input-field"
                  placeholder="e.g. ACC-001234"
                  value={accountRef}
                  onChange={e => setAccountRef(e.target.value)}
                />
              </div>

              {useJson ? (
                <div className="input-group">
                  <label className="input-label">Feature JSON</label>
                  <textarea
                    className="input-field"
                    rows={12}
                    placeholder={'{\n  "F3898": 0.5,\n  "F3805": 1.2,\n  ...\n}'}
                    value={jsonInput}
                    onChange={e => setJsonInput(e.target.value)}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: 12, resize: 'vertical' }}
                  />
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    Paste a JSON object with feature names as keys and numeric values.
                    Missing features default to 0.
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 12 }}>
                    Enter values for the top 10 most important features (others default to 0):
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    {FEATURE_NAMES.slice(0, 10).map(f => (
                      <div className="input-group" key={f} style={{ marginBottom: 0 }}>
                        <label className="input-label" style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{f}</label>
                        <input
                          className="input-field"
                          type="number"
                          step="any"
                          placeholder="0.0"
                          value={features[f] ?? ''}
                          onChange={e => setFeatures(prev => ({ ...prev, [f]: e.target.value }))}
                          style={{ fontFamily: 'var(--font-mono)' }}
                        />
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 12 }}>
                    💡 Switch to <b>JSON</b> mode to enter all 50 features at once.
                  </div>
                </div>
              )}

              {error && (
                <div style={{ padding: '10px 14px', background: 'var(--red-dim)', border: '1px solid var(--border-red)', borderRadius: 8, color: 'var(--red)', fontSize: 13, marginTop: 16 }}>
                  ❌ {error}
                </div>
              )}

              <button
                type="submit"
                className="btn btn-primary w-full"
                style={{ marginTop: 20 }}
                disabled={loading}
              >
                {loading
                  ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Analyzing…</>
                  : '🎯 Run Fraud Analysis'
                }
              </button>
            </form>
          </div>

          {/* Feature reference */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, color: 'var(--text-secondary)' }}>
              📋 Top 10 Feature Names (by RF Importance)
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {FEATURE_NAMES.slice(0, 10).map((f, i) => (
                <span key={f} style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 11,
                  padding: '3px 8px',
                  background: 'var(--bg-elevated)',
                  borderRadius: 4,
                  color: 'var(--text-accent)',
                  border: '1px solid var(--border)',
                }}>
                  #{i+1} {f}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Result panel */}
        {result && <ResultPanel result={result} />}
      </div>
    </div>
  );
}
