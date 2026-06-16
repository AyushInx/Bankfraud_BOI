// src/components/ShapChart.jsx — SHAP waterfall bar chart using recharts

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell
} from 'recharts';

function ShapTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: '#151e35',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 13,
    }}>
      <div style={{ color: '#f0f4ff', fontWeight: 700, marginBottom: 6 }}>{d.name}</div>
      <div style={{ color: '#8b9cc8', marginBottom: 4 }}>
        Value: <span style={{ color: '#f0f4ff', fontFamily: 'monospace' }}>{d.value?.toFixed(4)}</span>
      </div>
      <div style={{ color: '#8b9cc8' }}>
        SHAP Impact:{' '}
        <span style={{ color: d.shap_impact >= 0 ? '#ff4757' : '#2ed573', fontWeight: 700 }}>
          {d.shap_impact >= 0 ? '+' : ''}{d.shap_impact?.toFixed(4)}
        </span>
      </div>
    </div>
  );
}

export default function ShapChart({ features = [], maxFeatures = 10 }) {
  if (!features || features.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📊</div>
        <div className="empty-state-title">No SHAP data available</div>
      </div>
    );
  }

  const data = features
    .slice(0, maxFeatures)
    .sort((a, b) => b.shap_impact - a.shap_impact)
    .map(f => ({
      name:        f.name,
      value:       f.value,
      shap_impact: f.shap_impact,
      absImpact:   Math.abs(f.shap_impact),
    }));

  return (
    <div style={{ width: '100%' }}>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 12, color: '#8b9cc8' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 12, borderRadius: 2, background: '#ff4757', display: 'inline-block' }} />
          Pushes toward fraud
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 12, borderRadius: 2, background: '#2ed573', display: 'inline-block' }} />
          Pushes toward legitimate
        </span>
      </div>

      <ResponsiveContainer width="100%" height={Math.max(250, data.length * 38)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 60, left: 20, bottom: 4 }}
        >
          <CartesianGrid
            horizontal={false}
            stroke="rgba(255,255,255,0.05)"
          />
          <XAxis
            type="number"
            tick={{ fill: '#4a5580', fontSize: 11 }}
            axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={70}
            tick={{ fill: '#8b9cc8', fontSize: 11, fontFamily: 'JetBrains Mono' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<ShapTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
          <ReferenceLine x={0} stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
          <Bar dataKey="shap_impact" radius={[0, 4, 4, 0]} maxBarSize={22}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.shap_impact >= 0 ? '#ff4757' : '#2ed573'}
                fillOpacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
