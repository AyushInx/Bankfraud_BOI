// src/components/RiskGauge.jsx — Animated SVG semi-circle risk gauge

import { useEffect, useRef } from 'react';

const TIER_COLORS = {
  LOW:      '#2ed573',
  MEDIUM:   '#1e90ff',
  HIGH:     '#ffa502',
  CRITICAL: '#ff4757',
};

function getRiskTier(score) {
  if (score >= 90) return 'CRITICAL';
  if (score >= 70) return 'HIGH';
  if (score >= 40) return 'MEDIUM';
  return 'LOW';
}

export default function RiskGauge({ score = 0, size = 220 }) {
  const pathRef = useRef(null);
  const tier    = getRiskTier(score);
  const color   = TIER_COLORS[tier];

  const cx = size / 2;
  const cy = size / 2;
  const r  = (size / 2) - 22;

  // Arc math — full gauge is a 180° semi-circle (left to right)
  const startAngle = Math.PI;       // 9 o'clock position (left)
  const endAngle   = 0;             // 3 o'clock (right)

  function polarToCart(angle) {
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    };
  }

  const start = polarToCart(startAngle);
  const end   = polarToCart(endAngle);
  const trackPath = `M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`;

  // Filled arc: from 180° sweeping clockwise by score/100 * 180°
  const fillAngle = startAngle - (score / 100) * Math.PI;
  const fillEnd   = polarToCart(fillAngle);
  const largeArc  = score > 50 ? 1 : 0;
  const fillPath  = `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 1 ${fillEnd.x} ${fillEnd.y}`;

  // Needle pointer
  const needleAngle = startAngle - (score / 100) * Math.PI;
  const needleLen   = r - 10;
  const needleTip = {
    x: cx + needleLen * Math.cos(needleAngle),
    y: cy + needleLen * Math.sin(needleAngle),
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg
        width={size}
        height={size / 2 + 40}
        viewBox={`0 0 ${size} ${size / 2 + 40}`}
        style={{ overflow: 'visible' }}
      >
        {/* Background glow */}
        <defs>
          <filter id="gauge-glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
          <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#2ed573" />
            <stop offset="40%"  stopColor="#1e90ff" />
            <stop offset="70%"  stopColor="#ffa502" />
            <stop offset="100%" stopColor="#ff4757" />
          </linearGradient>
        </defs>

        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={18}
          strokeLinecap="round"
        />

        {/* Gradient fill */}
        <path
          d={trackPath}
          fill="none"
          stroke="url(#gauge-gradient)"
          strokeWidth={18}
          strokeLinecap="round"
          strokeDasharray={`${(score / 100) * Math.PI * r} ${Math.PI * r}`}
          style={{ transition: 'stroke-dasharray 1s cubic-bezier(0.4,0,0.2,1)' }}
        />

        {/* Needle */}
        <line
          x1={cx} y1={cy}
          x2={needleTip.x} y2={needleTip.y}
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          filter="url(#gauge-glow)"
          style={{ transition: 'all 1s cubic-bezier(0.4,0,0.2,1)' }}
        />
        <circle cx={cx} cy={cy} r={8} fill={color} />

        {/* Score text */}
        <text
          x={cx} y={cy + 36}
          textAnchor="middle"
          fill={color}
          fontSize={32}
          fontWeight={800}
          fontFamily="'JetBrains Mono', monospace"
          style={{ transition: 'fill 0.5s' }}
        >
          {score}
        </text>
        <text
          x={cx} y={cy + 56}
          textAnchor="middle"
          fill="rgba(255,255,255,0.4)"
          fontSize={11}
          fontWeight={600}
          fontFamily="'Inter', sans-serif"
          letterSpacing="1"
        >
          RISK SCORE
        </text>

        {/* Min/Max labels */}
        <text x={start.x - 6} y={cy + 6} fill="rgba(255,255,255,0.25)" fontSize={10} textAnchor="end">0</text>
        <text x={end.x + 6}   y={cy + 6} fill="rgba(255,255,255,0.25)" fontSize={10} textAnchor="start">100</text>
      </svg>

      {/* Tier badge */}
      <span className={`badge badge-${tier.toLowerCase()}`} style={{ fontSize: 12 }}>
        {tier}
      </span>
    </div>
  );
}
