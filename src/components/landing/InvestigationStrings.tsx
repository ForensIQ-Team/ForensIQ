import React, { useEffect, useState } from 'react';

export interface StringConnection {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  sag?: number;
}

interface InvestigationStringsProps {
  connections: StringConnection[];
  className?: string;
}

export const InvestigationStrings: React.FC<InvestigationStringsProps> = ({
  connections,
  className = '',
}) => {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    // Staggered animation trigger
    const timer = setTimeout(() => {
      setAnimated(true);
    }, 150);
    return () => clearTimeout(timer);
  }, []);

  return (
    <svg
      className={`absolute inset-0 w-full h-full pointer-events-none z-15 ${className}`}
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        {/* Red String Shadow */}
        <filter id="string-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="1" dy="3" stdDeviation="2" floodColor="#450a0a" floodOpacity="0.35" />
        </filter>
        {/* Red String Gradient */}
        <linearGradient id="red-string-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#dc2626" />
          <stop offset="50%" stopColor="#b91c1c" />
          <stop offset="100%" stopColor="#991b1b" />
        </linearGradient>
      </defs>

      {connections.map((conn, idx) => {
        const sag = conn.sag ?? 25;
        const midX = (conn.x1 + conn.x2) / 2;
        const midY = (conn.y1 + conn.y2) / 2 + sag;
        const pathData = `M ${conn.x1} ${conn.y1} Q ${midX} ${midY} ${conn.x2} ${conn.y2}`;

        // Approximate curve length for smooth dash animation
        const dx = conn.x2 - conn.x1;
        const dy = conn.y2 - conn.y1;
        const length = Math.sqrt(dx * dx + dy * dy) + sag * 1.5;

        return (
          <g key={conn.id || idx}>
            {/* Darker under-string shadow curve */}
            <path
              d={pathData}
              fill="none"
              stroke="#450a0a"
              strokeWidth="2.8"
              strokeOpacity="0.25"
              transform="translate(1, 3)"
            />

            {/* Main Red Forensic Thread */}
            <path
              d={pathData}
              fill="none"
              stroke="url(#red-string-grad)"
              strokeWidth="2.4"
              strokeLinecap="round"
              filter="url(#string-shadow)"
              style={{
                strokeDasharray: length,
                strokeDashoffset: animated ? 0 : length,
                transition: `stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1) ${idx * 0.18}s`,
              }}
            />

            {/* Little knots at ends */}
            <circle cx={conn.x1} cy={conn.y1} r="3" fill="#7f1d1d" opacity="0.8" />
            <circle cx={conn.x2} cy={conn.y2} r="3" fill="#7f1d1d" opacity="0.8" />
          </g>
        );
      })}
    </svg>
  );
};
