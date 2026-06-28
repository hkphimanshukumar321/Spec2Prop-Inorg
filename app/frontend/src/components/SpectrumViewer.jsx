import React, { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function SpectrumViewer({ ramanSpectrum, isScanning }) {
  // Subsample data to prevent browser lag (2048 points is a lot for recharts)
  const chartData = useMemo(() => {
    if (!ramanSpectrum) return [];
    const step = Math.ceil(ramanSpectrum.x.length / 500); // Max 500 points
    const data = [];
    for (let i = 0; i < ramanSpectrum.x.length; i += step) {
      data.push({
        x: Math.round(ramanSpectrum.x[i]),
        y: ramanSpectrum.y[i]
      });
    }
    return data;
  }, [ramanSpectrum]);

  if (isScanning || !ramanSpectrum) {
    return (
      <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-xl h-80 flex flex-col items-center justify-center text-slate-500">
        <svg className="w-12 h-12 mb-3 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
        <p>{isScanning ? 'Acquiring spectrum data...' : 'Waiting for sample selection...'}</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
          </svg>
          Processed Raman Spectrum
        </h2>
        <span className="text-xs text-slate-400">Baseline corrected & Normalized</span>
      </div>

      <div className="h-64 w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis 
              dataKey="x" 
              stroke="#94a3b8" 
              tick={{ fill: '#94a3b8' }} 
              label={{ value: 'Raman Shift (cm⁻¹)', position: 'insideBottomRight', offset: -5, fill: '#94a3b8' }} 
            />
            <YAxis 
              stroke="#94a3b8" 
              tick={{ fill: '#94a3b8' }} 
              domain={['auto', 'auto']}
              label={{ value: 'Intensity', angle: -90, position: 'insideLeft', fill: '#94a3b8' }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#f8fafc' }}
              itemStyle={{ color: '#38bdf8' }}
              formatter={(value) => value.toFixed(3)}
              labelFormatter={(label) => `${label} cm⁻¹`}
            />
            <Line 
              type="monotone" 
              dataKey="y" 
              stroke="#38bdf8" 
              strokeWidth={1.5} 
              dot={false} 
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
