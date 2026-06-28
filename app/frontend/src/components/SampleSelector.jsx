import React from 'react';
import { motion } from 'framer-motion';

export default function SampleSelector({ 
  samples, 
  selectedSample, 
  onSelectSample, 
  onRandomSample, 
  revealLabel, 
  setRevealLabel,
  disabled,
  isStaticMode 
}) {
  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-xl">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
          Sample Selection
        </h2>
        <div className="flex gap-2">
          {isStaticMode ? (
            <span className="bg-purple-500/20 text-purple-300 text-xs px-2.5 py-1 rounded-full font-medium border border-purple-500/30">
              Static GitHub Demo
            </span>
          ) : (
            <span className="bg-emerald-500/20 text-emerald-300 text-xs px-2.5 py-1 rounded-full font-medium border border-emerald-500/30">
              Local Live Inference
            </span>
          )}
          <span className="bg-blue-500/20 text-blue-300 text-xs px-2.5 py-1 rounded-full font-medium border border-blue-500/30">
            Real Held-Out Test Sample
          </span>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-400 mb-1">Select from Database</label>
          <div className="flex gap-2">
            <select
              disabled={disabled}
              value={selectedSample?.sample_id || ''}
              onChange={(e) => {
                const s = samples.find(x => x.sample_id === e.target.value);
                if (s) onSelectSample(s);
              }}
              className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="" disabled>-- Select a test sample --</option>
              {samples.map(s => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.sample_id} - {s.mineral_name} {s.has_xrd ? '(Raman + XRD)' : '(Raman)'}
                </option>
              ))}
            </select>
            <button
              onClick={onRandomSample}
              disabled={disabled}
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded-lg font-medium transition-colors border border-slate-600 disabled:opacity-50"
            >
              Random
            </button>
          </div>
        </div>

        {selectedSample && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-slate-900/50 p-4 rounded-lg border border-slate-700/50 space-y-2"
          >
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-slate-400">Sample ID:</div>
              <div className="text-slate-200 font-mono">{selectedSample.sample_id}</div>
              <div className="text-slate-400">Mineral:</div>
              <div className="text-slate-200 font-medium">{selectedSample.mineral_name}</div>
              <div className="text-slate-400">Data Available:</div>
              <div className="text-slate-200">
                {selectedSample.has_raman && <span className="mr-2 text-green-400">✓ Raman</span>}
                {selectedSample.has_xrd ? <span className="text-green-400">✓ XRD</span> : <span className="text-slate-500">✗ XRD</span>}
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-700/50 flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={revealLabel}
                  onChange={(e) => setRevealLabel(e.target.checked)}
                  className="rounded border-slate-600 text-blue-500 focus:ring-blue-500/50 bg-slate-800"
                />
                <span className="text-sm text-slate-300 select-none">Reveal True Label</span>
              </label>
              {revealLabel && (
                <div className="flex gap-2">
                  <span className="text-xs font-medium text-slate-400 bg-slate-800 px-2 py-0.5 rounded border border-slate-600/50">
                    Orig: {selectedSample.original_12class_label}
                  </span>
                  <span className="text-xs font-medium text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                    True: {selectedSample.true_5class_label}
                  </span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
