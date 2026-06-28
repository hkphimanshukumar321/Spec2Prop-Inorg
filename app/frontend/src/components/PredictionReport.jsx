import React from 'react';
import { motion } from 'framer-motion';

export default function PredictionReport({ result, onReset }) {
  if (!result) return null;

  const isHighConfidence = result.prediction_quality === 'High';
  const colorClass = isHighConfidence ? 'text-emerald-400' : (result.prediction_quality === 'Medium' ? 'text-amber-400' : 'text-rose-400');
  const bgClass = isHighConfidence ? 'bg-emerald-400/10' : (result.prediction_quality === 'Medium' ? 'bg-amber-400/10' : 'bg-rose-400/10');
  const borderClass = isHighConfidence ? 'border-emerald-400/20' : (result.prediction_quality === 'Medium' ? 'border-amber-400/20' : 'border-rose-400/20');

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-xl relative overflow-hidden"
    >
      <div className={`absolute -right-20 -top-20 w-64 h-64 rounded-full blur-[80px] opacity-20 ${bgClass.replace('/10', '')}`} />

      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
            Prediction Results
            {isHighConfidence && (
              <svg className="w-6 h-6 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
          </h2>
          <div className="flex gap-2 mt-1">
            <span className="bg-slate-700 text-slate-300 text-xs px-2 py-0.5 rounded border border-slate-600">Runtime Preprocessing</span>
            <span className="bg-slate-700 text-slate-300 text-xs px-2 py-0.5 rounded border border-slate-600">Trained Model Output</span>
          </div>
          <p className="text-slate-400 text-sm mt-1">
            Processed by {result.model_name} in {result.inference_time_ms}ms ({result.feature_dim} features)
          </p>
        </div>
        <button
          onClick={onReset}
          className="text-slate-400 hover:text-white px-3 py-1.5 rounded bg-slate-700 hover:bg-slate-600 transition-colors text-sm"
        >
          New Scan
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Broad 5-Class Prediction */}
        <div className={`p-5 rounded-xl border ${borderClass} ${bgClass} flex flex-col justify-center`}>
          <div className="text-sm font-medium text-slate-400 mb-1">Broad Family Group</div>
          <div className={`text-4xl font-bold ${colorClass} mb-2 tracking-tight`}>
            {result.predicted_5class_label}
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="text-slate-300">Confidence:</div>
            <div className="font-mono font-medium text-white">{(result.predicted_5class_confidence * 100).toFixed(1)}%</div>
            <div className={`px-2 py-0.5 rounded text-xs font-medium border ${borderClass} ${colorClass}`}>
              {result.prediction_quality} Quality
            </div>
          </div>
        </div>

        {/* Fine 9-Class Candidate Ranking */}
        <div className="bg-slate-900/50 p-4 rounded-xl border border-slate-700">
          <div className="text-sm font-medium text-slate-400 mb-3">Fine 9-Class Candidate Ranking</div>
          <div className="space-y-3">
            {result.top3_9class && result.top3_9class.map((item, idx) => (
              <div key={item.label} className="relative">
                <div className="flex justify-between text-sm mb-1">
                  <span className={idx === 0 ? 'text-white font-medium flex gap-1' : 'text-slate-300 flex gap-1'}>
                    <span className="text-slate-500">{idx + 1}.</span> {item.label}
                  </span>
                  <span className="font-mono text-slate-400">{(item.probability * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${item.probability * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.1 * idx }}
                    className={`h-1.5 rounded-full ${idx === 0 ? (isHighConfidence ? 'bg-emerald-500' : 'bg-amber-500') : 'bg-slate-600'}`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Actionable Recommendation */}
      <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg flex items-start gap-3">
        <svg className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <div>
          <div className="text-sm font-medium text-blue-300">Scientific Suggestion</div>
          <div className="text-sm text-blue-100/80 mt-1">{result.recommendation}</div>
        </div>
      </div>
    </motion.div>
  );
}
