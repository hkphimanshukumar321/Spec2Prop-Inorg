import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function MachineScanSimulation({ isScanning, onScanComplete }) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('Idle');

  useEffect(() => {
    if (!isScanning) {
      setProgress(0);
      setStatus('Idle');
      return;
    }

    const steps = [
      { p: 10, msg: 'Initializing Raman laser (532nm)...', wait: 500 },
      { p: 35, msg: 'Acquiring spectral response...', wait: 1200 },
      { p: 70, msg: 'Applying baseline correction...', wait: 800 },
      { p: 100, msg: 'Scan complete. Spectral fingerprint acquired.', wait: 500 },
    ];

    let currentStep = 0;
    
    const runStep = async () => {
      if (currentStep >= steps.length) {
        setTimeout(() => onScanComplete(), 500);
        return;
      }
      const step = steps[currentStep];
      setStatus(step.msg);
      setProgress(step.p);
      currentStep++;
      setTimeout(runStep, step.wait);
    };

    runStep();

  }, [isScanning, onScanComplete]);

  return (
    <div className="bg-slate-800 rounded-xl p-5 border border-slate-700 shadow-xl overflow-hidden relative">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          <svg className="w-5 h-5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
          </svg>
          Instrument Control
        </h2>
        <span className="text-xs text-slate-500 italic">Visual simulation</span>
      </div>

      <div className="relative h-48 bg-slate-900 rounded-lg border border-slate-700 flex flex-col items-center justify-center p-4">
        {/* Animated Laser/Scanner Graphic */}
        <div className="relative w-32 h-32 mb-4">
          <div className="absolute inset-0 border-4 border-slate-700 rounded-full"></div>
          
          {isScanning && (
            <>
              {/* Spinning scanning ring */}
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="absolute inset-0 border-4 border-transparent border-t-indigo-500 rounded-full"
              />
              {/* Laser beam */}
              <motion.div 
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: [0.5, 1, 0.5], height: '100%' }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="absolute left-1/2 -ml-0.5 top-0 w-1 bg-indigo-500 blur-[1px]"
              />
            </>
          )}
          
          <div className="absolute inset-0 flex items-center justify-center">
            <svg className={`w-10 h-10 ${isScanning ? 'text-indigo-400' : 'text-slate-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full max-w-xs">
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span>{status}</span>
            <span>{progress}%</span>
          </div>
          <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
            <motion.div 
              className="bg-indigo-500 h-1.5 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ type: 'spring', bounce: 0, duration: 0.5 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
