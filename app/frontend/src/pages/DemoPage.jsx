import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { getSamples, getRandomSample, getSampleDetails, runInference } from '../api/client';
import SampleSelector from '../components/SampleSelector';
import MachineScanSimulation from '../components/MachineScanSimulation';
import SpectrumViewer from '../components/SpectrumViewer';
import PredictionReport from '../components/PredictionReport';

export default function DemoPage() {
  const [isStaticMode, setIsStaticMode] = useState(false);
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [sampleDetails, setSampleDetails] = useState(null);
  
  const [isScanning, setIsScanning] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  
  const [inferenceResult, setInferenceResult] = useState(null);
  const [pendingStaticResult, setPendingStaticResult] = useState(null);
  const [revealLabel, setRevealLabel] = useState(false);
  
  const [error, setError] = useState(null);

  // Initial load: Try live backend first, fallback to static JSON
  useEffect(() => {
    getSamples(100)
      .then(data => {
        setSamples(data.samples);
        setIsStaticMode(false);
      })
      .catch(async () => {
        // Fallback to static mode
        setIsStaticMode(true);
        try {
          const res = await axios.get(`${import.meta.env.BASE_URL}demo_samples/index.json`);
          setSamples(res.data.samples);
        } catch (err) {
          setError('Failed to load live backend and no static demo files found.');
        }
      });
  }, []);

  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setSampleDetails(null);
    setInferenceResult(null);
    setPendingStaticResult(null);
    setRevealLabel(false);
    setError(null);
    
    try {
      if (isStaticMode) {
        const res = await axios.get(`${import.meta.env.BASE_URL}demo_samples/${sample.sample_id}.json`);
        setSampleDetails({
          raman_spectrum: {
            x: res.data.processed_raman_x,
            y: res.data.processed_raman_y
          }
        });
        // Store in pending until user hits Start Scan
        setPendingStaticResult(res.data);
      } else {
        const details = await getSampleDetails(sample.sample_id);
        setSampleDetails(details);
      }
    } catch (err) {
      setError('Failed to load sample details.');
    }
  };

  const handleRandomSample = async () => {
    try {
      if (isStaticMode) {
        const idx = Math.floor(Math.random() * samples.length);
        handleSelectSample(samples[idx]);
      } else {
        const data = await getRandomSample();
        handleSelectSample(data.sample);
      }
    } catch (err) {
      setError('Failed to load random sample.');
    }
  };

  const startScan = () => {
    if (!selectedSample) return;
    setIsScanning(true);
    setInferenceResult(null);
  };

  const onScanComplete = async () => {
    setIsScanning(false);
    setIsInferring(true);
    
    try {
      await new Promise(r => setTimeout(r, 600)); // UI delay
      
      if (!isStaticMode) {
        const result = await runInference({
          sample_id: selectedSample.sample_id,
          task: 'family',
          modality: 'raman'
        });
        setInferenceResult(result);
      } else {
        // If static mode, populate inferenceResult from the pending payload
        setInferenceResult(pendingStaticResult);
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference failed.');
    } finally {
      setIsInferring(false);
    }
  };

  const resetFlow = () => {
    setSelectedSample(null);
    setSampleDetails(null);
    setInferenceResult(null);
    setRevealLabel(false);
    setError(null);
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-4 space-y-6">
      
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-100 tracking-tight">Live Material Screening</h1>
        <p className="text-slate-400 mt-2">
          Select a sample from the held-out test set, acquire its Raman spectrum, and run the hybrid LightGBM-DART deployment model for candidate prioritization.
        </p>
      </div>

      {error && (
        <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-lg flex items-center gap-3">
          <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}

      {/* Main Grid: Controls vs Visuals */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Flow Control */}
        <div className="space-y-6 flex flex-col">
          <SampleSelector 
            samples={samples} 
            selectedSample={selectedSample}
            onSelectSample={handleSelectSample}
            onRandomSample={handleRandomSample}
            revealLabel={revealLabel}
            setRevealLabel={setRevealLabel}
            disabled={isScanning || isInferring}
            isStaticMode={isStaticMode}
          />

          <MachineScanSimulation 
            isScanning={isScanning} 
            onScanComplete={onScanComplete} 
          />
          
          <div className="mt-auto pt-6 flex flex-col gap-3">
            <button
              onClick={startScan}
              disabled={!selectedSample || isScanning || isInferring || inferenceResult !== null}
              className={`w-full py-3 rounded-lg font-bold shadow-lg transition-all ${
                !selectedSample || isScanning || isInferring || inferenceResult !== null
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed shadow-none'
                  : 'bg-blue-600 hover:bg-blue-500 text-white hover:shadow-blue-500/25 active:scale-95'
              }`}
            >
              {isScanning ? 'Acquiring Spectrum...' : (isInferring ? 'Running Inference...' : '▶ Start Spectral Scan')}
            </button>
          </div>
        </div>

        {/* Right Column: Visuals & Results */}
        <div className="lg:col-span-2 space-y-6">
          <SpectrumViewer 
            ramanSpectrum={sampleDetails?.raman_spectrum} 
            isScanning={isScanning} 
          />

          {isInferring && (
            <div className="bg-slate-800 rounded-xl p-8 border border-slate-700 shadow-xl flex flex-col items-center justify-center space-y-4">
              <div className="w-10 h-10 border-4 border-slate-700 border-t-blue-500 rounded-full animate-spin"></div>
              <div className="text-slate-300 font-medium animate-pulse">Extracting Domain & Prototype Features...</div>
              <div className="text-xs text-slate-500 font-mono">Invoking LightGBM-DART Pipeline...</div>
            </div>
          )}

          {inferenceResult && !isScanning && !isInferring && (
            <PredictionReport 
              result={inferenceResult} 
              onReset={resetFlow}
            />
          )}
        </div>
      </div>
      
      {/* Footer disclaimer */}
      <div className="mt-12 text-center text-xs text-slate-500">
        {isStaticMode ? (
          "This GitHub-hosted demo is static. The instrument animation is simulated, but the displayed spectra and predictions are generated from real held-out test samples using the trained Spec2Prop-Edge pipeline."
        ) : (
          "Demo intended for research evaluation. Predictions are derived solely from Raman spectra using Spec2Prop-InorgBench."
        )}
      </div>
    </div>
  );
}
