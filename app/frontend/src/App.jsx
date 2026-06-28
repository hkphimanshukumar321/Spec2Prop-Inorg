import React, { useState } from 'react';
import Navbar from './components/Navbar';
import DemoPage from './pages/DemoPage';

function App() {
  const [currentView, setView] = useState('demo');

  return (
    <div className="min-h-screen bg-slate-950 font-sans selection:bg-blue-500/30">
      <Navbar currentView={currentView} setView={setView} />
      
      {currentView === 'demo' ? (
        <DemoPage />
      ) : (
        <div className="max-w-4xl mx-auto py-20 px-4 text-center">
          <h1 className="text-5xl font-extrabold text-white tracking-tight mb-6">
            Intelligent Inorganic <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500">
              Material Screening
            </span>
          </h1>
          <p className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto">
            A production-ready machine learning pipeline for rapid, non-destructive candidate prioritization and confidence-aware decision support using raw Raman spectroscopy.
          </p>
          <button 
            onClick={() => setView('demo')}
            className="px-8 py-4 bg-white text-slate-900 font-bold rounded-full hover:bg-slate-200 transition-transform hover:scale-105 shadow-xl shadow-white/10"
          >
            Launch Live Demo
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
