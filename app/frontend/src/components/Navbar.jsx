import React from 'react';

export default function Navbar({ currentView, setView }) {
  return (
    <nav className="bg-slate-900 border-b border-slate-800 p-4">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center font-bold text-white shadow-lg shadow-blue-500/20">
            S2P
          </div>
          <h1 className="text-xl font-bold text-slate-100 tracking-tight">
            Spec2Prop<span className="text-blue-400">-Edge</span>
          </h1>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => setView('home')}
            className={`text-sm font-medium transition-colors ${currentView === 'home' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-200'}`}
          >
            Home
          </button>
          <button
            onClick={() => setView('demo')}
            className={`text-sm font-medium transition-colors ${currentView === 'demo' ? 'text-blue-400' : 'text-slate-400 hover:text-slate-200'}`}
          >
            Live Demo
          </button>
        </div>
      </div>
    </nav>
  );
}
