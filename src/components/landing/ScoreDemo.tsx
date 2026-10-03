import React, { useEffect, useState } from 'react';
import { Search, ShieldCheck, RefreshCw, Cpu } from 'lucide-react';

export const ScoreDemo: React.FC = () => {
  const [scanStep, setScanStep] = useState<number>(0);

  // Automated scanning loop sequence
  useEffect(() => {
    const interval = setInterval(() => {
      setScanStep((prev) => (prev + 1) % 4);
    }, 2800);

    return () => clearInterval(interval);
  }, []);

  const resetScan = () => {
    setScanStep(0);
  };

  return (
    <section id="score-demo" className="py-20 px-4 sm:px-6 lg:px-12 max-w-7xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-12 space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase">
          <span>AUTOMATED DEMONSTRATION</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif font-bold text-stone-900 tracking-tight">
          Live Forensic Verification Sweep
        </h2>
        <p className="text-stone-600 text-sm leading-relaxed">
          Watch the automated forensic scanner evaluate spatial noise residuals and stamp final authenticity verdict.
        </p>
      </div>

      {/* Main Score Demo Stage */}
      <div className="bg-[#f5f0e6] border-2 border-[#e6dfd1] rounded-3xl p-6 sm:p-10 shadow-2xl relative overflow-hidden max-w-4xl mx-auto">
        {/* Header Bar inside Demo */}
        <div className="flex items-center justify-between border-b border-stone-300 pb-4 mb-6">
          <div className="flex items-center gap-2 font-mono text-xs text-stone-700">
            <Cpu className="w-4 h-4 text-red-600 animate-spin" />
            <span className="font-bold">INSPECTION TARGET: DEMO_CANDIDATE_44.JPG</span>
          </div>

          <button
            onClick={resetScan}
            className="px-3 py-1.5 bg-stone-900 text-stone-100 text-xs font-mono rounded hover:bg-stone-800 transition-colors flex items-center gap-1.5 cursor-none"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>RESTART SWEEP</span>
          </button>
        </div>

        {/* Pinned Image Stage */}
        <div className="relative w-full h-80 sm:h-96 bg-stone-900 rounded-lg overflow-hidden border border-stone-700 shadow-xl flex items-center justify-center">
          {/* Target Sample Photo */}
          <img
            src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=1000&q=80"
            alt="Demo Target"
            className="w-full h-full object-cover filter contrast-105"
          />

          {/* Sweep Grid Line overlay */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(239,68,68,0.1)_1px,transparent_1px),linear-gradient(to_bottom,rgba(239,68,68,0.1)_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />

          {/* Automatic Sweeping Magnifying Glass */}
          <div
            className="absolute z-20 pointer-events-none transition-all duration-1000 ease-in-out"
            style={{
              top: scanStep === 0 ? '20%' : scanStep === 1 ? '50%' : scanStep === 2 ? '40%' : '30%',
              left: scanStep === 0 ? '20%' : scanStep === 1 ? '70%' : scanStep === 2 ? '45%' : '50%',
            }}
          >
            <div className="relative -translate-x-1/2 -translate-y-1/2">
              <div className="w-32 h-32 rounded-full border-4 border-red-600 bg-red-950/20 backdrop-blur-[2px] shadow-2xl flex items-center justify-center relative overflow-hidden animate-pulse">
                <Search className="w-8 h-8 text-red-400" />
                <div className="absolute inset-0 bg-gradient-to-tr from-red-500/30 to-transparent" />
              </div>
            </div>
          </div>

          {/* Live Scanning Status Overlay */}
          <div className="absolute top-4 left-4 z-30 bg-black/80 text-stone-100 px-3 py-1.5 rounded font-mono text-xs border border-stone-700 backdrop-blur-xs space-y-0.5">
            <div className="text-red-400 font-bold flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
              <span>
                {scanStep === 0 && 'STEP 1: INITIALIZING SCAN...'}
                {scanStep === 1 && 'STEP 2: CHECKING ELA RESIDUALS...'}
                {scanStep === 2 && 'STEP 3: COMPUTING ENSEMBLE SCORE...'}
                {scanStep >= 3 && 'VERDICT: SCAN COMPLETE'}
              </span>
            </div>
            <div className="text-[10px] text-stone-400">
              PRNU SENSOR MATCH: 98.4% | FREQUENCY VARIANCE: OK
            </div>
          </div>

          {/* Rubber Stamp Impact Animation when step >= 3 */}
          {scanStep >= 3 && (
            <div className="absolute inset-0 z-40 flex items-center justify-center pointer-events-none">
              <div className="border-8 border-dashed border-emerald-600 text-emerald-500 px-8 py-4 rounded-md font-mono font-black text-4xl sm:text-6xl tracking-widest uppercase transform -rotate-12 bg-emerald-950/80 backdrop-blur-xs shadow-2xl animate-[bounce_0.6s_ease-out]">
                <div className="flex items-center gap-3">
                  <ShieldCheck className="w-10 h-10 sm:w-16 sm:h-16 text-emerald-400" />
                  <div>
                    <div>94% AUTHENTIC</div>
                    <div className="text-xs sm:text-base tracking-normal text-emerald-300 font-normal">
                      CRYPTOGRAPHICALLY VERIFIED
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
