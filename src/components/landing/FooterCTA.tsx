import React from 'react';
import { ArrowRight, ShieldCheck, Lock, ExternalLink } from 'lucide-react';

interface FooterCTAProps {
  onLaunchPlatform: () => void;
}

export const FooterCTA: React.FC<FooterCTAProps> = ({ onLaunchPlatform }) => {
  return (
    <section className="py-24 bg-[#0f172a] text-white border-t-4 border-red-700 relative overflow-hidden">
      {/* Background Subtle Forensic Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:32px_32px] pointer-events-none" />

      <div className="max-w-5xl mx-auto px-4 text-center relative z-10 space-y-8">
        <div className="w-14 h-14 rounded-2xl bg-red-950/80 border-2 border-red-600 text-red-400 mx-auto flex items-center justify-center shadow-2xl">
          <ShieldCheck className="w-8 h-8" />
        </div>

        <h2 className="text-3xl sm:text-5xl font-serif font-black tracking-tight text-stone-100 uppercase">
          JOIN THE INVESTIGATION
        </h2>

        <p className="text-stone-300 text-sm sm:text-base max-w-xl mx-auto leading-relaxed font-sans">
          Deploy multi-model neural ensemble diagnostics, robust EOT perceptual watermarking, and reverse pHash dissemination tracking across your organization.
        </p>

        {/* Large Interactive Forensic Red Ink Stamp / Button */}
        <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
          <button
            onClick={onLaunchPlatform}
            className="group relative inline-flex items-center gap-3 px-8 py-4 bg-red-700 hover:bg-red-600 text-white font-mono font-black text-base sm:text-lg rounded-full shadow-2xl transition-all duration-300 transform hover:rotate-2 hover:scale-95 active:scale-90 cursor-none border-2 border-red-500 uppercase tracking-wider"
          >
            <span>LAUNCH PLATFORM</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>

        {/* Forensic C2PA Compliance Badges */}
        <div className="pt-8 flex flex-wrap items-center justify-center gap-6 text-xs font-mono text-stone-400 border-t border-stone-800 max-w-2xl mx-auto">
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4 text-red-500" />
            <span>C2PA STANDARDS COMPLIANT</span>
          </div>
          <span>•</span>
          <div>SHA-256 EVIDENCE AUDIT</div>
          <span>•</span>
          <div>VERSION 2.4 STABLE</div>
        </div>
      </div>

      {/* Footer Navigation & Copyright */}
      <footer className="mt-20 pt-8 border-t border-stone-800 text-xs font-mono text-stone-500 max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-red-500" />
          <span className="font-bold text-stone-300">FORENSIQ ENGINE</span>
          <span>© 2026 Digital Forensic Intelligence</span>
        </div>

        <div className="flex items-center gap-6">
          <a href="#how-it-works" className="hover:text-stone-300 transition-colors">
            HOW IT WORKS
          </a>
          <a href="#modules" className="hover:text-stone-300 transition-colors">
            MODULES
          </a>
          <a href="#propagation" className="hover:text-stone-300 transition-colors">
            3D PROPAGATION
          </a>
          <button onClick={onLaunchPlatform} className="text-red-400 hover:text-red-300 transition-colors flex items-center gap-1 font-bold">
            <span>PLATFORM WORKSPACE</span>
            <ExternalLink className="w-3 h-3" />
          </button>
        </div>
      </footer>
    </section>
  );
};
