import React, { useEffect, useRef, useState } from 'react';
import { Upload, Search, CheckCircle2, FileText, ArrowRight } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.25 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const steps = [
    {
      step: '01',
      title: 'UPLOAD PHOTO',
      icon: Upload,
      description: 'Submit suspicious digital media for forensic examination.',
      visual: (
        <div className="w-full h-32 bg-stone-900 rounded-sm p-3 flex flex-col items-center justify-center border border-stone-700 text-stone-300 relative overflow-hidden group">
          <div className="w-10 h-10 rounded-full bg-red-950/60 border border-red-700 flex items-center justify-center text-red-400 mb-2">
            <Upload className="w-5 h-5 animate-pulse" />
          </div>
          <span className="text-xs font-mono text-stone-950">DROP MEDIA HERE (.JPG, .PNG)</span>
          <span className="text-[11px] font-mono text-red-400 mt-1">SHA-256 HASH CALCULATING...</span>
        </div>
      ),
    },
    {
      step: '02',
      title: 'AI SCANS',
      icon: Search,
      description: 'Multi-model neural ensemble inspects latent patterns & ELA residual maps.',
      visual: (
        <div className="w-full h-32 bg-stone-900 rounded-sm p-2 relative overflow-hidden border border-stone-700">
          <img
            src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=400&q=80"
            alt="Scanning"
            className="w-full h-full object-cover opacity-60"
          />
          {/* Animated Magnifying Lens Sweep */}
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-16 h-16 rounded-full border-2 border-red-500 bg-red-500/20 backdrop-blur-xs flex items-center justify-center shadow-lg shadow-red-900/50 animate-bounce">
              <Search className="w-6 h-6 text-red-300" />
            </div>
          </div>
          <div className="absolute bottom-1 right-1 bg-black/80 text-red-400 text-[11px] font-mono px-1.5 py-0.5 rounded">
            SCANNING... 88%
          </div>
        </div>
      ),
    },
    {
      step: '03',
      title: 'AUTHENTICITY SCORE',
      icon: CheckCircle2,
      description: 'Neural models calculate final fused authenticity rating & confidence metrics.',
      visual: (
        <div className="w-full h-32 bg-stone-900 rounded-sm p-3 flex flex-col items-center justify-center border border-stone-700 text-stone-100">
          <div className="text-3xl font-black font-mono text-emerald-400 tracking-tight">94%</div>
          <div className="text-xs font-mono font-bold text-emerald-300 uppercase tracking-widest mt-1">
            LIKELY AUTHENTIC
          </div>
          <div className="w-full bg-stone-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div className="bg-emerald-500 h-full w-[94%]" />
          </div>
          <span className="text-[11px] font-mono text-stone-950 mt-2">HIGH MODEL AGREEMENT</span>
        </div>
      ),
    },
    {
      step: '04',
      title: 'CASE REPORT',
      icon: FileText,
      description: 'Generate cryptographically signed forensic audit binder with metadata traces.',
      visual: (
        <div className="w-full h-32 bg-stone-900 rounded-sm p-3 border border-stone-700 flex flex-col justify-between relative overflow-hidden">
          <div className="flex justify-between items-center text-xs font-mono text-stone-950">
            <span>BINDER #FQ-2026</span>
            <span className="text-emerald-400">PASSED</span>
          </div>
          <div className="space-y-1 my-1">
            <div className="w-3/4 h-1.5 bg-stone-700 rounded" />
            <div className="w-full h-1.5 bg-stone-800 rounded" />
            <div className="w-2/3 h-1.5 bg-stone-700 rounded" />
          </div>
          {/* Distressed Stamp Overlay */}
          <div className="absolute bottom-2 right-2 border-2 border-dashed border-emerald-500 text-emerald-400 text-xs font-mono font-black px-1.5 py-0.5 transform -rotate-12 bg-emerald-950/40">
            VERIFIED
          </div>
        </div>
      ),
    },
  ];

  return (
    <section id="how-it-works" ref={sectionRef} className="py-20 px-4 sm:px-6 lg:px-12 max-w-7xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-14 space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase">
          <span>INVESTIGATION PIPELINE</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif font-bold text-stone-900 tracking-tight">
          How ForensIQ Authenticates Media
        </h2>
        <p className="text-stone-950 text-sm leading-relaxed">
          Four interconnected forensic stages executed autonomously from initial upload to final cryptographic evidence sign-off.
        </p>
      </div>

      {/* Cards Container with Connecting Red Strings */}
      <div className="relative">
        {/* Animated Connecting String SVG line across desktop */}
        <svg
          className="absolute top-1/2 left-0 w-full h-8 -translate-y-1/2 hidden md:block pointer-events-none z-10"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M 120 16 Q 360 4 600 16 T 1080 16"
            fill="none"
            stroke="#dc2626"
            strokeWidth="3"
            strokeDasharray="1000"
            strokeDashoffset={isVisible ? 0 : 1000}
            style={{ transition: 'stroke-dashoffset 2s ease-in-out' }}
          />
        </svg>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8 relative z-20">
          {steps.map((item, idx) => {
            const Icon = item.icon;
            return (
              <div
                key={item.step}
                className={`bg-[#faf8f5] border border-stone-300 p-5 rounded-md shadow-lg transition-all duration-500 relative ${
                  isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
                }`}
                style={{ transitionDelay: `${idx * 150}ms` }}
              >
                {/* Top Pushpin */}
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 z-30">
                  <div className="w-4 h-4 rounded-full bg-red-700 border border-red-900 shadow-md" />
                </div>

                <div className="flex items-center justify-between border-b border-stone-200 pb-2 mb-3">
                  <span className="font-mono font-black text-xl text-red-700">{item.step}</span>
                  <div className="w-7 h-7 rounded bg-stone-900 text-stone-100 flex items-center justify-center">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                </div>

                <h3 className="font-serif font-bold text-stone-900 text-base mb-1">{item.title}</h3>
                <p className="text-xs text-stone-950 mb-4 h-10 leading-snug">{item.description}</p>

                {/* Card Visual Stage */}
                {item.visual}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};
