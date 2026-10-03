import React from 'react';
import {
  FileCode,
  Layers,
  Eye,
  Copy,
  Radio,
  UserX,
  Cpu,
  ShieldCheck,
} from 'lucide-react';

export const DetectionModules: React.FC = () => {
  const modules = [
    {
      id: 'mod-1',
      title: 'EXIF ANALYSIS',
      icon: FileCode,
      tag: 'METADATA AUDIT',
      rotation: -2,
      frontDesc: 'Camera Hardware & EXIF Structural Integrity',
      backDesc:
        'Audits embedded XMP metadata headers, camera hardware strings, lens profiles, and creation timestamps for splicing anomalies.',
    },
    {
      id: 'mod-2',
      title: 'ERROR LEVEL ANALYSIS',
      icon: Layers,
      tag: 'FREQUENCY ELA',
      rotation: 3,
      frontDesc: 'JPEG Quantization Disparity Map',
      backDesc:
        'Re-compresses image frequencies to isolate non-uniform compression error levels across localized image patches.',
    },
    {
      id: 'mod-3',
      title: 'GRAD-CAM',
      icon: Eye,
      tag: 'NEURAL ATTENTION',
      rotation: -3,
      frontDesc: 'Convolutional Heatmap Localization',
      backDesc:
        'Visualizes neural attention heatmaps to identify specific pixel regions driving synthetic generation flags.',
    },
    {
      id: 'mod-4',
      title: 'COPY-MOVE DETECTION',
      icon: Copy,
      tag: 'CLONING PATTERNS',
      rotation: 2,
      frontDesc: 'Keypoint & Block-Matching Analysis',
      backDesc:
        'Identifies duplicated pixel regions, cloned textures, and copy-pasted document elements using SIFT keypoints.',
    },
    {
      id: 'mod-5',
      title: 'SENSOR NOISE',
      icon: Radio,
      tag: 'PRNU FINGERPRINT',
      rotation: -4,
      frontDesc: 'Photo-Response Non-Uniformity Audit',
      backDesc:
        'Extracts physical sensor noise signatures to verify hardware camera origin and detect localized image insertion.',
    },
    {
      id: 'mod-6',
      title: 'DEEPFAKE DETECTION',
      icon: UserX,
      tag: 'FACIAL ENSEMBLE',
      rotation: 4,
      frontDesc: 'Generative Artifact Neural Classifier',
      backDesc:
        'AI-based analysis identifies visual patterns associated with manipulated or synthetically generated media.',
    },
  ];

  return (
    <section id="modules" className="py-20 px-4 sm:px-6 lg:px-12 max-w-7xl mx-auto">
      <div className="text-center max-w-2xl mx-auto mb-14 space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase">
          <span>DIAGNOSTIC ENGINE MATRIX</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-serif font-bold text-stone-900 tracking-tight">
          Forensic Detection Modules
        </h2>
        <p className="text-stone-600 text-sm leading-relaxed">
          Hover over any evidence card to examine the specific detection methodology.
        </p>
      </div>

      {/* Grid of 3D Flipping Evidence Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
        {modules.map((mod) => {
          const Icon = mod.icon;
          return (
            <div
              key={mod.id}
              style={{ transform: `rotate(${mod.rotation}deg)` }}
              className="group h-64 [perspective:1000px] cursor-pointer relative"
            >
              {/* Top Red Pushpin */}
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
                <div className="w-4 h-4 rounded-full bg-red-700 border border-red-900 shadow-md" />
              </div>

              {/* Card Container for 3D Flip */}
              <div className="relative h-full w-full rounded-sm border border-stone-300 shadow-xl transition-all duration-500 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)] bg-[#faf8f5]">
                {/* FRONT FACE */}
                <div className="absolute inset-0 h-full w-full p-6 [backface-visibility:hidden] flex flex-col justify-between rounded-sm">
                  <div className="flex items-center justify-between border-b border-stone-200 pb-3">
                    <span className="font-mono text-[10px] font-bold text-red-700 bg-red-50 border border-red-200 px-2 py-0.5 rounded uppercase">
                      {mod.tag}
                    </span>
                    <div className="w-8 h-8 rounded bg-stone-900 text-stone-100 flex items-center justify-center">
                      <Icon className="w-4 h-4" />
                    </div>
                  </div>

                  <div className="my-auto space-y-2">
                    <h3 className="font-serif font-bold text-xl text-stone-900">{mod.title}</h3>
                    <p className="text-xs text-stone-600 font-mono">{mod.frontDesc}</p>
                  </div>

                  <div className="pt-2 border-t border-stone-200 text-[10px] font-mono text-stone-400 flex items-center justify-between">
                    <span>STATUS: ACTIVE MODULE</span>
                    <span className="text-red-700 font-bold group-hover:underline">HOVER TO FLIP ➔</span>
                  </div>
                </div>

                {/* BACK FACE */}
                <div className="absolute inset-0 h-full w-full p-6 [backface-visibility:hidden] [transform:rotateY(180deg)] bg-stone-900 text-stone-100 rounded-sm flex flex-col justify-between border border-red-900">
                  <div className="flex items-center justify-between border-b border-stone-800 pb-2">
                    <span className="font-mono text-[10px] text-red-400 font-bold">METHODOLOGY SPEC</span>
                    <Icon className="w-4 h-4 text-red-500" />
                  </div>

                  <div className="my-auto">
                    <h4 className="font-serif font-bold text-base text-red-400 mb-2">{mod.title}</h4>
                    <p className="text-xs text-stone-300 leading-relaxed font-sans">{mod.backDesc}</p>
                  </div>

                  <div className="text-[9px] font-mono text-stone-500 pt-2 border-t border-stone-800">
                    FORENSIQ PARALLEL INFERENCE PIPELINE
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Central Authenticity Engine Node Visual */}
      <div className="mt-16 relative flex flex-col items-center">
        {/* SVG Red Connection Lines */}
        <svg className="w-full h-16 pointer-events-none mb-2" viewBox="0 0 800 60">
          <path d="M 200 10 Q 400 50 400 50" fill="none" stroke="#dc2626" strokeWidth="2" strokeDasharray="4 4" />
          <path d="M 600 10 Q 400 50 400 50" fill="none" stroke="#dc2626" strokeWidth="2" strokeDasharray="4 4" />
          <circle cx="400" cy="50" r="4" fill="#dc2626" />
        </svg>

        <div className="bg-[#0f172a] text-white px-6 py-3 rounded-full border-2 border-red-600 shadow-2xl flex items-center gap-3">
          <Cpu className="w-5 h-5 text-red-500 animate-spin" />
          <div className="font-mono text-xs">
            <span className="font-bold text-stone-200">AUTHENTICITY ENGINE</span>
            <span className="mx-2 text-stone-500">•</span>
            <span className="text-red-400 font-bold">PARALLEL ENSEMBLE FUSION</span>
          </div>
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
        </div>
      </div>
    </section>
  );
};
