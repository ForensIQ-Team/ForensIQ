import React, { useState, useRef } from 'react';

export interface EvidencePhotoProps {
  id: string;
  title: string;
  caseNumber: string;
  imageUrl: string;
  tamperedUrl?: string;
  rotation: number;
  positionClass?: string;
  isCenter?: boolean;
  isFake?: boolean;
  fakeStampText?: string;
  tag?: string;
  onHoverChange?: (isHovered: boolean) => void;
}

export const EvidencePhoto: React.FC<EvidencePhotoProps> = ({
  id,
  title,
  caseNumber,
  imageUrl,
  tamperedUrl,
  rotation,
  positionClass = '',
  isCenter = false,
  isFake = true,
  fakeStampText = 'FAKE',
  onHoverChange,
}) => {
  const [isHovered, setIsHovered] = useState(false);
  const [localPos, setLocalPos] = useState({ x: -100, y: -100 });
  const photoRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!photoRef.current) return;
    const rect = photoRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setLocalPos({ x, y });
  };

  const handleMouseEnter = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsHovered(true);
    if (onHoverChange) onHoverChange(true);
    handleMouseMove(e);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    if (onHoverChange) onHoverChange(false);
  };

  return (
    <div
      ref={photoRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      style={{
        transform: `rotate(${rotation}deg) translateZ(0)`,
      }}
      className={`relative group cursor-none select-none transition-transform duration-300 ${
        isCenter
          ? 'w-56 sm:w-68 md:w-76 z-20 hover:z-30 hover:scale-105'
          : 'w-36 sm:w-44 md:w-48 z-10 hover:z-25 hover:scale-105'
      } ${positionClass}`}
    >
      {/* Small Red Pushpin Dot at top center with subtle contact shadow */}
      <div className="absolute -top-3 left-1/2 -translate-x-1/2 z-30 pointer-events-none">
        <div className="relative flex items-center justify-center">
          {/* Subtle contact shadow under pin */}
          <div className="w-3 h-1.5 bg-black/45 rounded-full blur-[1px] absolute top-3.5 left-0.5" />
          {/* Needle shadow */}
          <div className="w-1 h-2.5 bg-stone-900/60 rounded-full blur-[0.5px] absolute top-2 left-1 rotate-12" />
          {/* Small Red Pushpin Knob */}
          <div
            data-pushpin-knob={id}
            className="w-4 h-4 rounded-full bg-gradient-to-tr from-red-900 via-red-600 to-rose-400 border border-red-950 shadow-[0_3px_6px_rgba(0,0,0,0.4)] flex items-center justify-center relative"
          >
            <div className="w-1 h-1 rounded-full bg-white/80 translate-x-[-1px] translate-y-[-1px]" />
          </div>
        </div>
      </div>

      {/* Crisp White Polaroid Frame with subtle Lift Shadow on Cork */}
      <div
        className={`bg-[#fdfdfd] p-2.5 pt-3 pb-6 border border-stone-200/90 rounded-2xs shadow-[0_12px_24px_-4px_rgba(20,10,5,0.32)] transition-shadow duration-300 ${
          isHovered ? 'shadow-[0_20px_35px_-4px_rgba(20,10,5,0.45)] border-red-300' : ''
        }`}
      >
        {/* Image Display Area with Two-Layer Magnifying Reveal */}
        <div
          className={`relative overflow-hidden bg-stone-900 rounded-2xs ${
            isCenter ? 'h-52 sm:h-64 md:h-72' : 'h-32 sm:h-40 md:h-44'
          }`}
        >
          {/* Layer 1: Clean Photograph */}
          <img
            src={imageUrl}
            alt={title}
            className="w-full h-full object-cover filter contrast-[1.03] brightness-[0.97]"
          />

          {/* Center photo red distressed FAKE rubber stamp placed diagonally across bottom */}
          {isCenter && (
            <div
              className={`absolute bottom-3 right-3 z-10 pointer-events-none transition-all duration-300 ${
                isHovered
                  ? 'scale-110 opacity-100 text-red-600 border-red-600'
                  : 'scale-100 opacity-90 text-red-700/90 border-red-700/90'
              }`}
            >
              <div
                className="border-3 border-dashed rounded-xs px-2.5 py-0.5 font-mono font-black text-xl sm:text-2xl tracking-widest uppercase transform -rotate-12 bg-red-100/30 backdrop-blur-[1px] shadow-sm"
                style={{
                  boxShadow: 'inset 0 0 4px rgba(185, 28, 28, 0.2)',
                }}
              >
                {fakeStampText}
              </div>
            </div>
          )}

          {/* Layer 2: Revealed Forensic Layer under Magnifying Glass Lens */}
          {isHovered && (
            <div
              className="absolute inset-0 z-20 pointer-events-none bg-slate-950"
              style={{
                clipPath: `circle(70px at ${localPos.x}px ${localPos.y}px)`,
              }}
            >
              <img
                src={tamperedUrl || imageUrl}
                alt={title}
                className="w-full h-full object-cover filter brightness-125 contrast-125 saturate-150 mix-blend-hard-light"
              />

              {/* Forensic Grid / Thermal Heatmap Overlay */}
              <div className="absolute inset-0 bg-gradient-to-tr from-red-900/40 via-amber-600/30 to-teal-500/20 mix-blend-overlay" />
              
              {/* Digital Anomaly Scanlines */}
              <div className="absolute inset-0 opacity-40 bg-[linear-gradient(rgba(255,0,0,0.15)_1px,transparent_1px)] bg-[size:100%_4px]" />

              {/* Revealed Forensic Indicators inside lens */}
              <div className="absolute inset-0 p-2.5 flex flex-col justify-between font-mono text-[9px] text-red-300 font-bold drop-shadow-md">
                <div className="flex justify-between items-start bg-black/60 p-1 rounded backdrop-blur-xs">
                  <span className="text-red-400 font-extrabold animate-pulse">
                    ⚠️ {isFake ? 'MANIPULATION DETECTED' : 'AUTHENTIC CAPTURE'}
                  </span>
                  <span className="text-amber-300">
                    {isFake ? 'EOT: 98.2%' : 'PRNU: MATCH'}
                  </span>
                </div>

                {isCenter && (
                  <div className="self-center bg-red-600/90 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-widest border border-white/40 uppercase shadow-lg transform -rotate-6">
                    🚨 STAMP VERIFIED FAKE
                  </div>
                )}

                <div className="flex justify-between items-end bg-black/60 p-1 rounded backdrop-blur-xs text-[8px]">
                  <span className="text-teal-300">SPECTRAL ANOMALY</span>
                  <span className="text-slate-300">X:{Math.round(localPos.x)} Y:{Math.round(localPos.y)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
