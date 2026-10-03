import React, { useState, useEffect, useRef } from 'react';
import { ShieldCheck, ArrowRight } from 'lucide-react';
import { EvidencePhoto } from './EvidencePhoto';
import { StringConnection } from './InvestigationStrings';

interface DetectiveBoardProps {
  onLaunchPlatform: () => void;
  onHoverPhotoChange: (isHovered: boolean) => void;
}

export const DetectiveBoard: React.FC<DetectiveBoardProps> = ({
  onLaunchPlatform,
  onHoverPhotoChange,
}) => {
  const [connections, setConnections] = useState<StringConnection[]>([]);
  const boardRef = useRef<HTMLDivElement>(null);

  // Exact 8 polaroid photos arranged in a clean, non-overlapping radial ring around center photo:
  const radialPhotos = [
    // 1. TOP (0°) - Green foliage / organic pattern
    {
      id: 'radial-1',
      title: 'FOLIAGE_CANOPY.JPG',
      caseNumber: 'CASE #FQ-901',
      imageUrl: 'https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?auto=format&fit=crop&w=600&q=80',
      rotation: 1,
      positionClass: 'absolute top-[3%] left-1/2 -translate-x-1/2',
      tag: 'ORGANIC PATTERN',
      isFake: true,
    },
    // 2. TOP RIGHT (45°) - Hooded subject / surveillance
    {
      id: 'radial-2',
      title: 'HOODED_SUBJECT.JPG',
      caseNumber: 'CASE #FQ-902',
      imageUrl: 'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?auto=format&fit=crop&w=600&q=80',
      rotation: 3,
      positionClass: 'absolute top-[6%] right-[6%]',
      tag: 'SURVEILLANCE',
      isFake: true,
    },
    // 3. MID RIGHT (90°) - Camera lens PRNU sensor
    {
      id: 'radial-3',
      title: 'LENS_PRNU_SENSOR.JPG',
      caseNumber: 'CASE #FQ-903',
      imageUrl: 'https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=600&q=80',
      rotation: -2,
      positionClass: 'absolute top-1/2 -translate-y-1/2 right-[3%]',
      tag: 'PRNU SENSOR',
      isFake: false,
    },
    // 4. BOTTOM RIGHT (135°) - Smiling male portrait
    {
      id: 'radial-4',
      title: 'SUSPECT_MALE_FACE.JPG',
      caseNumber: 'CASE #FQ-904',
      imageUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=600&q=80',
      rotation: 4,
      positionClass: 'absolute bottom-[6%] right-[6%]',
      tag: 'SYNTHETIC FACE',
      isFake: true,
    },
    // 5. BOTTOM (180°) - Man wearing hat
    {
      id: 'radial-5',
      title: 'HAT_SUBJECT_INTERCEPT.JPG',
      caseNumber: 'CASE #FQ-905',
      imageUrl: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?auto=format&fit=crop&w=600&q=80',
      rotation: -1,
      positionClass: 'absolute bottom-[3%] left-1/2 -translate-x-1/2',
      tag: 'EOT WATERMARK',
      isFake: false,
    },
    // 6. BOTTOM LEFT (225°) - Interior scene scan
    {
      id: 'radial-6',
      title: 'INTERIOR_SCENE_SCAN.JPG',
      caseNumber: 'CASE #FQ-906',
      imageUrl: 'https://images.unsplash.com/photo-1513694203232-719a280e022f?auto=format&fit=crop&w=600&q=80',
      rotation: -4,
      positionClass: 'absolute bottom-[6%] left-[6%]',
      tag: 'EXPOSURE WARP',
      isFake: true,
    },
    // 7. MID LEFT (270°) - Fingerprint macro
    {
      id: 'radial-7',
      title: 'FINGERPRINT_MACRO.JPG',
      caseNumber: 'CASE #FQ-907',
      imageUrl: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=600&q=80',
      rotation: 2,
      positionClass: 'absolute top-1/2 -translate-y-1/2 left-[3%]',
      tag: 'LATENT RIDGE',
      isFake: false,
    },
    // 8. TOP LEFT (315°) - Business woman with folder
    {
      id: 'radial-8',
      title: 'EXHIBIT_A_DOC.JPG',
      caseNumber: 'CASE #FQ-908',
      imageUrl: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=600&q=80',
      rotation: -3,
      positionClass: 'absolute top-[6%] left-[6%]',
      tag: 'METADATA AUDIT',
      isFake: false,
    },
  ];

  // Center primary target photo
  const centerPhoto = {
    id: 'center-photo',
    title: 'TARGET_PRIMARY_FACE.JPG',
    caseNumber: 'TARGET CASE #FQ-8091',
    imageUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=800&q=80',
    rotation: 0,
    positionClass: 'absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20',
    isCenter: true,
    isFake: true,
    fakeStampText: 'FAKE',
    tag: 'PRIMARY EVIDENCE',
  };

  // Calculate clean, straight red strings connecting from every radial pushpin to center pushpin
  useEffect(() => {
    const updateStringCoords = () => {
      if (!boardRef.current) return;
      const boardRect = boardRef.current.getBoundingClientRect();

      const centerPinEl = boardRef.current.querySelector('[data-pushpin-knob="center-photo"]');
      if (!centerPinEl) return;
      const centerPinRect = centerPinEl.getBoundingClientRect();
      const cx = centerPinRect.left + centerPinRect.width / 2 - boardRect.left;
      const cy = centerPinRect.top + centerPinRect.height / 2 - boardRect.top;

      const newConns: StringConnection[] = [];

      radialPhotos.forEach((photo) => {
        const pinEl = boardRef.current?.querySelector(`[data-pushpin-knob="${photo.id}"]`);
        if (pinEl) {
          const pinRect = pinEl.getBoundingClientRect();
          if (pinRect.width > 0 && pinRect.height > 0) {
            const px = pinRect.left + pinRect.width / 2 - boardRect.left;
            const py = pinRect.top + pinRect.height / 2 - boardRect.top;

            newConns.push({
              id: `string-${photo.id}`,
              x1: cx,
              y1: cy,
              x2: px,
              y2: py,
              sag: 0, // Straight red string per reference
            });
          }
        }
      });

      setConnections(newConns);
    };

    const timer1 = setTimeout(updateStringCoords, 120);
    const timer2 = setTimeout(updateStringCoords, 450);

    window.addEventListener('resize', updateStringCoords);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      window.removeEventListener('resize', updateStringCoords);
    };
  }, []);

  return (
    <section id="about" className="relative w-full min-h-[92vh] pt-4 pb-12 px-2 sm:px-6 lg:px-12 flex flex-col justify-between overflow-hidden">
      {/* Warm Spotlight Cone in Top Right Corner */}
      <div className="absolute -top-12 -right-12 w-[600px] h-[600px] bg-gradient-radial from-amber-100/50 via-amber-200/15 to-transparent rounded-full pointer-events-none z-0 filter blur-2xl" />

      {/* 1. FULL STYLED NAVBAR TOP OF PAGE */}
      <header className="relative z-50 max-w-7xl mx-auto w-full py-4 px-4 sm:px-6">
        <div className="w-full bg-stone-900/5 backdrop-blur-sm border border-stone-400/40 rounded-full px-5 sm:px-7 py-2.5 flex items-center justify-between shadow-xs">
          {/* Left: Clean Brand Text */}
          <a href="#" className="font-serif font-black text-xl sm:text-2xl text-slate-950 tracking-tight uppercase cursor-none flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-600 animate-pulse" />
            <span>FORENSIQ</span>
          </a>

          {/* Center: Evenly Spaced Navigation Links */}
          <nav className="hidden md:flex items-center gap-6 lg:gap-8 font-sans font-semibold text-xs text-stone-800">
            <a href="#about" className="px-3 py-1 rounded-full hover:bg-stone-800/10 hover:text-slate-950 transition-all cursor-none">
              About
            </a>
            <a href="#score-demo" className="px-3 py-1 rounded-full hover:bg-stone-800/10 hover:text-slate-950 transition-all cursor-none">
              Live Demo
            </a>
            <a href="#modules" className="px-3 py-1 rounded-full hover:bg-stone-800/10 hover:text-slate-950 transition-all cursor-none">
              Detection Modules
            </a>
            <a href="#use-cases" className="px-3 py-1 rounded-full hover:bg-stone-800/10 hover:text-slate-950 transition-all cursor-none">
              Case Studies
            </a>
            <a href="#footer-cta" className="px-3 py-1 rounded-full hover:bg-stone-800/10 hover:text-slate-950 transition-all cursor-none">
              Contact
            </a>
          </nav>

          {/* Far Right: Launch Platform Pill Button */}
          <button
            onClick={onLaunchPlatform}
            className="px-5 py-2 bg-[#6b6255] hover:bg-[#524b40] text-white text-xs font-semibold rounded-full transition-all duration-300 cursor-none flex items-center gap-2 shadow-md hover:shadow-lg border border-stone-500/40 uppercase tracking-wider"
          >
            <span>LAUNCH PLATFORM</span>
          </button>
        </div>
      </header>

      {/* 2. HERO — SPLIT-SCREEN LAYOUT */}
      <div className="relative max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-center py-6 px-4 sm:px-6 lg:px-8 my-auto">
        {/* LEFT COLUMN (~35-40% width): Bold Headline & Content */}
        <div className="lg:col-span-5 flex flex-col justify-center text-left z-30 space-y-5 pr-2 lg:pr-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-100/90 border border-red-200 text-red-900 font-mono text-xs font-bold uppercase w-fit">
            <ShieldCheck className="w-3.5 h-3.5 text-red-700" />
            <span>AI-POWERED FORENSIC PLATFORM</span>
          </div>

          <h1 className="font-serif font-black text-5xl sm:text-6xl lg:text-7xl text-slate-950 tracking-tight leading-none uppercase">
            FORENSIQ
          </h1>

          <h2 className="text-base sm:text-lg lg:text-xl font-serif font-bold text-stone-800 leading-snug">
            Digital Media Authenticity &amp; Forensic Investigation Platform
          </h2>

          <p className="text-xs sm:text-sm text-stone-600 leading-relaxed font-sans max-w-md">
            AI-powered forensic analysis tracing digital media from origin to viral spread. Detect deepfakes, verify authenticity, protect truth.
          </p>

          <div className="pt-2">
            <button
              onClick={onLaunchPlatform}
              className="px-7 py-3.5 bg-slate-950 hover:bg-red-900 text-white font-mono font-bold text-xs sm:text-sm rounded-full shadow-xl transition-all duration-300 flex items-center gap-3 group border border-slate-700 uppercase tracking-wider cursor-none"
            >
              <span>Start Investigation</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform text-red-400" />
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN (~60-65% width): Corkboard Scene & Side Props */}
        <div className="lg:col-span-7 relative flex items-center justify-center min-h-[540px] sm:min-h-[600px] w-full">
          {/* LEFT SIDE WALL PROPS (FULLY VISIBLE ON WALL SPACE TO LEFT OF BOARD) */}
          <div className="hidden lg:flex flex-col items-end gap-10 absolute -left-32 xl:-left-40 top-6 z-30 pointer-events-none w-44">
            {/* Torn Newspaper Clipping "DEEPFAKE SCANDAL..." */}
            <div className="relative transform -rotate-6 bg-[#e6dfcc] p-3 pt-4 border border-stone-400/80 rounded-xs shadow-[0_14px_28px_rgba(0,0,0,0.25)] w-44 text-stone-900 font-serif">
              <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 z-30">
                <div className="w-3.5 h-3.5 rounded-full bg-gradient-to-tr from-stone-800 to-stone-500 border border-stone-900 shadow-md" />
              </div>
              <div className="border-b border-stone-800 pb-1 mb-1 text-[7px] font-sans font-bold tracking-widest uppercase text-stone-700">
                DAILY GAZETTE • BREAKING
              </div>
              <h3 className="font-black text-xs uppercase leading-tight font-serif tracking-tighter">
                DEEPFAKE SCANDAL IN OIL...
              </h3>
              <p className="text-[8px] font-sans text-stone-700 leading-snug mt-1 opacity-90 line-clamp-3">
                Synthesized facial composite bypasses biometric security systems across corporate networks...
              </p>
            </div>

            {/* Torn Sticky Note "247 CASES SOLVED" */}
            <div className="relative transform rotate-4 bg-[#f3ecda] p-3 pt-3.5 border border-amber-200/90 shadow-[0_10px_20px_rgba(0,0,0,0.18)] w-36 text-stone-900 font-sans">
              <div className="absolute -top-2.5 left-1/2 -translate-x-1/2 z-30">
                <div className="w-3.5 h-3.5 rounded-full bg-gradient-to-tr from-stone-800 to-stone-500 border border-stone-900 shadow-md" />
              </div>
              <div className="text-center">
                <div className="font-serif font-black text-xl tracking-tighter leading-none text-stone-900">
                  247
                </div>
                <div className="font-mono font-bold text-[9px] tracking-widest uppercase text-stone-700 mt-0.5">
                  CASES SOLVED
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT SIDE WALL PROPS (RESTING OUTSIDE BOARD FRAME) */}
          <div className="hidden lg:flex flex-col items-start gap-10 absolute -right-8 xl:-right-14 top-8 z-10 pointer-events-none w-44">
            {/* Sticky Note "98.2% ACCURACY" */}
            <div className="relative transform rotate-3 bg-[#f5efde] p-2.5 pt-3 border border-amber-200/90 shadow-[0_10px_20px_rgba(0,0,0,0.18)] w-32 text-stone-900 font-sans">
              <div className="absolute -top-2 left-1/2 -translate-x-1/2 w-10 h-3 bg-amber-200/70 transform -rotate-1 border-x border-amber-300/50 shadow-xs" />
              <div className="text-center">
                <div className="font-serif font-black text-lg tracking-tighter text-stone-900">
                  98.2%
                </div>
                <div className="font-mono font-bold text-[8px] tracking-widest uppercase text-stone-700">
                  ACCURACY
                </div>
              </div>
            </div>

            {/* Red Yarn Spool Prop */}
            <div className="relative ml-2 mt-2">
              <div className="w-9 h-13 bg-gradient-to-b from-red-800 via-red-600 to-red-900 rounded-xs shadow-lg border-x-2 border-amber-950 flex flex-col justify-between p-0.5 relative">
                <div className="w-full h-1 bg-amber-900/70 rounded-xs" />
                <div className="w-full h-1 bg-amber-900/70 rounded-xs" />
                <div className="absolute -top-1.5 inset-x-0.5 h-1.5 bg-amber-800 rounded-t-xs border-t border-amber-900" />
                <div className="absolute -bottom-1.5 inset-x-0.5 h-1.5 bg-amber-800 rounded-b-xs border-b border-amber-900" />
              </div>
            </div>

            {/* Confidential Case-File Folder Tab Peeking in from Far Right Edge */}
            <div className="absolute -right-8 top-44 transform -rotate-2 bg-[#d8be92] p-3 pt-5 border-l-4 border-amber-900 rounded-l-md shadow-xl w-32 text-stone-900 font-sans">
              <div className="border border-red-800/80 px-1.5 py-0.5 text-center font-mono font-black text-[9px] tracking-widest text-red-800 uppercase transform -rotate-6 bg-red-100/40 mb-1.5">
                CONFIDENTIAL
              </div>
              <div className="text-[7px] font-mono text-stone-800 space-y-0.5">
                <div>FILE: #FQ-2026-X</div>
                <div>CLASSIFIED FORENSIC</div>
              </div>
            </div>
          </div>

          {/* REALISTIC CORK PINBOARD WITH MATTE DARK WALNUT FRAME */}
          <div
            ref={boardRef}
            className="relative w-full max-w-xl sm:max-w-2xl lg:max-w-3xl rounded-xs border-[14px] sm:border-[16px] border-[#2c1a12] shadow-[0_30px_70px_-10px_rgba(20,10,5,0.52)] overflow-hidden h-[500px] sm:h-[560px] md:h-[620px] flex items-center justify-center p-4 z-20"
            style={{
              backgroundColor: '#ba8551',
              backgroundImage: `
                radial-gradient(circle at 50% 40%, rgba(255,245,230,0.24) 0%, rgba(186,133,81,0.1) 60%, rgba(120,70,30,0.32) 100%),
                radial-gradient(rgba(130, 80, 35, 0.45) 1.2px, transparent 0),
                radial-gradient(rgba(70, 35, 10, 0.35) 1.8px, transparent 0)
              `,
              backgroundSize: '100% 100%, 7px 7px, 13px 13px',
            }}
          >
            {/* Beveled Frame Drop Shadow inside Cork */}
            <div className="absolute inset-0 pointer-events-none shadow-[inset_0_0_40px_rgba(20,10,5,0.55)]" />

            {/* SVG Straight Red Strings Layer */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none z-15">
              {connections.map((conn) => (
                <line
                  key={conn.id}
                  x1={conn.x1}
                  y1={conn.y1}
                  x2={conn.x2}
                  y2={conn.y2}
                  stroke="#c81e1e"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  style={{
                    filter: 'drop-shadow(0px 2px 3px rgba(0,0,0,0.45))',
                  }}
                />
              ))}
            </svg>

            {/* Evidence Photos Stage */}
            <div className="relative w-full h-full">
              {/* 8 Radial Surrounding Photos */}
              {radialPhotos.map((photo) => (
                <div key={photo.id} className={photo.positionClass}>
                  <EvidencePhoto
                    {...photo}
                    onHoverChange={onHoverPhotoChange}
                  />
                </div>
              ))}

              {/* Center Primary Photo */}
              <div className={centerPhoto.positionClass}>
                <EvidencePhoto
                  {...centerPhoto}
                  onHoverChange={onHoverPhotoChange}
                />

                {/* Vintage Brass Compass Prop Pinned at Bottom-Left of Center Photo */}
                <div className="absolute -bottom-3 -left-3 z-30 pointer-events-none drop-shadow-lg">
                  <div className="relative flex items-center justify-center">
                    <div className="w-9 h-9 rounded-full border-2 border-amber-700 bg-amber-950/70 backdrop-blur-[1px] shadow-md flex items-center justify-center relative">
                      <div className="w-2 h-2 rounded-full border border-amber-600 absolute -top-1.5 bg-amber-900/80" />
                      <div className="w-7 h-7 rounded-full border border-amber-600/70 bg-amber-100/10 flex items-center justify-center relative">
                        <span className="absolute top-0 text-[5px] font-mono font-bold text-amber-300">N</span>
                        <span className="absolute bottom-0 text-[5px] font-mono font-bold text-amber-300">S</span>
                        <div className="w-0.5 h-4 bg-red-600 transform rotate-45 rounded-full shadow-xs" />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
