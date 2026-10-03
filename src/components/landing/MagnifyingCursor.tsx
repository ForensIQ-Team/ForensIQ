import React, { useEffect, useState, useRef } from 'react';

interface MagnifyingCursorProps {
  isHoveringPhoto?: boolean;
}

export const MagnifyingCursor: React.FC<MagnifyingCursorProps> = ({ isHoveringPhoto = false }) => {
  const [pos, setPos] = useState<{ x: number; y: number }>({ x: -100, y: -100 });
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const reqRef = useRef<number | null>(null);
  const mousePos = useRef<{ x: number; y: number }>({ x: -100, y: -100 });

  useEffect(() => {
    // Hide default cursor on landing page
    const originalCursor = document.body.style.cursor;
    document.body.style.cursor = 'none';

    const handleMouseMove = (e: MouseEvent) => {
      mousePos.current = { x: e.clientX, y: e.clientY };
      if (!isVisible) setIsVisible(true);
    };

    const handleMouseLeave = () => {
      setIsVisible(false);
    };

    const handleMouseEnter = () => {
      setIsVisible(true);
    };

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);
    document.addEventListener('mouseenter', handleMouseEnter);

    const updatePosition = () => {
      setPos((prev) => {
        const dx = mousePos.current.x - prev.x;
        const dy = mousePos.current.y - prev.y;
        return {
          x: prev.x + dx * 0.35,
          y: prev.y + dy * 0.35,
        };
      });
      reqRef.current = requestAnimationFrame(updatePosition);
    };

    reqRef.current = requestAnimationFrame(updatePosition);

    return () => {
      document.body.style.cursor = originalCursor || 'auto';
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      document.removeEventListener('mouseenter', handleMouseEnter);
      if (reqRef.current) cancelAnimationFrame(reqRef.current);
    };
  }, []);

  if (!isVisible) return null;

  // Realistic small diameter: 40px normal, 46px hovered
  const lensSize = isHoveringPhoto ? 46 : 40;

  return (
    <div
     className="fixed pointer-events-none z-[9999]"
      style={{
        transform: `translate3d(${pos.x}px, ${pos.y}px, 0)`,
        left: 0,
        top: 0,
      }}
    >
      {/* Small Realistic Detective Magnifying Lens */}
      <div
        className={`relative -translate-x-1/2 -translate-y-1/2 transition-transform duration-300 ease-out ${
          isHoveringPhoto ? 'scale-115' : 'scale-100'
        }`}
      >
        

        {/* Thin Bronze / Brass Rim */}
        <div
          className={`relative rounded-full border-2 transition-colors duration-300 backdrop-blur-[1px] flex items-center justify-center ${
            isHoveringPhoto
              ? 'border-red-600/90 shadow-[0_0_12px_rgba(220,38,38,0.4)] bg-red-950/10'
              : 'border-amber-800/80 shadow-md bg-stone-900/5'
          }`}
          style={{
            width: `${lensSize}px`,
            height: `${lensSize}px`,
          }}
        >
          {/* Glassy Lens Specular Highlight */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-br from-white/45 via-transparent to-transparent pointer-events-none" />
          <div className="absolute inset-0.5 rounded-full border border-white/20 pointer-events-none" />

          {/* Crosshair Ticks */}
          <div className="absolute inset-0 flex items-center justify-center opacity-25 pointer-events-none">
            <div className={`w-full h-[1px] ${isHoveringPhoto ? 'bg-red-500' : 'bg-stone-700'}`} />
            <div className={`h-full w-[1px] absolute ${isHoveringPhoto ? 'bg-red-500' : 'bg-stone-700'}`} />
          </div>

          {/* Precision Focal Point */}
          <div
            className={`w-1 h-1 rounded-full transition-colors duration-200 ${
              isHoveringPhoto ? 'bg-red-600 shadow-xs shadow-red-500' : 'bg-stone-800'
            }`}
          />
        </div>

        {/* Short Vintage Handle at 45deg bottom-right */}
        <div
          className="absolute left-1/2 top-1/2 origin-top-left pointer-events-none"
          style={{
            transform: `translate(${lensSize * 0.32}px, ${lensSize * 0.32}px) rotate(45deg)`,
          }}
        >
          {/* Short Dark Wood & Brass Ferrule Handle */}
          <div className="w-2 h-5 bg-gradient-to-b from-amber-950 via-stone-800 to-stone-900 rounded-b-xs shadow-sm border-x border-amber-900/40 relative">
            <div className="absolute top-0 inset-x-0 h-0.5 bg-amber-500/80" />
          </div>
        </div>
      </div>
    </div>
  );
};
