import React, { useEffect, useRef } from 'react';

interface MagnifyingCursorProps {
  isHoveringPhoto?: boolean;
}

// Lens center in loupe.png (512x512) is at about 36.5% from left/top
const LENS_CENTER = '36.5%';

export const MagnifyingCursor: React.FC<MagnifyingCursorProps> = ({ isHoveringPhoto = false }) => {
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const originalCursor = document.body.style.cursor;
    document.body.style.cursor = 'none';

    const target = { x: -100, y: -100 };
    const current = { x: -100, y: -100 };
    let raf = 0;

    const handleMouseMove = (e: MouseEvent) => {
      target.x = e.clientX;
      target.y = e.clientY;
      if (wrapRef.current) wrapRef.current.style.opacity = '1';
    };
    const handleMouseLeave = () => {
      if (wrapRef.current) wrapRef.current.style.opacity = '0';
    };

    const loop = () => {
      current.x += (target.x - current.x) * 0.35;
      current.y += (target.y - current.y) * 0.35;
      if (wrapRef.current) {
        wrapRef.current.style.transform = `translate3d(${current.x}px, ${current.y}px, 0)`;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    window.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      document.body.style.cursor = originalCursor || 'auto';
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className="fixed left-0 top-0 pointer-events-none z-[9999]"
      style={{ opacity: 0 }}
    >
      <img
        src="/loupe.png"
        alt=""
        draggable={false}
        style={{
          width: 44,
          height: 44,
          opacity: 0.6,
          transform: `translate(-${LENS_CENTER}, -${LENS_CENTER}) scale(${isHoveringPhoto ? 1.2 : 1})`,
          transformOrigin: `${LENS_CENTER} ${LENS_CENTER}`,
          transition: 'transform 200ms ease-out',
          userSelect: 'none',
        }}
      />
    </div>
  );
};