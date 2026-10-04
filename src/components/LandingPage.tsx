import React, { useState, Suspense, lazy } from 'react';
import { NavItem, UserRole, ForensicAnalysis } from '../types';
import { MagnifyingCursor } from './landing/MagnifyingCursor';
import { DetectiveBoard } from './landing/DetectiveBoard';
import { HowItWorks } from './landing/HowItWorks';
import { DetectionModules } from './landing/DetectionModules';
import { ScoreDemo } from './landing/ScoreDemo';
import { UseCases } from './landing/UseCases';
import { FooterCTA } from './landing/FooterCTA';

// Lazy load 3D React Three Fiber Canvas section to prevent initial bundle blocking
const PropagationMap = lazy(() =>
  import('./landing/PropagationMap').then((m) => ({ default: m.PropagationMap }))
);

interface LandingPageProps {
  setActiveTab: (tab: NavItem) => void;
  setUserRole: (role: UserRole) => void;
  userRole: UserRole;
  onOpenViewer: (analysis: ForensicAnalysis) => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({
  setActiveTab,
  setUserRole,
  userRole,
  onOpenViewer,
}) => {
  const [isHoveringPhoto, setIsHoveringPhoto] = useState(false);

  // Isolated launch platform action navigation
  const handleLaunchPlatform = () => {
    setActiveTab('check-media');
  };

  return (
    <div className="min-h-screen bg-graph-paper text-stone-900 font-sans selection:bg-red-900 selection:text-white relative overflow-x-hidden">
      {/* Custom Magnifying Glass Lens Cursor (Active ONLY on Landing Page) */}
      <MagnifyingCursor isHoveringPhoto={isHoveringPhoto} />

      {/* 1. HERO — DETECTIVE CORKBOARD */}
      <DetectiveBoard
        onLaunchPlatform={handleLaunchPlatform}
        onHoverPhotoChange={setIsHoveringPhoto}
      />

      {/* 2. HOW IT WORKS — HORIZONTAL INVESTIGATION PIPELINE */}
      <HowItWorks />

      {/* 3. DETECTION MODULES — 3D FLIPPABLE EVIDENCE CARDS */}
      <DetectionModules />

      {/* 4. 3D PROPAGATION MAP — REACT THREE FIBER (SUSPENSE LAZY LOADED) */}
      <Suspense
        fallback={
          <div className="py-20 text-center font-mono text-xs text-stone-950">
            Initializing 3D Propagation Canvas...
          </div>
        }
      >
        <PropagationMap />
      </Suspense>

      {/* 5. SCORE DEMO — AUTOMATED SWEEP & RUBBER STAMP IMPACT */}
      <ScoreDemo />

      {/* 6. USE CASES — PINNED CASE FILE SECTORS */}
      <UseCases />

      {/* 7. FOOTER CTA — FINAL STAMP ACTION & C2PA BADGING */}
      <FooterCTA onLaunchPlatform={handleLaunchPlatform} />
    </div>
  );
};
