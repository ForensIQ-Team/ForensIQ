import React, { useState } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sliders,
  Grid,
  Layers,
  FileText,
  ShieldAlert,
  ShieldCheck,
  Search,
  Lock,
  ArrowLeft,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { ForensicAnalysis, NavItem, UserRole } from '../types';

interface ForensicViewerProps {
  analysis: ForensicAnalysis;
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onBack?: () => void;
  onAddToInvestigation?: (analysis: ForensicAnalysis) => void;
}

export const ForensicViewer: React.FC<ForensicViewerProps> = ({
  analysis,
  userRole,
  setActiveTab,
  onBack,
  onAddToInvestigation,
}) => {
  const [zoomLevel, setZoomLevel] = useState<number>(100);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [showRegions, setShowRegions] = useState<boolean>(true);
  const [showGrid, setShowGrid] = useState<boolean>(false);
  const [compareMode, setCompareMode] = useState<boolean>(false);

  // Collapsible accordion states
  const [metaOpen, setMetaOpen] = useState<boolean>(true);
  const [signalsOpen, setSignalsOpen] = useState<boolean>(true);
  const [modelsOpen, setModelsOpen] = useState<boolean>(false);

  const handleZoomIn = () => setZoomLevel((prev) => Math.min(prev + 25, 250));
  const handleZoomOut = () => setZoomLevel((prev) => Math.max(prev - 25, 50));
  const handleResetZoom = () => {
    setZoomLevel(100);
    setOverlayOpacity(0.65);
    setShowGrid(false);
  };

  return (
    <div className="space-y-6 max-w-[1700px] mx-auto pb-12">
      {/* Workspace Header Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          {onBack && (
            <button
              onClick={onBack}
              className="p-1.5 rounded-lg text-stone-950 hover:bg-slate-100 transition-colors cursor-pointer"
              title="Back"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
          )}

          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-indigo-900 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200">
                ACTIVE MEDIA • FILE ID: {analysis.id}
              </span>
              <span className="text-xs text-stone-950">• {analysis.mediaName}</span>
            </div>
            <h2 className="text-lg font-bold text-slate-900 mt-0.5">
              Interactive Forensic Workspace
            </h2>
          </div>
        </div>

        {/* Quick Action Top Buttons */}
        <div className="flex items-center gap-2">
          {userRole === 'investigator' && (
            <button
              onClick={() => {
                if (onAddToInvestigation) onAddToInvestigation(analysis);
              }}
              className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-stone-950 text-xs font-semibold rounded-lg border border-slate-300 transition-colors cursor-pointer"
            >
              Add to Investigation
            </button>
          )}

          <button
            onClick={() => setActiveTab('report')}
            className="px-3.5 py-1.5 bg-indigo-950 hover:bg-indigo-900 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5 text-indigo-300" />
            <span>Generate Forensic Report</span>
          </button>
        </div>
      </div>

      {/* Main 2-Column Forensic Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* CENTER / LEFT: Large Interactive Canvas Viewer */}
        <div className="lg:col-span-2 bg-slate-950 rounded-xl border border-slate-800 shadow-md flex flex-col justify-between overflow-hidden relative min-h-[520px]">
          {/* Top Viewer Control Bar */}
          <div className="bg-slate-900/90 border-b border-slate-800 p-3 flex flex-wrap items-center justify-between text-xs text-slate-300 gap-2 z-10">
            <div className="flex items-center gap-2 font-mono text-xs">
              <span className="text-indigo-400 font-bold">VIEWPORT:</span>
              <span>{zoomLevel}% Zoom</span>
              <span>•</span>
              <span>{showGrid ? 'Grid Active' : 'No Grid'}</span>
            </div>

            <div className="flex items-center gap-1.5">
              <button
                onClick={handleZoomIn}
                className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-white transition-colors cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={handleZoomOut}
                className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-white transition-colors cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={handleResetZoom}
                className="p-1.5 rounded hover:bg-slate-800 text-slate-300 hover:text-white transition-colors cursor-pointer"
                title="Reset View"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
              <span className="h-4 w-px bg-slate-700 mx-1" />
              <button
                onClick={() => setShowGrid(!showGrid)}
                className={`p-1.5 rounded text-xs font-medium cursor-pointer transition-colors ${
                  showGrid ? 'bg-indigo-600 text-white' : 'hover:bg-slate-800 text-stone-950'
                }`}
                title="Toggle Forensic Grid"
              >
                <Grid className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Central Stage Canvas */}
          <div className="relative flex-1 flex items-center justify-center p-6 overflow-hidden min-h-[400px]">
            {/* Grid Overlay */}
            {showGrid && (
              <div
                className="absolute inset-0 pointer-events-none opacity-20"
                style={{
                  backgroundImage:
                    'linear-gradient(to right, #6366f1 1px, transparent 1px), linear-gradient(to bottom, #6366f1 1px, transparent 1px)',
                  backgroundSize: '40px 40px',
                }}
              />
            )}

            {/* Media Image Container */}
            <div
              className="relative transition-transform duration-150 ease-out max-w-full max-h-[450px] flex items-center justify-center"
              style={{ transform: `scale(${zoomLevel / 100})` }}
            >
              {!compareMode ? (
                <>
                  <img
                    src={analysis.mediaUrl}
                    alt={analysis.mediaName}
                    className="max-h-[420px] w-auto object-contain rounded shadow-lg border border-slate-800"
                  />
                  {showRegions && analysis.heatmapUrl && (
                    <img
                      src={analysis.heatmapUrl}
                      alt="Heatmap"
                      style={{ opacity: overlayOpacity }}
                      className="absolute inset-0 w-full h-full object-cover rounded pointer-events-none mix-blend-screen transition-opacity"
                    />
                  )}
                </>
              ) : (
                <div className="grid grid-cols-2 gap-2 w-full h-full max-w-2xl">
                  <div className="relative border border-slate-800 rounded overflow-hidden">
                    <img
                      src={analysis.mediaUrl}
                      alt="Original"
                      className="w-full h-64 object-cover"
                    />
                    <span className="absolute bottom-1 left-1 px-2 py-0.5 bg-slate-900/90 text-white text-xs font-bold rounded">
                      Original
                    </span>
                  </div>
                  <div className="relative border border-slate-800 rounded overflow-hidden">
                    <img
                      src={analysis.elaUrl || analysis.heatmapUrl || analysis.mediaUrl}
                      alt="Analysis"
                      className="w-full h-64 object-cover"
                    />
                    <span className="absolute bottom-1 left-1 px-2 py-0.5 bg-red-950/90 text-white text-xs font-bold rounded">
                      ELA Residual
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Bottom Viewer Control Dock */}
          <div className="bg-slate-900/95 border-t border-slate-800 p-3 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-300 z-10">
            {/* Opacity Slider */}
            <div className="flex items-center gap-3">
              <span className="text-stone-950 font-medium">Overlay Opacity:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-32 accent-indigo-500 cursor-pointer"
              />
              <span className="font-mono text-indigo-300 font-bold">{Math.round(overlayOpacity * 100)}%</span>
            </div>

            {/* Quick Feature Toggles */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setCompareMode(!compareMode)}
                className={`px-3 py-1 rounded text-xs font-semibold cursor-pointer border transition-colors ${
                  compareMode
                    ? 'bg-indigo-600 border-indigo-500 text-white'
                    : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                }`}
              >
                Compare View
              </button>

              <button
                onClick={() => setShowRegions(!showRegions)}
                className={`px-3 py-1 rounded text-xs font-semibold cursor-pointer border transition-colors ${
                  showRegions
                    ? 'bg-slate-800 border-slate-600 text-indigo-300'
                    : 'bg-slate-800 border-slate-700 text-stone-950'
                }`}
              >
                {showRegions ? 'Regions Active' : 'Hide Heatmap'}
              </button>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE FORENSIC PANEL (Reference 1 inspired) */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
          {/* 1. AUTHENTICITY ASSESSMENT SCORE CARD */}
          <div className="border-b border-slate-100 pb-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-stone-950 uppercase tracking-wider">
                AUTHENTICITY ASSESSMENT
              </span>
              <span
                className={`px-2 py-0.5 rounded text-xs font-bold uppercase border ${
                  analysis.riskLevel === 'low'
                    ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                    : 'bg-red-50 text-red-800 border-red-200'
                }`}
              >
                Risk: {analysis.riskLevel.toUpperCase()}
              </span>
            </div>

            <div className="flex items-center gap-4 bg-slate-50 p-3 rounded-lg border border-slate-200">
              <div className="w-16 h-16 rounded-full border-4 border-indigo-950 bg-white flex flex-col items-center justify-center font-black shadow-xs">
                <span className="text-lg leading-none text-slate-900">{analysis.overallScore}%</span>
                <span className="text-[11px] text-stone-950 uppercase font-semibold">Authentic</span>
              </div>

              <div className="space-y-0.5">
                <h4 className="font-extrabold text-slate-900 text-base capitalize">
                  {analysis.authenticityStatus.replace(/-/g, ' ')}
                </h4>
                <p className="text-xs text-stone-950 font-medium">
                  Confidence Rating: <span className="font-bold text-stone-950">{analysis.confidence}%</span>
                </p>
              </div>
            </div>
          </div>

          {/* 2. AI DETECTION HEATMAP MINI PREVIEW */}
          <div className="space-y-2 border-b border-slate-100 pb-4">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-900">AI Detection Heatmap</span>
              <span className="text-xs text-indigo-700 font-semibold">Active Layer</span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center text-xs font-bold text-stone-950">
              <div className="p-1 bg-slate-50 border border-slate-200 rounded">
                <img
                  src={analysis.mediaUrl}
                  alt="Original"
                  className="w-full h-16 object-cover rounded mb-1"
                />
                Original
              </div>
              <div className="p-1 bg-slate-50 border border-slate-200 rounded">
                <img
                  src={analysis.heatmapUrl || analysis.mediaUrl}
                  alt="Heatmap"
                  className="w-full h-16 object-cover rounded mb-1"
                />
                Heatmap
              </div>
            </div>
          </div>

          {/* 3. COLLAPSIBLE METADATA TABLE */}
          <div className="border-b border-slate-100 pb-3">
            <button
              onClick={() => setMetaOpen(!metaOpen)}
              className="w-full flex items-center justify-between font-bold text-xs text-slate-900 uppercase tracking-wider py-1 cursor-pointer"
            >
              <span>Metadata & EXIF</span>
              {metaOpen ? <ChevronUp className="w-4 h-4 text-stone-950" /> : <ChevronDown className="w-4 h-4 text-stone-950" />}
            </button>

            {metaOpen && (
              <div className="mt-2 divide-y divide-slate-100 text-xs">
                <div className="py-1.5 flex justify-between">
                  <span className="text-stone-950">Dimensions</span>
                  <span className="font-mono font-semibold text-slate-900">{analysis.metadata.dimensions}</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-stone-950">Camera</span>
                  <span className="font-medium text-stone-950">{analysis.metadata.cameraModel || 'None'}</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-stone-950">Software</span>
                  <span className="font-medium text-stone-950">{analysis.metadata.software}</span>
                </div>
                <div className="py-1.5 flex justify-between">
                  <span className="text-stone-950">pHash</span>
                  <span className="font-mono text-xs text-stone-950">{analysis.metadata.pHash}</span>
                </div>
              </div>
            )}
          </div>

          {/* 4. COLLAPSIBLE MANIPULATION SIGNALS */}
          <div className="border-b border-slate-100 pb-3">
            <button
              onClick={() => setSignalsOpen(!signalsOpen)}
              className="w-full flex items-center justify-between font-bold text-xs text-slate-900 uppercase tracking-wider py-1 cursor-pointer"
            >
              <span>Manipulation Signals</span>
              {signalsOpen ? <ChevronUp className="w-4 h-4 text-stone-950" /> : <ChevronDown className="w-4 h-4 text-stone-950" />}
            </button>

            {signalsOpen && (
              <div className="mt-2 space-y-2">
                {analysis.manipulationSignals.map((sig) => (
                  <div key={sig.id} className="p-2 bg-slate-50 rounded border border-slate-200 text-xs">
                    <div className="flex items-center justify-between font-bold text-stone-950">
                      <span>{sig.name}</span>
                      <span className="text-xs text-stone-950 font-mono">{sig.score}</span>
                    </div>
                    <p className="text-xs text-stone-950 mt-0.5">{sig.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Action Row */}
          <div className="pt-2 grid grid-cols-2 gap-2">
            <button
              onClick={() => setActiveTab('protect-image')}
              className="p-2 bg-indigo-950 text-white rounded-lg text-xs font-semibold hover:bg-indigo-900 flex items-center justify-center gap-1 cursor-pointer"
            >
              <Lock className="w-3.5 h-3.5 text-indigo-300" />
              <span>Protect</span>
            </button>
            <button
              onClick={() => setActiveTab('find-misuse')}
              className="p-2 bg-indigo-600 text-white rounded-lg text-xs font-semibold hover:bg-indigo-500 flex items-center justify-center gap-1 cursor-pointer"
            >
              <Search className="w-3.5 h-3.5" />
              <span>Find Misuse</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
