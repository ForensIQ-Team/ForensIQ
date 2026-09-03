import React, { useState } from 'react';
import { ExplainabilityData } from '../../types';
import { Eye, Sliders, Sparkles, Layers, Maximize2, ShieldAlert } from 'lucide-react';

interface ExplainabilityPanelProps {
  explainability: ExplainabilityData;
  mediaName: string;
}

export const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({
  explainability,
  mediaName,
}) => {
  const [viewMode, setViewMode] = useState<'gradcam' | 'vit' | 'overlay' | 'original'>('gradcam');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.75);
  const [showRegions, setShowRegions] = useState<boolean>(true);

  // Active layer URL
  const activeHeatmapUrl =
    viewMode === 'vit' ? explainability.vitAttentionUrl : explainability.gradCamUrl;

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-5">
      {/* Title & View Mode Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4">
        <div>
          <h4 className="font-bold text-slate-900 text-sm flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-indigo-600" />
            <span>AI Explainability & Attention Visualizer</span>
          </h4>
          <p className="text-xs text-slate-500 mt-0.5">
            Inspecting Grad-CAM class activation maps and Vision Transformer (ViT) spatial attention matrices
          </p>
        </div>

        {/* View Mode Buttons */}
        <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg border border-slate-200">
          <button
            onClick={() => setViewMode('gradcam')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'gradcam'
                ? 'bg-indigo-950 text-white shadow-xs'
                : 'text-slate-700 hover:text-slate-900'
            }`}
          >
            Grad-CAM
          </button>

          <button
            onClick={() => setViewMode('vit')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'vit'
                ? 'bg-indigo-950 text-white shadow-xs'
                : 'text-slate-700 hover:text-slate-900'
            }`}
          >
            ViT Attention
          </button>

          <button
            onClick={() => setViewMode('overlay')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'overlay'
                ? 'bg-indigo-950 text-white shadow-xs'
                : 'text-slate-700 hover:text-slate-900'
            }`}
          >
            Overlay
          </button>

          <button
            onClick={() => setViewMode('original')}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              viewMode === 'original'
                ? 'bg-indigo-950 text-white shadow-xs'
                : 'text-slate-700 hover:text-slate-900'
            }`}
          >
            Original
          </button>
        </div>
      </div>

      {/* Main Canvas Viewer */}
      <div className="relative bg-slate-950 rounded-xl overflow-hidden border border-slate-800 min-h-[380px] flex items-center justify-center">
        {/* Original Base Image */}
        <img
          src={explainability.originalUrl}
          alt={mediaName}
          className="w-full h-96 object-cover"
        />

        {/* Heatmap Overlay Layer */}
        {(viewMode === 'gradcam' || viewMode === 'vit' || viewMode === 'overlay') && (
          <img
            src={activeHeatmapUrl}
            alt="Neural Heatmap"
            style={{ opacity: viewMode === 'overlay' ? overlayOpacity : 0.85 }}
            className="absolute inset-0 w-full h-96 object-cover mix-blend-screen transition-opacity duration-200"
          />
        )}

        {/* Region ROI Bounding Boxes */}
        {showRegions &&
          explainability.highlightedRegions.map((region, idx) => (
            <div
              key={idx}
              style={{
                left: `${region.bounds.x}%`,
                top: `${region.bounds.y}%`,
                width: `${region.bounds.width}%`,
                height: `${region.bounds.height}%`,
              }}
              className="absolute border-2 border-red-500 bg-red-500/20 rounded shadow-xs pointer-events-auto transition-all hover:bg-red-500/30 group"
            >
              <span className="absolute -top-6 left-0 bg-red-950 text-white font-mono font-bold text-[10px] px-1.5 py-0.5 rounded shadow-xs whitespace-nowrap border border-red-800">
                {region.label}
              </span>

              {/* Tooltip on hover */}
              <div className="hidden group-hover:block absolute top-full mt-1 left-0 z-20 bg-slate-900 text-white text-[11px] p-2 rounded shadow-lg max-w-xs border border-slate-700">
                <p className="font-bold text-red-400 mb-0.5">{region.label}</p>
                <p className="text-slate-300">{region.description}</p>
              </div>
            </div>
          ))}

        {/* Canvas Watermark Badge */}
        <div className="absolute bottom-3 left-3 bg-slate-900/90 text-white text-[10px] font-mono px-2.5 py-1 rounded border border-slate-700 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span>LAYER: {viewMode.toUpperCase()}</span>
        </div>
      </div>

      {/* Control Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-slate-50 p-3.5 rounded-lg border border-slate-200 text-xs">
        {/* Opacity Slider */}
        <div className="flex items-center gap-3 min-w-[240px]">
          <Sliders className="w-4 h-4 text-slate-500" />
          <span className="font-semibold text-slate-700">Heatmap Opacity:</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={overlayOpacity}
            onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
            className="w-32 accent-indigo-950 cursor-pointer"
          />
          <span className="font-mono text-indigo-950 font-bold">
            {Math.round(overlayOpacity * 100)}%
          </span>
        </div>

        {/* ROI Regions Toggle */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-slate-700 font-semibold cursor-pointer">
            <input
              type="checkbox"
              checked={showRegions}
              onChange={(e) => setShowRegions(e.target.checked)}
              className="rounded text-indigo-900 focus:ring-indigo-900 accent-indigo-900"
            />
            <span>Highlight Suspicious ROIs</span>
          </label>
        </div>
      </div>

      {/* Legend Footer */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs pt-2">
        <div className="flex items-center gap-2 p-2.5 rounded bg-slate-50 border border-slate-200">
          <span className="w-3.5 h-3.5 rounded bg-red-600 shrink-0" />
          <div>
            <p className="font-bold text-slate-900 text-[11px]">Red / Warm Regions</p>
            <p className="text-[10px] text-slate-500">High neural network feature activation</p>
          </div>
        </div>

        <div className="flex items-center gap-2 p-2.5 rounded bg-slate-50 border border-slate-200">
          <span className="w-3.5 h-3.5 rounded bg-amber-500 shrink-0" />
          <div>
            <p className="font-bold text-slate-900 text-[11px]">Yellow / Amber Regions</p>
            <p className="text-[10px] text-slate-500">Moderate feature transition gradient</p>
          </div>
        </div>

        <div className="flex items-center gap-2 p-2.5 rounded bg-slate-50 border border-slate-200">
          <span className="w-3.5 h-3.5 rounded bg-indigo-600 shrink-0" />
          <div>
            <p className="font-bold text-slate-900 text-[11px]">Blue / Dark Regions</p>
            <p className="text-[10px] text-slate-500">Baseline authentic background texture</p>
          </div>
        </div>
      </div>
    </div>
  );
};
