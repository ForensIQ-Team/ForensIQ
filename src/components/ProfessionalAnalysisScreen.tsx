import React, { useState } from 'react';
import {
  Shield,
  Layers,
  Upload,
  FileCheck2,
  Sliders,
  CheckCircle2,
  FileText,
  Plus,
  Info,
  Maximize2,
  Eye,
} from 'lucide-react';
import { ForensicAnalysis, UserRole, NavItem } from '../types';
import { MOCK_ANALYSES, SAMPLE_IMAGES, MOCK_INCIDENT } from '../data/mockData';
import { PropagationTimeline } from './PropagationTimeline';

interface ProfessionalAnalysisScreenProps {
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onOpenViewer: (analysis: ForensicAnalysis) => void;
}

export const ProfessionalAnalysisScreen: React.FC<ProfessionalAnalysisScreenProps> = ({
  userRole,
  setActiveTab,
  onOpenViewer,
}) => {
  const [activeAnalysis, setActiveAnalysis] = useState<ForensicAnalysis>(MOCK_ANALYSES['FQ-8091']);
  const [compareMedia, setCompareMedia] = useState<ForensicAnalysis>(MOCK_ANALYSES['FQ-4022']);
  const [splitSlider, setSplitSlider] = useState<number>(50);
  const [investigationSuccess, setInvestigationSuccess] = useState<boolean>(false);

  const handleAddToInvestigation = () => {
    setInvestigationSuccess(true);
    setTimeout(() => setInvestigationSuccess(false), 3000);
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Top Banner Header */}
      <div className="bg-[#1e1b18]/95 text-white rounded-xl p-6 shadow-xs border border-stone-800 backdrop-blur-md flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-red-900/40 text-red-300 text-xs font-mono font-bold uppercase tracking-wider border border-red-700/40 mb-2">
            <Shield className="w-3.5 h-3.5 text-amber-300" />
            <span>VERIFIED INVESTIGATOR SUITE ACTIVE</span>
          </div>
          <h2 className="text-xl font-bold tracking-tight">Professional Forensic Inspection &amp; Compare</h2>
          <p className="text-xs text-stone-300 mt-1 max-w-2xl leading-relaxed">
            Advanced diagnostic workspace for verified digital investigators. Access raw residual noise maps, double JPEG compression matrices, and side-by-side asset comparison.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onOpenViewer(activeAnalysis)}
            className="px-4 py-2 bg-red-800 hover:bg-red-700 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer flex items-center gap-1.5 shadow-xs"
          >
            <Eye className="w-4 h-4" />
            <span>Open Interactive Viewer</span>
          </button>
          <button
            onClick={() => setActiveTab('report')}
            className="px-4 py-2 glass-input text-white text-xs font-semibold rounded-lg hover:bg-white/10 transition-colors cursor-pointer flex items-center gap-1.5"
          >
            <FileText className="w-4 h-4 text-amber-300" />
            <span>Audit Report</span>
          </button>
        </div>
      </div>

      {investigationSuccess && (
        <div className="p-4 bg-emerald-50/90 border border-emerald-200 rounded-xl text-xs text-emerald-900 flex items-center gap-2 shadow-xs">
          <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
          <span className="font-bold">Added asset to Active Investigation INC-2026-042 Binder!</span>
        </div>
      )}

      {/* Main 2-Asset Compare Tool */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#e2d8c3]/80 pb-4">
          <div>
            <h3 className="font-bold text-stone-950 text-base">Dual-Asset Compare &amp; Splicing Inspector</h3>
            <p className="text-xs text-stone-950">
              Compare baseline registered asset against suspicious derivative target.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={handleAddToInvestigation}
              className="px-3.5 py-1.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
            >
              Add to Investigation
            </button>
          </div>
        </div>

        {/* Compare Stage Canvas */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Asset 1 */}
          <div className="space-y-3 p-4 glass-card rounded-xl">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-stone-950">ASSET A (Baseline Registered)</span>
              <span className="px-2 py-0.5 rounded bg-emerald-50/80 text-emerald-800 text-xs font-bold border border-emerald-200">
                Score: 94% Authentic
              </span>
            </div>

            <div className="relative h-64 rounded-lg overflow-hidden border border-stone-300 bg-stone-900 flex items-center justify-center">
              <img
                src={activeAnalysis.mediaUrl}
                alt="Asset A"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="text-xs space-y-1 text-stone-950">
              <p className="font-bold text-stone-950">{activeAnalysis.mediaName}</p>
              <p className="text-xs text-stone-950 font-mono">
                SHA256: {activeAnalysis.metadata.hashSHA256.substring(0, 32)}...
              </p>
            </div>
          </div>

          {/* Asset 2 */}
          <div className="space-y-3 p-4 glass-card rounded-xl">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-stone-950">ASSET B (Suspicious Target)</span>
              <span className="px-2 py-0.5 rounded bg-red-50/80 text-red-800 text-xs font-bold border border-red-200">
                Score: 18% Synthetic
              </span>
            </div>

            <div className="relative h-64 rounded-lg overflow-hidden border border-stone-300 bg-stone-900 flex items-center justify-center">
              <img
                src={compareMedia.mediaUrl}
                alt="Asset B"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="text-xs space-y-1 text-stone-950">
              <p className="font-bold text-stone-950">{compareMedia.mediaName}</p>
              <p className="text-xs text-stone-950 font-mono">
                SHA256: {compareMedia.metadata.hashSHA256.substring(0, 32)}...
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Propagation Lineage Timeline */}
      <PropagationTimeline userRole={userRole} />
    </div>
  );
};
