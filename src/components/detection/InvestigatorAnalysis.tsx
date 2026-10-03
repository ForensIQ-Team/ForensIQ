import React from 'react';
import { DetectionResult, NavItem } from '../../types';
import { ModelScoreCard } from './ModelScoreCard';
import { ExplainabilityPanel } from './ExplainabilityPanel';
import { VideoAnalysisTimeline } from './VideoAnalysisTimeline';
import {
  ShieldAlert,
  FileText,
  Clock,
  Cpu,
  Hash,
  Database,
  FileCheck2,
  AlertTriangle,
  Download,
  Eye,
  Camera,
  Layers,
} from 'lucide-react';

interface InvestigatorAnalysisProps {
  result: DetectionResult;
  setActiveTab: (tab: NavItem) => void;
  onOpenViewer: (analysis: DetectionResult) => void;
  onOpenReport?: () => void;
}

export const InvestigatorAnalysis: React.FC<InvestigatorAnalysisProps> = ({
  result,
  setActiveTab,
  onOpenViewer,
  onOpenReport,
}) => {
  return (
    <div className="space-y-6">
      {/* Top Investigator Summary Card */}
      <div className="bg-slate-950 text-white rounded-xl p-6 border border-slate-800 shadow-md space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-mono font-bold uppercase border border-indigo-500/30">
                FORENSIQ INVESTIGATOR SUITE
              </span>
              <span className="text-[11px] font-mono text-slate-400">ID: {result.id}</span>
            </div>
            <h3 className="text-2xl font-black tracking-tight text-white flex items-center gap-2 mt-1">
              <span>Technical Forensic Analysis Panel</span>
            </h3>
            <p className="text-xs text-slate-400">
              Timestamp: {result.uploadDate} • Inference Latency: {result.processingTimeMs}ms
            </p>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onOpenViewer(result)}
              className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
            >
              <Eye className="w-4 h-4 text-indigo-200" />
              <span>Interactive Viewer</span>
            </button>
            <button
              onClick={() => {
                if (onOpenReport) {
                  onOpenReport();
                } else {
                  setActiveTab('report');
                }
              }}
              className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold rounded-lg border border-slate-700 flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
            >
              <FileText className="w-4 h-4 text-indigo-300" />
              <span>Generate Audit PDF</span>
            </button>
          </div>
        </div>

        {/* 4 Metrics Box Banner */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold">Classification</span>
            <p className="text-lg font-black text-white">{result.classification}</p>
            <span className="text-[10px] text-indigo-300">Confidence: {result.confidenceScore}%</span>
          </div>

          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold">Authenticity Rating</span>
            <p className="text-lg font-black text-emerald-400">{result.authenticityScore}%</p>
            <span className="text-[10px] text-slate-400">Risk Level: {result.riskLevel.toUpperCase()}</span>
          </div>

          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold">Faces Tracked</span>
            <p className="text-lg font-black text-white">{result.facesDetected} Face(s)</p>
            <span className="text-[10px] text-slate-400">MTCNN Alignment</span>
          </div>

          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-bold">Model Agreement</span>
            <p className="text-sm font-bold text-amber-300 truncate">{result.modelAgreement}</p>
            <span className="text-[10px] text-slate-400">Bayesian Fusion</span>
          </div>
        </div>
      </div>

      {/* Model Score Breakdown & Explainability Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <ModelScoreCard
            scores={result.detailedModelScores}
            modelAgreement={result.modelAgreement}
            modelAgreementDescription={result.modelAgreementDescription}
          />
        </div>

        <div className="lg:col-span-2">
          <ExplainabilityPanel
            explainability={result.explainability}
            mediaName={result.mediaName}
          />
        </div>
      </div>

      {/* Video Analysis Timeline if video */}
      {result.mediaType === 'video' && result.videoData && (
        <VideoAnalysisTimeline
          videoData={result.videoData}
          mediaUrl={result.mediaUrl}
          mediaName={result.mediaName}
        />
      )}

      {/* Technical Forensic Signals & EXIF Metadata Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* EXIF Metadata Table */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider border-b border-slate-100 pb-2 flex items-center gap-2">
            <Camera className="w-4 h-4 text-indigo-600" />
            <span>EXIF & FILE STRUCTURE METADATA</span>
          </h4>

          <div className="divide-y divide-slate-100 text-xs">
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">File Identifier / Name:</span>
              <span className="font-mono font-semibold text-slate-900">{result.metadata.filename}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">File Size & Format:</span>
              <span className="font-mono text-slate-800">{result.metadata.filesize} ({result.metadata.filetype})</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">Dimensions / Resolution:</span>
              <span className="font-mono text-slate-800">{result.metadata.dimensions}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">Camera / Hardware:</span>
              <span className="font-medium text-slate-800">{result.metadata.cameraModel || 'None / Stripped'}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">Software / Export String:</span>
              <span className="font-medium text-slate-800">{result.metadata.software || 'None'}</span>
            </div>
            <div className="py-2 flex justify-between">
              <span className="text-slate-500">SHA-256 Hash:</span>
              <span className="font-mono text-[10px] text-slate-600 truncate max-w-[200px]">{result.metadata.hashSHA256}</span>
            </div>
          </div>
        </div>

        {/* Forensic Indicators List */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider border-b border-slate-100 pb-2 flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-indigo-600" />
            <span>FORENSIC MANIPULATION INDICATORS</span>
          </h4>

          <div className="space-y-3">
            {result.manipulationSignals.map((signal) => (
              <div
                key={signal.id}
                className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-slate-900">{signal.name}</span>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase border ${
                      signal.status === 'passed'
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                        : signal.status === 'warning'
                        ? 'bg-amber-50 text-amber-800 border-amber-200'
                        : 'bg-red-50 text-red-800 border-red-200'
                    }`}
                  >
                    {signal.score}
                  </span>
                </div>
                <p className="text-xs text-slate-600 leading-snug">{signal.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
