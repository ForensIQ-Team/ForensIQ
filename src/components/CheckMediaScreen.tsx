import React, { useState } from 'react';
import {
  Upload,
  FileCheck2,
  Sparkles,
  RefreshCw,
  Eye,
  ShieldCheck,
  Film,
  UserCheck,
  ShieldAlert,
  ArrowRight,
  AlertTriangle,
  Play,
  CheckCircle2,
} from 'lucide-react';
import { DetectionResult, UserRole, NavItem, PipelineStage } from '../types';
import { detectionService, DEFAULT_PIPELINE_STAGES, PRESET_DETECTIONS } from '../services/detectionService';
import { historyService } from '../services/historyService';
import { DetectionPipeline } from './detection/DetectionPipeline';
import { NormalUserResult } from './detection/NormalUserResult';
import { InvestigatorAnalysis } from './detection/InvestigatorAnalysis';
import { SAMPLE_IMAGES } from '../data/mockData';

interface CheckMediaScreenProps {
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onOpenViewer: (analysis: DetectionResult) => void;
  onOpenReport?: (analysis?: DetectionResult) => void;
}

export const CheckMediaScreen: React.FC<CheckMediaScreenProps> = ({
  userRole,
  setActiveTab,
  onOpenViewer,
  onOpenReport,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mediaPreviewUrl, setMediaPreviewUrl] = useState<string | null>(null);
  const [analysisState, setAnalysisState] = useState<'idle' | 'analyzing' | 'complete'>('complete');
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(
    DEFAULT_PIPELINE_STAGES.map((s) => ({ ...s, status: 'pending' }))
  );
  const [detectionResult, setDetectionResult] = useState<DetectionResult>(PRESET_DETECTIONS.real);
  const [activePresetKey, setActivePresetKey] = useState<string>('real');
  const [viewRole, setViewRole] = useState<UserRole>(userRole);

  // Synchronize viewRole if userRole changes
  React.useEffect(() => {
    setViewRole(userRole);
  }, [userRole]);

  // Execute detection pipeline on preset select or upload
  const runDetectionPipeline = async (input: File | string) => {
    setAnalysisState('analyzing');
    setCurrentStageIndex(0);

    try {
      const result = await detectionService.analyzeMedia(input, (stageIndex, updatedStages) => {
        setCurrentStageIndex(stageIndex);
        setPipelineStages(updatedStages);
      });

      setDetectionResult(result);
      setAnalysisState('complete');

      // Auto-log into persistent historyService
      historyService.addHistoryItem(result);
    } catch (err) {
      console.error('Detection pipeline error:', err);
      setAnalysisState('complete');
    }
  };

  // Handle Preset Select
  const handlePresetSelect = (key: string) => {
    setActivePresetKey(key);
    setSelectedFile(null);
    setMediaPreviewUrl(null);
    runDetectionPipeline(key);
  };

  // Handle File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setMediaPreviewUrl(url);
      runDetectionPipeline(file);
    }
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Title & View Switcher Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-serif font-bold text-stone-950 tracking-tight">AI Media Detection Engine</h2>
          <p className="text-stone-600 text-xs sm:text-sm mt-1">
            Detect deepfakes, synthetic AI media, and face swaps using multi-model score fusion &amp; explainability heatmaps.
          </p>
        </div>

        {/* View Mode Toggle (Normal vs Investigator) */}
        <div className="flex items-center gap-2 glass-panel rounded-xl p-1.5 shadow-xs">
          <span className="text-xs font-semibold text-stone-500 pl-2">Presentation View:</span>
          <div className="flex rounded-lg p-0.5" style={{ background: 'rgba(230, 220, 200, 0.28)' }}>
            <button
              onClick={() => setViewRole('normal')}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-all cursor-pointer ${
                viewRole === 'normal' ? 'text-white shadow-xs' : 'text-stone-700 hover:text-stone-950'
              }`}
              style={viewRole === 'normal' ? { background: 'rgba(20, 18, 15, 0.82)' } : {}}
            >
              Normal View
            </button>
            <button
              onClick={() => setViewRole('investigator')}
              className={`px-3 py-1 rounded-md text-xs font-bold transition-all cursor-pointer ${
                viewRole === 'investigator' ? 'text-white shadow-xs' : 'text-stone-700 hover:text-stone-950'
              }`}
              style={viewRole === 'investigator' ? { background: 'rgba(20, 18, 15, 0.82)' } : {}}
            >
              Investigator View
            </button>
          </div>
        </div>
      </div>

      {/* Upload Zone & Sample Selector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Box Dropzone — forensic scanning surface */}
        <div className="lg:col-span-2 glass-panel rounded-xl p-6 shadow-xs space-y-4">
          <div className="glass-upload rounded-xl p-8 text-center flex flex-col items-center justify-center min-h-[220px]">
            <div className="w-12 h-12 rounded-full flex items-center justify-center text-stone-900 mb-3" style={{ background: 'rgba(230, 215, 185, 0.35)', border: '1px solid rgba(180, 158, 118, 0.40)' }}>
              <Upload className="w-5 h-5 text-stone-800" />
            </div>

            <p className="text-sm font-semibold text-stone-950">
              Drag &amp; drop media file here, or{' '}
              <label className="text-red-700 hover:text-red-800 underline cursor-pointer font-bold">
                Choose a file
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </p>

            <p className="text-xs text-stone-500 mt-1">
              Supports JPG, PNG, WEBP, MP4, MOV (up to 100 MB)
            </p>

            {selectedFile && (
              <div className="mt-4 px-3 py-1.5 glass-input text-stone-950 rounded-md text-xs font-semibold flex items-center gap-2">
                <FileCheck2 className="w-4 h-4 text-red-700" />
                <span>Uploaded: {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)</span>
              </div>
            )}
          </div>
        </div>

        {/* Quick Sample Preset Selector */}
        <div className="glass-panel rounded-xl p-5 shadow-xs space-y-3 flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs font-mono font-bold text-stone-950 uppercase tracking-wider mb-2">
              <Sparkles className="w-4 h-4 text-red-700" />
              <span>Or Select Preset Demo Scenario</span>
            </div>
            <p className="text-xs text-stone-600 mb-3">
              Test detection pipeline accuracy across deterministic sample cases:
            </p>

            <div className="space-y-2">
              <button
                onClick={() => handlePresetSelect('real')}
                className="w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-all cursor-pointer"
                style={{
                  background: activePresetKey === 'real' && !selectedFile ? 'rgba(220, 200, 160, 0.45)' : 'rgba(255, 250, 240, 0.18)',
                  border: activePresetKey === 'real' && !selectedFile ? '1px solid rgba(100, 75, 40, 0.55)' : '1px solid rgba(200, 185, 155, 0.30)',
                }}
              >
                <img
                  src={SAMPLE_IMAGES.authenticPortrait}
                  alt="Real"
                  className="w-10 h-10 rounded object-cover border border-[#d8ccb6]"
                />
                <div className="text-xs">
                  <p className="font-bold text-stone-950">1. Authentic Camera Portrait</p>
                  <p className="text-[10px] text-emerald-800 font-semibold">Expected: Real (94% Score)</p>
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('deepfake')}
                className="w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-all cursor-pointer"
                style={{
                  background: activePresetKey === 'deepfake' && !selectedFile ? 'rgba(220, 200, 160, 0.45)' : 'rgba(255, 250, 240, 0.18)',
                  border: activePresetKey === 'deepfake' && !selectedFile ? '1px solid rgba(100, 75, 40, 0.55)' : '1px solid rgba(200, 185, 155, 0.30)',
                }}
              >
                <img
                  src={SAMPLE_IMAGES.aiSyntheticFace}
                  alt="Deepfake"
                  className="w-10 h-10 rounded object-cover border border-[#d8ccb6]"
                />
                <div className="text-xs">
                  <p className="font-bold text-stone-950">2. Face-Swap Deepfake</p>
                  <p className="text-[10px] text-red-700 font-semibold">Expected: Deepfake (95% Fake)</p>
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('aiGenerated')}
                className="w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-all cursor-pointer"
                style={{
                  background: activePresetKey === 'aiGenerated' && !selectedFile ? 'rgba(220, 200, 160, 0.45)' : 'rgba(255, 250, 240, 0.18)',
                  border: activePresetKey === 'aiGenerated' && !selectedFile ? '1px solid rgba(100, 75, 40, 0.55)' : '1px solid rgba(200, 185, 155, 0.30)',
                }}
              >
                <img
                  src={SAMPLE_IMAGES.aiSyntheticFace}
                  alt="GenAI"
                  className="w-10 h-10 rounded object-cover border border-[#d8ccb6]"
                />
                <div className="text-xs">
                  <p className="font-bold text-stone-950">3. GenAI Synthetic Profile</p>
                  <p className="text-[10px] text-red-700 font-semibold">Expected: AI-Generated</p>
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('uncertain')}
                className="w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-all cursor-pointer"
                style={{
                  background: activePresetKey === 'uncertain' && !selectedFile ? 'rgba(220, 200, 160, 0.45)' : 'rgba(255, 250, 240, 0.18)',
                  border: activePresetKey === 'uncertain' && !selectedFile ? '1px solid rgba(100, 75, 40, 0.55)' : '1px solid rgba(200, 185, 155, 0.30)',
                }}
              >
                <img
                  src={SAMPLE_IMAGES.manipulatedDocument}
                  alt="Uncertain"
                  className="w-10 h-10 rounded object-cover border border-[#d8ccb6]"
                />
                <div className="text-xs">
                  <p className="font-bold text-stone-950">4. Low-Res Compressed Media</p>
                  <p className="text-[10px] text-amber-800 font-semibold">Expected: Uncertain (Model Split)</p>
                </div>
              </button>

              <button
                onClick={() => handlePresetSelect('video')}
                className="w-full p-2.5 rounded-lg text-left flex items-center gap-3 transition-all cursor-pointer"
                style={{
                  background: activePresetKey === 'video' && !selectedFile ? 'rgba(220, 200, 160, 0.45)' : 'rgba(255, 250, 240, 0.18)',
                  border: activePresetKey === 'video' && !selectedFile ? '1px solid rgba(100, 75, 40, 0.55)' : '1px solid rgba(200, 185, 155, 0.30)',
                }}
              >
                <div className="w-10 h-10 rounded bg-[#1e1b18] text-amber-300 flex items-center justify-center font-bold shrink-0">
                  <Film className="w-5 h-5" />
                </div>
                <div className="text-xs">
                  <p className="font-bold text-stone-950">5. Video Deepfake (MP4)</p>
                  <p className="text-[10px] text-red-700 font-semibold">Expected: Deepfake (24 Frames)</p>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Animated 10-Stage Pipeline Processing */}
      {analysisState === 'analyzing' && (
        <DetectionPipeline
          stages={pipelineStages}
          currentStageIndex={currentStageIndex}
        />
      )}

      {/* Completed Results Stage */}
      {analysisState === 'complete' && (
        <div className="space-y-6">
          {/* Render View based on viewRole */}
          {viewRole === 'normal' ? (
            <NormalUserResult
              result={detectionResult}
              onExploreDetails={() => setViewRole('investigator')}
              onOpenReport={() => onOpenReport && onOpenReport(detectionResult)}
            />
          ) : (
            <InvestigatorAnalysis
              result={detectionResult}
              setActiveTab={setActiveTab}
              onOpenViewer={onOpenViewer}
              onOpenReport={() => onOpenReport && onOpenReport(detectionResult)}
            />
          )}
        </div>
      )}
    </div>
  );
};
