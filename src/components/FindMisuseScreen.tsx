import React, { useState } from 'react';
import {
  Search,
  Globe,
  Upload,
  RefreshCw,
  ExternalLink,
  ShieldAlert,
  FileCheck2,
  CheckCircle2,
  AlertTriangle,
  FolderPlus,
  FileText,
  Info,
  Sparkles,
} from 'lucide-react';
import { MisuseMatch, UserRole, NavItem } from '../types';
import { MOCK_MISUSE_MATCHES, SAMPLE_IMAGES } from '../data/mockData';

interface FindMisuseScreenProps {
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onOpenReport?: () => void;
}

export const FindMisuseScreen: React.FC<FindMisuseScreenProps> = ({
  userRole,
  setActiveTab,
  onOpenReport,
}) => {
  const [selectedMethod, setSelectedMethod] = useState<'reverse' | 'web' | 'submission'>('reverse');
  const [searchState, setSearchState] = useState<'idle' | 'scanning' | 'complete'>('complete');
  const [scanStep, setScanStep] = useState<string>('Searching index...');
  const [matches, setMatches] = useState<MisuseMatch[]>(MOCK_MISUSE_MATCHES);
  const [savedMatches, setSavedMatches] = useState<Record<string, boolean>>({});
  const [incidentCreated, setIncidentCreated] = useState<boolean>(false);

  const handleStartSearch = () => {
    setSearchState('scanning');
    const steps = [
      'Searching public web index...',
      'Comparing candidate vectors...',
      'Checking similarity thresholds...',
      'Verifying ownership watermark signals...',
      'Analyzing potential manipulation regions...',
    ];

    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length) {
        setScanStep(steps[stepIdx]);
        stepIdx++;
      } else {
        clearInterval(interval);
        setSearchState('complete');
      }
    }, 600);
  };

  const toggleSaveMatch = (id: string) => {
    setSavedMatches((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const handleCreateIncident = () => {
    setIncidentCreated(true);
    setTimeout(() => setIncidentCreated(false), 3500);
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Header Bar */}
      <div>
        <h2 className="text-2xl font-bold text-stone-950 tracking-tight">Find Misuse</h2>
        <p className="text-stone-600 text-xs sm:text-sm mt-1">
          Look for public copies, modified versions, or possible deepfakes of your registered image.
        </p>
      </div>

      {/* Discovery Methods Selector */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-5">
        <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2">
          SELECT DISCOVERY METHOD
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Method 1 */}
          <button
            onClick={() => setSelectedMethod('reverse')}
            className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
              selectedMethod === 'reverse'
                ? 'border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                : 'border-[#e2d8c3]/80 glass-card text-stone-800 hover:bg-[#faf7f2]/60'
            }`}
          >
            <div className="flex items-center gap-2 font-bold text-xs mb-1">
              <Search className="w-4 h-4 text-amber-300" />
              <span>Reverse Vector Search</span>
            </div>
            <p
              className={`text-[11px] leading-relaxed ${
                selectedMethod === 'reverse' ? 'text-amber-200' : 'text-stone-500'
              }`}
            >
              Find visually similar copies across supported search services and perceptual registries.
            </p>
          </button>

          {/* Method 2 */}
          <button
            onClick={() => setSelectedMethod('web')}
            className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
              selectedMethod === 'web'
                ? 'border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                : 'border-[#e2d8c3]/80 glass-card text-stone-800 hover:bg-[#faf7f2]/60'
            }`}
          >
            <div className="flex items-center gap-2 font-bold text-xs mb-1">
              <Globe className="w-4 h-4 text-amber-300" />
              <span>Public Web Search</span>
            </div>
            <p
              className={`text-[11px] leading-relaxed ${
                selectedMethod === 'web' ? 'text-amber-200' : 'text-stone-500'
              }`}
            >
              Scan selected publicly accessible websites, repositories, and forums for matching media.
            </p>
          </button>

          {/* Method 3 */}
          <button
            onClick={() => setSelectedMethod('submission')}
            className={`p-4 rounded-xl border text-left transition-all cursor-pointer ${
              selectedMethod === 'submission'
                ? 'border-stone-950 bg-[#1e1b18] text-white shadow-xs'
                : 'border-[#e2d8c3]/80 glass-card text-stone-800 hover:bg-[#faf7f2]/60'
            }`}
          >
            <div className="flex items-center gap-2 font-bold text-xs mb-1">
              <Upload className="w-4 h-4 text-amber-300" />
              <span>User Submission Match</span>
            </div>
            <p
              className={`text-[11px] leading-relaxed ${
                selectedMethod === 'submission' ? 'text-amber-200' : 'text-stone-500'
              }`}
            >
              Upload an image you found yourself and compare it with your registered media baseline.
            </p>
          </button>
        </div>

        {/* Selected Image Baseline & Start Search */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-[#e2d8c3]/80">
          <div className="flex items-center gap-3">
            <img
              src={SAMPLE_IMAGES.authenticPortrait}
              alt="Registered baseline"
              className="w-10 h-10 object-cover rounded border border-[#d8ccb6]"
            />
            <div className="text-xs">
              <p className="font-bold text-stone-950">Portrait_Authentic_01.jpg</p>
              <p className="text-[11px] text-stone-500">Registered Baseline • ID: REG-2026-884192</p>
            </div>
          </div>

          <button
            onClick={handleStartSearch}
            disabled={searchState === 'scanning'}
            className="w-full sm:w-auto px-6 py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs disabled:opacity-50"
          >
            {searchState === 'scanning' ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-amber-300" />
                <span>Scanning Index...</span>
              </>
            ) : (
              <>
                <Search className="w-4 h-4 text-amber-300" />
                <span>Start Misuse Search</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Scanning Progress Screen */}
      {searchState === 'scanning' && (
        <div className="glass-panel rounded-xl p-10 text-center shadow-xs space-y-4">
          <RefreshCw className="w-8 h-8 text-red-700 animate-spin mx-auto" />
          <h3 className="font-bold text-stone-950 text-lg">Scanning Public Sources &amp; Web Indices</h3>
          <p className="text-xs font-mono text-red-800 glass-input px-3 py-1 rounded inline-block font-bold">
            {scanStep}
          </p>
        </div>
      )}

      {/* Search Results Display */}
      {searchState === 'complete' && (
        <div className="space-y-6">
          {/* Results Summary Header */}
          <div className="glass-panel rounded-xl p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-stone-950 text-lg">5 Possible Matches Found</span>
                <span className="px-2 py-0.5 rounded bg-amber-50/80 text-amber-800 text-xs font-bold border border-amber-200">
                  3 With Watermark Signal
                </span>
              </div>
              <p className="text-xs text-stone-500 mt-0.5">
                Results depend on publicly accessible search services and registered index coverage.
              </p>
            </div>

            {/* Quick Action Buttons */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleCreateIncident}
                className="px-3.5 py-2 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
              >
                <FolderPlus className="w-4 h-4 text-amber-300" />
                <span>Create Incident Record</span>
              </button>

              <button
                onClick={() => setActiveTab('report')}
                className="px-3.5 py-2 glass-input hover:bg-[#e8decb]/60 text-stone-800 text-xs font-semibold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <FileText className="w-4 h-4 text-stone-600" />
                <span>Generate Ready Report</span>
              </button>
            </div>
          </div>

          {/* Incident Creation Toast Callout */}
          {incidentCreated && (
            <div className="p-4 bg-emerald-50/90 border border-emerald-200 rounded-xl text-xs text-emerald-900 flex items-center justify-between shadow-xs">
              <div className="flex items-center gap-2.5">
                <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                <div>
                  <p className="font-bold text-sm">Incident INC-2026-042 Successfully Created</p>
                  <p className="text-emerald-700">
                    Grouped 5 discovered matches, timestamps, and forensic evidence into active investigation binder.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setActiveTab('report')}
                className="px-3 py-1.5 bg-emerald-700 text-white rounded font-semibold hover:bg-emerald-800 cursor-pointer"
              >
                View Incident Report
              </button>
            </div>
          )}

          {/* Results List Cards */}
          <div className="space-y-4">
            {matches.map((item) => (
              <div
                key={item.id}
                className="glass-panel rounded-xl p-5 shadow-xs hover:border-stone-400/80 transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-5"
              >
                {/* Left Thumbnail & Info */}
                <div className="flex items-start gap-4">
                  <img
                    src={item.thumbnailUrl}
                    alt={item.sourceName}
                    className="w-20 h-20 object-cover rounded-lg border border-[#d8ccb6] shrink-0"
                  />

                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase border ${
                          item.classification === 'CONFIRMED MATCH'
                            ? 'bg-emerald-50/80 text-emerald-800 border-emerald-200'
                            : item.classification === 'AI MANIPULATED' || item.classification === 'TAMPERED'
                            ? 'bg-red-50/80 text-red-800 border-red-200'
                            : 'bg-amber-50/80 text-amber-800 border-amber-200'
                        }`}
                      >
                        {item.classification}
                      </span>

                      <span className="text-[11px] font-bold text-stone-800 font-mono">
                        {item.similarity}% Similarity
                      </span>
                    </div>

                    <h4 className="font-bold text-stone-950 text-sm">{item.sourceName}</h4>

                    <a
                      href={item.sourceUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-red-700 hover:underline font-mono flex items-center gap-1 truncate max-w-md"
                    >
                      <span>{item.sourceUrl}</span>
                      <ExternalLink className="w-3 h-3" />
                    </a>

                    <div className="flex flex-wrap items-center gap-3 text-[11px] text-stone-500 pt-1">
                      <span>Found: {item.foundDate}</span>
                      <span>•</span>
                      <span>Category: {item.platformCategory}</span>
                      <span>•</span>
                      <span className="font-semibold text-stone-700">
                        Watermark: {item.watermarkStatus}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Right Action Buttons */}
                <div className="flex flex-wrap items-center gap-2 shrink-0 w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-[#e2d8c3]/80">
                  <button
                    onClick={() => toggleSaveMatch(item.id)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                      savedMatches[item.id]
                        ? 'bg-emerald-50/90 border-emerald-200 text-emerald-800'
                        : 'glass-input text-stone-700 hover:bg-[#e8decb]/60'
                    }`}
                  >
                    {savedMatches[item.id] ? '✓ Saved Evidence' : 'Save Evidence'}
                  </button>

                  <button
                    onClick={() =>
                      alert(
                        `Opening ready-to-submit DMCA/Takedown evidence document for ${item.sourceName}. ForensIQ assists with evidence gathering; submission occurs directly on host reporting forms.`
                      )
                    }
                    className="px-3 py-1.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
                  >
                    Report Content
                  </button>
                </div>
              </div>
            ))}
          </div>

          {/* DMCA / Platform Disclaimer Notice */}
          <div className="p-4 glass-input rounded-xl text-xs text-stone-600 flex items-start gap-3">
            <Info className="w-5 h-5 text-stone-500 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Notice:</strong> ForensIQ provides evidence gathering and provenance reports to assist media owners. ForensIQ does not directly modify or remove content hosted on third-party websites or platforms.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
