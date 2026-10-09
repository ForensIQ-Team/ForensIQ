import React, { useState, useRef, useCallback } from 'react';
import {
  Search,
  Globe,
  Upload,
  RefreshCw,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  FolderPlus,
  FileText,
  Info,
  ImagePlus,
  X,
  Globe2,
  Eye,
  ShieldAlert,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { MisuseMatch, UserRole, NavItem } from '../types';
import { MOCK_MISUSE_MATCHES, SAMPLE_IMAGES } from '../data/mockData';
import { useAuth } from '../services/authContext';
import { investigatorService } from '../services/investigatorService';
import {
  searchPropagation,
  matchTypeLabel,
  matchTypeBadgeClasses,
  PROPAGATION_STEPS,
  PropagationSearchResponse,
} from '../services/propagationService';
import { PropagationNetwork } from './PropagationNetwork';

interface FindMisuseScreenProps {
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onOpenReport?: () => void;
}

type SelectedMethod = 'reverse' | 'web' | 'submission';
type SearchState    = 'idle' | 'scanning' | 'complete' | 'error';

export const FindMisuseScreen: React.FC<FindMisuseScreenProps> = ({
  userRole,
  setActiveTab,
  onOpenReport,
}) => {
  const [selectedMethod, setSelectedMethod] = useState<SelectedMethod>('reverse');

  const [legacySearchState, setLegacySearchState] = useState<'idle' | 'scanning' | 'complete'>('complete');
  const [legacyScanStep, setLegacyScanStep]       = useState<string>('Searching index...');
  const [legacyMatches, setLegacyMatches]          = useState<MisuseMatch[]>(MOCK_MISUSE_MATCHES);
  const [savedMatches, setSavedMatches]            = useState<Record<string, boolean>>({});
  const [incidentCreated, setIncidentCreated]      = useState<boolean>(false);

  const [submittedFile, setSubmittedFile]            = useState<File | null>(null);
  const [submittedPreview, setSubmittedPreview]      = useState<string | null>(null);
  const [submissionSearchState, setSubmissionSearchState] = useState<SearchState>('idle');
  const [submissionStep, setSubmissionStep]          = useState<string>('');
  const [submissionStepIdx, setSubmissionStepIdx]    = useState<number>(0);
  const [submissionResults, setSubmissionResults]    = useState<PropagationSearchResponse | null>(null);
  const [submissionError, setSubmissionError]        = useState<string | null>(null);
  const [propSavedItems, setPropSavedItems]          = useState<Record<string, boolean>>({});
  const [showPropagationNetwork, setShowPropagationNetwork] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropRef      = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging]                 = useState(false);

  const { user } = useAuth();
  const [caseCreating, setCaseCreating] = useState(false);
  const [createdCaseMsg, setCreatedCaseMsg] = useState<string | null>(null);

  const handleCreateCaseFromResults = async (resultsCount: number) => {
    try {
      setCaseCreating(true);
      const created = await investigatorService.createCase({
        title: `Web Misuse Investigation (${resultsCount} Discovered Sources)`,
        description: `Investigating unauthorized media dissemination and viral propagation discovered across ${resultsCount} online sources.`,
        severity: 'high',
        tags: ['web-misuse', 'propagation', 'disinformation'],
      });
      setCreatedCaseMsg(`✓ Case "${created.title}" successfully opened.`);
      setTimeout(() => {
        setActiveTab('investigator-workspace');
      }, 1000);
    } catch (err: any) {
      alert(err.message || 'Failed to create case');
    } finally {
      setCaseCreating(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024)       return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleFileSelect = useCallback((file: File) => {
    setSubmissionResults(null);
    setSubmissionError(null);
    setSubmissionSearchState('idle');
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!allowed.includes(file.type)) {
      setSubmissionError('Unsupported format. Please upload JPG, JPEG, PNG, or WEBP.');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setSubmissionError('Image exceeds 20 MB limit.');
      return;
    }
    setSubmittedFile(file);
    const reader = new FileReader();
    reader.onload = e => setSubmittedPreview(e.target?.result as string ?? null);
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const handleDragLeave = () => setIsDragging(false);

  const clearFile = () => {
    setSubmittedFile(null);
    setSubmittedPreview(null);
    setSubmissionResults(null);
    setSubmissionError(null);
    setSubmissionSearchState('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleStartSubmissionSearch = async () => {
    if (!submittedFile) return;
    setSubmissionSearchState('scanning');
    setSubmissionError(null);
    setSubmissionResults(null);
    setSubmissionStep(PROPAGATION_STEPS[0]);
    setSubmissionStepIdx(0);

    try {
      const result = await searchPropagation(
        submittedFile,
        (step, idx) => {
          setSubmissionStep(step);
          setSubmissionStepIdx(idx);
        }
      );
      setSubmissionResults(result);
      setSubmissionSearchState('complete');
    } catch (err: any) {
      setSubmissionError(err?.message ?? 'An unknown error occurred.');
      setSubmissionSearchState('error');
    }
  };

  const handleStartLegacySearch = () => {
    setLegacySearchState('scanning');
    const steps = [
      'Searching public web index...',
      'Comparing candidate vectors...',
      'Checking similarity thresholds...',
      'Verifying ownership watermark signals...',
      'Analyzing potential manipulation regions...',
    ];
    let stepIdx = 0;
    const interval = setInterval(() => {
      if (stepIdx < steps.length) { setLegacyScanStep(steps[stepIdx]); stepIdx++; }
      else { clearInterval(interval); setLegacySearchState('complete'); }
    }, 600);
  };

  const toggleSaveMatch = (id: string) =>
    setSavedMatches(prev => ({ ...prev, [id]: !prev[id] }));

  const togglePropSave = (key: string) =>
    setPropSavedItems(prev => ({ ...prev, [key]: !prev[key] }));

  const handleCreateIncident = () => {
    setIncidentCreated(true);
    setTimeout(() => setIncidentCreated(false), 3500);
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-stone-950 tracking-tight">Find Misuse</h2>
        <p className="text-stone-600 text-xs sm:text-sm mt-1">
          Look for public copies, modified versions, or possible deepfakes of your registered image.
        </p>
      </div>

      {/* ---- Discovery Method Selector ---- */}
      <div className="glass-panel rounded-xl p-6 shadow-xs space-y-5">
        <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-2">
          SELECT DISCOVERY METHOD
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
            <p className={`text-[11px] leading-relaxed ${selectedMethod === 'reverse' ? 'text-amber-200' : 'text-stone-500'}`}>
              Find visually similar copies across supported search services and perceptual registries.
            </p>
          </button>

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
            <p className={`text-[11px] leading-relaxed ${selectedMethod === 'web' ? 'text-amber-200' : 'text-stone-500'}`}>
              Scan selected publicly accessible websites, repositories, and forums for matching media.
            </p>
          </button>

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
            <p className={`text-[11px] leading-relaxed ${selectedMethod === 'submission' ? 'text-amber-200' : 'text-stone-500'}`}>
              Upload a suspicious image you found online. ForensIQ will search the public web for visually matching copies.
            </p>
          </button>
        </div>

        {selectedMethod !== 'submission' && (
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-[#e2d8c3]/80">
            <div className="flex items-center gap-3">
              <img
                src={SAMPLE_IMAGES.authenticPortrait}
                alt="Registered baseline"
                className="w-10 h-10 object-cover rounded border border-[#d8ccb6]"
              />
              <div className="text-xs">
                <p className="font-bold text-stone-950">Portrait_Authentic_01.jpg</p>
                <p className="text-[11px] text-stone-500">Registered Baseline · ID: REG-2026-884192</p>
              </div>
            </div>
            <button
              onClick={handleStartLegacySearch}
              disabled={legacySearchState === 'scanning'}
              className="w-full sm:w-auto px-6 py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs disabled:opacity-50"
            >
              {legacySearchState === 'scanning' ? (
                <><RefreshCw className="w-4 h-4 animate-spin text-amber-300" /><span>Scanning Index...</span></>
              ) : (
                <><Search className="w-4 h-4 text-amber-300" /><span>Start Misuse Search</span></>
              )}
            </button>
          </div>
        )}
      </div>

      {/* ===================== USER SUBMISSION PANEL ===================== */}
      {selectedMethod === 'submission' && (
        <div className="space-y-6">
          <div className="glass-panel rounded-xl p-6 shadow-xs space-y-5">
            <div>
              <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider">
                Upload Suspicious Image
              </h3>
              <p className="text-xs text-stone-500 mt-1">
                Upload an image you discovered online and let ForensIQ search the public web for visually matching copies using Google Lens.
              </p>
            </div>

            {!submittedFile ? (
              <div
                ref={dropRef}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={`glass-upload rounded-xl flex flex-col items-center justify-center gap-4 py-14 cursor-pointer transition-all ${
                  isDragging ? 'border-red-700 bg-[#fff5f0]/40' : ''
                }`}
              >
                <div className={`w-14 h-14 rounded-full bg-stone-950/5 flex items-center justify-center transition-transform ${isDragging ? 'scale-110' : ''}`}>
                  <ImagePlus className={`w-7 h-7 transition-colors ${isDragging ? 'text-red-700' : 'text-stone-500'}`} />
                </div>
                <div className="text-center">
                  <p className="font-semibold text-stone-800 text-sm">
                    {isDragging ? 'Drop image here' : 'Drag & drop your image here'}
                  </p>
                  <p className="text-xs text-stone-500 mt-1">or click to browse files</p>
                  <p className="text-[11px] text-stone-400 mt-2">JPG · JPEG · PNG · WEBP · Max 20 MB</p>
                </div>
              </div>
            ) : (
              <div className="glass-card rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center gap-4">
                <img
                  src={submittedPreview ?? ''}
                  alt="Preview"
                  className="w-20 h-20 object-cover rounded-lg border border-[#d8ccb6] shrink-0"
                />
                <div className="flex-1 min-w-0 space-y-0.5">
                  <p className="font-bold text-stone-950 text-sm truncate">{submittedFile.name}</p>
                  <p className="text-xs text-stone-500">
                    {submittedFile.type.replace('image/', '').toUpperCase()} · {formatBytes(submittedFile.size)}
                  </p>
                  <p className="text-[11px] text-stone-400">Ready to search</p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="px-3 py-1.5 glass-input text-stone-700 text-xs font-semibold rounded-lg hover:bg-[#e8decb]/60 cursor-pointer transition-colors"
                  >
                    Change
                  </button>
                  <button
                    onClick={clearFile}
                    className="p-1.5 glass-input text-stone-500 rounded-lg hover:bg-red-50/50 hover:text-red-700 cursor-pointer transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}

            <input
              ref={fileInputRef}
              type="file"
              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFileSelect(f); }}
            />

            {submissionError && submissionSearchState !== 'error' && (
              <div className="flex items-center gap-2 text-xs text-red-800 bg-red-50/80 border border-red-200 rounded-lg px-3 py-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{submissionError}</span>
              </div>
            )}

            <div className="flex justify-end pt-2">
              <button
                onClick={handleStartSubmissionSearch}
                disabled={!submittedFile || submissionSearchState === 'scanning'}
                className="px-6 py-2.5 bg-[#1e1b18] hover:bg-stone-900 disabled:opacity-40 text-white text-xs font-semibold rounded-lg flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
              >
                {submissionSearchState === 'scanning' ? (
                  <><RefreshCw className="w-4 h-4 animate-spin text-amber-300" /><span>Searching...</span></>
                ) : (
                  <><Search className="w-4 h-4 text-amber-300" /><span>Search Public Web</span></>
                )}
              </button>
            </div>
          </div>

          {submissionSearchState === 'scanning' && (
            <div className="glass-panel rounded-xl p-10 text-center shadow-xs space-y-5">
              <div className="flex justify-center gap-1.5 mb-2">
                {PROPAGATION_STEPS.map((_, idx) => (
                  <div
                    key={idx}
                    className={`h-1 rounded-full transition-all duration-500 ${
                      idx <= submissionStepIdx ? 'bg-stone-950 w-8' : 'bg-stone-300 w-4'
                    }`}
                  />
                ))}
              </div>
              <RefreshCw className="w-8 h-8 text-red-700 animate-spin mx-auto" />
              <h3 className="font-bold text-stone-950 text-lg">Searching Public Web Sources</h3>
              <p className="text-xs font-mono text-red-800 glass-input px-3 py-1 rounded inline-block font-bold">
                {submissionStep}
              </p>
              <p className="text-xs text-stone-500">
                This may take 10–30 seconds. ForensIQ is querying Google Lens for visual matches.
              </p>
            </div>
          )}

          {submissionSearchState === 'error' && submissionError && (
            <div className="glass-panel rounded-xl p-6 shadow-xs space-y-3">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-6 h-6 text-red-700 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-red-800 text-sm">Search Failed</h4>
                  <p className="text-xs text-red-700 mt-1">{submissionError}</p>
                </div>
              </div>
              <button
                onClick={handleStartSubmissionSearch}
                disabled={!submittedFile}
                className="px-4 py-2 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg flex items-center gap-2 cursor-pointer"
              >
                <RefreshCw className="w-4 h-4 text-amber-300" />
                Retry Search
              </button>
            </div>
          )}

          {submissionSearchState === 'complete' && submissionResults && (
            <div className="space-y-6">
              {submissionResults.status === 'no_results' && (
                <div className="glass-panel rounded-xl p-10 text-center shadow-xs space-y-3">
                  <Eye className="w-8 h-8 text-stone-400 mx-auto" />
                  <h4 className="font-bold text-stone-800 text-base">No Matches Found</h4>
                  <p className="text-xs text-stone-500 max-w-sm mx-auto">
                    Google Lens did not find any publicly visible copies of this image.
                    This may mean the image is new, not indexed, or its distribution is limited.
                  </p>
                </div>
              )}

              {submissionResults.status === 'success' && (
                <>
                  <div className="glass-panel rounded-xl p-5 shadow-xs flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-3 flex-wrap">
                        <span className="font-bold text-stone-950 text-lg">
                          {submissionResults.total} Possible Match{submissionResults.total !== 1 ? 'es' : ''} Found
                        </span>
                        {submissionResults.exact_matches.length > 0 && (
                          <span className="px-2 py-0.5 rounded bg-emerald-50/80 text-emerald-800 text-xs font-bold border border-emerald-200">
                            {submissionResults.exact_matches.length} Exact
                          </span>
                        )}
                        {submissionResults.visual_matches.length > 0 && (
                          <span className="px-2 py-0.5 rounded bg-amber-50/80 text-amber-800 text-xs font-bold border border-amber-200">
                            {submissionResults.visual_matches.length} Visual
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-stone-500 mt-0.5">
                        Discovered via Google Lens · Search Discovery (not forensic verification)
                      </p>
                    </div>
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
                        <span>Generate Report</span>
                      </button>
                    </div>
                  </div>

                  {incidentCreated && (
                    <div className="p-4 bg-emerald-50/90 border border-emerald-200 rounded-xl text-xs text-emerald-900 flex items-center justify-between shadow-xs">
                      <div className="flex items-center gap-2.5">
                        <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                        <div>
                          <p className="font-bold text-sm">Incident INC-2026-{Math.floor(Math.random() * 999).toString().padStart(3, '0')} Created</p>
                          <p className="text-emerald-700">
                            Grouped {submissionResults.total} discovered matches into active investigation binder.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Propagation Network */}
                  <div>
                    <button
                      onClick={() => setShowPropagationNetwork(v => !v)}
                      className="w-full flex items-center justify-between px-4 py-3 glass-input hover:bg-[#e8decb]/60 rounded-xl border border-stone-300/60 text-stone-800 font-semibold text-sm transition-colors cursor-pointer"
                    >
                      <span className="flex items-center gap-2">
                        <Globe className="w-4 h-4 text-amber-600" />
                        View Propagation Network
                        <span className="text-[10px] text-stone-500 font-normal ml-1">
                          — visual media spread map
                        </span>
                      </span>
                      {showPropagationNetwork
                        ? <ChevronUp className="w-4 h-4 text-stone-500" />
                        : <ChevronDown className="w-4 h-4 text-stone-500" />}
                    </button>

                    {showPropagationNetwork && (
                      <div className="mt-4">
                        <PropagationNetwork
                          results={submissionResults.all_results}
                          searchedImagePreview={submittedPreview}
                        />
                      </div>
                    )}
                  </div>

                  <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div>
                      <h3 className="font-bold text-stone-950 text-sm uppercase tracking-wider">
                        Discovered Results ({submissionResults.all_results.length})
                      </h3>
                      <p className="text-xs text-stone-500 mt-0.5">
                        Detailed source records and investigative actions for all indexed matches.
                      </p>
                    </div>

                    {user && (user.role === 'investigator' || user.role === 'admin') && (
                      <button
                        onClick={() => handleCreateCaseFromResults(submissionResults.all_results.length)}
                        disabled={caseCreating}
                        className="px-3.5 py-2 bg-stone-950 hover:bg-stone-900 text-white text-xs font-bold rounded-xl flex items-center gap-1.5 transition-all shadow-xs cursor-pointer shrink-0 disabled:opacity-50"
                      >
                        <FolderPlus className="w-4 h-4 text-red-400" />
                        <span>{caseCreating ? 'Opening Case...' : 'Create Case from Results'}</span>
                      </button>
                    )}
                  </div>

                  {createdCaseMsg && (
                    <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900 text-xs font-semibold flex items-center gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                      <span>{createdCaseMsg}</span>
                    </div>
                  )}

                  <div className="space-y-4">
                    {submissionResults.all_results.map((item, idx) => {
                      const saveKey = item.url ?? `result_${idx}`;
                      return (
                        <div
                          key={saveKey}
                          className="glass-panel rounded-xl p-5 shadow-xs hover:border-stone-400/80 transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-5"
                        >
                          <div className="flex items-start gap-4">
                            <div className="w-20 h-20 rounded-lg border border-[#d8ccb6] shrink-0 overflow-hidden bg-stone-100/50 flex items-center justify-center">
                              {item.thumbnail ? (
                                <img
                                  src={item.thumbnail}
                                  alt="Thumbnail"
                                  className="w-full h-full object-cover"
                                  onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                />
                              ) : (
                                <Globe2 className="w-8 h-8 text-stone-300" />
                              )}
                            </div>

                            <div className="space-y-1 min-w-0">
                              <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-extrabold uppercase border ${matchTypeBadgeClasses(item.match_type)}`}>
                                {matchTypeLabel(item.match_type)}
                              </span>

                              <h4 className="font-bold text-stone-950 text-sm leading-snug">
                                {item.title ?? item.domain ?? 'Untitled Source'}
                              </h4>

                              {item.url && (
                                <a
                                  href={item.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-xs text-red-700 hover:underline font-mono flex items-center gap-1 truncate max-w-md"
                                >
                                  <span className="truncate">{item.url}</span>
                                  <ExternalLink className="w-3 h-3 shrink-0" />
                                </a>
                              )}

                              <div className="flex flex-wrap items-center gap-3 text-[11px] text-stone-500 pt-0.5">
                                {item.domain && (
                                  <span className="font-semibold text-stone-700 flex items-center gap-1">
                                    <Globe2 className="w-3 h-3" /> {item.domain}
                                  </span>
                                )}
                                {item.date && <><span>·</span><span>Found: {item.date}</span></>}
                                {item.region && <><span>·</span><span>Region: {item.region}</span></>}
                                {item.snippet && (
                                  <p className="w-full text-stone-400 truncate max-w-lg mt-0.5">{item.snippet}</p>
                                )}
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-2 shrink-0 w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-[#e2d8c3]/80">
                            <button
                              onClick={() => togglePropSave(saveKey)}
                              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                                propSavedItems[saveKey]
                                  ? 'bg-emerald-50/90 border-emerald-200 text-emerald-800'
                                  : 'glass-input text-stone-700 hover:bg-[#e8decb]/60'
                              }`}
                            >
                              {propSavedItems[saveKey] ? '[OK] Saved Evidence' : 'Save Evidence'}
                            </button>
                            {item.url && (
                              <a
                                href={item.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="px-3 py-1.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
                              >
                                Open Source
                              </a>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          )}

          <div className="p-4 glass-input rounded-xl text-xs text-stone-600 flex items-start gap-3">
            <Info className="w-5 h-5 text-stone-500 shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong>Notice:</strong> ForensIQ provides evidence gathering and provenance reports to assist media owners.
              ForensIQ does not directly modify or remove content hosted on third-party websites or platforms.
            </p>
          </div>
        </div>
      )}

      {/* ===================== LEGACY RESULTS ===================== */}
      {selectedMethod !== 'submission' && (
        <>
          {legacySearchState === 'scanning' && (
            <div className="glass-panel rounded-xl p-10 text-center shadow-xs space-y-4">
              <RefreshCw className="w-8 h-8 text-red-700 animate-spin mx-auto" />
              <h3 className="font-bold text-stone-950 text-lg">Scanning Public Sources &amp; Web Indices</h3>
              <p className="text-xs font-mono text-red-800 glass-input px-3 py-1 rounded inline-block font-bold">
                {legacyScanStep}
              </p>
            </div>
          )}

          {legacySearchState === 'complete' && (
            <div className="space-y-6">
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

              {incidentCreated && (
                <div className="p-4 bg-emerald-50/90 border border-emerald-200 rounded-xl text-xs text-emerald-900 flex items-center justify-between shadow-xs">
                  <div className="flex items-center gap-2.5">
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0" />
                    <div>
                      <p className="font-bold text-sm">Incident INC-2026-042 Successfully Created</p>
                      <p className="text-emerald-700">Grouped 5 discovered matches into active investigation binder.</p>
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

              <div className="space-y-4">
                {legacyMatches.map((item) => (
                  <div
                    key={item.id}
                    className="glass-panel rounded-xl p-5 shadow-xs hover:border-stone-400/80 transition-all flex flex-col md:flex-row items-start md:items-center justify-between gap-5"
                  >
                    <div className="flex items-start gap-4">
                      <img
                        src={item.thumbnailUrl}
                        alt={item.sourceName}
                        className="w-20 h-20 object-cover rounded-lg border border-[#d8ccb6] shrink-0"
                      />
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-extrabold uppercase border ${
                            item.classification === 'CONFIRMED MATCH'
                              ? 'bg-emerald-50/80 text-emerald-800 border-emerald-200'
                              : item.classification === 'AI MANIPULATED' || item.classification === 'TAMPERED'
                              ? 'bg-red-50/80 text-red-800 border-red-200'
                              : 'bg-amber-50/80 text-amber-800 border-amber-200'
                          }`}>
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
                          <span>·</span>
                          <span>Category: {item.platformCategory}</span>
                          <span>·</span>
                          <span className="font-semibold text-stone-700">Watermark: {item.watermarkStatus}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 shrink-0 w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-[#e2d8c3]/80">
                      <button
                        onClick={() => toggleSaveMatch(item.id)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-colors cursor-pointer ${
                          savedMatches[item.id]
                            ? 'bg-emerald-50/90 border-emerald-200 text-emerald-800'
                            : 'glass-input text-stone-700 hover:bg-[#e8decb]/60'
                        }`}
                      >
                        {savedMatches[item.id] ? '[OK] Saved Evidence' : 'Save Evidence'}
                      </button>
                      <button
                        onClick={() => alert(`Opening DMCA/Takedown evidence document for ${item.sourceName}.`)}
                        className="px-3 py-1.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-semibold rounded-lg transition-colors cursor-pointer"
                      >
                        Report Content
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <div className="p-4 glass-input rounded-xl text-xs text-stone-600 flex items-start gap-3">
                <Info className="w-5 h-5 text-stone-500 shrink-0 mt-0.5" />
                <p className="leading-relaxed">
                  <strong>Notice:</strong> ForensIQ provides evidence gathering and provenance reports to assist media owners.
                  ForensIQ does not directly modify or remove content hosted on third-party websites or platforms.
                </p>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};