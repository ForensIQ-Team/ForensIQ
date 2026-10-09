import React, { useState, useRef } from 'react';
import {
  Download,
  ShieldCheck,
  ExternalLink,
  Printer,
  Upload,
  RefreshCw,
  FileText,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Cpu,
} from 'lucide-react';
import { ForensicReportData, UserRole, NavItem } from '../types';
import { reportService, GenerateReportResponse } from '../services/reportService';

interface ReportScreenProps {
  report?: ForensicReportData;
  reportId?: string;
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
}

export const ReportScreen: React.FC<ReportScreenProps> = ({
  report,
  reportId: initialReportId,
  userRole,
  setActiveTab,
}) => {
  // Use passed reportId, or detect from report.reportId, or fallback to latest generated ID
  const defaultReportId = initialReportId || (report?.reportId?.startsWith('FX-') ? report.reportId : 'FX-20261007-9FC1');
  const [activeReportId, setActiveReportId] = useState<string>(defaultReportId);
  const [reportResult, setReportResult] = useState<GenerateReportResponse | null>(null);

  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [generationStep, setGenerationStep] = useState<string>('');
  const [generationError, setGenerationError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    await generateNewReport(file);
  };

  const generateNewReport = async (file: File) => {
    setIsGenerating(true);
    setGenerationError(null);
    setGenerationStep('Uploading media to forensic server...');

    try {
      const stepTimer1 = setTimeout(() => {
        setGenerationStep('Running multi-model ensemble (Xception, EfficientNet, ViT)...');
      }, 1500);

      const stepTimer2 = setTimeout(() => {
        setGenerationStep('Extracting face crops and computing Grad-CAM heatmaps...');
      }, 5000);

      const stepTimer3 = setTimeout(() => {
        setGenerationStep('Synthesizing forensic findings with Groq LLM (Llama / GPT)...');
      }, 10000);

      const res = await reportService.generateReport(file);

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);
      clearTimeout(stepTimer3);

      setReportResult(res);
      setActiveReportId(res.report_id);
      setIsGenerating(false);
      setGenerationStep('');
    } catch (err: unknown) {
      setIsGenerating(false);
      setGenerationStep('');
      setGenerationError(err instanceof Error ? err.message : 'Report generation failed');
    }
  };

  const reportHtmlUrl = activeReportId ? reportService.getReportHtmlUrl(activeReportId) : '';
  const reportDownloadUrl = activeReportId ? reportService.getReportDownloadUrl(activeReportId) : '';

  const handlePrint = () => {
    if (iframeRef.current && iframeRef.current.contentWindow) {
      iframeRef.current.contentWindow.focus();
      iframeRef.current.contentWindow.print();
    } else {
      window.print();
    }
  };

  return (
    <div className="space-y-6 max-w-[1700px] mx-auto pb-12">
      {/* Header Bar */}
      <div className="glass-panel rounded-xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-4 print:hidden">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#1e1b18] text-amber-400 flex items-center justify-center font-bold shadow-xs">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-bold text-stone-900 glass-input px-2.5 py-0.5 rounded">
                REPORT ID: {activeReportId || 'NOT GENERATED'}
              </span>
              <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 text-[10px] font-bold border border-emerald-200 uppercase flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-amber-500" />
                GROQ NARRATIVE + ML GRAD-CAM
              </span>
            </div>
            <h2 className="text-lg font-serif font-bold text-stone-950 mt-1">
              ForensIQ Forensic Inspection Report
            </h2>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept="image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/quicktime,video/avi,video/webm"
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isGenerating}
            className="px-3.5 py-2 bg-stone-100 hover:bg-stone-200 text-stone-800 text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer border border-stone-300"
          >
            {isGenerating ? (
              <RefreshCw className="w-4 h-4 animate-spin text-stone-600" />
            ) : (
              <Upload className="w-4 h-4 text-stone-700" />
            )}
            <span>{isGenerating ? 'Generating...' : 'Generate New Report'}</span>
          </button>

          {activeReportId && (
            <>
              <a
                href={reportHtmlUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-2 glass-input hover:bg-white text-stone-800 text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors"
                title="Open Report in Full Tab"
              >
                <ExternalLink className="w-4 h-4 text-stone-600" />
                <span>Open in Tab</span>
              </a>

              <button
                onClick={handlePrint}
                className="px-3 py-2 glass-input hover:bg-white text-stone-800 text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer"
                title="Print or Save as PDF"
              >
                <Printer className="w-4 h-4 text-stone-600" />
                <span>Print PDF</span>
              </button>

              <a
                href={reportDownloadUrl}
                download
                className="px-4 py-2 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors shadow-xs cursor-pointer"
              >
                <Download className="w-4 h-4 text-amber-300" />
                <span>Download Report (.html)</span>
              </a>
            </>
          )}
        </div>
      </div>

      {/* Generation Status Overlay / Banner */}
      {isGenerating && (
        <div className="glass-panel rounded-xl p-6 border-amber-300/60 bg-amber-50/70 text-amber-950 flex items-center gap-4 animate-pulse">
          <div className="w-10 h-10 rounded-full bg-amber-500/20 flex items-center justify-center shrink-0">
            <Cpu className="w-6 h-6 text-amber-700 animate-spin" />
          </div>
          <div className="space-y-1">
            <h4 className="font-bold text-sm">Generating Forensic Report...</h4>
            <p className="text-xs text-amber-800 font-mono">{generationStep}</p>
          </div>
        </div>
      )}

      {/* Generation Error Alert */}
      {generationError && (
        <div className="glass-panel rounded-xl p-4 border-red-300 bg-red-50/80 text-red-950 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 shrink-0" />
          <div>
            <h4 className="font-bold text-xs">Report Generation Failed</h4>
            <p className="text-xs text-red-700">{generationError}</p>
          </div>
        </div>
      )}

      {/* Main Report View Container */}
      {activeReportId ? (
        <div className="w-full rounded-xl overflow-hidden shadow-lg border border-stone-200/80 bg-white">
          <iframe
            ref={iframeRef}
            src={reportHtmlUrl}
            title="ForensIQ Forensic Report"
            className="w-full h-[900px] border-none block"
            style={{ minHeight: '850px' }}
          />
        </div>
      ) : (
        <div className="glass-panel-strong rounded-xl p-12 text-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-stone-100 flex items-center justify-center mx-auto text-stone-400">
            <FileText className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-stone-900">No Forensic Report Generated Yet</h3>
          <p className="text-xs text-stone-600 max-w-md mx-auto">
            Upload an image or video above to generate a full digital forensics report powered by our
            multi-model ensemble, Grad-CAM attention heatmaps, and Groq narrative analysis.
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-6 py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-bold rounded-lg inline-flex items-center gap-2 transition-colors cursor-pointer shadow-xs"
          >
            <Upload className="w-4 h-4 text-amber-300" />
            <span>Select Media to Generate Report</span>
          </button>
        </div>
      )}
    </div>
  );
};
