import React, { useState } from 'react';
import { DetectionResult } from '../../types';
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  FileText,
  ArrowRight,
  RefreshCw,
  ExternalLink,
  Download,
} from 'lucide-react';
import {
  generateForensicReport,
  buildReportHtmlUrl,
  buildReportDownloadUrl,
  GenerateReportResponse,
} from '../../services/reportService';

interface NormalUserResultProps {
  result: DetectionResult;
  selectedFile?: File | null;
  onExploreDetails: () => void;
  onOpenReport?: () => void;
}

export const NormalUserResult: React.FC<NormalUserResultProps> = ({
  result,
  selectedFile,
  onExploreDetails,
}) => {
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [reportResult, setReportResult] = useState<GenerateReportResponse | null>(null);

  const handleGenerateReport = async () => {
    if (!selectedFile) {
      setReportError('Please upload an image or video first.');
      return;
    }

    setReportLoading(true);
    setReportError(null);

    try {
      const res = await generateForensicReport(selectedFile);
      setReportResult(res);
      // Immediately open report in a new tab
      window.open(buildReportHtmlUrl(res.report_id), '_blank');
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Report generation failed.';
      setReportError(msg);
    } finally {
      setReportLoading(false);
    }
  };

  const isReal = result.classification === 'Real';
  const isUncertain = result.classification === 'Uncertain';
  const isFakeOrAI = result.classification === 'Deepfake' || result.classification === 'AI-Generated';

  const confValue = result.confidenceScore ?? result.confidence ?? 90;




  return (
    <div className="glass-panel-strong rounded-xl p-6 shadow-md space-y-6">
      {/* Header Banner: FORENSIQ ANALYSIS COMPLETE */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4" style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.28)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-[#1e1b18] text-red-500 flex items-center justify-center font-bold">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-red-700 block">
              FORENSIQ ANALYSIS COMPLETE
            </span>
            <h3 className="text-lg font-serif font-black text-stone-950 truncate max-w-md">
              {result.mediaName}
            </h3>
          </div>
        </div>

        {/* Thumbnail Preview */}
        {result.mediaUrl && (
          <img
            src={result.mediaUrl}
            alt={result.mediaName}
            className="w-14 h-14 object-cover rounded-lg border border-[#dcd0b9] shadow-2xs"
          />
        )}
      </div>

      {/* Final Result Card — translucent glass tinted by classification */}
      <div
        className="p-5 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4"
        style={{
          background: isReal
            ? 'rgba(220, 250, 225, 0.30)'
            : isUncertain
            ? 'rgba(255, 240, 200, 0.30)'
            : 'rgba(255, 215, 215, 0.30)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: isReal
            ? '1px solid rgba(52, 168, 83, 0.35)'
            : isUncertain
            ? '1px solid rgba(200, 145, 30, 0.35)'
            : '1px solid rgba(196, 30, 30, 0.35)',
        }}
      >
        <div className="flex items-start gap-4">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 font-bold text-white shadow-xs ${
              isReal ? 'bg-emerald-700' : isUncertain ? 'bg-amber-700' : 'bg-red-700'
            }`}
          >
            {isReal ? (
              <ShieldCheck className="w-6 h-6" />
            ) : isUncertain ? (
              <AlertTriangle className="w-6 h-6" />
            ) : (
              <ShieldAlert className="w-6 h-6" />
            )}
          </div>

          <div className="space-y-1">
            <span className={`text-[10px] font-mono font-bold uppercase tracking-wider opacity-75 ${
              isReal ? 'text-emerald-900' : isUncertain ? 'text-amber-900' : 'text-red-900'
            }`}>
              FINAL RESULT
            </span>
            <h3 className={`text-2xl font-serif font-black uppercase tracking-tight ${
              isReal ? 'text-emerald-950' : isUncertain ? 'text-amber-950' : 'text-red-950'
            }`}>
              {result.classification}
            </h3>
            <p className="text-xs font-semibold opacity-90 leading-relaxed max-w-xl text-stone-800">
              {result.summaryText}
            </p>
          </div>
        </div>

        {/* Confidence Badge — glass, not white */}
        <div
          className="p-4 rounded-xl text-center shrink-0 min-w-[140px]"
          style={{
            background: 'rgba(255, 252, 244, 0.48)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            border: '1px solid rgba(200, 185, 155, 0.38)',
          }}
        >
          <span className="text-[10px] font-mono font-bold uppercase text-stone-500 block">Confidence</span>
          <span className="text-3xl font-serif font-black text-stone-950">{confValue.toFixed(1)}%</span>
          <span className="text-[10px] font-semibold text-stone-600 block mt-0.5">
            Authenticity: {result.authenticityScore}%
          </span>
        </div>
      </div>




      {/* Report Action Buttons */}
      <div className="pt-3 border-t border-[#f0e6d6]/80 flex flex-wrap items-center justify-between gap-3">
        <button
          onClick={onExploreDetails}
          className="px-4 py-2 glass-input hover:bg-[#e8decb]/80 text-stone-900 rounded-lg text-xs font-bold flex items-center gap-2 cursor-pointer transition-colors"
        >
          <span>Examine Detailed Evidence</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={handleGenerateReport}
            disabled={reportLoading || !selectedFile}
            className="px-4 py-2.5 bg-[#1e1b18] hover:bg-stone-900 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg text-xs font-bold flex items-center gap-2 cursor-pointer shadow-xs transition-colors"
          >
            {reportLoading ? (
              <>
                <RefreshCw className="w-4 h-4 text-amber-300 animate-spin" />
                <span>Generating report…</span>
              </>
            ) : (
              <>
                <FileText className="w-4 h-4 text-amber-300" />
                <span>Generate Forensic Report</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Success Banner */}
      {reportResult && (
        <div className="p-4 rounded-xl bg-stone-900/90 border border-stone-800 text-stone-100 flex flex-wrap items-center justify-between gap-3 text-xs shadow-md">
          <div className="flex items-center gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span className="font-semibold text-stone-300">Report Ready:</span>
            <span
              className={`px-2 py-0.5 rounded font-mono font-bold uppercase text-[11px] ${
                reportResult.verdict === 'REAL'
                  ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                  : reportResult.verdict === 'FAKE'
                  ? 'bg-red-950 text-red-300 border border-red-800'
                  : 'bg-amber-950 text-amber-300 border border-amber-800'
              }`}
            >
              {reportResult.verdict}
            </span>
            <span className="font-mono text-stone-400">
              ({(reportResult.confidence * 100).toFixed(1)}%)
            </span>
          </div>

          <div className="flex items-center gap-3">
            <a
              href={buildReportHtmlUrl(reportResult.report_id)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-amber-300 hover:text-amber-200 font-bold inline-flex items-center gap-1 hover:underline"
            >
              <span>Open report</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <span className="text-stone-600">•</span>
            <a
              href={buildReportDownloadUrl(reportResult.report_id)}
              className="text-stone-300 hover:text-white font-bold inline-flex items-center gap-1 hover:underline"
            >
              <span>Download report</span>
              <Download className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      )}

      {/* Error Message */}
      {reportError && (
        <div className="p-3 rounded-lg bg-red-950/60 border border-red-800/80 text-red-200 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
          <span>{reportError}</span>
        </div>
      )}
    </div>
  );
};
