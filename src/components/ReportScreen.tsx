import React, { useState, useRef } from 'react';
import {
  Download,
  ShieldCheck,
  Share2,
  Copy,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import { ForensicReportData, UserRole, NavItem } from '../types';
import { MOCK_REPORT } from '../data/mockData';

interface ReportScreenProps {
  report?: ForensicReportData;
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
}

export const ReportScreen: React.FC<ReportScreenProps> = ({
  report = MOCK_REPORT,
  userRole,
  setActiveTab,
}) => {
  const [copiedAudit, setCopiedAudit] = useState<boolean>(false);
  const [generatingPdf, setGeneratingPdf] = useState<boolean>(false);
  const reportDocRef = useRef<HTMLDivElement>(null);

  const handleCopyHash = () => {
    if (report.auditHash) {
      navigator.clipboard.writeText(report.auditHash);
      setCopiedAudit(true);
      setTimeout(() => setCopiedAudit(false), 2000);
    }
  };

  const handleExportPdf = async () => {
    if (!reportDocRef.current) return;
    setGeneratingPdf(true);
    try {
      const html2canvas = (await import('html2canvas')).default;
      const jsPDF = (await import('jspdf')).default;

      const canvas = await html2canvas(reportDocRef.current, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#faf8f5',
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const imgWidth = pageWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let yOffset = 0;
      let remainingHeight = imgHeight;

      while (remainingHeight > 0) {
        pdf.addImage(imgData, 'PNG', 0, -yOffset, imgWidth, imgHeight);
        remainingHeight -= pageHeight;
        yOffset += pageHeight;
        if (remainingHeight > 0) pdf.addPage();
      }

      pdf.save(`ForensIQ_Report_${report.reportId}.pdf`);
    } catch (err) {
      console.error('PDF generation failed:', err);
    } finally {
      setGeneratingPdf(false);
    }
  };

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* Top Header Controls */}
      <div className="glass-panel rounded-xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-4 print:hidden">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs font-bold text-stone-900 glass-input px-2 py-0.5 rounded">
              REPORT ID: {report.reportId}
            </span>
            <span className="px-2 py-0.5 rounded bg-emerald-50/80 text-emerald-800 text-xs font-bold border border-emerald-200 uppercase">
              {report.status}
            </span>
          </div>
          <h2 className="text-lg font-serif font-bold text-stone-950 mt-1">Forensic Evidence Report Preview</h2>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleExportPdf}
            disabled={generatingPdf}
            className="px-4 py-2 bg-[#1e1b18] hover:bg-stone-900 disabled:opacity-60 disabled:cursor-not-allowed text-white text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
          >
            {generatingPdf ? (
              <Loader2 className="w-4 h-4 text-amber-300 animate-spin" />
            ) : (
              <Download className="w-4 h-4 text-amber-300" />
            )}
            <span>{generatingPdf ? 'Generating PDF...' : 'Download PDF'}</span>
          </button>
        </div>
      </div>

      {/* Main Document & Side Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Printable Executive Forensic Report Document */}
        <div ref={reportDocRef} className="lg:col-span-2 glass-panel-strong rounded-xl p-8 shadow-xs space-y-6 text-stone-900 font-sans print:border-none print:shadow-none print:p-0">
          {/* Document Header */}
          <div className="border-b-2 border-stone-950 pb-6 flex items-start justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded bg-[#1e1b18] text-red-500 flex items-center justify-center font-bold">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <span className="font-serif font-black text-2xl tracking-tight text-stone-950">ForensIQ</span>
              </div>
              <p className="text-xs font-mono font-bold uppercase tracking-wider text-stone-950 pt-1">
                DIGITAL MEDIA FORENSICS &amp; PROVENANCE INSPECTION
              </p>
            </div>

            <div className="text-right text-xs font-mono">
              <p className="font-bold text-red-700">{report.reportId}</p>
              <p className="text-stone-950">{report.createdDate}</p>
              <p className="text-emerald-800 font-bold uppercase">{report.status}</p>
            </div>
          </div>

          {/* Metadata Block */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 p-4 glass-card rounded-lg text-xs">
            <div>
              <p className="text-xs text-stone-950 font-mono font-bold uppercase">TARGET MEDIA</p>
              <p className="font-bold text-stone-950 truncate">{report.mediaName}</p>
            </div>
            <div>
              <p className="text-xs text-stone-950 font-mono font-bold uppercase">AUTHOR / INVESTIGATOR</p>
              <p className="font-semibold text-stone-950">{report.authorRole || 'Forensic Media Specialist (Level III)'}</p>
            </div>
            <div>
              <p className="text-xs text-stone-950 font-mono font-bold uppercase">ENSEMBLE VERSION</p>
              <p className="font-mono text-stone-950">ForensIQ v2.4 (2026)</p>
            </div>
          </div>

          {/* Section I: Executive Summary */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              I. EXECUTIVE SUMMARY
            </h3>
            <p className="text-xs text-stone-950 leading-relaxed font-sans">{report.summary}</p>
          </div>

          {/* Section II: Detection Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              II. DETECTION FINDINGS &amp; MODEL CONSENSUS
            </h3>
            <ul className="space-y-1.5 text-xs text-stone-950 font-mono">
              {report.detectionFindings.map((f, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className="text-red-700 font-bold">•</span>
                  <span>{f}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Section III: Visual & Heatmap Analysis */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              III. VISUAL &amp; HEATMAP ANALYSIS
            </h3>
            <p className="text-xs text-stone-950 leading-relaxed">{report.visualFindings}</p>
          </div>

          {/* Section IV: Metadata Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              IV. METADATA &amp; FILE STRUCTURE FINDINGS
            </h3>
            <p className="text-xs text-stone-950 leading-relaxed">{report.metadataFindings}</p>
          </div>

          {/* Section V: Similarity & Ownership Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              V. SIMILARITY &amp; OWNERSHIP FINDINGS
            </h3>
            <p className="text-xs text-stone-950 leading-relaxed">{report.similarityFindings}</p>
          </div>

          {/* Section VI: Propagation Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              VI. PROPAGATION FINDINGS
            </h3>
            <p className="text-xs text-stone-950 leading-relaxed">{report.propagationFindings}</p>
          </div>

          {/* Section VII: Evidence Audit Sign-off */}
          <div className="pt-6 border-t-2 border-stone-950 space-y-2 text-xs">
            <div className="flex justify-between items-end">
              <div>
                <p className="font-serif font-bold text-stone-950">ForensIQ Forensic Verification Sign-off</p>
                <p className="text-stone-950 font-mono text-xs">Analysis ID: {report.reportId}</p>
                <p className="text-stone-950 font-mono text-xs">Engine: ForensIQ Core v2.4</p>
              </div>
              <div className="text-right">
                <p className="font-bold text-stone-950">{report.author || 'Forensic Media Specialist'}</p>
                <p className="text-stone-950 text-xs">Verified Forensic Specialist (Level III)</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Side Report Actions & Audit Info */}
        <div className="space-y-6 print:hidden">
          {/* Actions Box */}
          <div className="glass-panel rounded-xl p-5 shadow-xs space-y-4">
            <h4 className="font-mono font-bold text-stone-950 text-xs uppercase tracking-wider border-b border-[#f0e6d6]/80 pb-2">
              REPORT ACTIONS
            </h4>

            <div className="space-y-2">
              <button
                onClick={handleExportPdf}
                disabled={generatingPdf}
                className="w-full py-2.5 bg-[#1e1b18] hover:bg-stone-900 disabled:opacity-60 disabled:cursor-not-allowed text-white text-xs font-bold rounded-lg flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
              >
                {generatingPdf ? (
                  <Loader2 className="w-4 h-4 text-amber-300 animate-spin" />
                ) : (
                  <Download className="w-4 h-4 text-amber-300" />
                )}
                <span>{generatingPdf ? 'GENERATING PDF...' : 'DOWNLOAD PDF'}</span>
              </button>
            </div>
          </div>

          {/* Audit Trail Info Card */}
          <div className="glass-panel rounded-xl p-5 shadow-xs space-y-3">
            <h4 className="font-mono font-bold text-stone-950 text-xs uppercase tracking-wider border-b border-[#f0e6d6]/80 pb-2">
              AUDIT INFORMATION
            </h4>

            <div className="space-y-2 text-xs text-stone-950">
              <div className="flex justify-between">
                <span>Created:</span>
                <span className="font-mono text-stone-950">{report.createdDate}</span>
              </div>
              <div className="flex justify-between">
                <span>Last Updated:</span>
                <span className="font-mono text-stone-950">{report.lastUpdated}</span>
              </div>
              <div className="flex justify-between">
                <span>Analysis Engine:</span>
                <span className="font-bold text-stone-950">ForensIQ Core v2.4</span>
              </div>
            </div>

            <div className="pt-2 border-t border-[#f0e6d6]/80 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-stone-950 font-mono font-bold uppercase">Audit Hash</span>
                <button
                  onClick={handleCopyHash}
                  className="text-xs text-red-700 font-bold hover:underline cursor-pointer"
                >
                  Copy
                </button>
              </div>
              <p className="font-mono text-xs text-stone-950 glass-input p-2 rounded break-all">
                {report.auditHash}
              </p>
              {copiedAudit && (
                <p className="text-xs text-emerald-800 font-bold text-right">Hash copied!</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
