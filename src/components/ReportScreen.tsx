import React, { useState } from 'react';
import {
  Download,
  ShieldCheck,
  Share2,
  Copy,
  CheckCircle2,
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

  const handleCopyHash = () => {
    if (report.auditHash) {
      navigator.clipboard.writeText(report.auditHash);
      setCopiedAudit(true);
      setTimeout(() => setCopiedAudit(false), 2000);
    }
  };

  const handleExportPdf = () => {
    window.print();
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
            <span className="px-2 py-0.5 rounded bg-emerald-50/80 text-emerald-800 text-[10px] font-bold border border-emerald-200 uppercase">
              {report.status}
            </span>
          </div>
          <h2 className="text-lg font-serif font-bold text-stone-950 mt-1">Forensic Evidence Report Preview</h2>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleExportPdf}
            className="px-4 py-2 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-bold rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-xs"
          >
            <Download className="w-4 h-4 text-amber-300" />
            <span>Export / Print PDF</span>
          </button>
        </div>
      </div>

      {/* Main Document & Side Panel Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Printable Executive Forensic Report Document */}
        <div className="lg:col-span-2 glass-panel-strong rounded-xl p-8 shadow-xs space-y-6 text-stone-900 font-sans print:border-none print:shadow-none print:p-0">
          {/* Document Header */}
          <div className="border-b-2 border-stone-950 pb-6 flex items-start justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded bg-[#1e1b18] text-red-500 flex items-center justify-center font-bold">
                  <ShieldCheck className="w-5 h-5" />
                </div>
                <span className="font-serif font-black text-2xl tracking-tight text-stone-950">ForensIQ</span>
              </div>
              <p className="text-xs font-mono font-bold uppercase tracking-wider text-stone-600 pt-1">
                DIGITAL MEDIA FORENSICS &amp; PROVENANCE INSPECTION
              </p>
            </div>

            <div className="text-right text-xs font-mono">
              <p className="font-bold text-red-700">{report.reportId}</p>
              <p className="text-stone-600">{report.createdDate}</p>
              <p className="text-emerald-800 font-bold uppercase">{report.status}</p>
            </div>
          </div>

          {/* Metadata Block */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 p-4 glass-card rounded-lg text-xs">
            <div>
              <p className="text-[10px] text-stone-500 font-mono font-bold uppercase">TARGET MEDIA</p>
              <p className="font-bold text-stone-950 truncate">{report.mediaName}</p>
            </div>
            <div>
              <p className="text-[10px] text-stone-500 font-mono font-bold uppercase">AUTHOR / INVESTIGATOR</p>
              <p className="font-semibold text-stone-800">{report.authorRole || 'Forensic Media Specialist (Level III)'}</p>
            </div>
            <div>
              <p className="text-[10px] text-stone-500 font-mono font-bold uppercase">ENSEMBLE VERSION</p>
              <p className="font-mono text-stone-800">ForensIQ v2.4 (2026)</p>
            </div>
          </div>

          {/* Section I: Executive Summary */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              I. EXECUTIVE SUMMARY
            </h3>
            <p className="text-xs text-stone-800 leading-relaxed font-sans">{report.summary}</p>
          </div>

          {/* Section II: Detection Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              II. DETECTION FINDINGS &amp; MODEL CONSENSUS
            </h3>
            <ul className="space-y-1.5 text-xs text-stone-800 font-mono">
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
            <p className="text-xs text-stone-800 leading-relaxed">{report.visualFindings}</p>
          </div>

          {/* Section IV: Metadata Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              IV. METADATA &amp; FILE STRUCTURE FINDINGS
            </h3>
            <p className="text-xs text-stone-800 leading-relaxed">{report.metadataFindings}</p>
          </div>

          {/* Section V: Similarity & Ownership Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              V. SIMILARITY &amp; OWNERSHIP FINDINGS
            </h3>
            <p className="text-xs text-stone-800 leading-relaxed">{report.similarityFindings}</p>
          </div>

          {/* Section VI: Propagation Findings */}
          <div className="space-y-2">
            <h3 className="text-xs font-mono font-extrabold text-stone-950 uppercase tracking-wider border-b border-[#e2d8c3]/80 pb-1">
              VI. PROPAGATION FINDINGS
            </h3>
            <p className="text-xs text-stone-800 leading-relaxed">{report.propagationFindings}</p>
          </div>

          {/* Section VII: Evidence Audit Sign-off */}
          <div className="pt-6 border-t-2 border-stone-950 space-y-2 text-xs">
            <div className="flex justify-between items-end">
              <div>
                <p className="font-serif font-bold text-stone-950">ForensIQ Forensic Verification Sign-off</p>
                <p className="text-stone-500 font-mono text-[10px]">Analysis ID: {report.reportId}</p>
                <p className="text-stone-500 font-mono text-[10px]">Engine: ForensIQ Core v2.4</p>
              </div>
              <div className="text-right">
                <p className="font-bold text-stone-950">{report.author || 'Forensic Media Specialist'}</p>
                <p className="text-stone-600 text-[11px]">Verified Forensic Specialist (Level III)</p>
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
                className="w-full py-2.5 bg-[#1e1b18] hover:bg-stone-900 text-white text-xs font-bold rounded-lg flex items-center justify-center gap-2 transition-colors cursor-pointer shadow-xs"
              >
                <Download className="w-4 h-4 text-amber-300" />
                <span>DOWNLOAD PDF FORMAT</span>
              </button>
            </div>
          </div>

          {/* Audit Trail Info Card */}
          <div className="glass-panel rounded-xl p-5 shadow-xs space-y-3">
            <h4 className="font-mono font-bold text-stone-950 text-xs uppercase tracking-wider border-b border-[#f0e6d6]/80 pb-2">
              AUDIT INFORMATION
            </h4>

            <div className="space-y-2 text-xs text-stone-700">
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
                <span className="text-[10px] text-stone-500 font-mono font-bold uppercase">Audit Hash</span>
                <button
                  onClick={handleCopyHash}
                  className="text-[10px] text-red-700 font-bold hover:underline cursor-pointer"
                >
                  Copy
                </button>
              </div>
              <p className="font-mono text-[10px] text-stone-800 glass-input p-2 rounded break-all">
                {report.auditHash}
              </p>
              {copiedAudit && (
                <p className="text-[10px] text-emerald-800 font-bold text-right">Hash copied!</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
