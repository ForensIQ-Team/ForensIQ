import React, { useState } from 'react';
import {
  History,
  Search,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ExternalLink,
} from 'lucide-react';
import { UserRole, NavItem, DetectionResult } from '../types';
import { historyService } from '../services/historyService';

interface HistoryScreenProps {
  userRole: UserRole;
  setActiveTab: (tab: NavItem) => void;
  onOpenReport?: (analysis?: DetectionResult) => void;
}

const glassPanelStyle: React.CSSProperties = {
  background: 'rgba(255, 252, 244, 0.78)',
  backdropFilter: 'blur(20px) saturate(130%)',
  WebkitBackdropFilter: 'blur(20px) saturate(130%)',
  border: '1px solid rgba(255, 255, 255, 0.7)',
  boxShadow: '0 8px 32px rgba(40, 30, 15, 0.08)',
};

const glassCardStyle: React.CSSProperties = {
  background: 'rgba(245, 238, 222, 0.45)',
  border: '1px solid rgba(226, 216, 195, 0.7)',
};

const glassInputStyle: React.CSSProperties = {
  background: 'rgba(255, 255, 255, 0.65)',
  border: '1px solid rgba(226, 216, 195, 0.8)',
};

export const HistoryScreen: React.FC<HistoryScreenProps> = ({
  userRole,
  setActiveTab,
  onOpenReport,
}) => {
  const [filter, setFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const historyItems = historyService.getHistoryItems();

  const filteredHistory = historyItems.filter((item) => {
    if (filter === 'checked' && item.action !== 'Check Media') return false;
    if (filter === 'protected' && item.action !== 'Protect Image') return false;
    if (filter === 'misuse' && item.action !== 'Find Misuse') return false;

    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      return (
        item.mediaName.toLowerCase().includes(q) ||
        item.id.toLowerCase().includes(q) ||
        item.action.toLowerCase().includes(q) ||
        item.resultSummary.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="space-y-6 max-w-[1700px] mx-auto pb-12">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-serif font-bold text-stone-950 tracking-tight">Activity &amp; Media History</h2>
        <p className="text-stone-600 text-xs sm:text-sm mt-1">
          Unified audit trail of checked media, protected assets, misuse searches, and dynamic forensic reports.
        </p>
      </div>

      {/* Filter Tabs & Search Bar */}
      <div className="rounded-xl p-4 shadow-xs flex flex-col md:flex-row items-center justify-between gap-4" style={glassPanelStyle}>
        {/* Filter Pills */}
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <button
            onClick={() => setFilter('all')}
            style={filter !== 'all' ? glassInputStyle : undefined}
            className={`px-3 py-1.5 rounded-lg font-bold transition-colors cursor-pointer ${
              filter === 'all'
                ? 'bg-[#1e1b18] text-white shadow-xs'
                : 'text-stone-800 hover:bg-[#eae0d0]/60'
            }`}
          >
            All Activity
          </button>
          <button
            onClick={() => setFilter('checked')}
            style={filter !== 'checked' ? glassInputStyle : undefined}
            className={`px-3 py-1.5 rounded-lg font-bold transition-colors cursor-pointer ${
              filter === 'checked'
                ? 'bg-[#1e1b18] text-white shadow-xs'
                : 'text-stone-800 hover:bg-[#eae0d0]/60'
            }`}
          >
            Checked Media
          </button>
          <button
            onClick={() => setFilter('protected')}
            style={filter !== 'protected' ? glassInputStyle : undefined}
            className={`px-3 py-1.5 rounded-lg font-bold transition-colors cursor-pointer ${
              filter === 'protected'
                ? 'bg-[#1e1b18] text-white shadow-xs'
                : 'text-stone-800 hover:bg-[#eae0d0]/60'
            }`}
          >
            Protected Images
          </button>
          <button
            onClick={() => setFilter('misuse')}
            style={filter !== 'misuse' ? glassInputStyle : undefined}
            className={`px-3 py-1.5 rounded-lg font-bold transition-colors cursor-pointer ${
              filter === 'misuse'
                ? 'bg-[#1e1b18] text-white shadow-xs'
                : 'text-stone-800 hover:bg-[#eae0d0]/60'
            }`}
          >
            Misuse Searches
          </button>
        </div>

        {/* Search */}
        <div className="relative w-full md:w-64">
          <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-stone-400" />
          <input
            type="text"
            placeholder="Search analysis ID, filename..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={glassInputStyle}
            className="w-full pl-9 pr-3 py-1.5 rounded-lg text-xs focus:outline-none focus:ring-1 focus:ring-stone-900 text-stone-800 placeholder-stone-400"
          />
        </div>
      </div>

      {/* History Table Container */}
      <div className="rounded-xl shadow-xs overflow-hidden" style={glassPanelStyle}>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(200, 185, 155, 0.28)' }} className="text-stone-500 font-mono font-bold uppercase tracking-wider">
                <th className="py-3 px-4">Analysis ID &amp; Media</th>
                <th className="py-3 px-4">Action</th>
                <th className="py-3 px-4">Detection Result</th>
                <th className="py-3 px-4">Timestamp</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 text-right">Report</th>
              </tr>
            </thead>
            <tbody className="divide-y text-stone-800" style={{ borderColor: 'rgba(200, 185, 155, 0.20)' }}>
              {filteredHistory.map((item) => (
                <tr
                  key={item.id}
                  className="transition-colors"
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(240, 228, 200, 0.22)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-3">
                      <img
                        src={item.mediaUrl}
                        alt={item.mediaName}
                        className="w-10 h-10 object-cover rounded-md border border-[#dcd0b9]"
                      />
                      <div>
                        <p className="font-bold text-stone-950">{item.mediaName}</p>
                        <p className="text-[10px] font-mono text-red-700 font-bold">{item.id}</p>
                      </div>
                    </div>
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-1 rounded text-stone-900 font-semibold text-[11px]" style={glassInputStyle}>
                      {item.action}
                    </span>
                  </td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      {item.risk === 'high' || item.status === 'Flagged' ? (
                        <AlertTriangle className="w-4 h-4 text-red-700 shrink-0" />
                      ) : (
                        <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                      )}
                      <span className="font-bold text-stone-950">{item.resultSummary}</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 text-stone-600 font-mono text-[11px]">{item.date}</td>
                  <td className="py-3.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                        item.status === 'Completed'
                          ? 'bg-emerald-50/80 text-emerald-800 border-emerald-200'
                          : 'bg-red-50/80 text-red-800 border-red-200'
                      }`}
                    >
                      {item.status}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <button
                      onClick={() => {
                        if (onOpenReport && item.detectionResult) {
                          onOpenReport(item.detectionResult);
                        } else {
                          setActiveTab('report');
                        }
                      }}
                      className="px-3 py-1.5 bg-[#1e1b18] hover:bg-stone-900 text-white rounded-lg text-xs font-bold inline-flex items-center gap-1.5 cursor-pointer shadow-xs transition-colors"
                      title="View Forensic Report"
                    >
                      <FileText className="w-3.5 h-3.5 text-amber-300" />
                      <span>VIEW REPORT</span>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};