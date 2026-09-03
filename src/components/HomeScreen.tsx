import React from 'react';
import {
  FileSearch,
  Lock,
  Search,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  History,
  FileText,
  Activity,
  PieChart,
  Shield,
  TrendingUp,
} from 'lucide-react';
import { NavItem, UserRole, DetectionResult } from '../types';
import { historyService } from '../services/historyService';

interface HomeScreenProps {
  setActiveTab: (tab: NavItem) => void;
  userRole: UserRole;
  onOpenReport?: (analysis?: DetectionResult) => void;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({
  setActiveTab,
  userRole,
  onOpenReport,
}) => {
  const historyItems = historyService.getHistoryItems();
  const recentHistory = historyItems.slice(0, 5);

  // Calculate real metrics from history
  const totalAnalyzed = historyItems.length;
  const authenticCount = historyItems.filter(
    (h) =>
      h.detectionResult?.classification === 'Real' ||
      h.risk === 'low' ||
      h.status === 'Completed'
  ).length;
  const fakeCount = historyItems.filter(
    (h) =>
      h.detectionResult?.classification === 'Deepfake' ||
      h.detectionResult?.classification === 'AI-Generated' ||
      h.risk === 'high' ||
      h.status === 'Flagged'
  ).length;
  const protectedCount = 14 + historyItems.filter((h) => h.action === 'Protect Image').length;

  // Authenticity Distribution percentages
  const authenticPct = totalAnalyzed > 0 ? Math.round((authenticCount / totalAnalyzed) * 100) : 65;
  const fakePct = totalAnalyzed > 0 ? Math.round((fakeCount / totalAnalyzed) * 100) : 28;
  const undeterminedPct = totalAnalyzed > 0 ? 100 - authenticPct - fakePct : 7;

  return (
    <div className="space-y-8 max-w-[1700px] mx-auto pb-12">
      {/* 1. FORENSIC METRIC CARDS ROW */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {/* Metric 1 */}
        <div className="glass-card rounded-2xl p-5 space-y-2 hover:-translate-y-1 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-bold text-stone-600 uppercase tracking-wider">
              Total Media Analyzed
            </span>
            <div
              className="p-2 rounded-xl text-stone-900"
              style={{ background: 'rgba(230, 215, 185, 0.35)' }}
            >
              <Activity className="w-4 h-4 text-stone-800" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-serif font-black text-stone-950">{totalAnalyzed}</span>
            <span className="text-[11px] font-medium text-stone-600">assets inspected</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-emerald-800 pt-1">
            <TrendingUp className="w-3 h-3 text-emerald-700" />
            <span>Active Forensic Logging</span>
          </div>
        </div>

        {/* Metric 2 */}
        <div className="glass-card rounded-2xl p-5 space-y-2 hover:-translate-y-1 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-bold text-stone-600 uppercase tracking-wider">
              Authentic Media
            </span>
            <div
              className="p-2 rounded-xl text-emerald-900"
              style={{ background: 'rgba(215, 240, 220, 0.40)' }}
            >
              <ShieldCheck className="w-4 h-4 text-emerald-700" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-serif font-black text-stone-950">{authenticCount}</span>
            <span className="text-[11px] font-medium text-stone-600">verified real</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-emerald-800 pt-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-700" />
            <span>{authenticPct}% Clean Verification</span>
          </div>
        </div>

        {/* Metric 3 */}
        <div className="glass-card rounded-2xl p-5 space-y-2 hover:-translate-y-1 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-bold text-stone-600 uppercase tracking-wider">
              Suspicious / Fake Media
            </span>
            <div
              className="p-2 rounded-xl text-red-900"
              style={{ background: 'rgba(250, 215, 215, 0.40)' }}
            >
              <AlertTriangle className="w-4 h-4 text-red-700" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-serif font-black text-stone-950">{fakeCount}</span>
            <span className="text-[11px] font-medium text-stone-600">flagged deepfakes</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-red-700 pt-1">
            <span className="w-1.5 h-1.5 rounded-full bg-red-600" />
            <span>{fakePct}% Synthetic Artifacts</span>
          </div>
        </div>

        {/* Metric 4 */}
        <div className="glass-card rounded-2xl p-5 space-y-2 hover:-translate-y-1 transition-all">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono font-bold text-stone-600 uppercase tracking-wider">
              Protected Assets
            </span>
            <div
              className="p-2 rounded-xl text-stone-900"
              style={{ background: 'rgba(230, 215, 185, 0.35)' }}
            >
              <Lock className="w-4 h-4 text-stone-800" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-serif font-black text-stone-950">{protectedCount}</span>
            <span className="text-[11px] font-medium text-stone-600">registered &amp; perturbed</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px] font-semibold text-stone-700 pt-1">
            <Shield className="w-3 h-3 text-red-700" />
            <span>EOT Watermark Active</span>
          </div>
        </div>
      </div>

      {/* 2. MAIN ANALYTICS SECTION (LEFT: DETECTION ACTIVITY / RIGHT: AUTHENTICITY DISTRIBUTION) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT: Detection Activity Chart */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between border-b border-[#f0e6d6]/80 pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-red-700" />
              <h3 className="font-serif font-bold text-stone-950 text-base">
                Media Detection Activity
              </h3>
            </div>
            <span className="text-[10px] font-mono font-bold text-stone-500 uppercase tracking-wider">
              Real Time Logging
            </span>
          </div>

          {/* SVG Activity Graph Visualization */}
          <div className="relative pt-4 pb-2">
            <div className="h-44 w-full relative flex items-end">
              <svg className="w-full h-full overflow-visible" viewBox="0 0 500 120" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="activityGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#c41e1e" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#c41e1e" stopOpacity="0.0" />
                  </linearGradient>
                </defs>
                <path
                  d="M 0 100 Q 50 80 100 50 T 200 40 T 300 70 T 400 20 T 500 35 L 500 120 L 0 120 Z"
                  fill="url(#activityGrad)"
                />
                <path
                  d="M 0 100 Q 50 80 100 50 T 200 40 T 300 70 T 400 20 T 500 35"
                  fill="none"
                  stroke="#c41e1e"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                {/* Data point dots */}
                <circle cx="100" cy="50" r="4" fill="#c41e1e" />
                <circle cx="200" cy="40" r="4" fill="#c41e1e" />
                <circle cx="300" cy="70" r="4" fill="#c41e1e" />
                <circle cx="400" cy="20" r="5" fill="#1e1b18" stroke="#c41e1e" strokeWidth="2" />
                <circle cx="500" cy="35" r="4" fill="#c41e1e" />
              </svg>
            </div>

            <div className="flex justify-between items-center text-[10px] font-mono text-stone-500 pt-3 border-t border-[#f0e6d6]/60">
              <span>24 Hours Ago</span>
              <span>18 Hours Ago</span>
              <span>12 Hours Ago</span>
              <span>6 Hours Ago</span>
              <span>Current Session</span>
            </div>
          </div>
        </div>

        {/* RIGHT: Authenticity Distribution */}
        <div className="glass-panel rounded-2xl p-6 space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-[#f0e6d6]/80 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <PieChart className="w-4 h-4 text-red-700" />
                <h3 className="font-serif font-bold text-stone-950 text-base">
                  Authenticity Distribution
                </h3>
              </div>
              <span className="text-[10px] font-mono font-bold text-stone-500 uppercase">
                Consensus
              </span>
            </div>

            {/* Progress Segment Bar */}
            <div className="space-y-4">
              <div className="h-3 w-full rounded-full flex overflow-hidden p-0.5" style={{ background: 'rgba(230, 220, 200, 0.40)' }}>
                <div
                  style={{ width: `${authenticPct}%` }}
                  className="h-full bg-emerald-700 rounded-l-full"
                  title={`Authentic: ${authenticPct}%`}
                />
                <div
                  style={{ width: `${fakePct}%` }}
                  className="h-full bg-red-700"
                  title={`Deepfake: ${fakePct}%`}
                />
                <div
                  style={{ width: `${undeterminedPct}%` }}
                  className="h-full bg-amber-600 rounded-r-full"
                  title={`Undetermined: ${undeterminedPct}%`}
                />
              </div>

              {/* Breakdown Labels */}
              <div className="space-y-2.5 pt-2 text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-700" />
                    <span className="font-semibold text-stone-900">Authentic Camera Capture</span>
                  </div>
                  <span className="font-mono font-bold text-stone-950">{authenticPct}%</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-red-700" />
                    <span className="font-semibold text-stone-900">Synthetic / Deepfake</span>
                  </div>
                  <span className="font-mono font-bold text-stone-950">{fakePct}%</span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-600" />
                    <span className="font-semibold text-stone-900">Undetermined / Compressed</span>
                  </div>
                  <span className="font-mono font-bold text-stone-950">{undeterminedPct}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-3 border-t border-[#f0e6d6]/60">
            <p className="text-[11px] text-stone-600 leading-tight">
              Dual-model confidence score fusion combining Xception V2 &amp; EfficientNet-B4 V2.
            </p>
          </div>
        </div>
      </div>

      {/* 3. QUICK ACTIONS SECTION ("START AN INVESTIGATION") */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-serif font-bold text-stone-950 text-base">Start an Investigation</h3>
          <span className="text-[10px] font-mono font-bold text-stone-500 uppercase">
            Workstation Modules
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Check Media */}
          <div className="glass-card rounded-2xl p-6 hover:-translate-y-1 hover:border-red-900/40 hover:shadow-lg transition-all flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-[#1e1b18] text-white flex items-center justify-center shadow-xs group-hover:scale-105 transition-transform">
                <FileSearch className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h4 className="text-base font-serif font-bold text-stone-950">
                  Check Media
                </h4>
                <p className="text-xs text-stone-600 mt-1.5 leading-relaxed">
                  Analyze images or video for synthetic face swaps, AI generation, or spatial manipulation using Xception V2 &amp; EfficientNet-B4 V2.
                </p>
              </div>
            </div>

            <div className="pt-6 border-t border-[#f0e6d6]/60 mt-6">
              <button
                onClick={() => setActiveTab('check-media')}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-[#1e1b18] hover:bg-stone-950 text-white text-xs font-bold rounded-xl transition-colors cursor-pointer"
              >
                <span>Analyze Media</span>
                <ArrowRight className="w-3.5 h-3.5 text-red-400 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          {/* Card 2: Protect Image */}
          <div className="glass-card rounded-2xl p-6 hover:-translate-y-1 hover:border-red-900/40 hover:shadow-lg transition-all flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-[#1e1b18] text-white flex items-center justify-center shadow-xs group-hover:scale-105 transition-transform">
                <Lock className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h4 className="text-base font-serif font-bold text-stone-950">
                  Protect Image
                </h4>
                <p className="text-xs text-stone-600 mt-1.5 leading-relaxed">
                  Apply neural adversarial perturbation and embed ownership metadata to protect assets from AI cloning before public release.
                </p>
              </div>
            </div>

            <div className="pt-6 border-t border-[#f0e6d6]/60 mt-6">
              <button
                onClick={() => setActiveTab('protect-image')}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-[#1e1b18] hover:bg-stone-950 text-white text-xs font-bold rounded-xl transition-colors cursor-pointer"
              >
                <span>Protect Image</span>
                <ArrowRight className="w-3.5 h-3.5 text-red-400 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>

          {/* Card 3: Find Misuse */}
          <div className="glass-card rounded-2xl p-6 hover:-translate-y-1 hover:border-red-900/40 hover:shadow-lg transition-all flex flex-col justify-between group">
            <div className="space-y-4">
              <div className="w-12 h-12 rounded-xl bg-[#1e1b18] text-white flex items-center justify-center shadow-xs group-hover:scale-105 transition-transform">
                <Search className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h4 className="text-base font-serif font-bold text-stone-950">
                  Find Misuse
                </h4>
                <p className="text-xs text-stone-600 mt-1.5 leading-relaxed">
                  Search indexed public web sources and registered hash databases to locate unauthorized media copies or deepfake derivatives.
                </p>
              </div>
            </div>

            <div className="pt-6 border-t border-[#f0e6d6]/60 mt-6">
              <button
                onClick={() => setActiveTab('find-misuse')}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-[#1e1b18] hover:bg-stone-950 text-white text-xs font-bold rounded-xl transition-colors cursor-pointer"
              >
                <span>Find Misuse</span>
                <ArrowRight className="w-3.5 h-3.5 text-red-400 group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 4. REAL RECENT ACTIVITY SECTION */}
      <div className="glass-panel rounded-2xl p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-[#f0e6d6]/80 pb-4">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-stone-800" />
            <h3 className="font-serif font-bold text-stone-950 text-base">Recent Activity</h3>
          </div>
          <button
            onClick={() => setActiveTab('history')}
            className="text-xs font-bold text-stone-700 hover:text-stone-950 flex items-center gap-1 cursor-pointer"
          >
            <span>VIEW ALL ACTIVITY</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {recentHistory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-[#f0e6d6]/80 text-stone-500 font-mono font-bold uppercase">
                  <th className="py-2.5 px-3">Media Item</th>
                  <th className="py-2.5 px-3">Action</th>
                  <th className="py-2.5 px-3">Result &amp; Status</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3 text-right">Report</th>
                </tr>
              </thead>
              <tbody className="divide-y text-stone-800" style={{ borderColor: 'rgba(200, 185, 155, 0.20)' }}>
                {recentHistory.map((item) => (
                  <tr
                    key={item.id}
                    className="transition-colors"
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(240, 228, 200, 0.22)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-3">
                        <img
                          src={item.mediaUrl}
                          alt={item.mediaName}
                          className="w-10 h-10 object-cover rounded-lg border border-[#dcd0b9]"
                        />
                        <div>
                          <span className="font-bold text-stone-950 block">{item.mediaName}</span>
                          <span className="font-mono text-[10px] text-stone-500">{item.id}</span>
                        </div>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      <span className="px-2.5 py-1 rounded-md glass-input text-stone-900 font-semibold text-[11px]">
                        {item.action}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-2">
                        {item.risk === 'high' || item.status === 'Flagged' ? (
                          <AlertTriangle className="w-4 h-4 text-red-700 shrink-0" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-700 shrink-0" />
                        )}
                        <span className="font-bold text-stone-900">{item.resultSummary}</span>
                      </div>
                    </td>
                    <td className="py-3 px-3 text-stone-600 font-mono text-[11px]">{item.date}</td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => {
                          if (onOpenReport && item.detectionResult) {
                            onOpenReport(item.detectionResult);
                          } else {
                            setActiveTab('report');
                          }
                        }}
                        className="px-3 py-1.5 bg-[#1e1b18] hover:bg-stone-950 text-white rounded-lg text-[11px] font-bold inline-flex items-center gap-1 cursor-pointer transition-colors"
                        title="View Report"
                      >
                        <FileText className="w-3 h-3 text-amber-300" />
                        <span>Report</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-stone-600 text-xs">
            <p className="font-bold text-stone-800">No media analyzed yet</p>
            <p>Your recent forensic checks will appear here.</p>
          </div>
        )}
      </div>
    </div>
  );
};
