import React, { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  FolderKanban,
  FileSearch,
  Bell,
  Eye,
  Globe2,
  BarChart3,
  ScrollText,
  Users,
  Settings,
  Plus,
  Search,
  Filter,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Lock,
  Unlock,
  Download,
  ExternalLink,
  ChevronRight,
  ArrowLeft,
  Clock,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Send,
  Trash2,
  RefreshCw,
  Copy,
  Check,
  UserCheck,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  AreaChart,
  Area,
  CartesianGrid,
} from 'recharts';

import { useAuth } from '../../services/authContext';
import { investigatorService } from '../../services/investigatorService';
import {
  InvestigatorCase,
  CaseFindingItem,
  CaseTimelineItem,
  CaseNoteItem,
  WatchlistItem,
  WatchlistAlertItem,
  SourceReputationItem,
  AuditLogItem,
  OrgMemberItem,
  DashboardStatsData,
  EvidenceLockItem,
  NavItem,
} from '../../types';

type SectionKey =
  | 'dashboard'
  | 'cases'
  | 'findings'
  | 'alerts'
  | 'watchlists'
  | 'sources'
  | 'analytics'
  | 'audit'
  | 'team'
  | 'settings';

interface InvestigatorWorkspaceProps {
  setActiveTab: (tab: NavItem) => void;
  onOpenReport?: (analysis?: any, reportId?: string) => void;
}

export const InvestigatorWorkspace: React.FC<InvestigatorWorkspaceProps> = ({
  setActiveTab,
  onOpenReport,
}) => {
  const { user, currentOrg } = useAuth();
  const [activeSection, setActiveSection] = useState<SectionKey>('dashboard');

  // Dashboard Data
  const [dashboardData, setDashboardData] = useState<{
    stats: DashboardStatsData | null;
    recent_cases: InvestigatorCase[];
    recent_findings: CaseFindingItem[];
    recent_alerts: WatchlistAlertItem[];
  }>({
    stats: null,
    recent_cases: [],
    recent_findings: [],
    recent_alerts: [],
  });

  // Cases State
  const [casesList, setCasesList] = useState<InvestigatorCase[]>([]);
  const [selectedCase, setSelectedCase] = useState<InvestigatorCase | null>(null);
  const [caseFilterStatus, setCaseFilterStatus] = useState<string>('all');
  const [caseFilterSeverity, setCaseFilterSeverity] = useState<string>('all');
  const [isNewCaseModalOpen, setIsNewCaseModalOpen] = useState(false);
  const [newCaseForm, setNewCaseForm] = useState({ title: '', description: '', severity: 'medium', tags: '' });

  // Case Detail State
  const [caseTab, setCaseTab] = useState<'overview' | 'findings' | 'timeline' | 'notes' | 'exports'>('overview');
  const [caseFindings, setCaseFindings] = useState<CaseFindingItem[]>([]);
  const [caseTimeline, setCaseTimeline] = useState<CaseTimelineItem[]>([]);
  const [caseNotes, setCaseNotes] = useState<CaseNoteItem[]>([]);
  const [newNoteText, setNewNoteText] = useState('');
  const [isAttachModalOpen, setIsAttachModalOpen] = useState(false);
  const [attachFile, setAttachFile] = useState<File | null>(null);
  const [attachVerdict, setAttachVerdict] = useState('FAKE');
  const [attachLoading, setAttachLoading] = useState(false);

  // Global Findings State
  const [allFindings, setAllFindings] = useState<CaseFindingItem[]>([]);
  const [selectedFinding, setSelectedFinding] = useState<CaseFindingItem | null>(null);
  const [custodyChain, setCustodyChain] = useState<{
    locks: EvidenceLockItem[];
    timeline_entries: CaseTimelineItem[];
  } | null>(null);
  const [findingVerdictFilter, setFindingVerdictFilter] = useState('all');

  // Alerts & Watchlists
  const [alertsList, setAlertsList] = useState<WatchlistAlertItem[]>([]);
  const [watchlists, setWatchlists] = useState<WatchlistItem[]>([]);
  const [isNewWatchlistOpen, setIsNewWatchlistOpen] = useState(false);
  const [newWatchlistForm, setNewWatchlistForm] = useState({ name: '', target_type: 'social_crawl', target_query: '', description: '' });

  // Sources & Analytics
  const [sources, setSources] = useState<SourceReputationItem[]>([]);
  const [analyticsOverview, setAnalyticsOverview] = useState<any>(null);
  const [analyticsTimeline, setAnalyticsTimeline] = useState<any[]>([]);

  // Team & Audit
  const [teamMembers, setTeamMembers] = useState<OrgMemberItem[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogItem[]>([]);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('analyst');
  const [generatedInviteLink, setGeneratedInviteLink] = useState<string | null>(null);
  const [copiedLink, setCopiedLink] = useState(false);

  // Loading & notification banner
  const [loading, setLoading] = useState(false);
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  const showNotice = (msg: string) => {
    setActionNotice(msg);
    setTimeout(() => setActionNotice(null), 4000);
  };

  // ---------------- Fetch initial data based on active section ----------------
  const loadDashboard = async () => {
    try {
      setLoading(true);
      const res = await investigatorService.getDashboard();
      setDashboardData({
        stats: res.stats,
        recent_cases: res.recent_cases || [],
        recent_findings: res.recent_findings || [],
        recent_alerts: res.recent_alerts || [],
      });
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadCases = async () => {
    try {
      setLoading(true);
      const res = await investigatorService.listCases({
        status: caseFilterStatus,
        severity: caseFilterSeverity,
      });
      setCasesList(res.cases);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadCaseDetail = async (caseId: string) => {
    try {
      const [c, findings, timeline, notes] = await Promise.all([
        investigatorService.getCase(caseId),
        investigatorService.getCaseFindings(caseId),
        investigatorService.getCaseTimeline(caseId),
        investigatorService.getCaseNotes(caseId),
      ]);
      setSelectedCase(c);
      setCaseFindings(findings);
      setCaseTimeline(timeline);
      setCaseNotes(notes);
    } catch (e) {
      console.error(e);
    }
  };

  const loadAllFindings = async () => {
    try {
      setLoading(true);
      const findings = await investigatorService.getAllFindings({ verdict: findingVerdictFilter });
      setAllFindings(findings);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadAlerts = async () => {
    try {
      setLoading(true);
      const alerts = await investigatorService.listAlerts();
      setAlertsList(alerts);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadWatchlists = async () => {
    try {
      setLoading(true);
      const w = await investigatorService.listWatchlists();
      setWatchlists(w);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadSources = async () => {
    try {
      setLoading(true);
      const s = await investigatorService.listSources();
      setSources(s);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadAnalytics = async () => {
    try {
      setLoading(true);
      const [overview, timeline] = await Promise.all([
        investigatorService.getAnalyticsOverview(),
        investigatorService.getAnalyticsTimeline(14),
      ]);
      setAnalyticsOverview(overview);
      setAnalyticsTimeline(timeline.points || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const loadTeamAndAudit = async () => {
    try {
      setLoading(true);
      const [members, logs] = await Promise.all([
        investigatorService.listTeamMembers(),
        investigatorService.getAuditLog(),
      ]);
      setTeamMembers(members);
      setAuditLogs(logs);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeSection === 'dashboard') loadDashboard();
    else if (activeSection === 'cases') loadCases();
    else if (activeSection === 'findings') loadAllFindings();
    else if (activeSection === 'alerts') loadAlerts();
    else if (activeSection === 'watchlists') loadWatchlists();
    else if (activeSection === 'sources') loadSources();
    else if (activeSection === 'analytics') loadAnalytics();
    else if (activeSection === 'audit') loadTeamAndAudit();
    else if (activeSection === 'team') loadTeamAndAudit();
  }, [activeSection, caseFilterStatus, caseFilterSeverity, findingVerdictFilter]);

  // ---------------- Handlers ----------------
  const handleCreateCase = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCaseForm.title.trim()) return;
    try {
      const tags = newCaseForm.tags
        .split(',')
        .map((t) => t.trim())
        .filter(Boolean);
      const created = await investigatorService.createCase({
        title: newCaseForm.title,
        description: newCaseForm.description,
        severity: newCaseForm.severity,
        tags,
      });
      setIsNewCaseModalOpen(false);
      setNewCaseForm({ title: '', description: '', severity: 'medium', tags: '' });
      showNotice(`Case "${created.title}" initialized.`);
      loadCases();
      setSelectedCase(created);
    } catch (err: any) {
      showNotice(err.message || 'Failed to create case');
    }
  };

  const handleAttachDetection = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedCase) return;
    setAttachLoading(true);
    try {
      const res = await investigatorService.attachDetection(selectedCase.id, {
        file: attachFile || undefined,
        verdict: attachVerdict,
      });
      setIsAttachModalOpen(false);
      setAttachFile(null);
      showNotice('Detection attached to case findings.');
      loadCaseDetail(selectedCase.id);
    } catch (err: any) {
      showNotice(err.message || 'Failed to attach detection');
    } finally {
      setAttachLoading(false);
    }
  };

  const handleAddNote = async () => {
    if (!selectedCase || !newNoteText.trim()) return;
    try {
      await investigatorService.addCaseNote(selectedCase.id, newNoteText.trim());
      setNewNoteText('');
      loadCaseDetail(selectedCase.id);
      showNotice('Note added to case record.');
    } catch (err: any) {
      showNotice(err.message || 'Failed to add note');
    }
  };

  const handleToggleFreezeFinding = async (finding: CaseFindingItem) => {
    try {
      if (finding.frozen) {
        await investigatorService.unfreezeFinding(finding.id, 'Investigator unlocked finding');
        showNotice('Evidence finding unlocked.');
      } else {
        await investigatorService.freezeFinding(finding.id, 'Formal evidentiary freeze applied');
        showNotice('Evidence finding locked into custody chain.');
      }
      if (selectedCase) loadCaseDetail(selectedCase.id);
      if (activeSection === 'findings') loadAllFindings();
      if (selectedFinding) {
        const updated = await investigatorService.getFinding(selectedFinding.id);
        setSelectedFinding(updated);
        const chain = await investigatorService.getFindingCustody(selectedFinding.id);
        setCustodyChain(chain);
      }
    } catch (err: any) {
      showNotice(err.message || 'Operation failed');
    }
  };

  const handleViewFindingCustody = async (finding: CaseFindingItem) => {
    setSelectedFinding(finding);
    try {
      const chain = await investigatorService.getFindingCustody(finding.id);
      setCustodyChain(chain);
    } catch (e) {
      console.error(e);
    }
  };

  const handleExportCase = async (caseId: string) => {
    try {
      const exportRes = await investigatorService.exportCase(caseId, 'html');
      showNotice('Certified forensic report dossier generated.');
      window.open(exportRes.download_url, '_blank');
    } catch (err: any) {
      showNotice(err.message || 'Export failed');
    }
  };

  const handleTriageAlert = async (alertId: string, status: string) => {
    try {
      await investigatorService.triageAlert(alertId, status);
      showNotice(`Alert status updated to ${status}.`);
      loadAlerts();
    } catch (err: any) {
      showNotice(err.message || 'Failed to update alert');
    }
  };

  const handleEscalateAlert = async (alertId: string) => {
    try {
      const res = await investigatorService.escalateAlert(alertId);
      showNotice('Alert escalated into new case investigation.');
      loadAlerts();
      setActiveSection('cases');
      loadCaseDetail(res.case_id);
    } catch (err: any) {
      showNotice(err.message || 'Failed to escalate alert');
    }
  };

  const handleRunWatchlist = async (id: string) => {
    try {
      await investigatorService.runWatchlistNow(id);
      showNotice('Watchlist scan finished. New alert dispatched.');
      loadWatchlists();
    } catch (err: any) {
      showNotice(err.message || 'Failed to execute watchlist');
    }
  };

  const handleCreateWatchlist = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWatchlistForm.name.trim() || !newWatchlistForm.target_query.trim()) return;
    try {
      await investigatorService.createWatchlist(newWatchlistForm);
      setIsNewWatchlistOpen(false);
      setNewWatchlistForm({ name: '', target_type: 'social_crawl', target_query: '', description: '' });
      showNotice('Watchlist established.');
      loadWatchlists();
    } catch (err: any) {
      showNotice(err.message || 'Failed to create watchlist');
    }
  };

  const handleInviteTeam = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail.trim()) return;
    try {
      const res = await investigatorService.inviteMember(inviteEmail.trim(), inviteRole);
      setGeneratedInviteLink(res.invite_link);
      showNotice('Invitation generated.');
      loadTeamAndAudit();
    } catch (err: any) {
      showNotice(err.message || 'Failed to invite');
    }
  };

  // Nav items for sidebar
  const navItems: { key: SectionKey; label: string; icon: any; count?: number }[] = [
    { key: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { key: 'cases', label: 'Cases', icon: FolderKanban, count: dashboardData.stats?.total_cases },
    { key: 'findings', label: 'Findings', icon: FileSearch },
    { key: 'alerts', label: 'Alerts', icon: Bell, count: dashboardData.stats?.unread_alerts },
    { key: 'watchlists', label: 'Watchlists', icon: Eye },
    { key: 'sources', label: 'Sources', icon: Globe2 },
    { key: 'analytics', label: 'Analytics', icon: BarChart3 },
    { key: 'audit', label: 'Audit Log', icon: ScrollText },
    { key: 'team', label: 'Team', icon: Users },
    { key: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="flex flex-col lg:flex-row gap-6 max-w-[1700px] mx-auto pb-16">
      {/* ---------------- Sidebar Navigation ---------------- */}
      <aside className="w-full lg:w-64 shrink-0 glass-panel rounded-2xl p-4 shadow-sm flex flex-col justify-between">
        <div className="space-y-4">
          <div className="px-2 py-1 border-b border-[rgba(200,185,155,0.25)] pb-3">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-red-600" />
              <span className="font-serif font-black text-sm text-stone-950 uppercase tracking-wider">
                Investigator Hub
              </span>
            </div>
            <p className="text-[11px] text-stone-600 mt-1 truncate">
              Org: <span className="font-semibold">{currentOrg?.name || 'Verified Forensic Org'}</span>
            </p>
          </div>

          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.key;
              return (
                <button
                  key={item.key}
                  onClick={() => {
                    setActiveSection(item.key);
                    if (item.key === 'cases') setSelectedCase(null);
                  }}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                    isActive
                      ? 'bg-stone-950 text-white shadow-xs'
                      : 'text-stone-800 hover:text-stone-950 hover:bg-[#ede4d4]/40'
                  }`}
                >
                  <div className="flex items-center gap-2.5">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-red-400' : 'text-stone-600'}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.count !== undefined && item.count > 0 && (
                    <span
                      className={`text-[10px] font-mono font-bold px-1.5 py-0.2 rounded ${
                        isActive ? 'bg-red-900/60 text-red-200' : 'bg-stone-200 text-stone-700'
                      }`}
                    >
                      {item.count}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        <div className="pt-4 border-t border-[rgba(200,185,155,0.25)] px-2">
          <button
            onClick={() => setActiveTab('profile')}
            className="w-full flex items-center justify-between py-1.5 text-[11px] font-semibold text-stone-700 hover:text-stone-950 cursor-pointer"
          >
            <span>My Profile &amp; Tier</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </aside>

      {/* ---------------- Main Content Workspace ---------------- */}
      <main className="flex-1 min-w-0 space-y-6">
        {/* Banner notification */}
        {actionNotice && (
          <div className="p-3 rounded-xl bg-stone-950 text-white text-xs flex items-center justify-between shadow-md animate-fade-in">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>{actionNotice}</span>
            </div>
            <button onClick={() => setActionNotice(null)} className="text-stone-400 hover:text-white text-xs">
              &times;
            </button>
          </div>
        )}

        {/* ============================================================== */}
        {/* 1. DASHBOARD SECTION */}
        {/* ============================================================== */}
        {activeSection === 'dashboard' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Investigator Dashboard</h2>
                <p className="text-xs text-stone-600">
                  Forensic telemetry, active cases, and verified evidentiary custody overview.
                </p>
              </div>
              <button
                onClick={() => {
                  setActiveSection('cases');
                  setIsNewCaseModalOpen(true);
                }}
                className="flex items-center gap-2 px-3.5 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer hover:bg-stone-900"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>New Case</span>
              </button>
            </div>

            {/* Stat Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
              <div className="glass-card rounded-2xl p-4 shadow-2xs space-y-1">
                <span className="text-[10px] font-mono uppercase text-stone-500 font-bold">Total Cases</span>
                <p className="text-2xl font-serif font-black text-stone-950">
                  {dashboardData.stats?.total_cases ?? 0}
                </p>
              </div>

              <div className="glass-card rounded-2xl p-4 shadow-2xs space-y-1">
                <span className="text-[10px] font-mono uppercase text-stone-500 font-bold">Open Cases</span>
                <p className="text-2xl font-serif font-black text-amber-700">
                  {dashboardData.stats?.open_cases ?? 0}
                </p>
              </div>

              <div className="glass-card rounded-2xl p-4 shadow-2xs space-y-1">
                <span className="text-[10px] font-mono uppercase text-stone-500 font-bold">Findings (Month)</span>
                <p className="text-2xl font-serif font-black text-stone-950">
                  {dashboardData.stats?.findings_month ?? 0}
                </p>
              </div>

              <div className="glass-card rounded-2xl p-4 shadow-2xs space-y-1">
                <span className="text-[10px] font-mono uppercase text-stone-500 font-bold">Fake vs Real (Month)</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-xl font-serif font-black text-red-600">
                    {dashboardData.stats?.fake_findings_month ?? 0}
                  </span>
                  <span className="text-xs text-stone-400 font-mono">/</span>
                  <span className="text-xl font-serif font-black text-emerald-600">
                    {dashboardData.stats?.real_findings_month ?? 0}
                  </span>
                </div>
              </div>

              <div className="glass-card rounded-2xl p-4 shadow-2xs space-y-1">
                <span className="text-[10px] font-mono uppercase text-stone-500 font-bold">Unread Alerts</span>
                <p className="text-2xl font-serif font-black text-red-600">
                  {dashboardData.stats?.unread_alerts ?? 0}
                </p>
              </div>
            </div>

            {/* Recent Cases & Findings Columns */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Recent Cases Table */}
              <div className="glass-panel rounded-2xl p-5 shadow-xs space-y-3">
                <div className="flex items-center justify-between border-b border-[rgba(200,185,155,0.25)] pb-2.5">
                  <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900 flex items-center gap-2">
                    <FolderKanban className="w-4 h-4 text-stone-700" />
                    <span>Recent Investigation Cases</span>
                  </h3>
                  <button
                    onClick={() => setActiveSection('cases')}
                    className="text-[11px] font-semibold text-stone-700 hover:text-stone-950 cursor-pointer"
                  >
                    View All &rarr;
                  </button>
                </div>

                <div className="space-y-2">
                  {dashboardData.recent_cases.length === 0 ? (
                    <p className="text-xs text-stone-500 py-6 text-center">No cases created yet.</p>
                  ) : (
                    dashboardData.recent_cases.map((c) => (
                      <div
                        key={c.id}
                        onClick={() => {
                          setActiveSection('cases');
                          loadCaseDetail(c.id);
                        }}
                        className="p-3 rounded-xl glass-card hover:bg-[#faf6ee]/70 transition-all cursor-pointer flex items-center justify-between"
                      >
                        <div className="space-y-0.5">
                          <p className="font-bold text-xs text-stone-900 truncate max-w-xs">{c.title}</p>
                          <div className="flex items-center gap-2 text-[10px] text-stone-500 font-mono">
                            <span>Status: {c.status}</span>
                            <span>&bull;</span>
                            <span>Findings: {c.findings_count}</span>
                          </div>
                        </div>
                        <span
                          className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                            c.severity === 'critical' || c.severity === 'high'
                              ? 'bg-red-100 text-red-900'
                              : 'bg-stone-200 text-stone-800'
                          }`}
                        >
                          {c.severity}
                        </span>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Recent Findings Table */}
              <div className="glass-panel rounded-2xl p-5 shadow-xs space-y-3">
                <div className="flex items-center justify-between border-b border-[rgba(200,185,155,0.25)] pb-2.5">
                  <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900 flex items-center gap-2">
                    <FileSearch className="w-4 h-4 text-stone-700" />
                    <span>Recent Case Findings</span>
                  </h3>
                  <button
                    onClick={() => setActiveSection('findings')}
                    className="text-[11px] font-semibold text-stone-700 hover:text-stone-950 cursor-pointer"
                  >
                    View All &rarr;
                  </button>
                </div>

                <div className="space-y-2">
                  {dashboardData.recent_findings.length === 0 ? (
                    <p className="text-xs text-stone-500 py-6 text-center">No findings attached yet.</p>
                  ) : (
                    dashboardData.recent_findings.map((f) => (
                      <div
                        key={f.id}
                        onClick={() => handleViewFindingCustody(f)}
                        className="p-3 rounded-xl glass-card hover:bg-[#faf6ee]/70 transition-all cursor-pointer flex items-center justify-between"
                      >
                        <div className="space-y-0.5">
                          <p className="font-bold text-xs text-stone-900 truncate max-w-xs">{f.original_filename}</p>
                          <p className="text-[10px] text-stone-500 font-mono">
                            SHA: {f.sha256.slice(0, 16)}... &bull; Conf: {Math.round(f.confidence * 100)}%
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                              f.verdict === 'FAKE'
                                ? 'bg-red-100 text-red-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {f.verdict}
                          </span>
                          {f.frozen && <span title="Locked"><Lock className="w-3.5 h-3.5 text-stone-600" /></span>}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>

            {/* Recent Alerts Card */}
            <div className="glass-panel rounded-2xl p-5 shadow-xs space-y-3">
              <div className="flex items-center justify-between border-b border-[rgba(200,185,155,0.25)] pb-2.5">
                <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900 flex items-center gap-2">
                  <Bell className="w-4 h-4 text-stone-700" />
                  <span>Monitoring Feed &amp; Recent Alerts</span>
                </h3>
                <button
                  onClick={() => setActiveSection('alerts')}
                  className="text-[11px] font-semibold text-stone-700 hover:text-stone-950 cursor-pointer"
                >
                  Manage Alerts &rarr;
                </button>
              </div>

              <div className="space-y-2">
                {dashboardData.recent_alerts.length === 0 ? (
                  <p className="text-xs text-stone-500 py-4 text-center">No monitoring alerts triggered.</p>
                ) : (
                  dashboardData.recent_alerts.map((a) => (
                    <div
                      key={a.id}
                      className="p-3 rounded-xl glass-card flex items-center justify-between gap-4"
                    >
                      <div className="space-y-0.5 min-w-0">
                        <p className="font-bold text-xs text-stone-900 truncate">{a.source_url}</p>
                        <p className="text-[10px] text-stone-500 font-mono">
                          Watchlist: {a.watchlist_name || 'Global'} &bull; Status: {a.status}
                        </p>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <span
                          className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                            a.severity === 'high' ? 'bg-red-100 text-red-900' : 'bg-amber-100 text-amber-900'
                          }`}
                        >
                          {a.severity}
                        </span>
                        <button
                          onClick={() => handleTriageAlert(a.id, 'triaged')}
                          className="px-2 py-1 rounded text-[10px] font-bold bg-stone-900 text-white cursor-pointer"
                        >
                          Triage
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 2. CASES SECTION & CASE DETAIL */}
        {/* ============================================================== */}
        {activeSection === 'cases' && !selectedCase && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Investigation Cases</h2>
                <p className="text-xs text-stone-600">
                  Organize findings, forensic evidence chains, and team notes by investigation dossier.
                </p>
              </div>
              <button
                onClick={() => setIsNewCaseModalOpen(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer hover:bg-stone-900"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>New Case</span>
              </button>
            </div>

            {/* Filters */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-xs">
                <span className="font-semibold text-stone-600">Status:</span>
                <select
                  value={caseFilterStatus}
                  onChange={(e) => setCaseFilterStatus(e.target.value)}
                  className="glass-input rounded-xl px-2.5 py-1 text-xs focus:outline-none"
                >
                  <option value="all">All</option>
                  <option value="open">Open</option>
                  <option value="in_progress">In Progress</option>
                  <option value="closed">Closed</option>
                  <option value="archived">Archived</option>
                </select>
              </div>

              <div className="flex items-center gap-2 text-xs">
                <span className="font-semibold text-stone-600">Severity:</span>
                <select
                  value={caseFilterSeverity}
                  onChange={(e) => setCaseFilterSeverity(e.target.value)}
                  className="glass-input rounded-xl px-2.5 py-1 text-xs focus:outline-none"
                >
                  <option value="all">All</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>

            {/* Cases Table */}
            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Title</th>
                    <th className="p-3.5">Status</th>
                    <th className="p-3.5">Severity</th>
                    <th className="p-3.5">Findings</th>
                    <th className="p-3.5">Notes</th>
                    <th className="p-3.5">Created</th>
                    <th className="p-3.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {casesList.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="p-8 text-center text-stone-500">
                        No cases found matching filters.
                      </td>
                    </tr>
                  ) : (
                    casesList.map((c) => (
                      <tr
                        key={c.id}
                        onClick={() => loadCaseDetail(c.id)}
                        className="hover:bg-[rgba(255,250,240,0.5)] transition-colors cursor-pointer"
                      >
                        <td className="p-3.5 font-bold text-stone-950">
                          <div>{c.title}</div>
                          {c.description && (
                            <div className="text-[11px] text-stone-500 font-normal truncate max-w-sm">
                              {c.description}
                            </div>
                          )}
                        </td>
                        <td className="p-3.5">
                          <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase bg-stone-200/80 text-stone-800">
                            {c.status}
                          </span>
                        </td>
                        <td className="p-3.5">
                          <span
                            className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                              c.severity === 'critical' || c.severity === 'high'
                                ? 'bg-red-100 text-red-900'
                                : 'bg-stone-200 text-stone-800'
                            }`}
                          >
                            {c.severity}
                          </span>
                        </td>
                        <td className="p-3.5 font-mono">{c.findings_count}</td>
                        <td className="p-3.5 font-mono">{c.notes_count}</td>
                        <td className="p-3.5 text-stone-500 text-[11px] font-mono">
                          {c.created_at ? new Date(c.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                        <td className="p-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => loadCaseDetail(c.id)}
                            className="text-stone-700 hover:text-stone-950 font-bold text-xs"
                          >
                            Open &rarr;
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ---------------- Case Detail View ---------------- */}
        {activeSection === 'cases' && selectedCase && (
          <div className="space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setSelectedCase(null)}
                  className="p-2 rounded-xl glass-card text-stone-700 hover:text-stone-950 cursor-pointer"
                >
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-xl font-serif font-black text-stone-950">{selectedCase.title}</h2>
                    <span
                      className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                        selectedCase.severity === 'high' || selectedCase.severity === 'critical'
                          ? 'bg-red-100 text-red-900'
                          : 'bg-stone-200 text-stone-800'
                      }`}
                    >
                      {selectedCase.severity}
                    </span>
                  </div>
                  <p className="text-xs text-stone-500 font-mono mt-0.5">
                    Case ID: {selectedCase.id} &bull; Created by: {selectedCase.creator_name || 'Investigator'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsAttachModalOpen(true)}
                  className="px-3.5 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer flex items-center gap-1.5"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Attach Detection</span>
                </button>
                <button
                  onClick={() => handleExportCase(selectedCase.id)}
                  className="px-3.5 py-2 rounded-xl text-xs font-bold text-stone-900 glass-card shadow-xs cursor-pointer flex items-center gap-1.5 hover:bg-stone-100/70"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export Report</span>
                </button>
              </div>
            </div>

            {/* Case Tabs */}
            <div className="flex items-center gap-2 border-b border-[rgba(200,185,155,0.3)] pb-2 text-xs font-bold">
              {(['overview', 'findings', 'timeline', 'notes', 'exports'] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setCaseTab(tab)}
                  className={`px-3 py-1.5 rounded-xl capitalize transition-all cursor-pointer ${
                    caseTab === tab
                      ? 'bg-stone-950 text-white shadow-xs'
                      : 'text-stone-700 hover:text-stone-950'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab: Overview */}
            {caseTab === 'overview' && (
              <div className="glass-panel rounded-2xl p-6 space-y-6">
                <div>
                  <h3 className="text-xs font-mono font-bold uppercase text-stone-500 tracking-wider mb-1">Description</h3>
                  <p className="text-xs text-stone-800 leading-relaxed">
                    {selectedCase.description || 'No description provided.'}
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-[rgba(200,185,155,0.25)]">
                  <div>
                    <label className="text-[10px] font-mono font-bold text-stone-500 uppercase block mb-1">Status</label>
                    <select
                      value={selectedCase.status}
                      onChange={async (e) => {
                        const updated = await investigatorService.updateCaseStatus(selectedCase.id, e.target.value);
                        setSelectedCase(updated);
                        showNotice(`Status changed to ${e.target.value}`);
                      }}
                      className="glass-input rounded-xl px-2.5 py-1 text-xs w-full focus:outline-none"
                    >
                      <option value="open">Open</option>
                      <option value="in_progress">In Progress</option>
                      <option value="closed">Closed</option>
                      <option value="archived">Archived</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[10px] font-mono font-bold text-stone-500 uppercase block mb-1">Severity</label>
                    <p className="text-xs font-bold text-stone-900 uppercase">{selectedCase.severity}</p>
                  </div>

                  <div>
                    <label className="text-[10px] font-mono font-bold text-stone-500 uppercase block mb-1">Tags</label>
                    <div className="flex flex-wrap gap-1">
                      {selectedCase.tags.length > 0 ? (
                        selectedCase.tags.map((t, i) => (
                          <span key={i} className="text-[10px] font-mono bg-stone-200 text-stone-800 px-2 py-0.5 rounded">
                            {t}
                          </span>
                        ))
                      ) : (
                        <span className="text-xs text-stone-400">None</span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab: Findings */}
            {caseTab === 'findings' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-bold text-xs uppercase tracking-wider text-stone-800">
                    Attached Forensic Findings ({caseFindings.length})
                  </h3>
                  <button
                    onClick={() => setIsAttachModalOpen(true)}
                    className="px-3 py-1.5 rounded-xl text-xs font-bold bg-stone-900 text-white cursor-pointer"
                  >
                    + Attach Media
                  </button>
                </div>

                <div className="space-y-3">
                  {caseFindings.length === 0 ? (
                    <div className="glass-panel rounded-2xl p-8 text-center text-xs text-stone-500">
                      No forensic findings attached to this case yet. Click "+ Attach Media" to run detection.
                    </div>
                  ) : (
                    caseFindings.map((f) => (
                      <div
                        key={f.id}
                        className="glass-panel rounded-2xl p-4 flex items-center justify-between gap-4"
                      >
                        <div className="space-y-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-xs text-stone-950 truncate max-w-sm">
                              {f.original_filename}
                            </span>
                            <span
                              className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                                f.verdict === 'FAKE'
                                  ? 'bg-red-100 text-red-800'
                                  : 'bg-emerald-100 text-emerald-800'
                              }`}
                            >
                              {f.verdict}
                            </span>
                          </div>
                          <p className="text-[10px] text-stone-500 font-mono truncate">
                            SHA-256: {f.sha256} &bull; Conf: {Math.round(f.confidence * 100)}%
                          </p>
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                          <button
                            onClick={() => handleToggleFreezeFinding(f)}
                            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 cursor-pointer ${
                              f.frozen
                                ? 'bg-amber-100 text-amber-900 border border-amber-300'
                                : 'glass-card text-stone-800'
                            }`}
                          >
                            {f.frozen ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
                            <span>{f.frozen ? 'Locked' : 'Lock Evidence'}</span>
                          </button>
                          <button
                            onClick={() => handleViewFindingCustody(f)}
                            className="px-3 py-1.5 rounded-xl text-xs font-bold bg-stone-950 text-white cursor-pointer"
                          >
                            Custody Chain
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Tab: Timeline */}
            {caseTab === 'timeline' && (
              <div className="glass-panel rounded-2xl p-6 space-y-4">
                <h3 className="font-bold text-xs uppercase tracking-wider text-stone-800 border-b border-[rgba(200,185,155,0.25)] pb-2">
                  Chronological Case Activity Log
                </h3>
                <div className="space-y-3">
                  {caseTimeline.map((item) => (
                    <div key={item.id} className="flex items-start gap-3 text-xs">
                      <div className="w-2 h-2 rounded-full bg-red-600 mt-1.5 shrink-0" />
                      <div className="space-y-0.5">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-stone-900">{item.action.toUpperCase()}</span>
                          <span className="text-[10px] text-stone-500 font-mono">
                            by {item.actor_name || 'System'} at{' '}
                            {item.created_at ? new Date(item.created_at).toLocaleString() : ''}
                          </span>
                        </div>
                        {item.payload && Object.keys(item.payload).length > 0 && (
                          <pre className="text-[11px] text-stone-600 font-mono bg-stone-100/50 p-1.5 rounded-lg overflow-x-auto max-w-xl">
                            {JSON.stringify(item.payload, null, 2)}
                          </pre>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Tab: Notes */}
            {caseTab === 'notes' && (
              <div className="glass-panel rounded-2xl p-6 space-y-4">
                <div className="space-y-3">
                  {caseNotes.map((note) => (
                    <div key={note.id} className="p-3 rounded-xl glass-card text-xs space-y-1">
                      <div className="flex items-center justify-between text-[10px] text-stone-500 font-mono">
                        <span>{note.author_name || 'Investigator'}</span>
                        <span>{note.created_at ? new Date(note.created_at).toLocaleString() : ''}</span>
                      </div>
                      <p className="text-stone-900 leading-relaxed">{note.text}</p>
                    </div>
                  ))}
                </div>

                <div className="pt-4 border-t border-[rgba(200,185,155,0.25)] flex gap-2">
                  <input
                    type="text"
                    placeholder="Add an investigative note..."
                    value={newNoteText}
                    onChange={(e) => setNewNoteText(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddNote()}
                    className="flex-1 p-2 rounded-xl glass-input text-xs text-stone-900 focus:outline-none"
                  />
                  <button
                    onClick={handleAddNote}
                    className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 cursor-pointer"
                  >
                    Add Note
                  </button>
                </div>
              </div>
            )}

            {/* Tab: Exports */}
            {caseTab === 'exports' && (
              <div className="glass-panel rounded-2xl p-6 space-y-4">
                <div className="flex items-center justify-between border-b border-[rgba(200,185,155,0.25)] pb-3">
                  <div>
                    <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900">
                      Generated Forensic Export Dossiers
                    </h3>
                    <p className="text-xs text-stone-600">Export verified HTML reports with evidentiary hashes.</p>
                  </div>
                  <button
                    onClick={() => handleExportCase(selectedCase.id)}
                    className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 cursor-pointer flex items-center gap-1.5"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Generate New Export</span>
                  </button>
                </div>
                <p className="text-xs text-stone-500 py-4">
                  Dossier generator compiles all attached findings, SHA-256 hashes, custody locks, and timestamps into a legal record.
                </p>
              </div>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* 3. FINDINGS SECTION */}
        {/* ============================================================== */}
        {activeSection === 'findings' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Global Evidence Findings</h2>
                <p className="text-xs text-stone-600">
                  Cross-case evidence bank, cryptographic locks, and forensic analysis files.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold text-stone-600">Verdict:</span>
                <select
                  value={findingVerdictFilter}
                  onChange={(e) => setFindingVerdictFilter(e.target.value)}
                  className="glass-input rounded-xl px-2.5 py-1 text-xs focus:outline-none"
                >
                  <option value="all">All Verdicts</option>
                  <option value="fake">Fake Only</option>
                  <option value="real">Real Only</option>
                </select>
              </div>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Filename</th>
                    <th className="p-3.5">Verdict</th>
                    <th className="p-3.5">Confidence</th>
                    <th className="p-3.5">SHA-256</th>
                    <th className="p-3.5">Custody State</th>
                    <th className="p-3.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {allFindings.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="p-8 text-center text-stone-500">
                        No findings available.
                      </td>
                    </tr>
                  ) : (
                    allFindings.map((f) => (
                      <tr
                        key={f.id}
                        onClick={() => handleViewFindingCustody(f)}
                        className="hover:bg-[rgba(255,250,240,0.5)] transition-colors cursor-pointer"
                      >
                        <td className="p-3.5 font-bold text-stone-900">{f.original_filename}</td>
                        <td className="p-3.5">
                          <span
                            className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                              f.verdict === 'FAKE' ? 'bg-red-100 text-red-800' : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {f.verdict}
                          </span>
                        </td>
                        <td className="p-3.5 font-mono">{Math.round(f.confidence * 100)}%</td>
                        <td className="p-3.5 font-mono text-[11px] text-stone-600">{f.sha256.slice(0, 16)}...</td>
                        <td className="p-3.5">
                          {f.frozen ? (
                            <span className="text-[10px] font-mono font-bold text-amber-800 bg-amber-100 px-2 py-0.5 rounded flex items-center gap-1 w-max">
                              <Lock className="w-3 h-3" /> Locked
                            </span>
                          ) : (
                            <span className="text-[10px] font-mono text-stone-500">Unfrozen</span>
                          )}
                        </td>
                        <td className="p-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => handleViewFindingCustody(f)}
                            className="text-stone-800 hover:text-stone-950 font-bold"
                          >
                            Details &rarr;
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 4. ALERTS SECTION */}
        {/* ============================================================== */}
        {activeSection === 'alerts' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Monitoring Alerts Inbox</h2>
                <p className="text-xs text-stone-600">
                  Triaged detections from active watchlists, reverse image crawling, and social scrapers.
                </p>
              </div>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Source URL</th>
                    <th className="p-3.5">Watchlist</th>
                    <th className="p-3.5">Severity</th>
                    <th className="p-3.5">Match Score</th>
                    <th className="p-3.5">Status</th>
                    <th className="p-3.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {alertsList.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="p-8 text-center text-stone-500">
                        No alerts in inbox. Trigger a watchlist scan to generate alerts.
                      </td>
                    </tr>
                  ) : (
                    alertsList.map((a) => (
                      <tr key={a.id} className="hover:bg-[rgba(255,250,240,0.5)] transition-colors">
                        <td className="p-3.5 font-bold text-stone-900 truncate max-w-xs">{a.source_url}</td>
                        <td className="p-3.5 text-stone-600">{a.watchlist_name || 'General'}</td>
                        <td className="p-3.5">
                          <span
                            className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
                              a.severity === 'high' ? 'bg-red-100 text-red-900' : 'bg-stone-200 text-stone-800'
                            }`}
                          >
                            {a.severity}
                          </span>
                        </td>
                        <td className="p-3.5 font-mono">
                          {a.match_score ? `${Math.round(a.match_score * 100)}%` : 'N/A'}
                        </td>
                        <td className="p-3.5">
                          <span className="text-[10px] font-mono uppercase font-bold text-stone-700">
                            {a.status}
                          </span>
                        </td>
                        <td className="p-3.5 text-right space-x-2">
                          {a.status === 'new' && (
                            <button
                              onClick={() => handleTriageAlert(a.id, 'triaged')}
                              className="px-2 py-1 rounded text-[10px] font-bold bg-stone-900 text-white cursor-pointer"
                            >
                              Triage
                            </button>
                          )}
                          <button
                            onClick={() => handleEscalateAlert(a.id)}
                            className="px-2 py-1 rounded text-[10px] font-bold bg-red-700 text-white cursor-pointer"
                          >
                            Escalate to Case
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 5. WATCHLISTS SECTION */}
        {/* ============================================================== */}
        {activeSection === 'watchlists' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Active Watchlists</h2>
                <p className="text-xs text-stone-600">
                  Automated scraping and reverse query triggers for targets of interest.
                </p>
              </div>
              <button
                onClick={() => setIsNewWatchlistOpen(true)}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>New Watchlist</span>
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {watchlists.length === 0 ? (
                <div className="col-span-2 glass-panel p-8 text-center text-xs text-stone-500 rounded-2xl">
                  No watchlists configured. Click "New Watchlist" to set up target monitoring.
                </div>
              ) : (
                watchlists.map((w) => (
                  <div key={w.id} className="glass-panel rounded-2xl p-5 shadow-xs space-y-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="font-bold text-sm text-stone-950">{w.name}</h3>
                        <p className="text-xs text-stone-500 font-mono mt-0.5">Query: {w.target_query}</p>
                      </div>
                      <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 uppercase">
                        Active
                      </span>
                    </div>

                    <p className="text-xs text-stone-600">{w.description || 'Continuous monitoring target.'}</p>

                    <div className="pt-3 border-t border-[rgba(200,185,155,0.25)] flex items-center justify-between">
                      <span className="text-[11px] font-mono text-stone-500">
                        Alerts: {w.alerts_count || 0}
                      </span>
                      <button
                        onClick={() => handleRunWatchlist(w.id)}
                        className="px-3 py-1.5 rounded-xl text-xs font-bold text-white bg-stone-900 cursor-pointer flex items-center gap-1.5"
                      >
                        <RefreshCw className="w-3.5 h-3.5" />
                        <span>Run Now</span>
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 6. SOURCES SECTION */}
        {/* ============================================================== */}
        {activeSection === 'sources' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Source Reputation Index</h2>
              <p className="text-xs text-stone-600">
                Aggregated domain telemetry tracks known malicious spreaders and disinformation nodes.
              </p>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Domain</th>
                    <th className="p-3.5">Total Findings</th>
                    <th className="p-3.5">Fake Findings</th>
                    <th className="p-3.5">Fake Ratio</th>
                    <th className="p-3.5">Last Observed</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {sources.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="p-8 text-center text-stone-500">
                        No sources cataloged yet. Sources are automatically indexed as media is attached.
                      </td>
                    </tr>
                  ) : (
                    sources.map((s) => (
                      <tr key={s.id} className="hover:bg-[rgba(255,250,240,0.5)] transition-colors">
                        <td className="p-3.5 font-bold text-stone-900">{s.domain}</td>
                        <td className="p-3.5 font-mono">{s.total_findings}</td>
                        <td className="p-3.5 font-mono text-red-600 font-semibold">{s.fake_findings}</td>
                        <td className="p-3.5 font-mono font-bold">
                          {Math.round(s.fake_ratio * 100)}%
                        </td>
                        <td className="p-3.5 text-stone-500 font-mono text-[11px]">
                          {s.last_seen_at ? new Date(s.last_seen_at).toLocaleDateString() : 'Recent'}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 7. ANALYTICS SECTION */}
        {/* ============================================================== */}
        {activeSection === 'analytics' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Organization Analytics</h2>
              <p className="text-xs text-stone-600">
                14-day threat trajectory, verdict distributions, and severity vectors.
              </p>
            </div>

            {/* Time Series Chart */}
            <div className="glass-panel rounded-2xl p-6 shadow-xs space-y-4">
              <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900">
                Findings Per Day (Fake vs Real Timeline)
              </h3>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={analyticsTimeline}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e7e0d3" />
                    <XAxis dataKey="date" stroke="#78716c" fontSize={10} />
                    <YAxis stroke="#78716c" fontSize={10} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1c1917',
                        borderRadius: 12,
                        color: '#fff',
                        fontSize: 12,
                        border: 'none',
                      }}
                    />
                    <Area type="monotone" dataKey="fake" stroke="#dc2626" fill="#dc2626" fillOpacity={0.2} name="Fake" />
                    <Area type="monotone" dataKey="real" stroke="#059669" fill="#059669" fillOpacity={0.2} name="Real" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Cases By Status & Severity Charts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="glass-panel rounded-2xl p-6 shadow-xs space-y-4">
                <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900">
                  Cases By Status
                </h3>
                <div className="h-52 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={
                        analyticsOverview?.cases_by_status
                          ? Object.entries(analyticsOverview.cases_by_status).map(([k, v]) => ({
                              name: k,
                              count: v,
                            }))
                          : []
                      }
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e0d3" />
                      <XAxis dataKey="name" stroke="#78716c" fontSize={10} />
                      <YAxis stroke="#78716c" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#1c1917', borderRadius: 12, color: '#fff', fontSize: 12 }} />
                      <Bar dataKey="count" fill="#1c1917" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="glass-panel rounded-2xl p-6 shadow-xs space-y-4">
                <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900">
                  Cases By Severity
                </h3>
                <div className="h-52 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={
                        analyticsOverview?.cases_by_severity
                          ? Object.entries(analyticsOverview.cases_by_severity).map(([k, v]) => ({
                              name: k,
                              count: v,
                            }))
                          : []
                      }
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e7e0d3" />
                      <XAxis dataKey="name" stroke="#78716c" fontSize={10} />
                      <YAxis stroke="#78716c" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#1c1917', borderRadius: 12, color: '#fff', fontSize: 12 }} />
                      <Bar dataKey="count" fill="#dc2626" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 8. AUDIT LOG SECTION */}
        {/* ============================================================== */}
        {activeSection === 'audit' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Organization Audit Trail</h2>
              <p className="text-xs text-stone-600">
                Immutable chronological ledger of all user actions, case modifications, and evidentiary freezes.
              </p>
            </div>

            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Timestamp</th>
                    <th className="p-3.5">Actor</th>
                    <th className="p-3.5">Action</th>
                    <th className="p-3.5">Target</th>
                    <th className="p-3.5">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {auditLogs.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="p-8 text-center text-stone-500">
                        No audit records captured yet.
                      </td>
                    </tr>
                  ) : (
                    auditLogs.map((log) => (
                      <tr key={log.id} className="hover:bg-[rgba(255,250,240,0.5)] transition-colors">
                        <td className="p-3.5 font-mono text-stone-500 text-[11px]">
                          {log.created_at ? new Date(log.created_at).toLocaleString() : ''}
                        </td>
                        <td className="p-3.5 font-bold text-stone-900">{log.user_name || 'System'}</td>
                        <td className="p-3.5">
                          <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-stone-200 text-stone-800 uppercase">
                            {log.action}
                          </span>
                        </td>
                        <td className="p-3.5 font-mono text-stone-600">
                          {log.target_type ? `${log.target_type}:${log.target_id?.slice(0, 8)}...` : 'N/A'}
                        </td>
                        <td className="p-3.5 font-mono text-[11px] text-stone-500 truncate max-w-xs">
                          {JSON.stringify(log.metadata || {})}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 9. TEAM SECTION */}
        {/* ============================================================== */}
        {activeSection === 'team' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Organization Members</h2>
                <p className="text-xs text-stone-600">
                  Manage analysts, investigators, and access privileges for {currentOrg?.name}.
                </p>
              </div>
            </div>

            {/* Invite Box */}
            <div className="glass-panel rounded-2xl p-5 shadow-xs space-y-3">
              <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900">
                Invite Colleague to Organization
              </h3>
              <form onSubmit={handleInviteTeam} className="flex flex-col sm:flex-row gap-3">
                <input
                  type="email"
                  required
                  placeholder="colleague@organization.org"
                  value={inviteEmail}
                  onChange={(e) => setInviteEmail(e.target.value)}
                  className="flex-1 p-2 rounded-xl glass-input text-xs text-stone-900 focus:outline-none"
                />
                <select
                  value={inviteRole}
                  onChange={(e) => setInviteRole(e.target.value)}
                  className="glass-input rounded-xl px-3 py-2 text-xs focus:outline-none"
                >
                  <option value="analyst">Analyst</option>
                  <option value="admin">Admin</option>
                  <option value="viewer">Viewer</option>
                </select>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 shadow-sm cursor-pointer hover:bg-stone-900 shrink-0"
                >
                  Generate Invite
                </button>
              </form>

              {generatedInviteLink && (
                <div className="p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] text-amber-950 truncate">{generatedInviteLink}</span>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(generatedInviteLink);
                      setCopiedLink(true);
                      setTimeout(() => setCopiedLink(false), 2000);
                    }}
                    className="px-2.5 py-1 rounded-lg text-xs font-bold bg-stone-900 text-white cursor-pointer shrink-0 flex items-center gap-1"
                  >
                    {copiedLink ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedLink ? 'Copied' : 'Copy Link'}</span>
                  </button>
                </div>
              )}
            </div>

            {/* Members List */}
            <div className="glass-panel rounded-2xl overflow-hidden shadow-xs">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[rgba(200,185,155,0.3)] bg-[rgba(240,230,210,0.3)] text-stone-700 font-mono text-[11px] uppercase">
                    <th className="p-3.5">Member Name</th>
                    <th className="p-3.5">Email</th>
                    <th className="p-3.5">Role</th>
                    <th className="p-3.5">Joined Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[rgba(200,185,155,0.2)]">
                  {teamMembers.map((m) => (
                    <tr key={m.user_id} className="hover:bg-[rgba(255,250,240,0.5)] transition-colors">
                      <td className="p-3.5 font-bold text-stone-900">
                        {m.user?.display_name || 'Investigator'}
                      </td>
                      <td className="p-3.5 font-mono text-stone-600">{m.user?.email}</td>
                      <td className="p-3.5">
                        <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded uppercase bg-stone-200 text-stone-800">
                          {m.role}
                        </span>
                      </td>
                      <td className="p-3.5 font-mono text-stone-500 text-[11px]">
                        {m.joined_at ? new Date(m.joined_at).toLocaleDateString() : 'Active'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* 10. SETTINGS SECTION */}
        {/* ============================================================== */}
        {activeSection === 'settings' && (
          <div className="space-y-6 max-w-3xl">
            <div>
              <h2 className="text-2xl font-serif font-black text-stone-950 tracking-tight">Workspace Settings</h2>
              <p className="text-xs text-stone-600">
                Organization domain verification, credentials, and API access keys.
              </p>
            </div>

            <div className="glass-panel rounded-2xl p-6 shadow-xs space-y-4">
              <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900 border-b border-[rgba(200,185,155,0.25)] pb-2">
                Organization Profile
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="font-semibold text-stone-600 block mb-1">Organization Name</span>
                  <p className="font-bold text-stone-900">{currentOrg?.name || 'N/A'}</p>
                </div>
                <div>
                  <span className="font-semibold text-stone-600 block mb-1">Domain</span>
                  <p className="font-mono text-stone-900">{currentOrg?.domain || 'N/A'}</p>
                </div>
                <div>
                  <span className="font-semibold text-stone-600 block mb-1">Verification Method</span>
                  <p className="font-mono text-stone-900 capitalize">
                    {currentOrg?.verification_method || 'Email Domain Verification'}
                  </p>
                </div>
                <div>
                  <span className="font-semibold text-stone-600 block mb-1">Verification Date</span>
                  <p className="font-mono text-stone-900">
                    {currentOrg?.verified_at ? new Date(currentOrg.verified_at).toLocaleDateString() : 'Verified'}
                  </p>
                </div>
              </div>
            </div>

            {/* API Keys Stub */}
            <div className="glass-panel rounded-2xl p-6 shadow-xs space-y-3">
              <h3 className="font-bold text-xs uppercase tracking-wider text-stone-900 border-b border-[rgba(200,185,155,0.25)] pb-2">
                Programmatic API Access Keys
              </h3>
              <p className="text-xs text-stone-600 leading-relaxed">
                Connect external SIEM, SOAR, or OSINT ingestion pipelines using high-speed API keys.
              </p>
              <div className="flex gap-2 pt-2">
                <input
                  type="text"
                  readOnly
                  value="fq_live_94f8372091bc2e84d940a1b82"
                  className="flex-1 p-2 rounded-xl glass-input text-xs font-mono text-stone-700 select-all"
                />
                <button
                  onClick={() => showNotice('API key copied.')}
                  className="px-4 py-2 rounded-xl text-xs font-bold text-white bg-stone-950 cursor-pointer"
                >
                  Copy Key
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ============================================================== */}
      {/* MODAL: NEW CASE */}
      {/* ============================================================== */}
      {isNewCaseModalOpen && (
        <div className="fixed inset-0 z-50 glass-modal-bg flex items-center justify-center p-4">
          <div className="glass-panel rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-4">
            <h3 className="text-base font-serif font-black text-stone-950">Initialize Investigation Case</h3>
            <form onSubmit={handleCreateCase} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-stone-800 mb-1">Case Title *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. CEO Deepfake Impersonation Wire Fraud"
                  value={newCaseForm.title}
                  onChange={(e) => setNewCaseForm({ ...newCaseForm, title: e.target.value })}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div>
                <label className="block font-semibold text-stone-800 mb-1">Description &amp; Objective</label>
                <textarea
                  rows={3}
                  placeholder="Details and background of target incident..."
                  value={newCaseForm.description}
                  onChange={(e) => setNewCaseForm({ ...newCaseForm, description: e.target.value })}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Severity</label>
                  <select
                    value={newCaseForm.severity}
                    onChange={(e) => setNewCaseForm({ ...newCaseForm, severity: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>

                <div>
                  <label className="block font-semibold text-stone-800 mb-1">Tags (comma-separated)</label>
                  <input
                    type="text"
                    placeholder="election, deepfake, audio"
                    value={newCaseForm.tags}
                    onChange={(e) => setNewCaseForm({ ...newCaseForm, tags: e.target.value })}
                    className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsNewCaseModalOpen(false)}
                  className="px-4 py-2 rounded-xl glass-card font-semibold text-stone-700 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl font-bold text-white bg-stone-950 cursor-pointer shadow-sm"
                >
                  Create Case
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* MODAL: ATTACH DETECTION */}
      {/* ============================================================== */}
      {isAttachModalOpen && (
        <div className="fixed inset-0 z-50 glass-modal-bg flex items-center justify-center p-4">
          <div className="glass-panel rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-4">
            <h3 className="text-base font-serif font-black text-stone-950">Attach Forensic Media to Case</h3>
            <form onSubmit={handleAttachDetection} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-stone-800 mb-1">Media File Upload</label>
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={(e) => setAttachFile(e.target.files?.[0] || null)}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div>
                <label className="block font-semibold text-stone-800 mb-1">Preliminary Verdict</label>
                <select
                  value={attachVerdict}
                  onChange={(e) => setAttachVerdict(e.target.value)}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                >
                  <option value="FAKE">FAKE (Deepfake Detected)</option>
                  <option value="REAL">REAL (Authentic Media)</option>
                  <option value="INCONCLUSIVE">INCONCLUSIVE (Under Forensic Review)</option>
                </select>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsAttachModalOpen(false)}
                  className="px-4 py-2 rounded-xl glass-card font-semibold text-stone-700 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={attachLoading}
                  className="px-5 py-2 rounded-xl font-bold text-white bg-stone-950 cursor-pointer shadow-sm disabled:opacity-50"
                >
                  {attachLoading ? 'Running Analysis...' : 'Attach to Findings'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* MODAL: FINDING DETAIL & CUSTODY CHAIN */}
      {/* ============================================================== */}
      {selectedFinding && custodyChain && (
        <div className="fixed inset-0 z-50 glass-modal-bg flex items-center justify-center p-4">
          <div className="glass-panel rounded-2xl p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl space-y-5">
            <div className="flex items-start justify-between border-b border-[rgba(200,185,155,0.25)] pb-3">
              <div>
                <h3 className="text-base font-serif font-black text-stone-950">
                  {selectedFinding.original_filename}
                </h3>
                <p className="text-xs text-stone-500 font-mono mt-0.5">
                  Finding ID: {selectedFinding.id} &bull; SHA-256: {selectedFinding.sha256}
                </p>
              </div>
              <button
                onClick={() => {
                  setSelectedFinding(null);
                  setCustodyChain(null);
                }}
                className="text-stone-500 hover:text-stone-950 text-base font-bold cursor-pointer"
              >
                &times;
              </button>
            </div>

            {/* Verdict and Score */}
            <div className="grid grid-cols-2 gap-4">
              <div className="glass-card rounded-xl p-3">
                <span className="text-[10px] font-mono text-stone-500 uppercase font-bold">Verdict</span>
                <p
                  className={`text-lg font-serif font-black ${
                    selectedFinding.verdict === 'FAKE' ? 'text-red-600' : 'text-emerald-600'
                  }`}
                >
                  {selectedFinding.verdict}
                </p>
              </div>
              <div className="glass-card rounded-xl p-3">
                <span className="text-[10px] font-mono text-stone-500 uppercase font-bold">Confidence Score</span>
                <p className="text-lg font-serif font-black text-stone-950">
                  {Math.round(selectedFinding.confidence * 100)}%
                </p>
              </div>
            </div>

            {/* Evidence Lock Toggle */}
            <div className="p-4 rounded-xl border border-[rgba(200,185,155,0.4)] glass-card flex items-center justify-between">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  {selectedFinding.frozen ? (
                    <Lock className="w-4 h-4 text-amber-700" />
                  ) : (
                    <Unlock className="w-4 h-4 text-stone-500" />
                  )}
                  <span className="font-bold text-xs text-stone-900">
                    {selectedFinding.frozen ? 'Finding Locked (Evidence Chain Intact)' : 'Evidence Unlocked'}
                  </span>
                </div>
                <p className="text-[11px] text-stone-600">
                  Freezing creates an immutable cryptographic hash chain entry suitable for court submission.
                </p>
              </div>
              <button
                onClick={() => handleToggleFreezeFinding(selectedFinding)}
                className={`px-3 py-1.5 rounded-xl text-xs font-bold cursor-pointer ${
                  selectedFinding.frozen ? 'bg-amber-100 text-amber-900' : 'bg-stone-950 text-white'
                }`}
              >
                {selectedFinding.frozen ? 'Unlock' : 'Freeze Finding'}
              </button>
            </div>

            {/* Custody Chain Table */}
            <div className="space-y-2">
              <h4 className="font-mono text-xs font-bold uppercase text-stone-700">Cryptographic Custody Chain</h4>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {custodyChain.locks.length === 0 ? (
                  <p className="text-xs text-stone-500">No cryptographic locks created yet.</p>
                ) : (
                  custodyChain.locks.map((lock) => (
                    <div key={lock.id} className="p-3 rounded-xl glass-card text-xs space-y-1">
                      <div className="flex items-center justify-between font-mono text-[10px] text-stone-500">
                        <span>Locked by: {lock.locked_by_name || 'Investigator'}</span>
                        <span>{lock.locked_at ? new Date(lock.locked_at).toLocaleString() : ''}</span>
                      </div>
                      <p className="text-stone-800 font-semibold">{lock.reason || 'Evidence freeze'}</p>
                      <div className="font-mono text-[10px] text-stone-500 break-all">
                        Chain Hash: {lock.chain_hash}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => {
                  setSelectedFinding(null);
                  setCustodyChain(null);
                }}
                className="px-4 py-2 rounded-xl text-xs font-bold bg-stone-900 text-white cursor-pointer"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================== */}
      {/* MODAL: NEW WATCHLIST */}
      {/* ============================================================== */}
      {isNewWatchlistOpen && (
        <div className="fixed inset-0 z-50 glass-modal-bg flex items-center justify-center p-4">
          <div className="glass-panel rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-4">
            <h3 className="text-base font-serif font-black text-stone-950">Add Monitoring Watchlist</h3>
            <form onSubmit={handleCreateWatchlist} className="space-y-4 text-xs">
              <div>
                <label className="block font-semibold text-stone-800 mb-1">Watchlist Name *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Executive Impersonation Scan"
                  value={newWatchlistForm.name}
                  onChange={(e) => setNewWatchlistForm({ ...newWatchlistForm, name: e.target.value })}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div>
                <label className="block font-semibold text-stone-800 mb-1">Target Query / Keywords *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. CEO voice, leak video, deepfake"
                  value={newWatchlistForm.target_query}
                  onChange={(e) => setNewWatchlistForm({ ...newWatchlistForm, target_query: e.target.value })}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div>
                <label className="block font-semibold text-stone-800 mb-1">Description</label>
                <textarea
                  rows={2}
                  placeholder="Monitoring scope notes..."
                  value={newWatchlistForm.description}
                  onChange={(e) => setNewWatchlistForm({ ...newWatchlistForm, description: e.target.value })}
                  className="w-full p-2 rounded-xl glass-input text-stone-900 focus:outline-none"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsNewWatchlistOpen(false)}
                  className="px-4 py-2 rounded-xl glass-card font-semibold text-stone-700 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl font-bold text-white bg-stone-950 cursor-pointer shadow-sm"
                >
                  Create Watchlist
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
