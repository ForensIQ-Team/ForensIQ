import {
  InvestigatorCase,
  CaseFindingItem,
  CaseTimelineItem,
  CaseNoteItem,
  EvidenceLockItem,
  WatchlistItem,
  WatchlistAlertItem,
  SourceReputationItem,
  AuditLogItem,
  OrgMemberItem,
  DashboardStatsData,
} from '../types';

const getAuthHeaders = () => {
  const token = localStorage.getItem('forensiq_token');
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
};

export const investigatorService = {
  // ---------------- Dashboard ----------------
  async getDashboard() {
    const res = await fetch('/api/investigator/dashboard', { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch dashboard');
    return res.json();
  },

  // ---------------- Cases ----------------
  async listCases(params?: { status?: string; severity?: string; tag?: string }) {
    const query = new URLSearchParams();
    if (params?.status && params.status !== 'all') query.append('status', params.status);
    if (params?.severity && params.severity !== 'all') query.append('severity', params.severity);
    if (params?.tag) query.append('tag', params.tag);

    const res = await fetch(`/api/investigator/cases?${query.toString()}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch cases');
    return res.json() as Promise<{ cases: InvestigatorCase[]; total: number }>;
  },

  async createCase(data: { title: string; description?: string; severity?: string; tags?: string[] }) {
    const res = await fetch('/api/investigator/cases', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create case');
    return res.json() as Promise<InvestigatorCase>;
  },

  async getCase(id: string) {
    const res = await fetch(`/api/investigator/cases/${id}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch case');
    return res.json() as Promise<InvestigatorCase>;
  },

  async updateCase(id: string, data: Partial<InvestigatorCase>) {
    const res = await fetch(`/api/investigator/cases/${id}`, {
      method: 'PATCH',
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to update case');
    return res.json() as Promise<InvestigatorCase>;
  },

  async deleteCase(id: string) {
    const res = await fetch(`/api/investigator/cases/${id}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error('Failed to delete case');
    return res.json();
  },

  async updateCaseStatus(id: string, status: string) {
    const res = await fetch(`/api/investigator/cases/${id}/status`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error('Failed to update status');
    return res.json() as Promise<InvestigatorCase>;
  },

  async assignCase(id: string, assignedTo?: string) {
    const res = await fetch(`/api/investigator/cases/${id}/assign`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ assigned_to: assignedTo || null }),
    });
    if (!res.ok) throw new Error('Failed to assign case');
    return res.json() as Promise<InvestigatorCase>;
  },

  async getCaseTimeline(id: string) {
    const res = await fetch(`/api/investigator/cases/${id}/timeline`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch timeline');
    return res.json() as Promise<CaseTimelineItem[]>;
  },

  async getCaseNotes(id: string) {
    const res = await fetch(`/api/investigator/cases/${id}/notes`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch notes');
    return res.json() as Promise<CaseNoteItem[]>;
  },

  async addCaseNote(id: string, text: string) {
    const res = await fetch(`/api/investigator/cases/${id}/notes`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error('Failed to add note');
    return res.json() as Promise<CaseNoteItem>;
  },

  async deleteCaseNote(caseId: string, noteId: string) {
    const res = await fetch(`/api/investigator/cases/${caseId}/notes/${noteId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error('Failed to delete note');
    return res.json();
  },

  // ---------------- Findings ----------------
  async getCaseFindings(caseId: string) {
    const res = await fetch(`/api/investigator/cases/${caseId}/findings`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch findings');
    return res.json() as Promise<CaseFindingItem[]>;
  },

  async getAllFindings(params?: { verdict?: string; case_id?: string }) {
    const query = new URLSearchParams();
    if (params?.verdict && params.verdict !== 'all') query.append('verdict', params.verdict);
    if (params?.case_id) query.append('case_id', params.case_id);

    const res = await fetch(`/api/investigator/findings?${query.toString()}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch findings');
    return res.json() as Promise<CaseFindingItem[]>;
  },

  async getFinding(id: string) {
    const res = await fetch(`/api/investigator/findings/${id}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch finding');
    return res.json() as Promise<CaseFindingItem>;
  },

  async attachDetection(caseId: string, payload: {
    file?: File;
    detectionJson?: string;
    verdict?: string;
    confidence?: number;
    reportId?: string;
  }) {
    const token = localStorage.getItem('forensiq_token');
    const formData = new FormData();
    if (payload.file) formData.append('file', payload.file);
    if (payload.detectionJson) formData.append('detection_json', payload.detectionJson);
    if (payload.verdict) formData.append('verdict', payload.verdict);
    if (payload.confidence !== undefined) formData.append('confidence', String(payload.confidence));
    if (payload.reportId) formData.append('report_id', payload.reportId);

    const res = await fetch(`/api/investigator/cases/${caseId}/findings/attach-detection`, {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to attach detection');
    return res.json() as Promise<{ finding: CaseFindingItem; message: string }>;
  },

  async freezeFinding(id: string, reason?: string) {
    const res = await fetch(`/api/investigator/findings/${id}/freeze`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ reason }),
    });
    if (!res.ok) throw new Error('Failed to freeze finding');
    return res.json() as Promise<CaseFindingItem>;
  },

  async unfreezeFinding(id: string, reason?: string) {
    const res = await fetch(`/api/investigator/findings/${id}/unfreeze`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ reason }),
    });
    if (!res.ok) throw new Error('Failed to unfreeze finding');
    return res.json() as Promise<CaseFindingItem>;
  },

  // ---------------- Evidence & Custody ----------------
  async getFindingCustody(findingId: string) {
    const res = await fetch(`/api/evidence/findings/${findingId}/custody`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch custody chain');
    return res.json() as Promise<{
      finding_id: string;
      sha256: string;
      frozen: boolean;
      frozen_at?: string;
      locks: EvidenceLockItem[];
      timeline_entries: CaseTimelineItem[];
    }>;
  },

  // ---------------- Export ----------------
  async exportCase(caseId: string, exportType: string = 'html') {
    const res = await fetch(`/api/investigator/cases/${caseId}/export`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ export_type: exportType }),
    });
    if (!res.ok) throw new Error('Failed to export case');
    return res.json() as Promise<{
      id: string;
      case_id: string;
      export_type: string;
      storage_path: string;
      sha256: string;
      download_url: string;
      created_at: string;
    }>;
  },

  // ---------------- Monitoring & Alerts ----------------
  async listWatchlists() {
    const res = await fetch('/api/monitoring/watchlists', { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch watchlists');
    return res.json() as Promise<WatchlistItem[]>;
  },

  async createWatchlist(data: { name: string; description?: string; target_type: string; target_query: string }) {
    const res = await fetch('/api/monitoring/watchlists', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error('Failed to create watchlist');
    return res.json() as Promise<WatchlistItem>;
  },

  async deleteWatchlist(id: string) {
    const res = await fetch(`/api/monitoring/watchlists/${id}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error('Failed to delete watchlist');
    return res.json();
  },

  async runWatchlistNow(id: string) {
    const res = await fetch(`/api/monitoring/watchlists/${id}/run-now`, {
      method: 'POST',
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error('Failed to trigger scan');
    return res.json() as Promise<WatchlistAlertItem>;
  },

  async listAlerts(status?: string) {
    const query = status && status !== 'all' ? `?status=${status}` : '';
    const res = await fetch(`/api/monitoring/alerts${query}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch alerts');
    return res.json() as Promise<WatchlistAlertItem[]>;
  },

  async triageAlert(id: string, status: string) {
    const res = await fetch(`/api/monitoring/alerts/${id}/triage`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ status }),
    });
    if (!res.ok) throw new Error('Failed to triage alert');
    return res.json() as Promise<WatchlistAlertItem>;
  },

  async escalateAlert(id: string, options?: { title?: string; description?: string }) {
    const res = await fetch(`/api/monitoring/alerts/${id}/escalate`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(options || {}),
    });
    if (!res.ok) throw new Error('Failed to escalate alert');
    return res.json() as Promise<{ status: string; case_id: string; alert_id: string }>;
  },

  async listSources() {
    const res = await fetch('/api/monitoring/sources', { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch sources');
    return res.json() as Promise<SourceReputationItem[]>;
  },

  // ---------------- Team & Audit ----------------
  async listTeamMembers() {
    const res = await fetch('/api/team/members', { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch team members');
    return res.json() as Promise<OrgMemberItem[]>;
  },

  async inviteMember(email: string, role: string = 'analyst') {
    const res = await fetch('/api/team/invite', {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ email, role }),
    });
    if (!res.ok) throw new Error('Failed to invite member');
    return res.json() as Promise<{ status: string; invite_link: string; message: string }>;
  },

  async updateMemberRole(userId: string, role: string) {
    const res = await fetch(`/api/team/members/${userId}`, {
      method: 'PATCH',
      headers: getAuthHeaders(),
      body: JSON.stringify({ role }),
    });
    if (!res.ok) throw new Error('Failed to update member role');
    return res.json() as Promise<OrgMemberItem>;
  },

  async removeMember(userId: string) {
    const res = await fetch(`/api/team/members/${userId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error('Failed to remove member');
    return res.json();
  },

  async getAuditLog(action?: string) {
    const query = action && action !== 'all' ? `?action=${action}` : '';
    const res = await fetch(`/api/team/audit${query}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch audit log');
    return res.json() as Promise<AuditLogItem[]>;
  },

  // ---------------- Analytics ----------------
  async getAnalyticsOverview() {
    const res = await fetch('/api/analytics/org/overview', { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch analytics overview');
    return res.json();
  },

  async getAnalyticsTimeline(days: number = 14) {
    const res = await fetch(`/api/analytics/org/timeline?days=${days}`, { headers: getAuthHeaders() });
    if (!res.ok) throw new Error('Failed to fetch timeline');
    return res.json() as Promise<{ days: number; points: { date: string; fake: number; real: number; total: number }[] }>;
  },
};
