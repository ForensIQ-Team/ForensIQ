export type UserRole = 'normal' | 'investigator';

export type NavItem =
  | 'landing'
  | 'home'
  | 'check-media'
  | 'forensic-viewer'
  | 'protect-image'
  | 'find-misuse'
  | 'history'
  | 'professional-analysis'
  | 'report'
  | 'settings'
  | 'signup'
  | 'login'
  | 'profile'
  | 'investigator-workspace';

export type RiskLevel = 'low' | 'medium' | 'high';

export type ClassificationType = 'Real' | 'Deepfake' | 'AI-Generated' | 'Uncertain';

export type AuthenticityStatus =
  | 'likely-authentic'
  | 'uncertain'
  | 'likely-ai-generated'
  | 'tampered'
  | 'protected';

export interface DetailedModelScore {
  name: string; // 'Xception', 'EfficientNet-B4', 'Vision Transformer (ViT)', 'Fused Score'
  realScore: number; // 0 to 100
  fakeScore: number; // 0 to 100
  confidence: string; // 'High', 'Optimal', 'Low', etc.
}

export interface ModelScore {
  name: string;
  score: number; // 0 to 100
  confidence: string;
}

export interface MediaMetadata {
  filename: string;
  filesize: string;
  dimensions: string;
  filetype: string;
  cameraModel?: string;
  software?: string;
  modifiedDate: string;
  createdDate: string;
  colorSpace?: string;
  exifIntact: boolean;
  gpsData?: string;
  hashSHA256: string;
  pHash: string;
}

export interface ManipulationSignal {
  id: string;
  name: string;
  status: 'passed' | 'warning' | 'flagged';
  score: string;
  description: string;
  details?: string;
}

export interface PipelineStage {
  id: number;
  name: string;
  description: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  detail?: string;
}

export interface ExplainabilityRegion {
  label: string;
  description: string;
  bounds: { x: number; y: number; width: number; height: number }; // percentages 0-100
}

export interface ExplainabilityData {
  gradCamUrl: string;
  vitAttentionUrl: string;
  originalUrl: string;
  highlightedRegions: ExplainabilityRegion[];
}

export interface VideoFrameAnalysis {
  frameNumber: number;
  timestamp: string;
  fakeScore: number; // 0 to 100
  isSuspicious: boolean;
  anomalyType?: string;
  thumbnailUrl?: string;
}

export interface VideoAnalysisData {
  sampledFrames: number;
  facesDetected: number;
  framesAnalyzed: number;
  suspiciousFrameCount: number;
  frameResults: VideoFrameAnalysis[];
  fps: number;
  durationSeconds: number;
}

export interface DetectionResult {
  id: string;
  mediaName: string;
  mediaUrl: string;
  mediaType: 'image' | 'video';
  uploadDate: string;
  classification: ClassificationType;
  confidenceScore: number; // e.g. 92
  authenticityScore: number; // 0 to 100
  processingTimeMs: number; // e.g. 340
  facesDetected: number;
  modelAgreement: 'High Agreement' | 'Moderate Agreement' | 'Model Disagreement';
  modelAgreementDescription: string;
  riskLevel: RiskLevel;
  detailedModelScores: DetailedModelScore[];
  explainability: ExplainabilityData;
  metadata: MediaMetadata;
  manipulationSignals: ManipulationSignal[];
  videoData?: VideoAnalysisData;
  summaryText: string;
  warnings: string[];
  investigatorNotes?: string;

  // Backward compatibility fields for existing components:
  authenticityStatus: AuthenticityStatus;
  overallScore: number; // 0 to 100
  confidence: number; // 0 to 100
  modelScores: ModelScore[];
  heatmapUrl?: string;
  elaUrl?: string;
  whySignals: string[];
}

export interface ForensicAnalysis extends DetectionResult {}

export interface ProtectionRecord {
  id: string;
  registrationId: string;
  originalName: string;
  mediaUrl: string;
  protectedDate: string;
  protectionType: 'EOT Adversarial' | 'Robust Hybrid' | 'Metadata Provenance';
  watermarkSignal: 'Embedded' | 'Active';
  pHash: string;
  clipEmbeddingId: string;
  status: 'Registered & Protected';
  downloadUrl?: string;
}

export type MisuseClassification =
  | 'CONFIRMED MATCH'
  | 'PROBABLE MATCH'
  | 'TAMPERED'
  | 'AI MANIPULATED'
  | 'UNCERTAIN';

export interface MisuseMatch {
  id: string;
  sourceName: string;
  sourceUrl: string;
  foundDate: string;
  similarity: number; // e.g. 94%
  watermarkStatus: 'Detected' | 'Not recovered' | 'Partial';
  classification: MisuseClassification;
  thumbnailUrl: string;
  modifiedRegions?: string;
  risk: RiskLevel;
  firstSeen: string;
  platformCategory: string;
}

export interface PropagationNode {
  id: string;
  title: string;
  source: string;
  timestamp: string;
  similarity: number;
  classification: MisuseClassification;
  notes: string;
  type: 'original' | 'repost' | 'edited' | 'manipulated';
}

export interface IncidentRecord {
  id: string;
  incidentNumber: string;
  title: string;
  targetMediaName: string;
  targetMediaUrl: string;
  createdDate: string;
  matchesCount: number;
  status: 'Active Investigation' | 'Evidence Archived' | 'Report Generated' | 'Pending Review';
  matches: MisuseMatch[];
  notes?: string;
}

export interface ForensicReportData {
  reportId: string;
  title: string;
  createdDate: string;
  lastUpdated: string;
  author: string;
  authorRole: string;
  mediaName: string;
  mediaUrl: string;
  status: 'Draft' | 'Final / Verified';
  summary: string;
  detectionFindings: string[];
  visualFindings: string;
  metadataFindings: string;
  similarityFindings: string;
  propagationFindings: string;
  auditHash: string;
}

export interface HistoryItem {
  id: string;
  mediaName: string;
  mediaUrl: string;
  action: 'Check Media' | 'Protect Image' | 'Find Misuse' | 'Forensic Analysis' | 'Report Generated';
  resultSummary: string;
  date: string;
  status: 'Completed' | 'Processing' | 'Flagged';
  risk?: RiskLevel;
  detectionResult?: DetectionResult;
}

// ----------------------------------------------------------------
// Auth & Investigator Workspace Types
// ----------------------------------------------------------------

export interface AuthUser {
  id: string;
  email: string;
  display_name?: string;
  role: 'user' | 'investigator' | 'admin';
  tier: number;
  org_id?: string;
  is_active: boolean;
  created_at?: string;
  last_login_at?: string;
}

export interface OrganizationInfo {
  id: string;
  name: string;
  domain?: string;
  website?: string;
  verified_at?: string;
  verification_method?: string;
  created_at?: string;
}

export interface InvestigatorAppInfo {
  id: string;
  user_id: string;
  organization_name?: string;
  designation?: string;
  work_email?: string;
  org_website?: string;
  linkedin_url?: string;
  reason?: string;
  status: 'pending' | 'approved' | 'rejected';
  created_at?: string;
}

export interface InvestigatorCase {
  id: string;
  org_id: string;
  created_by: string;
  assigned_to?: string;
  title: string;
  description?: string;
  status: 'open' | 'in_progress' | 'closed' | 'escalated' | 'archived';
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  closed_at?: string;
  created_at?: string;
  updated_at?: string;
  creator_name?: string;
  assignee_name?: string;
  findings_count: number;
  notes_count: number;
}

export interface CaseFindingItem {
  id: string;
  case_id: string;
  uploaded_by: string;
  media_type: 'image' | 'video';
  original_filename: string;
  storage_path: string;
  sha256: string;
  file_size_bytes: number;
  verdict: 'FAKE' | 'REAL' | 'INCONCLUSIVE';
  confidence: number;
  analysis_json: any;
  gradcam_path?: string;
  report_id?: string;
  frozen: boolean;
  frozen_by?: string;
  frozen_at?: string;
  created_at?: string;
}

export interface CaseTimelineItem {
  id: number;
  case_id: string;
  actor_id?: string;
  action: string;
  payload: Record<string, any>;
  created_at?: string;
  actor_name?: string;
}

export interface CaseNoteItem {
  id: string;
  case_id: string;
  author_id: string;
  text: string;
  created_at?: string;
  author_name?: string;
}

export interface EvidenceLockItem {
  id: string;
  finding_id: string;
  locked_by: string;
  reason?: string;
  content_hash: string;
  chain_hash: string;
  locked_at?: string;
  locked_by_name?: string;
}

export interface WatchlistItem {
  id: string;
  org_id: string;
  created_by: string;
  name: string;
  description?: string;
  target_type: string;
  target_query: string;
  active: boolean;
  created_at?: string;
  alerts_count?: number;
}

export interface WatchlistAlertItem {
  id: string;
  watchlist_id: string;
  watchlist_name?: string;
  source_url: string;
  source_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  match_score?: number;
  snapshot_path?: string;
  status: 'new' | 'triaged' | 'dismissed' | 'escalated';
  triaged_by?: string;
  triaged_at?: string;
  case_id?: string;
  created_at?: string;
}

export interface SourceReputationItem {
  id: string;
  org_id: string;
  domain: string;
  total_findings: number;
  fake_findings: number;
  fake_ratio: number;
  last_seen_at?: string;
  notes?: string;
}

export interface AuditLogItem {
  id: number;
  user_id?: string;
  org_id?: string;
  action: string;
  target_type?: string;
  target_id?: string;
  metadata?: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  created_at?: string;
  user_name?: string;
}

export interface OrgMemberItem {
  org_id: string;
  user_id: string;
  role: 'owner' | 'admin' | 'analyst' | 'viewer';
  joined_at?: string;
  user?: AuthUser;
}

export interface DashboardStatsData {
  total_cases: number;
  open_cases: number;
  findings_month: number;
  fake_findings_month: number;
  real_findings_month: number;
  unread_alerts: number;
}

