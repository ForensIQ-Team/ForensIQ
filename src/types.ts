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
  | 'settings';

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
