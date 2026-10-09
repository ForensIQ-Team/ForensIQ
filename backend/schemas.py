import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# -------------------------------------------------------------
# Base Schema
# -------------------------------------------------------------
class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# -------------------------------------------------------------
# User Schemas
# -------------------------------------------------------------
class UserCreate(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(ORMModel):
    id: uuid.UUID
    email: str
    display_name: Optional[str] = None
    role: str
    tier: int
    org_id: Optional[uuid.UUID] = None
    is_active: bool
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# -------------------------------------------------------------
# Organization Schemas
# -------------------------------------------------------------
class OrgResponse(ORMModel):
    id: uuid.UUID
    name: str
    domain: Optional[str] = None
    website: Optional[str] = None
    verified_at: Optional[datetime] = None
    verification_method: Optional[str] = None
    created_at: Optional[datetime] = None


class OrgMemberResponse(ORMModel):
    org_id: uuid.UUID
    user_id: uuid.UUID
    role: str
    joined_at: Optional[datetime] = None
    user: Optional[UserResponse] = None


class InviteMember(BaseModel):
    email: str
    role: str = "analyst"


class UpdateMemberRole(BaseModel):
    role: str


# -------------------------------------------------------------
# Investigator Application Schemas
# -------------------------------------------------------------
class InvestigatorApply(BaseModel):
    organization_name: Optional[str] = None
    designation: Optional[str] = None
    work_email: str
    org_website: Optional[str] = None
    linkedin_url: Optional[str] = None
    reason: Optional[str] = None
    reference_email: Optional[str] = None


class InvestigatorApplicationResponse(ORMModel):
    id: uuid.UUID
    user_id: uuid.UUID
    organization_name: Optional[str] = None
    designation: Optional[str] = None
    work_email: Optional[str] = None
    org_website: Optional[str] = None
    linkedin_url: Optional[str] = None
    reason: Optional[str] = None
    reference_email: Optional[str] = None
    status: str
    reviewed_by: Optional[uuid.UUID] = None
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None
    created_at: Optional[datetime] = None


class InvestigatorStatus(BaseModel):
    role: str
    tier: int
    status: str
    organization: Optional[OrgResponse] = None
    application: Optional[InvestigatorApplicationResponse] = None


# -------------------------------------------------------------
# Case Schemas
# -------------------------------------------------------------
class CaseCreate(BaseModel):
    title: str
    description: Optional[str] = None
    severity: str = "medium"
    tags: Optional[List[str]] = Field(default_factory=list)
    assigned_to: Optional[uuid.UUID] = None


class CaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    tags: Optional[List[str]] = None
    assigned_to: Optional[uuid.UUID] = None


class AssignCaseRequest(BaseModel):
    assigned_to: Optional[uuid.UUID] = None


class ChangeStatusRequest(BaseModel):
    status: str


class CaseResponse(ORMModel):
    id: uuid.UUID
    org_id: uuid.UUID
    created_by: uuid.UUID
    assigned_to: Optional[uuid.UUID] = None
    title: str
    description: Optional[str] = None
    status: str
    severity: str
    tags: Optional[List[str]] = Field(default_factory=list)
    closed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    creator_name: Optional[str] = None
    assignee_name: Optional[str] = None
    findings_count: Optional[int] = 0
    notes_count: Optional[int] = 0


class CaseListResponse(BaseModel):
    cases: List[CaseResponse]
    total: int


# -------------------------------------------------------------
# Finding Schemas
# -------------------------------------------------------------
class FindingResponse(ORMModel):
    id: uuid.UUID
    case_id: uuid.UUID
    uploaded_by: uuid.UUID
    media_type: str
    original_filename: str
    storage_path: str
    sha256: str
    file_size_bytes: int
    verdict: str
    confidence: float
    analysis_json: Dict[str, Any]
    gradcam_path: Optional[str] = None
    report_id: Optional[str] = None
    frozen: bool = False
    frozen_by: Optional[uuid.UUID] = None
    frozen_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class AttachDetectionResponse(BaseModel):
    finding: FindingResponse
    message: str


# -------------------------------------------------------------
# Timeline & Note Schemas
# -------------------------------------------------------------
class TimelineEntryResponse(ORMModel):
    id: int
    case_id: uuid.UUID
    actor_id: Optional[uuid.UUID] = None
    action: str
    payload: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    actor_name: Optional[str] = None


class NoteCreate(BaseModel):
    text: str


class NoteResponse(ORMModel):
    id: uuid.UUID
    case_id: uuid.UUID
    author_id: uuid.UUID
    text: str
    created_at: Optional[datetime] = None
    author_name: Optional[str] = None


# -------------------------------------------------------------
# Evidence & Custody Schemas
# -------------------------------------------------------------
class LockFindingRequest(BaseModel):
    reason: Optional[str] = None


class EvidenceLockResponse(ORMModel):
    id: uuid.UUID
    finding_id: uuid.UUID
    locked_by: uuid.UUID
    reason: Optional[str] = None
    content_hash: str
    chain_hash: str
    locked_at: Optional[datetime] = None
    locked_by_name: Optional[str] = None


class CustodyChainResponse(BaseModel):
    finding_id: uuid.UUID
    sha256: str
    frozen: bool
    frozen_at: Optional[datetime] = None
    locks: List[EvidenceLockResponse]
    timeline_entries: List[TimelineEntryResponse]


# -------------------------------------------------------------
# Report Export Schemas
# -------------------------------------------------------------
class ExportCaseRequest(BaseModel):
    export_type: str = "html"
    finding_id: Optional[uuid.UUID] = None


class ReportExportResponse(ORMModel):
    id: uuid.UUID
    case_id: uuid.UUID
    finding_id: Optional[uuid.UUID] = None
    exported_by: uuid.UUID
    export_type: str
    storage_path: str
    sha256: str
    signed_hash: Optional[str] = None
    download_url: Optional[str] = None
    created_at: Optional[datetime] = None


# -------------------------------------------------------------
# Monitoring & Alerts Schemas
# -------------------------------------------------------------
class WatchlistCreate(BaseModel):
    name: str
    description: Optional[str] = None
    target_type: str = "custom"
    target_query: str
    active: bool = True


class WatchlistUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_type: Optional[str] = None
    target_query: Optional[str] = None
    active: Optional[bool] = None


class WatchlistResponse(ORMModel):
    id: uuid.UUID
    org_id: uuid.UUID
    created_by: uuid.UUID
    name: str
    description: Optional[str] = None
    target_type: str
    target_query: str
    active: bool
    created_at: Optional[datetime] = None
    alerts_count: Optional[int] = 0


class AlertResponse(ORMModel):
    id: uuid.UUID
    watchlist_id: uuid.UUID
    watchlist_name: Optional[str] = None
    source_url: str
    source_type: str
    severity: str
    match_score: Optional[float] = None
    snapshot_path: Optional[str] = None
    status: str
    triaged_by: Optional[uuid.UUID] = None
    triaged_at: Optional[datetime] = None
    case_id: Optional[uuid.UUID] = None
    created_at: Optional[datetime] = None


class AlertTriageRequest(BaseModel):
    status: str  # triaged | dismissed | new


class AlertEscalateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = "high"


# -------------------------------------------------------------
# Source Reputation Schemas
# -------------------------------------------------------------
class SourceReputationResponse(ORMModel):
    id: uuid.UUID
    org_id: uuid.UUID
    domain: str
    total_findings: int
    fake_findings: int
    fake_ratio: float
    last_seen_at: Optional[datetime] = None
    notes: Optional[str] = None


# -------------------------------------------------------------
# Audit Log Schemas
# -------------------------------------------------------------
class AuditLogResponse(ORMModel):
    id: int
    user_id: Optional[uuid.UUID] = None
    org_id: Optional[uuid.UUID] = None
    action: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: Optional[datetime] = None
    user_name: Optional[str] = None


# -------------------------------------------------------------
# Analytics Schemas
# -------------------------------------------------------------
class DashboardStats(BaseModel):
    total_cases: int
    open_cases: int
    findings_month: int
    fake_findings_month: int
    real_findings_month: int
    unread_alerts: int


class CaseStats(BaseModel):
    by_status: Dict[str, int]
    by_severity: Dict[str, int]
    total: int


class FindingStats(BaseModel):
    fake_count: int
    real_count: int
    inconclusive_count: int
    total: int


class OrgOverviewResponse(BaseModel):
    stats: DashboardStats
    recent_cases: List[CaseResponse]
    recent_findings: List[FindingResponse]
    recent_alerts: List[AlertResponse]
