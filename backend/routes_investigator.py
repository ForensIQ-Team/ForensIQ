import os
import uuid
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.database import get_db
from backend.models import (
    User,
    Organization,
    Case,
    CaseFinding,
    CaseTimeline,
    CaseNote,
    EvidenceLock,
    ReportExport,
    WatchlistAlert,
    SourceReputation,
)
from backend.deps import get_current_user, require_investigator, get_current_org
from backend.schemas import (
    CaseCreate,
    CaseUpdate,
    CaseResponse,
    CaseListResponse,
    AssignCaseRequest,
    ChangeStatusRequest,
    FindingResponse,
    AttachDetectionResponse,
    TimelineEntryResponse,
    NoteCreate,
    NoteResponse,
    LockFindingRequest,
    EvidenceLockResponse,
    ReportExportResponse,
    ExportCaseRequest,
    DashboardStats,
    CaseStats,
    FindingStats,
    SourceReputationResponse,
)
from backend import crud
from backend.utils.storage import save_org_finding, save_export, sha256_of_file
from backend.utils.hashing import sha256_of_bytes
from backend.utils.audit import log_audit

router = APIRouter()

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE_ROOT / "reports"


def _format_case(case: Case, db: Session) -> CaseResponse:
    findings_count = db.query(CaseFinding).filter(CaseFinding.case_id == case.id).count()
    notes_count = db.query(CaseNote).filter(CaseNote.case_id == case.id).count()
    creator = db.query(User).filter(User.id == case.created_by).first()
    assignee = db.query(User).filter(User.id == case.assigned_to).first() if case.assigned_to else None

    return CaseResponse(
        id=case.id,
        org_id=case.org_id,
        created_by=case.created_by,
        assigned_to=case.assigned_to,
        title=case.title,
        description=case.description,
        status=case.status,
        severity=case.severity,
        tags=case.tags or [],
        closed_at=case.closed_at,
        created_at=case.created_at,
        updated_at=case.updated_at,
        creator_name=creator.display_name if creator else None,
        assignee_name=assignee.display_name if assignee else None,
        findings_count=findings_count,
        notes_count=notes_count,
    )


# =============================================================
# Dashboard
# =============================================================
@router.get("/dashboard")
def get_dashboard(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    total_cases = db.query(Case).filter(Case.org_id == org.id).count()
    open_cases = db.query(Case).filter(Case.org_id == org.id, Case.status.in_(["open", "in_progress"])).count()

    # Findings this month
    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    month_findings = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(Case.org_id == org.id, CaseFinding.created_at >= month_start)
        .all()
    )
    findings_month = len(month_findings)
    fake_findings_month = sum(1 for f in month_findings if f.verdict == "FAKE")
    real_findings_month = sum(1 for f in month_findings if f.verdict == "REAL")

    unread_alerts = (
        db.query(WatchlistAlert)
        .join(crud.Watchlist, WatchlistAlert.watchlist_id == crud.Watchlist.id)
        .filter(crud.Watchlist.org_id == org.id, WatchlistAlert.status == "new")
        .count()
    )

    stats = DashboardStats(
        total_cases=total_cases,
        open_cases=open_cases,
        findings_month=findings_month,
        fake_findings_month=fake_findings_month,
        real_findings_month=real_findings_month,
        unread_alerts=unread_alerts,
    )

    recent_cases = [
        _format_case(c, db)
        for c in db.query(Case).filter(Case.org_id == org.id).order_by(desc(Case.updated_at)).limit(5).all()
    ]

    recent_findings = [
        FindingResponse.model_validate(f)
        for f in (
            db.query(CaseFinding)
            .join(Case, CaseFinding.case_id == Case.id)
            .filter(Case.org_id == org.id)
            .order_by(desc(CaseFinding.created_at))
            .limit(5)
            .all()
        )
    ]

    recent_alerts = [
        crud.list_alerts(db, org_id=org.id, limit=5)
    ][0]

    return {
        "stats": stats,
        "recent_cases": recent_cases,
        "recent_findings": recent_findings,
        "recent_alerts": recent_alerts,
    }


# =============================================================
# Cases
# =============================================================
@router.post("/cases", response_model=CaseResponse)
def create_case(
    data: CaseCreate,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.create_case(
        db=db,
        org_id=org.id,
        created_by=user.id,
        title=data.title,
        description=data.description,
        severity=data.severity,
        tags=data.tags,
        assigned_to=data.assigned_to or user.id,
    )

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="case_create",
        target_type="case",
        target_id=str(case.id),
        metadata={"title": case.title, "severity": case.severity},
        ip=client_ip,
        user_agent=user_agent,
    )

    return _format_case(case, db)


@router.get("/cases", response_model=CaseListResponse)
def get_cases(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    cases = crud.list_cases(
        db=db,
        org_id=org.id,
        status=status,
        severity=severity,
        tag=tag,
        limit=limit,
        offset=offset,
    )
    total = db.query(Case).filter(Case.org_id == org.id).count()
    return CaseListResponse(
        cases=[_format_case(c, db) for c in cases],
        total=total,
    )


@router.get("/cases/{case_id}", response_model=CaseResponse)
def get_case_detail(
    case_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return _format_case(case, db)


@router.patch("/cases/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: uuid.UUID,
    data: CaseUpdate,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    updated = crud.update_case(
        db=db,
        case=case,
        actor_id=user.id,
        title=data.title,
        description=data.description,
        status=data.status,
        severity=data.severity,
        tags=data.tags,
        assigned_to=data.assigned_to,
    )

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="case_update",
        target_type="case",
        target_id=str(case.id),
        metadata={"changes": data.model_dump(exclude_unset=True)},
        ip=client_ip,
        user_agent=user_agent,
    )

    return _format_case(updated, db)


@router.delete("/cases/{case_id}")
def delete_case(
    case_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    crud.soft_delete_case(db=db, case=case, actor_id=user.id)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="case_archive",
        target_type="case",
        target_id=str(case.id),
        ip=client_ip,
        user_agent=user_agent,
    )
    return {"status": "archived", "case_id": str(case_id)}


@router.post("/cases/{case_id}/assign", response_model=CaseResponse)
def assign_case(
    case_id: uuid.UUID,
    data: AssignCaseRequest,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    updated = crud.assign_case(db=db, case=case, actor_id=user.id, assigned_to=data.assigned_to)
    return _format_case(updated, db)


@router.post("/cases/{case_id}/status", response_model=CaseResponse)
def change_case_status(
    case_id: uuid.UUID,
    data: ChangeStatusRequest,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    updated = crud.change_status(db=db, case=case, actor_id=user.id, new_status=data.status)
    return _format_case(updated, db)


@router.get("/cases/{case_id}/timeline", response_model=List[TimelineEntryResponse])
def get_case_timeline(
    case_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    timeline = crud.list_timeline(db=db, case_id=case_id)
    results = []
    for item in timeline:
        actor = db.query(User).filter(User.id == item.actor_id).first() if item.actor_id else None
        results.append(TimelineEntryResponse(
            id=item.id,
            case_id=item.case_id,
            actor_id=item.actor_id,
            action=item.action,
            payload=item.payload or {},
            created_at=item.created_at,
            actor_name=actor.display_name if actor else "System",
        ))
    return results


@router.post("/cases/{case_id}/notes", response_model=NoteResponse)
def add_case_note(
    case_id: uuid.UUID,
    data: NoteCreate,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    note = crud.add_note(db=db, case_id=case_id, author_id=user.id, text=data.text)
    return NoteResponse(
        id=note.id,
        case_id=note.case_id,
        author_id=note.author_id,
        text=note.text,
        created_at=note.created_at,
        author_name=user.display_name,
    )


@router.get("/cases/{case_id}/notes", response_model=List[NoteResponse])
def get_case_notes(
    case_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    notes = crud.list_notes(db=db, case_id=case_id)
    results = []
    for n in notes:
        author = db.query(User).filter(User.id == n.author_id).first()
        results.append(NoteResponse(
            id=n.id,
            case_id=n.case_id,
            author_id=n.author_id,
            text=n.text,
            created_at=n.created_at,
            author_name=author.display_name if author else None,
        ))
    return results


@router.delete("/cases/{case_id}/notes/{note_id}")
def delete_case_note(
    case_id: uuid.UUID,
    note_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    deleted = crud.delete_note(db=db, note_id=note_id, user_id=user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found")
    return {"status": "deleted", "note_id": str(note_id)}


# =============================================================
# Findings
# =============================================================
@router.post("/cases/{case_id}/findings/attach-detection", response_model=AttachDetectionResponse)
async def attach_detection_to_case(
    case_id: uuid.UUID,
    file: Optional[UploadFile] = File(None),
    detection_json: Optional[str] = Form(None),
    verdict: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    report_id: Optional[str] = Form(None),
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    finding_id = uuid.uuid4()
    analysis_data = {}
    if detection_json:
        try:
            analysis_data = json.loads(detection_json)
        except Exception:
            analysis_data = {"raw": detection_json}

    final_verdict = verdict or analysis_data.get("classification") or analysis_data.get("verdict") or "INCONCLUSIVE"
    final_verdict = final_verdict.upper()
    if "DEEPFAKE" in final_verdict or "FAKE" in final_verdict or "TAMPERED" in final_verdict:
        final_verdict = "FAKE"
    elif "REAL" in final_verdict or "AUTHENTIC" in final_verdict:
        final_verdict = "REAL"
    else:
        final_verdict = "INCONCLUSIVE"

    final_conf = confidence
    if final_conf is None:
        raw_conf = analysis_data.get("confidenceScore") or analysis_data.get("confidence") or 0.85
        final_conf = float(raw_conf) / 100.0 if float(raw_conf) > 1.0 else float(raw_conf)

    original_filename = "attached_media"
    file_bytes = b""
    media_type = "image"

    if file:
        original_filename = file.filename or "media"
        file_bytes = await file.read()
        media_type = "video" if file.content_type and "video" in file.content_type else "image"

        # If analysis_data is empty, we can run the pipeline directly if available
        if not analysis_data:
            try:
                import predict_v2_facecrop as pipeline
                suffix = Path(original_filename).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = Path(tmp.name)
                try:
                    if media_type == "image":
                        pipeline_res = pipeline.analyze_image(tmp_path, save_gradcam=False)
                        final_verdict = "FAKE" if pipeline_res.get("verdict") == "FAKE" else "REAL"
                        final_conf = float(pipeline_res.get("confidence", 0.85))
                        analysis_data = pipeline_res
                finally:
                    if tmp_path.exists():
                        os.unlink(tmp_path)
            except Exception as e:
                # Fallback to defaults
                pass
    else:
        file_bytes = detection_json.encode() if detection_json else b"{}"

    sha256_hash = sha256_of_bytes(file_bytes)
    storage_path = str(save_org_finding(org.id, case.id, finding_id, file_bytes, original_filename))

    finding = crud.create_finding(
        db=db,
        case_id=case.id,
        uploaded_by=user.id,
        media_type=media_type,
        original_filename=original_filename,
        storage_path=storage_path,
        sha256=sha256_hash,
        file_size_bytes=len(file_bytes),
        verdict=final_verdict,
        confidence=final_conf,
        analysis_json=analysis_data,
        report_id=report_id or analysis_data.get("id"),
    )

    # Upsert source reputation if domain exists in analysis
    domain = analysis_data.get("sourceDomain") or "direct-upload.internal"
    crud.upsert_source_reputation(db=db, org_id=org.id, domain=domain, is_fake=(final_verdict == "FAKE"))

    return AttachDetectionResponse(
        finding=FindingResponse.model_validate(finding),
        message="Detection attached successfully to case.",
    )


@router.get("/cases/{case_id}/findings", response_model=List[FindingResponse])
def get_case_findings(
    case_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    findings = crud.list_findings_by_case(db=db, case_id=case_id)
    return [FindingResponse.model_validate(f) for f in findings]


@router.get("/findings", response_model=List[FindingResponse])
def get_all_findings(
    verdict: Optional[str] = None,
    case_id: Optional[uuid.UUID] = None,
    limit: int = 50,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    query = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(Case.org_id == org.id)
    )
    if verdict and verdict != "all":
        query = query.filter(CaseFinding.verdict == verdict.upper())
    if case_id:
        query = query.filter(CaseFinding.case_id == case_id)
    findings = query.order_by(desc(CaseFinding.created_at)).limit(limit).all()
    return [FindingResponse.model_validate(f) for f in findings]


@router.get("/findings/{finding_id}", response_model=FindingResponse)
def get_finding_detail(
    finding_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    finding = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(CaseFinding.id == finding_id, Case.org_id == org.id)
        .first()
    )
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")
    return FindingResponse.model_validate(finding)


@router.post("/findings/{finding_id}/freeze", response_model=FindingResponse)
def freeze_finding(
    finding_id: uuid.UUID,
    data: Optional[LockFindingRequest] = None,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    finding = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(CaseFinding.id == finding_id, Case.org_id == org.id)
        .first()
    )
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    reason = data.reason if data else "Evidence freeze requested by investigator"
    frozen = crud.freeze_finding(db=db, finding=finding, actor_id=user.id, reason=reason)
    return FindingResponse.model_validate(frozen)


@router.post("/findings/{finding_id}/unfreeze", response_model=FindingResponse)
def unfreeze_finding(
    finding_id: uuid.UUID,
    data: Optional[LockFindingRequest] = None,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    finding = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(CaseFinding.id == finding_id, Case.org_id == org.id)
        .first()
    )
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    reason = data.reason if data else "Unfreeze requested"
    unfrozen = crud.unfreeze_finding(db=db, finding=finding, actor_id=user.id, reason=reason)
    return FindingResponse.model_validate(unfrozen)


# =============================================================
# Analytics
# =============================================================
@router.get("/analytics/cases", response_model=CaseStats)
def get_analytics_cases(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    cases = db.query(Case).filter(Case.org_id == org.id).all()
    by_status = {}
    by_severity = {}
    for c in cases:
        by_status[c.status] = by_status.get(c.status, 0) + 1
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
    return CaseStats(
        by_status=by_status,
        by_severity=by_severity,
        total=len(cases),
    )


@router.get("/analytics/findings", response_model=FindingStats)
def get_analytics_findings(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    findings = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(Case.org_id == org.id)
        .all()
    )
    fake_count = sum(1 for f in findings if f.verdict == "FAKE")
    real_count = sum(1 for f in findings if f.verdict == "REAL")
    inconclusive = sum(1 for f in findings if f.verdict == "INCONCLUSIVE")
    return FindingStats(
        fake_count=fake_count,
        real_count=real_count,
        inconclusive_count=inconclusive,
        total=len(findings),
    )


@router.get("/analytics/sources", response_model=List[SourceReputationResponse])
def get_analytics_sources(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    sources = crud.list_sources(db=db, org_id=org.id)
    return [SourceReputationResponse.model_validate(s) for s in sources]


# =============================================================
# Cross-case correlation (stub)
# =============================================================
@router.post("/correlate")
def correlate_cases_stub(
    payload: Dict[str, Any],
    user: User = Depends(require_investigator),
):
    # TODO: pHash + CLIP
    return {
        "status": "completed",
        "matches_found": 0,
        "correlations": [],
        "message": "Cross-case correlation via CLIP embeddings is pending model indexing.",
    }


# =============================================================
# Forensic report export
# =============================================================
@router.post("/cases/{case_id}/export", response_model=ReportExportResponse)
def export_case_report(
    case_id: uuid.UUID,
    data: ExportCaseRequest,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    case = crud.get_case(db=db, case_id=case_id, org_id=org.id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    findings = crud.list_findings_by_case(db=db, case_id=case_id)
    timeline = crud.list_timeline(db=db, case_id=case_id)
    notes = crud.list_notes(db=db, case_id=case_id)

    export_id = uuid.uuid4()

    # Generate styled forensic HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ForensIQ Forensic Case Dossier - {case.title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f6f1e3; color: #1c1917; padding: 40px; margin: 0; }}
    .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; border: 1px solid #dcd3c1; border-radius: 12px; padding: 36px; box-shadow: 0 4px 20px rgba(0,0,0,0.06); }}
    .header {{ border-bottom: 2px solid #b91c1c; padding-bottom: 20px; margin-bottom: 24px; }}
    .title {{ font-size: 26px; font-weight: 800; color: #0c0a09; }}
    .badge {{ display: inline-block; padding: 4px 10px; font-size: 11px; font-weight: 700; border-radius: 6px; text-transform: uppercase; background: #fee2e2; color: #991b1b; }}
    .section {{ margin-top: 28px; }}
    .section-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #78716c; border-bottom: 1px solid #e7e5e4; padding-bottom: 6px; margin-bottom: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f5f5f4; }}
    th {{ background: #fafaf9; font-weight: 600; color: #44403c; }}
    .footer {{ margin-top: 40px; border-top: 1px solid #e7e5e4; padding-top: 16px; font-size: 11px; color: #a8a29e; text-align: center; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">ForensIQ Certified Dossier</div>
      <h1 class="title">{case.title}</h1>
      <p style="margin: 4px 0; font-size: 13px; color: #57534e;">Case ID: {case.id} &bull; Org: {org.name} &bull; Exported by: {user.display_name}</p>
      <p style="margin: 4px 0; font-size: 12px; color: #78716c;">Status: <strong>{case.status.upper()}</strong> &bull; Severity: <strong>{case.severity.upper()}</strong></p>
    </div>

    <div class="section">
      <div class="section-title">Case Description &amp; Scope</div>
      <p style="font-size: 14px; line-height: 1.6;">{case.description or "No description provided."}</p>
    </div>

    <div class="section">
      <div class="section-title">Attached Forensic Findings ({len(findings)})</div>
      <table>
        <thead>
          <tr>
            <th>Filename</th>
            <th>Verdict</th>
            <th>Confidence</th>
            <th>SHA-256 Hash</th>
            <th>Evidence Lock</th>
          </tr>
        </thead>
        <tbody>
          {"".join(f'''<tr>
            <td><strong>{f.original_filename}</strong></td>
            <td><span class="badge">{f.verdict}</span></td>
            <td>{round(f.confidence * 100, 1)}%</td>
            <td style="font-family: monospace; font-size: 11px;">{f.sha256[:16]}...</td>
            <td>{"LOCKED" if f.frozen else "UNFROZEN"}</td>
          </tr>''' for f in findings) if findings else '<tr><td colspan="5">No findings attached.</td></tr>'}
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="section-title">Investigator Notes ({len(notes)})</div>
      <ul>
        {"".join(f'<li style="font-size: 13px; margin-bottom: 8px;">{n.text}</li>' for n in notes) if notes else '<li style="font-size: 13px; color: #78716c;">No notes recorded.</li>'}
      </ul>
    </div>

    <div class="section">
      <div class="section-title">Chain of Custody &amp; Activity Log</div>
      <table>
        <thead>
          <tr><th>Timestamp</th><th>Action</th><th>Payload Summary</th></tr>
        </thead>
        <tbody>
          {"".join(f'''<tr>
            <td style="font-size: 11px;">{t.created_at}</td>
            <td><strong>{t.action}</strong></td>
            <td style="font-size: 12px;">{str(t.payload)}</td>
          </tr>''' for t in timeline[:10])}
        </tbody>
      </table>
    </div>

    <div class="footer">
      Generated automatically by ForensIQ Workstation v2.4 &bull; Cryptographically auditable record
    </div>
  </div>
</body>
</html>
"""
    export_bytes = html_content.encode("utf-8")
    saved_path = save_export(export_id, export_bytes, ext="html")
    sha256_hash = sha256_of_bytes(export_bytes)

    export_record = ReportExport(
        id=export_id,
        case_id=case.id,
        finding_id=data.finding_id,
        exported_by=user.id,
        export_type=data.export_type,
        storage_path=str(saved_path),
        sha256=sha256_hash,
    )
    db.add(export_record)
    db.commit()
    db.refresh(export_record)

    crud.add_timeline(db, case.id, user.id, "export", {
        "export_id": str(export_id),
        "export_type": data.export_type,
    })

    return ReportExportResponse(
        id=export_record.id,
        case_id=export_record.case_id,
        finding_id=export_record.finding_id,
        exported_by=export_record.exported_by,
        export_type=export_record.export_type,
        storage_path=export_record.storage_path,
        sha256=export_record.sha256,
        signed_hash=export_record.signed_hash,
        download_url=f"/api/investigator/exports/{export_id}/download",
        created_at=export_record.created_at,
    )


@router.get("/exports/{export_id}/download")
def download_export(
    export_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    export_rec = db.query(ReportExport).filter(ReportExport.id == export_id).first()
    if not export_rec:
        raise HTTPException(status_code=404, detail="Export not found")

    file_path = Path(export_rec.storage_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Exported file missing on disk")

    return FileResponse(
        path=str(file_path),
        media_type="text/html",
        filename=f"case_export_{export_id}.html",
    )
