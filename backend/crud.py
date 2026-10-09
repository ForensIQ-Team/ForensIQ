import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.models import (
    User,
    Organization,
    OrgMember,
    InvestigatorApplication,
    Case,
    CaseFinding,
    CaseTimeline,
    CaseNote,
    EvidenceLock,
    ReportExport,
    Watchlist,
    WatchlistAlert,
    SourceReputation,
    AuditLog,
)
from backend.security import hash_password


# =============================================================
# Users
# =============================================================
def create_user(db: Session, email: str, password: str, display_name: Optional[str] = None) -> User:
    pwd_hash = hash_password(password)
    user = User(
        email=email.strip().lower(),
        password_hash=pwd_hash,
        display_name=display_name or email.split("@")[0],
        role="user",
        tier=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email.strip().lower()).first()


def get_user_by_id(db: Session, user_id: Union[uuid.UUID, str]) -> Optional[User]:
    if isinstance(user_id, str):
        user_id = uuid.UUID(user_id)
    return db.query(User).filter(User.id == user_id).first()


def update_last_login(db: Session, user: User) -> None:
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()


def update_user_role_tier(
    db: Session,
    user: User,
    role: Optional[str] = None,
    tier: Optional[int] = None,
    org_id: Optional[uuid.UUID] = None,
) -> User:
    if role is not None:
        user.role = role
    if tier is not None:
        user.tier = tier
    if org_id is not None:
        user.org_id = org_id
    db.commit()
    db.refresh(user)
    return user


# =============================================================
# Organizations & Members
# =============================================================
def create_organization(
    db: Session,
    name: str,
    domain: Optional[str] = None,
    website: Optional[str] = None,
    verification_method: Optional[str] = None,
    verified_at: Optional[datetime] = None,
) -> Organization:
    org = Organization(
        name=name,
        domain=domain.lower() if domain else None,
        website=website,
        verification_method=verification_method,
        verified_at=verified_at,
    )
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


def get_org_by_domain(db: Session, domain: str) -> Optional[Organization]:
    return db.query(Organization).filter(Organization.domain == domain.lower()).first()


def get_org_by_id(db: Session, org_id: Union[uuid.UUID, str]) -> Optional[Organization]:
    if isinstance(org_id, str):
        org_id = uuid.UUID(org_id)
    return db.query(Organization).filter(Organization.id == org_id).first()


def add_org_member(db: Session, org_id: uuid.UUID, user_id: uuid.UUID, role: str = "analyst") -> OrgMember:
    member = db.query(OrgMember).filter(OrgMember.org_id == org_id, OrgMember.user_id == user_id).first()
    if not member:
        member = OrgMember(org_id=org_id, user_id=user_id, role=role)
        db.add(member)
    else:
        member.role = role
    db.commit()
    db.refresh(member)
    return member


def list_org_members(db: Session, org_id: uuid.UUID) -> List[OrgMember]:
    return db.query(OrgMember).filter(OrgMember.org_id == org_id).all()


def remove_org_member(db: Session, org_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    member = db.query(OrgMember).filter(OrgMember.org_id == org_id, OrgMember.user_id == user_id).first()
    if member:
        db.delete(member)
        user = get_user_by_id(db, user_id)
        if user and user.org_id == org_id:
            user.org_id = None
        db.commit()
        return True
    return False


# =============================================================
# Applications
# =============================================================
def create_application(
    db: Session,
    user_id: uuid.UUID,
    work_email: str,
    organization_name: Optional[str] = None,
    designation: Optional[str] = None,
    org_website: Optional[str] = None,
    linkedin_url: Optional[str] = None,
    reason: Optional[str] = None,
    reference_email: Optional[str] = None,
    status: str = "pending",
) -> InvestigatorApplication:
    app = InvestigatorApplication(
        user_id=user_id,
        work_email=work_email.strip().lower(),
        organization_name=organization_name,
        designation=designation,
        org_website=org_website,
        linkedin_url=linkedin_url,
        reason=reason,
        reference_email=reference_email.strip().lower() if reference_email else None,
        status=status,
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def get_application_by_user(db: Session, user_id: uuid.UUID) -> Optional[InvestigatorApplication]:
    return (
        db.query(InvestigatorApplication)
        .filter(InvestigatorApplication.user_id == user_id)
        .order_by(desc(InvestigatorApplication.created_at))
        .first()
    )


def list_pending_applications(db: Session) -> List[InvestigatorApplication]:
    return (
        db.query(InvestigatorApplication)
        .filter(InvestigatorApplication.status == "pending")
        .order_by(InvestigatorApplication.created_at.asc())
        .all()
    )


def approve_application(
    db: Session,
    app_id: uuid.UUID,
    reviewer_id: uuid.UUID,
    notes: Optional[str] = None,
) -> Optional[InvestigatorApplication]:
    app = db.query(InvestigatorApplication).filter(InvestigatorApplication.id == app_id).first()
    if not app:
        return None
    app.status = "approved"
    app.reviewed_by = reviewer_id
    app.reviewed_at = datetime.now(timezone.utc)
    app.reviewer_notes = notes
    db.commit()
    db.refresh(app)
    return app


def reject_application(
    db: Session,
    app_id: uuid.UUID,
    reviewer_id: uuid.UUID,
    notes: Optional[str] = None,
) -> Optional[InvestigatorApplication]:
    app = db.query(InvestigatorApplication).filter(InvestigatorApplication.id == app_id).first()
    if not app:
        return None
    app.status = "rejected"
    app.reviewed_by = reviewer_id
    app.reviewed_at = datetime.now(timezone.utc)
    app.reviewer_notes = notes
    db.commit()
    db.refresh(app)
    return app


# =============================================================
# Cases
# =============================================================
def create_case(
    db: Session,
    org_id: uuid.UUID,
    created_by: uuid.UUID,
    title: str,
    description: Optional[str] = None,
    severity: str = "medium",
    tags: Optional[List[str]] = None,
    assigned_to: Optional[uuid.UUID] = None,
) -> Case:
    case = Case(
        org_id=org_id,
        created_by=created_by,
        assigned_to=assigned_to,
        title=title,
        description=description,
        severity=severity,
        tags=tags or [],
        status="open",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    # Initial timeline entry
    add_timeline(db, case.id, created_by, "create", {"title": title, "severity": severity})

    return case


def get_case(db: Session, case_id: Union[uuid.UUID, str], org_id: Optional[uuid.UUID] = None) -> Optional[Case]:
    if isinstance(case_id, str):
        case_id = uuid.UUID(case_id)
    query = db.query(Case).filter(Case.id == case_id)
    if org_id:
        query = query.filter(Case.org_id == org_id)
    return query.first()


def list_cases(
    db: Session,
    org_id: uuid.UUID,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Case]:
    query = db.query(Case).filter(Case.org_id == org_id)
    if status and status != "all":
        query = query.filter(Case.status == status)
    if severity and severity != "all":
        query = query.filter(Case.severity == severity)
    if tag:
        query = query.filter(Case.tags.contains([tag]))
    return query.order_by(desc(Case.updated_at)).offset(offset).limit(limit).all()


def update_case(
    db: Session,
    case: Case,
    actor_id: uuid.UUID,
    title: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    tags: Optional[List[str]] = None,
    assigned_to: Optional[uuid.UUID] = None,
) -> Case:
    changes = {}
    if title is not None and title != case.title:
        changes["title"] = {"old": case.title, "new": title}
        case.title = title
    if description is not None and description != case.description:
        changes["description"] = {"old": case.description, "new": description}
        case.description = description
    if status is not None and status != case.status:
        changes["status"] = {"old": case.status, "new": status}
        case.status = status
        if status == "closed":
            case.closed_at = datetime.now(timezone.utc)
        elif case.closed_at is not None:
            case.closed_at = None
    if severity is not None and severity != case.severity:
        changes["severity"] = {"old": case.severity, "new": severity}
        case.severity = severity
    if tags is not None:
        changes["tags"] = {"old": case.tags, "new": tags}
        case.tags = tags
    if assigned_to is not None and assigned_to != case.assigned_to:
        changes["assigned_to"] = {"old": str(case.assigned_to) if case.assigned_to else None, "new": str(assigned_to)}
        case.assigned_to = assigned_to

    case.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(case)

    if changes:
        add_timeline(db, case.id, actor_id, "update", changes)

    return case


def soft_delete_case(db: Session, case: Case, actor_id: uuid.UUID) -> None:
    case.status = "archived"
    case.updated_at = datetime.now(timezone.utc)
    db.commit()
    add_timeline(db, case.id, actor_id, "status_change", {"status": "archived"})


def assign_case(db: Session, case: Case, actor_id: uuid.UUID, assigned_to: Optional[uuid.UUID]) -> Case:
    old_assignee = str(case.assigned_to) if case.assigned_to else None
    case.assigned_to = assigned_to
    case.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(case)
    add_timeline(db, case.id, actor_id, "assign", {
        "old_assigned_to": old_assignee,
        "new_assigned_to": str(assigned_to) if assigned_to else None,
    })
    return case


def change_status(db: Session, case: Case, actor_id: uuid.UUID, new_status: str) -> Case:
    old_status = case.status
    case.status = new_status
    if new_status == "closed":
        case.closed_at = datetime.now(timezone.utc)
    elif case.closed_at is not None:
        case.closed_at = None
    case.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(case)
    add_timeline(db, case.id, actor_id, "status_change", {
        "old_status": old_status,
        "new_status": new_status,
    })
    return case


# =============================================================
# Findings
# =============================================================
def create_finding(
    db: Session,
    case_id: uuid.UUID,
    uploaded_by: uuid.UUID,
    media_type: str,
    original_filename: str,
    storage_path: str,
    sha256: str,
    file_size_bytes: int,
    verdict: str,
    confidence: float,
    analysis_json: Dict[str, Any],
    gradcam_path: Optional[str] = None,
    report_id: Optional[str] = None,
) -> CaseFinding:
    finding = CaseFinding(
        case_id=case_id,
        uploaded_by=uploaded_by,
        media_type=media_type,
        original_filename=original_filename,
        storage_path=storage_path,
        sha256=sha256,
        file_size_bytes=file_size_bytes,
        verdict=verdict,
        confidence=confidence,
        analysis_json=analysis_json,
        gradcam_path=gradcam_path,
        report_id=report_id,
        frozen=False,
    )
    db.add(finding)
    db.commit()
    db.refresh(finding)

    # Timeline entry for upload/analysis
    add_timeline(db, case_id, uploaded_by, "upload", {
        "finding_id": str(finding.id),
        "filename": original_filename,
        "verdict": verdict,
        "confidence": confidence,
    })

    return finding


def get_finding(db: Session, finding_id: Union[uuid.UUID, str]) -> Optional[CaseFinding]:
    if isinstance(finding_id, str):
        finding_id = uuid.UUID(finding_id)
    return db.query(CaseFinding).filter(CaseFinding.id == finding_id).first()


def list_findings_by_case(db: Session, case_id: uuid.UUID) -> List[CaseFinding]:
    return (
        db.query(CaseFinding)
        .filter(CaseFinding.case_id == case_id)
        .order_by(desc(CaseFinding.created_at))
        .all()
    )


def freeze_finding(db: Session, finding: CaseFinding, actor_id: uuid.UUID, reason: Optional[str] = None) -> CaseFinding:
    finding.frozen = True
    finding.frozen_by = actor_id
    finding.frozen_at = datetime.now(timezone.utc)

    # Also record in evidence locks
    import hashlib
    # Compute chain hash based on previous locks if any
    prev_lock = (
        db.query(EvidenceLock)
        .filter(EvidenceLock.finding_id == finding.id)
        .order_by(desc(EvidenceLock.locked_at))
        .first()
    )
    prev_hash = prev_lock.chain_hash if prev_lock else "0" * 64
    chain_input = f"{prev_hash}:{finding.sha256}:{actor_id}:{reason or ''}"
    new_chain_hash = hashlib.sha256(chain_input.encode()).hexdigest()

    lock = EvidenceLock(
        finding_id=finding.id,
        locked_by=actor_id,
        reason=reason,
        content_hash=finding.sha256,
        chain_hash=new_chain_hash,
    )
    db.add(lock)
    db.commit()
    db.refresh(finding)

    add_timeline(db, finding.case_id, actor_id, "freeze", {
        "finding_id": str(finding.id),
        "chain_hash": new_chain_hash,
        "reason": reason,
    })

    return finding


def unfreeze_finding(db: Session, finding: CaseFinding, actor_id: uuid.UUID, reason: Optional[str] = None) -> CaseFinding:
    finding.frozen = False
    finding.frozen_by = None
    finding.frozen_at = None
    db.commit()
    db.refresh(finding)

    add_timeline(db, finding.case_id, actor_id, "unfreeze", {
        "finding_id": str(finding.id),
        "reason": reason,
    })

    return finding


# =============================================================
# Timeline
# =============================================================
def add_timeline(
    db: Session,
    case_id: uuid.UUID,
    actor_id: Optional[uuid.UUID],
    action: str,
    payload: Optional[Dict[str, Any]] = None,
) -> CaseTimeline:
    entry = CaseTimeline(
        case_id=case_id,
        actor_id=actor_id,
        action=action,
        payload=payload or {},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_timeline(db: Session, case_id: uuid.UUID) -> List[CaseTimeline]:
    return (
        db.query(CaseTimeline)
        .filter(CaseTimeline.case_id == case_id)
        .order_by(desc(CaseTimeline.created_at))
        .all()
    )


# =============================================================
# Notes
# =============================================================
def add_note(db: Session, case_id: uuid.UUID, author_id: uuid.UUID, text: str) -> CaseNote:
    note = CaseNote(
        case_id=case_id,
        author_id=author_id,
        text=text,
    )
    db.add(note)
    db.commit()
    db.refresh(note)

    add_timeline(db, case_id, author_id, "note", {"note_id": str(note.id)})
    return note


def list_notes(db: Session, case_id: uuid.UUID) -> List[CaseNote]:
    return (
        db.query(CaseNote)
        .filter(CaseNote.case_id == case_id)
        .order_by(desc(CaseNote.created_at))
        .all()
    )


def delete_note(db: Session, note_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    note = db.query(CaseNote).filter(CaseNote.id == note_id).first()
    if not note:
        return False
    case_id = note.case_id
    db.delete(note)
    db.commit()
    add_timeline(db, case_id, user_id, "update", {"deleted_note": str(note_id)})
    return True


# =============================================================
# Watchlists & Alerts
# =============================================================
def create_watchlist(
    db: Session,
    org_id: uuid.UUID,
    created_by: uuid.UUID,
    name: str,
    description: Optional[str],
    target_type: str,
    target_query: str,
    active: bool = True,
) -> Watchlist:
    w = Watchlist(
        org_id=org_id,
        created_by=created_by,
        name=name,
        description=description,
        target_type=target_type,
        target_query=target_query,
        active=active,
    )
    db.add(w)
    db.commit()
    db.refresh(w)
    return w


def list_watchlists(db: Session, org_id: uuid.UUID) -> List[Watchlist]:
    return db.query(Watchlist).filter(Watchlist.org_id == org_id).order_by(desc(Watchlist.created_at)).all()


def get_watchlist(db: Session, watchlist_id: Union[uuid.UUID, str]) -> Optional[Watchlist]:
    if isinstance(watchlist_id, str):
        watchlist_id = uuid.UUID(watchlist_id)
    return db.query(Watchlist).filter(Watchlist.id == watchlist_id).first()


def update_watchlist(
    db: Session,
    w: Watchlist,
    name: Optional[str] = None,
    description: Optional[str] = None,
    target_type: Optional[str] = None,
    target_query: Optional[str] = None,
    active: Optional[bool] = None,
) -> Watchlist:
    if name is not None:
        w.name = name
    if description is not None:
        w.description = description
    if target_type is not None:
        w.target_type = target_type
    if target_query is not None:
        w.target_query = target_query
    if active is not None:
        w.active = active
    db.commit()
    db.refresh(w)
    return w


def delete_watchlist(db: Session, watchlist_id: uuid.UUID) -> bool:
    w = db.query(Watchlist).filter(Watchlist.id == watchlist_id).first()
    if w:
        db.delete(w)
        db.commit()
        return True
    return False


def create_alert(
    db: Session,
    watchlist_id: uuid.UUID,
    source_url: str,
    source_type: str,
    severity: str,
    match_score: Optional[float] = None,
    snapshot_path: Optional[str] = None,
    status: str = "new",
) -> WatchlistAlert:
    alert = WatchlistAlert(
        watchlist_id=watchlist_id,
        source_url=source_url,
        source_type=source_type,
        severity=severity,
        match_score=match_score,
        snapshot_path=snapshot_path,
        status=status,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def list_alerts(
    db: Session,
    org_id: uuid.UUID,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[WatchlistAlert]:
    query = (
        db.query(WatchlistAlert)
        .join(Watchlist, WatchlistAlert.watchlist_id == Watchlist.id)
        .filter(Watchlist.org_id == org_id)
    )
    if status and status != "all":
        query = query.filter(WatchlistAlert.status == status)
    return query.order_by(desc(WatchlistAlert.created_at)).limit(limit).all()


def triage_alert(
    db: Session,
    alert_id: uuid.UUID,
    actor_id: uuid.UUID,
    status: str,
) -> Optional[WatchlistAlert]:
    alert = db.query(WatchlistAlert).filter(WatchlistAlert.id == alert_id).first()
    if not alert:
        return None
    alert.status = status
    alert.triaged_by = actor_id
    alert.triaged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert


def escalate_alert_to_case(
    db: Session,
    alert_id: uuid.UUID,
    actor_id: uuid.UUID,
    case_id: uuid.UUID,
) -> Optional[WatchlistAlert]:
    alert = db.query(WatchlistAlert).filter(WatchlistAlert.id == alert_id).first()
    if not alert:
        return None
    alert.status = "escalated"
    alert.case_id = case_id
    alert.triaged_by = actor_id
    alert.triaged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert


# =============================================================
# Source Reputation
# =============================================================
def upsert_source_reputation(
    db: Session,
    org_id: uuid.UUID,
    domain: str,
    is_fake: bool,
    notes: Optional[str] = None,
) -> SourceReputation:
    source = (
        db.query(SourceReputation)
        .filter(SourceReputation.org_id == org_id, SourceReputation.domain == domain.lower())
        .first()
    )
    if not source:
        source = SourceReputation(
            org_id=org_id,
            domain=domain.lower(),
            total_findings=1,
            fake_findings=1 if is_fake else 0,
            fake_ratio=1.0 if is_fake else 0.0,
            last_seen_at=datetime.now(timezone.utc),
            notes=notes,
        )
        db.add(source)
    else:
        source.total_findings += 1
        if is_fake:
            source.fake_findings += 1
        source.fake_ratio = source.fake_findings / source.total_findings
        source.last_seen_at = datetime.now(timezone.utc)
        if notes:
            source.notes = notes
    db.commit()
    db.refresh(source)
    return source


def list_sources(db: Session, org_id: uuid.UUID) -> List[SourceReputation]:
    return (
        db.query(SourceReputation)
        .filter(SourceReputation.org_id == org_id)
        .order_by(desc(SourceReputation.total_findings))
        .all()
    )


# =============================================================
# Audit Log
# =============================================================
def list_audit_entries(
    db: Session,
    org_id: Optional[uuid.UUID] = None,
    action: Optional[str] = None,
    limit: int = 100,
) -> List[AuditLog]:
    query = db.query(AuditLog)
    if org_id:
        query = query.filter(AuditLog.org_id == org_id)
    if action and action != "all":
        query = query.filter(AuditLog.action == action)
    return query.order_by(desc(AuditLog.created_at)).limit(limit).all()
