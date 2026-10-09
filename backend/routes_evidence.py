import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import get_db
from backend.models import User, CaseFinding, EvidenceLock, CaseTimeline, ReportExport
from backend.deps import require_investigator, get_current_user
from backend.schemas import (
    LockFindingRequest,
    EvidenceLockResponse,
    CustodyChainResponse,
    TimelineEntryResponse,
)
from backend import crud

router = APIRouter()


@router.post("/findings/{finding_id}/lock", response_model=EvidenceLockResponse)
def lock_finding(
    finding_id: uuid.UUID,
    data: Optional[LockFindingRequest] = None,
    user: User = Depends(require_investigator),
    db: Session = Depends(get_db),
):
    finding = crud.get_finding(db=db, finding_id=finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    reason = data.reason if data else "Forensic lock acquired by investigator"
    frozen = crud.freeze_finding(db=db, finding=finding, actor_id=user.id, reason=reason)

    latest_lock = (
        db.query(EvidenceLock)
        .filter(EvidenceLock.finding_id == finding.id)
        .order_by(desc(EvidenceLock.locked_at))
        .first()
    )

    return EvidenceLockResponse(
        id=latest_lock.id,
        finding_id=latest_lock.finding_id,
        locked_by=latest_lock.locked_by,
        reason=latest_lock.reason,
        content_hash=latest_lock.content_hash,
        chain_hash=latest_lock.chain_hash,
        locked_at=latest_lock.locked_at,
        locked_by_name=user.display_name,
    )


@router.post("/findings/{finding_id}/unlock")
def unlock_finding(
    finding_id: uuid.UUID,
    data: Optional[LockFindingRequest] = None,
    user: User = Depends(require_investigator),
    db: Session = Depends(get_db),
):
    finding = crud.get_finding(db=db, finding_id=finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    reason = data.reason if data else "Investigator unlocked finding"
    crud.unfreeze_finding(db=db, finding=finding, actor_id=user.id, reason=reason)
    return {"status": "unlocked", "finding_id": str(finding_id)}


@router.get("/findings/{finding_id}/custody", response_model=CustodyChainResponse)
def get_custody_chain(
    finding_id: uuid.UUID,
    user: User = Depends(require_investigator),
    db: Session = Depends(get_db),
):
    finding = crud.get_finding(db=db, finding_id=finding_id)
    if not finding:
        raise HTTPException(status_code=404, detail="Finding not found")

    locks = (
        db.query(EvidenceLock)
        .filter(EvidenceLock.finding_id == finding.id)
        .order_by(desc(EvidenceLock.locked_at))
        .all()
    )

    lock_responses = []
    for l in locks:
        locker = db.query(User).filter(User.id == l.locked_by).first()
        lock_responses.append(EvidenceLockResponse(
            id=l.id,
            finding_id=l.finding_id,
            locked_by=l.locked_by,
            reason=l.reason,
            content_hash=l.content_hash,
            chain_hash=l.chain_hash,
            locked_at=l.locked_at,
            locked_by_name=locker.display_name if locker else None,
        ))

    # Associated timeline entries for this case and finding
    all_timeline = crud.list_timeline(db=db, case_id=finding.case_id)
    timeline_responses = []
    for t in all_timeline:
        actor = db.query(User).filter(User.id == t.actor_id).first() if t.actor_id else None
        timeline_responses.append(TimelineEntryResponse(
            id=t.id,
            case_id=t.case_id,
            actor_id=t.actor_id,
            action=t.action,
            payload=t.payload or {},
            created_at=t.created_at,
            actor_name=actor.display_name if actor else "System",
        ))

    return CustodyChainResponse(
        finding_id=finding.id,
        sha256=finding.sha256,
        frozen=finding.frozen,
        frozen_at=finding.frozen_at,
        locks=lock_responses,
        timeline_entries=timeline_responses,
    )


@router.post("/exports/{export_id}/sign")
def sign_export_stub(
    export_id: uuid.UUID,
    user: User = Depends(require_investigator),
    db: Session = Depends(get_db),
):
    export_rec = db.query(ReportExport).filter(ReportExport.id == export_id).first()
    if not export_rec:
        raise HTTPException(status_code=404, detail="Export not found")

    import hashlib
    signature = hashlib.sha256(f"SIGNED:{export_rec.sha256}:{user.id}".encode()).hexdigest()
    export_rec.signed_hash = signature
    db.commit()

    return {
        "status": "signed",
        "export_id": str(export_id),
        "signed_hash": signature,
        "message": "Cryptographic signature stamp applied to export.",
    }
