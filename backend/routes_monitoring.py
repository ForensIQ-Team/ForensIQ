import uuid
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database import get_db
from backend.models import User, Organization, Watchlist, WatchlistAlert, SourceReputation, Case
from backend.deps import require_investigator, get_current_org
from backend.schemas import (
    WatchlistCreate,
    WatchlistUpdate,
    WatchlistResponse,
    AlertResponse,
    AlertTriageRequest,
    AlertEscalateRequest,
    SourceReputationResponse,
    CaseResponse,
)
from backend import crud

router = APIRouter()


@router.post("/watchlists", response_model=WatchlistResponse)
def create_watchlist(
    data: WatchlistCreate,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    w = crud.create_watchlist(
        db=db,
        org_id=org.id,
        created_by=user.id,
        name=data.name,
        description=data.description,
        target_type=data.target_type,
        target_query=data.target_query,
        active=data.active,
    )
    return WatchlistResponse(
        id=w.id,
        org_id=w.org_id,
        created_by=w.created_by,
        name=w.name,
        description=w.description,
        target_type=w.target_type,
        target_query=w.target_query,
        active=w.active,
        created_at=w.created_at,
        alerts_count=0,
    )


@router.get("/watchlists", response_model=List[WatchlistResponse])
def get_watchlists(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    watchlists = crud.list_watchlists(db=db, org_id=org.id)
    results = []
    for w in watchlists:
        alerts_count = db.query(WatchlistAlert).filter(WatchlistAlert.watchlist_id == w.id).count()
        results.append(WatchlistResponse(
            id=w.id,
            org_id=w.org_id,
            created_by=w.created_by,
            name=w.name,
            description=w.description,
            target_type=w.target_type,
            target_query=w.target_query,
            active=w.active,
            created_at=w.created_at,
            alerts_count=alerts_count,
        ))
    return results


@router.get("/watchlists/{watchlist_id}", response_model=WatchlistResponse)
def get_watchlist(
    watchlist_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    w = crud.get_watchlist(db=db, watchlist_id=watchlist_id)
    if not w or w.org_id != org.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    alerts_count = db.query(WatchlistAlert).filter(WatchlistAlert.watchlist_id == w.id).count()
    return WatchlistResponse(
        id=w.id,
        org_id=w.org_id,
        created_by=w.created_by,
        name=w.name,
        description=w.description,
        target_type=w.target_type,
        target_query=w.target_query,
        active=w.active,
        created_at=w.created_at,
        alerts_count=alerts_count,
    )


@router.patch("/watchlists/{watchlist_id}", response_model=WatchlistResponse)
def update_watchlist(
    watchlist_id: uuid.UUID,
    data: WatchlistUpdate,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    w = crud.get_watchlist(db=db, watchlist_id=watchlist_id)
    if not w or w.org_id != org.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    updated = crud.update_watchlist(
        db=db,
        w=w,
        name=data.name,
        description=data.description,
        target_type=data.target_type,
        target_query=data.target_query,
        active=data.active,
    )
    alerts_count = db.query(WatchlistAlert).filter(WatchlistAlert.watchlist_id == w.id).count()
    return WatchlistResponse(
        id=updated.id,
        org_id=updated.org_id,
        created_by=updated.created_by,
        name=updated.name,
        description=updated.description,
        target_type=updated.target_type,
        target_query=updated.target_query,
        active=updated.active,
        created_at=updated.created_at,
        alerts_count=alerts_count,
    )


@router.delete("/watchlists/{watchlist_id}")
def delete_watchlist(
    watchlist_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    w = crud.get_watchlist(db=db, watchlist_id=watchlist_id)
    if not w or w.org_id != org.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    crud.delete_watchlist(db=db, watchlist_id=watchlist_id)
    return {"status": "deleted", "watchlist_id": str(watchlist_id)}


# =============================================================
# Alerts
# =============================================================
@router.get("/alerts", response_model=List[AlertResponse])
def get_alerts(
    status: Optional[str] = None,
    limit: int = 50,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    alerts = crud.list_alerts(db=db, org_id=org.id, status=status, limit=limit)
    results = []
    for a in alerts:
        w = crud.get_watchlist(db=db, watchlist_id=a.watchlist_id)
        results.append(AlertResponse(
            id=a.id,
            watchlist_id=a.watchlist_id,
            watchlist_name=w.name if w else None,
            source_url=a.source_url,
            source_type=a.source_type,
            severity=a.severity,
            match_score=a.match_score,
            snapshot_path=a.snapshot_path,
            status=a.status,
            triaged_by=a.triaged_by,
            triaged_at=a.triaged_at,
            case_id=a.case_id,
            created_at=a.created_at,
        ))
    return results


@router.get("/alerts/{alert_id}", response_model=AlertResponse)
def get_alert_detail(
    alert_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    a = db.query(WatchlistAlert).filter(WatchlistAlert.id == alert_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Alert not found")
    w = crud.get_watchlist(db=db, watchlist_id=a.watchlist_id)
    if not w or w.org_id != org.id:
        raise HTTPException(status_code=404, detail="Alert not found")

    return AlertResponse(
        id=a.id,
        watchlist_id=a.watchlist_id,
        watchlist_name=w.name,
        source_url=a.source_url,
        source_type=a.source_type,
        severity=a.severity,
        match_score=a.match_score,
        snapshot_path=a.snapshot_path,
        status=a.status,
        triaged_by=a.triaged_by,
        triaged_at=a.triaged_at,
        case_id=a.case_id,
        created_at=a.created_at,
    )


@router.post("/alerts/{alert_id}/triage", response_model=AlertResponse)
def triage_alert(
    alert_id: uuid.UUID,
    data: AlertTriageRequest,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    a = db.query(WatchlistAlert).filter(WatchlistAlert.id == alert_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Alert not found")

    triaged = crud.triage_alert(db=db, alert_id=alert_id, actor_id=user.id, status=data.status)
    w = crud.get_watchlist(db=db, watchlist_id=triaged.watchlist_id)
    return AlertResponse(
        id=triaged.id,
        watchlist_id=triaged.watchlist_id,
        watchlist_name=w.name if w else None,
        source_url=triaged.source_url,
        source_type=triaged.source_type,
        severity=triaged.severity,
        match_score=triaged.match_score,
        snapshot_path=triaged.snapshot_path,
        status=triaged.status,
        triaged_by=triaged.triaged_by,
        triaged_at=triaged.triaged_at,
        case_id=triaged.case_id,
        created_at=triaged.created_at,
    )


@router.post("/alerts/{alert_id}/escalate")
def escalate_alert(
    alert_id: uuid.UUID,
    data: AlertEscalateRequest,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    a = db.query(WatchlistAlert).filter(WatchlistAlert.id == alert_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Alert not found")
    w = crud.get_watchlist(db=db, watchlist_id=a.watchlist_id)

    # Create new case for escalated alert
    case_title = data.title or f"Escalated Alert: {w.name if w else 'Threat Detection'} - {a.source_url}"
    case_desc = data.description or f"Auto-escalated from monitoring alert on {a.source_url}. Target query: {w.target_query if w else 'N/A'}"

    case = crud.create_case(
        db=db,
        org_id=org.id,
        created_by=user.id,
        title=case_title,
        description=case_desc,
        severity=data.severity or a.severity,
        tags=["escalated-alert", w.target_type if w else "monitoring"],
        assigned_to=user.id,
    )

    crud.escalate_alert_to_case(db=db, alert_id=alert_id, actor_id=user.id, case_id=case.id)

    return {
        "status": "escalated",
        "case_id": str(case.id),
        "alert_id": str(alert_id),
        "message": "Alert escalated to new investigation case.",
    }


@router.post("/watchlists/{watchlist_id}/run-now", response_model=AlertResponse)
def run_watchlist_now(
    watchlist_id: uuid.UUID,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    w = crud.get_watchlist(db=db, watchlist_id=watchlist_id)
    if not w or w.org_id != org.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    # Generate an alert triggered by immediate crawl/scan
    import random
    mock_sources = [
        f"https://x.com/viral_account/status/{random.randint(10000000, 99999999)}",
        f"https://reddit.com/r/deepfakes/comments/{uuid.uuid4().hex[:6]}",
        f"https://t.me/media_drop/{random.randint(100, 999)}",
        f"https://facebook.com/watch/?v={random.randint(100000, 999999)}",
    ]
    sampled_url = random.choice(mock_sources)
    domain = sampled_url.split("//")[-1].split("/")[0]

    alert = crud.create_alert(
        db=db,
        watchlist_id=w.id,
        source_url=sampled_url,
        source_type="social_crawl",
        severity="high" if "x.com" in sampled_url else "medium",
        match_score=0.92,
        snapshot_path=None,
        status="new",
    )

    crud.upsert_source_reputation(db=db, org_id=org.id, domain=domain, is_fake=True)

    return AlertResponse(
        id=alert.id,
        watchlist_id=alert.watchlist_id,
        watchlist_name=w.name,
        source_url=alert.source_url,
        source_type=alert.source_type,
        severity=alert.severity,
        match_score=alert.match_score,
        snapshot_path=alert.snapshot_path,
        status=alert.status,
        triaged_by=alert.triaged_by,
        triaged_at=alert.triaged_at,
        case_id=alert.case_id,
        created_at=alert.created_at,
    )


@router.get("/sources", response_model=List[SourceReputationResponse])
def get_sources(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    sources = crud.list_sources(db=db, org_id=org.id)
    return [SourceReputationResponse.model_validate(s) for s in sources]
