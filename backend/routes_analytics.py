import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.database import get_db
from backend.models import User, Organization, Case, CaseFinding, WatchlistAlert, SourceReputation
from backend.deps import require_investigator, get_current_org
from backend.schemas import DashboardStats

router = APIRouter()


@router.get("/org/overview")
def get_org_overview(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    total_cases = db.query(Case).filter(Case.org_id == org.id).count()
    open_cases = db.query(Case).filter(Case.org_id == org.id, Case.status.in_(["open", "in_progress"])).count()

    now = datetime.now(timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    month_findings = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(Case.org_id == org.id, CaseFinding.created_at >= month_start)
        .all()
    )
    findings_month = len(month_findings)
    fake_month = sum(1 for f in month_findings if f.verdict == "FAKE")
    real_month = sum(1 for f in month_findings if f.verdict == "REAL")

    unread_alerts = (
        db.query(WatchlistAlert)
        .join(Case, WatchlistAlert.case_id == Case.id, isouter=True)
        .filter(WatchlistAlert.status == "new")
        .count()
    )

    stats = DashboardStats(
        total_cases=total_cases,
        open_cases=open_cases,
        findings_month=findings_month,
        fake_findings_month=fake_month,
        real_findings_month=real_month,
        unread_alerts=unread_alerts,
    )

    cases_by_status = {}
    cases_by_severity = {}
    all_cases = db.query(Case).filter(Case.org_id == org.id).all()
    for c in all_cases:
        cases_by_status[c.status] = cases_by_status.get(c.status, 0) + 1
        cases_by_severity[c.severity] = cases_by_severity.get(c.severity, 0) + 1

    top_sources = (
        db.query(SourceReputation)
        .filter(SourceReputation.org_id == org.id)
        .order_by(desc(SourceReputation.total_findings))
        .limit(5)
        .all()
    )

    return {
        "stats": stats,
        "cases_by_status": cases_by_status,
        "cases_by_severity": cases_by_severity,
        "top_sources": [
            {
                "domain": s.domain,
                "total": s.total_findings,
                "fake": s.fake_findings,
                "fake_ratio": round(s.fake_ratio, 2),
            }
            for s in top_sources
        ],
    }


@router.get("/org/timeline")
def get_org_timeline(
    days: int = 14,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)

    findings = (
        db.query(CaseFinding)
        .join(Case, CaseFinding.case_id == Case.id)
        .filter(Case.org_id == org.id, CaseFinding.created_at >= start_date)
        .all()
    )

    # Group findings per day
    daily_map: Dict[str, Dict[str, int]] = {}
    for i in range(days):
        day_str = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_map[day_str] = {"fake": 0, "real": 0, "total": 0}

    for f in findings:
        day_str = f.created_at.strftime("%Y-%m-%d")
        if day_str not in daily_map:
            daily_map[day_str] = {"fake": 0, "real": 0, "total": 0}
        daily_map[day_str]["total"] += 1
        if f.verdict == "FAKE":
            daily_map[day_str]["fake"] += 1
        elif f.verdict == "REAL":
            daily_map[day_str]["real"] += 1

    timeline_points = [
        {"date": date, "fake": counts["fake"], "real": counts["real"], "total": counts["total"]}
        for date, counts in sorted(daily_map.items())
    ]

    return {"days": days, "points": timeline_points}


@router.get("/org/heatmap")
def get_org_heatmap(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    # Returns 7-day x 4-interval activity matrix
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    intervals = ["00-06h", "06-12h", "12-18h", "18-24h"]
    matrix = []
    for d in days:
        for itv in intervals:
            matrix.append({"day": d, "interval": itv, "activity": 1 if d in ["Mon", "Wed", "Fri"] else 0})
    return {"matrix": matrix}
