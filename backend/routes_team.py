import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Organization, OrgMember, AuditLog
from backend.deps import get_current_user, require_investigator, get_current_org, require_org_admin
from backend.schemas import (
    OrgMemberResponse,
    InviteMember,
    UpdateMemberRole,
    AuditLogResponse,
    UserResponse,
)
from backend import crud
from backend.utils.audit import log_audit

router = APIRouter()


@router.get("/members", response_model=List[OrgMemberResponse])
def get_team_members(
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    members = crud.list_org_members(db=db, org_id=org.id)
    results = []
    for m in members:
        u = crud.get_user_by_id(db, m.user_id)
        results.append(OrgMemberResponse(
            org_id=m.org_id,
            user_id=m.user_id,
            role=m.role,
            joined_at=m.joined_at,
            user=UserResponse.model_validate(u) if u else None,
        ))
    return results


@router.post("/invite")
def invite_member(
    data: InviteMember,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    # Check if user already exists
    target_user = crud.get_user_by_email(db, data.email)
    invite_token = uuid.uuid4().hex

    if target_user:
        crud.add_org_member(db=db, org_id=org.id, user_id=target_user.id, role=data.role)
        crud.update_user_role_tier(db=db, user=target_user, role="investigator", tier=1, org_id=org.id)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="team_invite",
        target_type="user",
        target_id=str(target_user.id) if target_user else None,
        metadata={"email": data.email, "role": data.role},
        ip=client_ip,
        user_agent=user_agent,
    )

    base_url = str(request.base_url).rstrip("/")
    invite_url = f"{base_url}/signup?invite={invite_token}&org={org.id}&role={data.role}&email={data.email}"

    return {
        "status": "invited",
        "email": data.email,
        "role": data.role,
        "invite_link": invite_url,
        "message": "Invite generated successfully.",
    }


@router.patch("/members/{member_user_id}", response_model=OrgMemberResponse)
def update_member_role(
    member_user_id: uuid.UUID,
    data: UpdateMemberRole,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(require_org_admin),
    db: Session = Depends(get_db),
):
    member = (
        db.query(OrgMember)
        .filter(OrgMember.org_id == org.id, OrgMember.user_id == member_user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="Member not found in organization")

    member.role = data.role
    db.commit()
    db.refresh(member)

    u = crud.get_user_by_id(db, member.user_id)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="team_role_change",
        target_type="user",
        target_id=str(member_user_id),
        metadata={"new_role": data.role},
        ip=client_ip,
        user_agent=user_agent,
    )

    return OrgMemberResponse(
        org_id=member.org_id,
        user_id=member.user_id,
        role=member.role,
        joined_at=member.joined_at,
        user=UserResponse.model_validate(u) if u else None,
    )


@router.delete("/members/{member_user_id}")
def remove_member(
    member_user_id: uuid.UUID,
    request: Request,
    user: User = Depends(require_investigator),
    org: Organization = Depends(require_org_admin),
    db: Session = Depends(get_db),
):
    if member_user_id == user.id:
        raise HTTPException(status_code=400, detail="Cannot remove yourself from organization")

    removed = crud.remove_org_member(db=db, org_id=org.id, user_id=member_user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Member not found in organization")

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id,
        action="team_member_remove",
        target_type="user",
        target_id=str(member_user_id),
        ip=client_ip,
        user_agent=user_agent,
    )

    return {"status": "removed", "user_id": str(member_user_id)}


@router.get("/audit", response_model=List[AuditLogResponse])
def get_team_audit_log(
    action: Optional[str] = None,
    limit: int = 100,
    user: User = Depends(require_investigator),
    org: Organization = Depends(get_current_org),
    db: Session = Depends(get_db),
):
    logs = crud.list_audit_entries(db=db, org_id=org.id, action=action, limit=limit)
    results = []
    for l in logs:
        u = crud.get_user_by_id(db, l.user_id) if l.user_id else None
        results.append(AuditLogResponse(
            id=l.id,
            user_id=l.user_id,
            org_id=l.org_id,
            action=l.action,
            target_type=l.target_type,
            target_id=l.target_id,
            metadata=l.audit_metadata or {},
            ip_address=str(l.ip_address) if l.ip_address else None,
            user_agent=l.user_agent,
            created_at=l.created_at,
            user_name=u.display_name if u else "System",
        ))
    return results
