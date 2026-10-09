import uuid
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Organization, OrgMember, InvestigatorApplication
from backend.security import verify_password, create_access_token
from backend.deps import get_current_user
from backend.schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    TokenResponse,
    InvestigatorApply,
    InvestigatorStatus,
    OrgResponse,
    InvestigatorApplicationResponse,
)
from backend.crud import (
    create_user,
    get_user_by_email,
    update_last_login,
    update_user_role_tier,
    create_organization,
    get_org_by_domain,
    add_org_member,
    create_application,
    get_application_by_user,
)
from backend.utils.audit import log_audit

router = APIRouter()

PUBLIC_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
    "protonmail.com",
    "aol.com",
    "live.com",
    "mail.com",
    "zoho.com",
    "yandex.com",
}


@router.post("/signup", response_model=TokenResponse)
def signup(data: UserCreate, request: Request, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, data.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    user = create_user(
        db=db,
        email=data.email,
        password=data.password,
        display_name=data.display_name,
    )

    token = create_access_token(str(user.id))

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=user.org_id,
        action="user_signup",
        target_type="user",
        target_id=str(user.id),
        metadata={"email": user.email},
        ip=client_ip,
        user_agent=user_agent,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.post("/login", response_model=TokenResponse)
def login(data: UserLogin, request: Request, db: Session = Depends(get_db)):
    user = get_user_by_email(db, data.email)
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    update_last_login(db, user)
    token = create_access_token(str(user.id))

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=user.org_id,
        action="user_login",
        target_type="user",
        target_id=str(user.id),
        metadata={"email": user.email},
        ip=client_ip,
        user_agent=user_agent,
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    return UserResponse.model_validate(user)


@router.post("/logout")
def logout(user: User = Depends(get_current_user)):
    return {"status": "logged_out", "message": "Successfully logged out"}


@router.post("/investigator/apply")
def apply_investigator(
    data: InvestigatorApply,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    work_email = data.work_email.strip().lower()
    email_domain = work_email.split("@")[-1] if "@" in work_email else ""

    if not email_domain:
        raise HTTPException(status_code=400, detail="Invalid work email format")

    is_public = email_domain in PUBLIC_EMAIL_DOMAINS
    status_str = "pending" if is_public else "approved"

    application = create_application(
        db=db,
        user_id=user.id,
        work_email=work_email,
        organization_name=data.organization_name,
        designation=data.designation,
        org_website=data.org_website,
        linkedin_url=data.linkedin_url,
        reason=data.reason,
        reference_email=data.reference_email,
        status=status_str,
    )

    org = None
    if not is_public:
        # Organization domain matching or auto-creation
        org = get_org_by_domain(db, email_domain)
        if not org:
            org_name = data.organization_name or email_domain.split(".")[0].capitalize()
            org = create_organization(
                db=db,
                name=org_name,
                domain=email_domain,
                website=data.org_website or f"https://{email_domain}",
                verification_method="email_domain",
                verified_at=datetime.now(timezone.utc),
            )

        # Set user as owner if first member, or analyst
        existing_members = db.query(OrgMember).filter(OrgMember.org_id == org.id).count()
        member_role = "owner" if existing_members == 0 else "analyst"
        add_org_member(db=db, org_id=org.id, user_id=user.id, role=member_role)

        # Upgrade user
        user = update_user_role_tier(db=db, user=user, role="investigator", tier=1, org_id=org.id)
    else:
        # User stays tier 0 pending review
        user = update_user_role_tier(db=db, user=user, role=user.role, tier=0)

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    log_audit(
        db=db,
        user_id=user.id,
        org_id=org.id if org else None,
        action="investigator_apply",
        target_type="investigator_application",
        target_id=str(application.id),
        metadata={"status": status_str, "domain": email_domain},
        ip=client_ip,
        user_agent=user_agent,
    )

    return {
        "user": UserResponse.model_validate(user),
        "organization": OrgResponse.model_validate(org) if org else None,
        "application": InvestigatorApplicationResponse.model_validate(application),
    }


@router.get("/investigator/status", response_model=InvestigatorStatus)
def investigator_status(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    application = get_application_by_user(db, user.id)
    org = db.query(Organization).filter(Organization.id == user.org_id).first() if user.org_id else None

    status_str = application.status if application else ("approved" if user.role == "investigator" else "none")

    return InvestigatorStatus(
        role=user.role,
        tier=user.tier,
        status=status_str,
        organization=OrgResponse.model_validate(org) if org else None,
        application=InvestigatorApplicationResponse.model_validate(application) if application else None,
    )


@router.post("/investigator/verify-dns")
def verify_dns_stub(user: User = Depends(get_current_user)):
    return {
        "status": "pending",
        "message": "DNS verification feature coming soon. Please contact administrator for manual verification.",
    }
