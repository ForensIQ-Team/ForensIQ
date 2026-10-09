import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Organization, OrgMember
from backend.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    if not token:
        return None
    user_id = decode_access_token(token)
    if not user_id:
        return None
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        return None
    return db.query(User).filter(User.id == user_uuid, User.is_active == True).first()


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    user_id = decode_access_token(token)
    if not user_id:
        raise credentials_exception
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_uuid).first()
    if not user or not user.is_active:
        raise credentials_exception
    return user


def require_role(required_role: str):
    def _role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires '{required_role}' role.",
            )
        return user
    return _role_checker


def require_investigator(user: User = Depends(get_current_user)) -> User:
    if user.role not in ("investigator", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Investigator access required.",
        )
    return user


def require_verified_investigator(user: User = Depends(get_current_user)) -> User:
    if user.role not in ("investigator", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Investigator access required.",
        )
    if user.tier < 1 and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verified investigator tier (Tier 1+) required.",
        )
    return user


def get_current_org(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Organization:
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not currently associated with an organization.",
        )
    org = db.query(Organization).filter(Organization.id == user.org_id).first()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found.",
        )
    return org


def require_org_admin(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Organization:
    if not user.org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not associated with an organization.",
        )
    if user.role == "admin":
        org = db.query(Organization).filter(Organization.id == user.org_id).first()
        if not org:
            raise HTTPException(status_code=404, detail="Organization not found")
        return org

    membership = (
        db.query(OrgMember)
        .filter(OrgMember.org_id == user.org_id, OrgMember.user_id == user.id)
        .first()
    )
    if not membership or membership.role not in ("owner", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization admin privileges required.",
        )
    org = db.query(Organization).filter(Organization.id == user.org_id).first()
    return org
