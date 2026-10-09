from typing import Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from backend.models import AuditLog


def log_audit(
    db: Session,
    user_id: Optional[Union[str, Any]],
    org_id: Optional[Union[str, Any]],
    action: str,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> AuditLog:
    entry = AuditLog(
        user_id=user_id,
        org_id=org_id,
        action=action,
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        audit_metadata=metadata or {},
        ip_address=ip if ip and ip != "testclient" else None,
        user_agent=user_agent,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
