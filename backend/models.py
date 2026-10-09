import uuid
from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    BigInteger,
    Float,
    DateTime,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET, ARRAY
from sqlalchemy.orm import relationship
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    display_name = Column(Text, nullable=True)
    role = Column(Text, nullable=False, default="user")  # user | investigator | admin
    tier = Column(Integer, nullable=False, default=0)    # 0..3
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    organization = relationship("Organization", foreign_keys=[org_id], back_populates="members_users")


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    domain = Column(Text, unique=True, nullable=True)
    website = Column(Text, nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    verification_method = Column(Text, nullable=True)  # email_domain | manual | dns
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    members_users = relationship("User", foreign_keys=[User.org_id], back_populates="organization")
    memberships = relationship("OrgMember", back_populates="organization", cascade="all, delete-orphan")


class OrgMember(Base):
    __tablename__ = "org_members"

    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role = Column(Text, nullable=False, default="analyst")  # owner | admin | analyst | viewer
    joined_at = Column(DateTime(timezone=True), server_default=func.now())

    organization = relationship("Organization", back_populates="memberships")
    user = relationship("User")


class InvestigatorApplication(Base):
    __tablename__ = "investigator_applications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    organization_name = Column(Text, nullable=True)
    designation = Column(Text, nullable=True)
    work_email = Column(Text, nullable=True)
    org_website = Column(Text, nullable=True)
    linkedin_url = Column(Text, nullable=True)
    reason = Column(Text, nullable=True)
    reference_email = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="pending")  # pending | approved | rejected
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])


class Case(Base):
    __tablename__ = "cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="open")  # open | in_progress | closed | escalated | archived
    severity = Column(Text, nullable=False, default="medium")  # low | medium | high | critical
    tags = Column(ARRAY(Text), nullable=True, default=list)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    organization = relationship("Organization")
    creator = relationship("User", foreign_keys=[created_by])
    assignee = relationship("User", foreign_keys=[assigned_to])
    findings = relationship("CaseFinding", back_populates="case", cascade="all, delete-orphan")
    timeline = relationship("CaseTimeline", back_populates="case", cascade="all, delete-orphan")
    notes = relationship("CaseNote", back_populates="case", cascade="all, delete-orphan")
    exports = relationship("ReportExport", back_populates="case", cascade="all, delete-orphan")


class CaseFinding(Base):
    __tablename__ = "case_findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False, index=True)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    media_type = Column(Text, nullable=False)  # image | video
    original_filename = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    sha256 = Column(Text, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    verdict = Column(Text, nullable=False)  # FAKE | REAL | INCONCLUSIVE
    confidence = Column(Float, nullable=False)
    analysis_json = Column(JSONB, nullable=False)
    gradcam_path = Column(Text, nullable=True)
    report_id = Column(Text, nullable=True)
    frozen = Column(Boolean, default=False)
    frozen_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    frozen_at = Column(DateTime(timezone=True), nullable=True)

    # TODO: pHash + CLIP
    # phash = Column(BigInteger, nullable=True)
    # clip_embedding = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="findings")
    uploader = relationship("User", foreign_keys=[uploaded_by])
    freezer = relationship("User", foreign_keys=[frozen_by])
    evidence_locks = relationship("EvidenceLock", back_populates="finding", cascade="all, delete-orphan")


class CaseTimeline(Base):
    __tablename__ = "case_timeline"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    actor_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(Text, nullable=False)  # create | update | upload | analysis | note | share | freeze | unfreeze | export | assign | status_change
    payload = Column(JSONB, nullable=True, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="timeline")
    actor = relationship("User")

    __table_args__ = (
        Index("ix_case_timeline_case_created", "case_id", created_at.desc()),
    )


class CaseNote(Base):
    __tablename__ = "case_notes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="notes")
    author = relationship("User")


class EvidenceLock(Base):
    __tablename__ = "evidence_locks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    finding_id = Column(UUID(as_uuid=True), ForeignKey("case_findings.id", ondelete="CASCADE"), nullable=False)
    locked_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    reason = Column(Text, nullable=True)
    content_hash = Column(Text, nullable=False)
    chain_hash = Column(Text, nullable=False)
    locked_at = Column(DateTime(timezone=True), server_default=func.now())

    finding = relationship("CaseFinding", back_populates="evidence_locks")
    locker = relationship("User")


class ReportExport(Base):
    __tablename__ = "report_exports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="CASCADE"), nullable=False)
    finding_id = Column(UUID(as_uuid=True), ForeignKey("case_findings.id", ondelete="SET NULL"), nullable=True)
    exported_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    export_type = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False)
    sha256 = Column(Text, nullable=False)
    signed_hash = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    case = relationship("Case", back_populates="exports")
    finding = relationship("CaseFinding")
    exporter = relationship("User")


class Watchlist(Base):
    __tablename__ = "watchlists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False)
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    target_type = Column(Text, nullable=False)  # celebrity | brand | media_hash | custom
    target_query = Column(Text, nullable=False)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    alerts = relationship("WatchlistAlert", back_populates="watchlist", cascade="all, delete-orphan")
    organization = relationship("Organization")
    creator = relationship("User")


class WatchlistAlert(Base):
    __tablename__ = "watchlist_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    watchlist_id = Column(UUID(as_uuid=True), ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False)
    source_url = Column(Text, nullable=False)
    source_type = Column(Text, nullable=False)
    severity = Column(Text, nullable=False)  # low | medium | high | critical
    match_score = Column(Float, nullable=True)
    snapshot_path = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="new")  # new | triaged | dismissed | escalated
    triaged_by = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    triaged_at = Column(DateTime(timezone=True), nullable=True)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    watchlist = relationship("Watchlist", back_populates="alerts")
    triager = relationship("User", foreign_keys=[triaged_by])
    case = relationship("Case")

    __table_args__ = (
        Index("ix_watchlist_alerts_status_created", "watchlist_id", "status", created_at.desc()),
    )


class SourceReputation(Base):
    __tablename__ = "source_reputations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False)
    domain = Column(Text, nullable=False)
    total_findings = Column(Integer, default=0)
    fake_findings = Column(Integer, default=0)
    fake_ratio = Column(Float, default=0.0)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)

    organization = relationship("Organization")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    action = Column(Text, nullable=False)
    target_type = Column(Text, nullable=True)
    target_id = Column(Text, nullable=True)
    audit_metadata = Column("metadata", JSONB, nullable=True)
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")
    organization = relationship("Organization")

    __table_args__ = (
        Index("ix_audit_log_org_created", "org_id", created_at.desc()),
    )

