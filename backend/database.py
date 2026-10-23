import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, INET, UUID
from dotenv import load_dotenv

logger = logging.getLogger("forensiq-db")

# SQLite compatibility handlers for PostgreSQL column types
@compiles(ARRAY, "sqlite")
def compile_array_sqlite(type_, compiler, **kw):
    return "TEXT"

@compiles(JSONB, "sqlite")
def compile_jsonb_sqlite(type_, compiler, **kw):
    return "JSON"

@compiles(INET, "sqlite")
def compile_inet_sqlite(type_, compiler, **kw):
    return "TEXT"

@compiles(UUID, "sqlite")
def compile_uuid_sqlite(type_, compiler, **kw):
    return "CHAR(36)"

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
if DATABASE_URL and "postgresql" in DATABASE_URL:
    try:
        pg_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            echo=False,
            connect_args={"connect_timeout": 2},
        )
        # Must actually execute a query — SQLAlchemy lazily establishes
        # connections so `with engine.connect(): pass` is a no-op.
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine = pg_engine
        print("[DB] Connected to PostgreSQL")
        logger.info("[DB] Connected to PostgreSQL")
    except Exception as e:
        print(f"[DB] PostgreSQL connection failed ({e}). Falling back to SQLite.")
        logger.warning(f"[DB] PostgreSQL connection failed ({e}). Falling back to SQLite.")

if engine is None:
    sqlite_path = Path(__file__).resolve().parent.parent / "forensiq.db"
    engine = create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    print(f"[DB] Using SQLite at {sqlite_path}")
    logger.info(f"[DB] Using SQLite database at {sqlite_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
