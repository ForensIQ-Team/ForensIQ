import os
from pathlib import Path
from typing import Union
from backend.utils.hashing import sha256_of_file

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE_ROOT = WORKSPACE_ROOT / "storage"


def ensure_storage_dirs():
    (STORAGE_ROOT / "users").mkdir(parents=True, exist_ok=True)
    (STORAGE_ROOT / "orgs").mkdir(parents=True, exist_ok=True)
    (STORAGE_ROOT / "exports").mkdir(parents=True, exist_ok=True)


def save_user_finding(user_id: Union[str, any], finding_id: Union[str, any], file_bytes: bytes, filename: str) -> Path:
    ensure_storage_dirs()
    dest_dir = STORAGE_ROOT / "users" / str(user_id) / "findings" / str(finding_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix or ".bin"
    file_path = dest_dir / f"original{ext}"
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path


def save_org_finding(org_id: Union[str, any], case_id: Union[str, any], finding_id: Union[str, any], file_bytes: bytes, filename: str) -> Path:
    ensure_storage_dirs()
    dest_dir = STORAGE_ROOT / "orgs" / str(org_id) / "cases" / str(case_id) / "findings" / str(finding_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix or ".bin"
    file_path = dest_dir / f"original{ext}"
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path


def save_export(export_id: Union[str, any], content_bytes: bytes, ext: str = "html") -> Path:
    ensure_storage_dirs()
    if not ext.startswith("."):
        ext = f".{ext}"
    dest_file = STORAGE_ROOT / "exports" / f"{export_id}{ext}"
    with open(dest_file, "wb") as f:
        f.write(content_bytes)
    return dest_file
