import hashlib
from pathlib import Path
from typing import Union


def sha256_of_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_of_file(path: Union[str, Path]) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# TODO: pHash + CLIP
def compute_phash_placeholder(path: Union[str, Path]) -> None:
    """Placeholder for pHash calculation. To be implemented later."""
    return None


# TODO: pHash + CLIP
def compute_clip_embedding_placeholder(path: Union[str, Path]) -> None:
    """Placeholder for CLIP embedding extraction. To be implemented later."""
    return None
