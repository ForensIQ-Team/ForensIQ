# ================================================================
# ForensIQ — Perceptual Hash (pHash) Verifier
#
# Detects exact and near-duplicate images regardless of resizing
# or re-compression. Uses a 256-bit (hash_size=16) perceptual hash.
# ================================================================

from __future__ import annotations

import logging
from typing import Optional

import imagehash
from PIL import Image

logger = logging.getLogger("forensiq-propagation")


def compute_phash(image: Image.Image) -> Optional[imagehash.ImageHash]:
    """
    Computes a 256-bit perceptual hash (hash_size=16) for the given PIL image.
    Returns None if hash computation fails.
    """
    try:
        return imagehash.phash(image.convert("RGB"), hash_size=16)
    except Exception as exc:
        logger.warning(f"[pHash] failed: {exc}")
        return None


def hamming_distance(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    """
    Computes Hamming distance between two image hashes.
    Subtraction between imagehash.ImageHash objects returns bit difference.
    """
    return int(h1 - h2)


def classify_phash(distance: int) -> str:
    """
    Classifies match severity based on pHash Hamming distance:
      distance <= 6  -> 'exact_duplicate'
      distance <= 14 -> 'near_duplicate'
      otherwise      -> 'no_match'
    """
    if distance <= 6:
        return "exact_duplicate"
    if distance <= 14:
        return "near_duplicate"
    return "no_match"
