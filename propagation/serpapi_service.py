# ================================================================
# ForensIQ — SerpApi / Google Lens Service
#
# This module is the ONLY place that communicates with SerpApi.
# It does NOT touch the detection pipeline.
#
# Architecture:
#   1. _upload_image_to_serpapi() — Uploads image to SerpApi's /image
#      endpoint (compressing/resizing via PIL if > 450 KB to stay within
#      the 500 KB limit) and gets an image_id.
#   2. search_google_lens()       — Queries SerpApi Google Lens with
#      image_id and returns normalised results.
#   3. _normalize_result()        — Converts raw SerpApi output to
#      clean internal schema.
#
# KEY DESIGN DECISIONS:
#   - SerpApi key is NEVER logged or returned to the frontend.
#   - We use the google-search-results SDK.
#   - All fields are treated as optional — missing fields are None.
#   - No similarity scores are invented; only returned values passed.
#   - This module is designed for pHash/CLIP to plug in later.
# ================================================================

from __future__ import annotations

import io
import os
import re
import logging
from pathlib import Path
from typing import Any

from PIL import Image
import requests
from serpapi import GoogleSearch

logger = logging.getLogger("forensiq-propagation")


# ----------------------------------------------------------------
# Key resolution
# ----------------------------------------------------------------

def _get_api_key() -> str:
    key = os.environ.get("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SERPAPI_KEY is not set. "
            "Add it to the .env file and restart the server."
        )
    return key


# ----------------------------------------------------------------
# Domain helpers
# ----------------------------------------------------------------

_CCTLD_MAP: dict[str, str] = {
    ".uk": "United Kingdom",
    ".us": "United States",
    ".au": "Australia",
    ".de": "Germany",
    ".fr": "France",
    ".jp": "Japan",
    ".cn": "China",
    ".in": "India",
    ".br": "Brazil",
    ".ca": "Canada",
    ".ru": "Russia",
    ".it": "Italy",
    ".es": "Spain",
    ".nl": "Netherlands",
    ".se": "Sweden",
    ".no": "Norway",
    ".pl": "Poland",
    ".za": "South Africa",
    ".mx": "Mexico",
    ".kr": "South Korea",
}


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        m = re.search(r"https?://([^/]+)", url)
        if m:
            host = m.group(1).lower().lstrip("www.")
            return host
    except Exception:
        pass
    return None


def _guess_region(domain: str | None) -> str | None:
    """
    Only returns a region when it can be reliably derived from a
    country-code TLD. Returns None (→ displayed as 'Unknown') otherwise.
    """
    if not domain:
        return None
    for cctld, country in _CCTLD_MAP.items():
        if domain.endswith(cctld):
            return country
    return None


# ----------------------------------------------------------------
# Result normalizer
# ----------------------------------------------------------------

def _normalize_result(raw: dict[str, Any], match_type: str) -> dict[str, Any]:
    """
    Convert a single raw SerpApi result dict into the ForensIQ schema.
    All fields are optional — missing ones become None.
    """
    url = raw.get("link") or raw.get("url") or None
    domain = _extract_domain(url)

    # Thumbnail: SerpApi may return 'thumbnail', 'image', or nothing
    thumbnail = (
        raw.get("thumbnail")
        or raw.get("image")
        or None
    )

    return {
        "title": raw.get("title") or None,
        "url": url,
        "domain": domain,
        "thumbnail": thumbnail,
        "source": raw.get("source") or domain,
        "match_type": match_type,
        "snippet": raw.get("snippet") or raw.get("description") or None,
        "date": raw.get("date") or None,
        "region": _guess_region(domain),
        # Placeholder slots for future pHash / CLIP verification
        "phash_verified": None,
        "clip_verified": None,
    }


# ----------------------------------------------------------------
# Image upload helper (SerpApi /image endpoint)
# ----------------------------------------------------------------

def _prepare_image_bytes(image_path: Path) -> tuple[bytes, str]:
    """
    Ensures image bytes are <= 480 KB (SerpApi limit is 500 KB).
    Resizes/compresses with PIL if needed.
    Returns (bytes, mime_type).
    """
    file_size = image_path.stat().st_size
    if file_size <= 450 * 1024:
        with open(image_path, "rb") as f:
            return f.read(), "image/jpeg"

    # Resize/compress if oversized
    logger.info(f"[SerpApi] Image is {file_size / 1024:.1f} KB; compressing to < 500 KB...")
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        # Scale down if large resolution
        max_dim = 1200
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        quality = 85
        while quality >= 40:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality, optimize=True)
            if buf.tell() <= 450 * 1024:
                return buf.getvalue(), "image/jpeg"
            quality -= 15

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=35, optimize=True)
        return buf.getvalue(), "image/jpeg"


def _upload_image_to_serpapi(image_path: Path) -> str:
    """
    Uploads a local image to SerpApi's /image endpoint to obtain an image_id.
    """
    api_key = _get_api_key()
    img_bytes, mime = _prepare_image_bytes(image_path)

    upload_url = "https://serpapi.com/image"
    try:
        resp = requests.post(
            upload_url,
            files={"image": ("upload.jpg", io.BytesIO(img_bytes), mime)},
            data={"api_key": api_key},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            image_id = data.get("image_id")
            if image_id:
                logger.info(f"[SerpApi] Image uploaded successfully, image_id: {image_id[:16]}...")
                return image_id

        err_detail = resp.text[:200]
        logger.warning(f"[SerpApi] Upload failed with status {resp.status_code}: {err_detail}")
        if resp.status_code == 401 or "invalid" in err_detail.lower():
            raise RuntimeError("SerpApi authentication failed. Please verify SERPAPI_KEY.")
        if "quota" in err_detail.lower() or "limit" in err_detail.lower():
            raise RuntimeError("SerpApi monthly quota exceeded.")
    except (RuntimeError, requests.RequestException):
        raise
    except Exception as exc:
        logger.warning(f"[SerpApi] Upload error: {exc}")

    raise RuntimeError(
        "Failed to upload image to search service. "
        "Please check your internet connection and API key."
    )


# ----------------------------------------------------------------
# Core SerpApi call — Google Lens
# ----------------------------------------------------------------

def search_google_lens(image_path: Path) -> dict[str, Any]:
    """
    Given a local image path, uploads to SerpApi and executes Google Lens search.
    Returns structured propagation results.
    """
    api_key = _get_api_key()

    logger.info(f"[SerpApi] Starting Google Lens search for: {image_path.name}")
    image_id = _upload_image_to_serpapi(image_path)

    params = {
        "engine": "google_lens",
        "image_id": image_id,
        "api_key": api_key,
    }

    try:
        search = GoogleSearch(params)
        raw = search.get_dict()
    except Exception as exc:
        logger.exception(f"[SerpApi] GoogleSearch call failed: {exc}")
        raise RuntimeError(f"Search query failed: {exc}")

    # ---- Error handling ----
    if "error" in raw:
        err_msg = str(raw["error"])
        if "quota" in err_msg.lower() or "limit" in err_msg.lower():
            raise RuntimeError(
                "Propagation search is temporarily unavailable "
                "because the search API quota has been reached."
            )
        if "invalid api" in err_msg.lower() or "authenticate" in err_msg.lower():
            raise RuntimeError("SerpApi authentication failed. Check SERPAPI_KEY.")
        raise RuntimeError(f"SerpApi error: {err_msg}")

    return _parse_lens_response(raw)


def _parse_lens_response(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Parse the Google Lens SerpApi response into ForensIQ schema.
    Returns:
        {
            "total": int,
            "exact_matches": [...],
            "visual_matches": [...],
            "all_results": [...],
        }
    """
    exact_raw = raw.get("exact_matches", []) or []
    visual_raw = raw.get("visual_matches", []) or []

    # Fallbacks if keys differ in some responses
    if not visual_raw:
        visual_raw = raw.get("lens_results", []) or raw.get("organic_results", []) or []

    exact_matches = [_normalize_result(r, "exact") for r in exact_raw]
    visual_matches = [_normalize_result(r, "visual") for r in visual_raw]

    all_results = exact_matches + visual_matches
    total = len(all_results)

    logger.info(
        f"[SerpApi] Lens returned {len(exact_matches)} exact + "
        f"{len(visual_matches)} visual = {total} total results."
    )

    return {
        "total": total,
        "exact_matches": exact_matches,
        "visual_matches": visual_matches,
        "all_results": all_results,
    }

