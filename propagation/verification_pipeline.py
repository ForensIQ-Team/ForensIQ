# ================================================================
# ForensIQ — Identity Verification Pipeline
#
# Orchestrates parallel discovery (Google Lens + Yandex Images)
# followed by 2-stage verification:
#   1. Perceptual hash (pHash) for exact/near duplicates
#   2. Face embedding (MTCNN + InceptionResnetV1) for facial identity
# ================================================================

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from PIL import Image

from propagation.face_verifier import (
    FaceEmbeddingResult,
    classify_face_match,
    cosine_similarity,
    extract_face_embedding,
)
from propagation.phash_verifier import (
    classify_phash,
    compute_phash,
    hamming_distance,
)
from propagation.serpapi_service import (
    search_google_lens,
    search_yandex_images,
)

logger = logging.getLogger("forensiq-propagation")

MAX_CANDIDATES = 60          # merged pool cap
PER_ENGINE_LIMIT = 40        # per-engine cap
FETCH_TIMEOUT = 8            # seconds
MAX_IMAGE_BYTES = 8 * 1024 * 1024
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (ForensIQ/1.0)"}
MAX_FETCH_WORKERS = 6


def _fetch_image(url: str) -> Optional[Image.Image]:
    """
    Downloads candidate image bytes from url.
    Rejects non-images or files exceeding MAX_IMAGE_BYTES.
    Returns RGB PIL Image or None on any failure.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        resp = requests.get(
            url,
            timeout=FETCH_TIMEOUT,
            stream=True,
            headers=HTTP_HEADERS,
        )
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("Content-Type", "").lower()
        if content_type and not content_type.startswith("image/"):
            return None

        buf = io.BytesIO()
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                buf.write(chunk)
                total_bytes += len(chunk)
                if total_bytes > MAX_IMAGE_BYTES:
                    return None

        buf.seek(0)
        img = Image.open(buf)
        img.load()
        return img.convert("RGB")
    except Exception as exc:
        logger.debug(f"[pipeline] _fetch_image failed for {url[:50]}: {exc}")
        return None


def run_verified_search(
    image_path: Path,
    progress_callback: Optional[Callable[[str, int], None]] = None,
) -> dict[str, Any]:
    """
    Runs the identity-verified search pipeline for an uploaded image.
    Returns candidate statistics and verified match categories.
    """
    if progress_callback:
        progress_callback("Searching public web indexes...", 0)

    # ----------------------------------------------------------------
    # STEP 1 — DISCOVERY (parallel Google Lens + Yandex Images)
    # ----------------------------------------------------------------
    stats: dict[str, Any] = {
        "google": 0,
        "yandex": 0,
        "google_error": None,
        "yandex_error": None,
    }
    raw_results: list[tuple[str, list[dict[str, Any]]]] = []

    def run_google() -> tuple[str, list[dict[str, Any]], Optional[str]]:
        try:
            r = search_google_lens(image_path)
            return "google", r.get("all_results", []), None
        except Exception as exc:
            return "google", [], str(exc)

    def run_yandex() -> tuple[str, list[dict[str, Any]], Optional[str]]:
        try:
            r = search_yandex_images(image_path)
            return "yandex", r.get("all_results", []), None
        except Exception as exc:
            return "yandex", [], str(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_google), pool.submit(run_yandex)]
        for f in as_completed(futures):
            engine, items, err = f.result()
            if err:
                stats[f"{engine}_error"] = err
                logger.warning(f"[pipeline] Discovery {engine} failed: {err}")
            else:
                stats[engine] = len(items)
                raw_results.append((engine, items))

    # Merge & deduplicate candidates
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for engine, items in raw_results:
        engine_count = 0
        for item in items:
            if engine_count >= PER_ENGINE_LIMIT:
                break
            url = item.get("url") or item.get("thumbnail") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            tagged = dict(item)
            tagged["_engine"] = engine
            candidates.append(tagged)
            engine_count += 1
            if len(candidates) >= MAX_CANDIDATES:
                break
        if len(candidates) >= MAX_CANDIDATES:
            break

    logger.info(
        f"[pipeline] discovery: google={stats['google']} yandex={stats['yandex']} "
        f"merged_unique={len(candidates)}"
    )

    # ----------------------------------------------------------------
    # STEP 2 — QUERY EMBEDDINGS
    # ----------------------------------------------------------------
    if progress_callback:
        progress_callback("Analyzing query face...", 1)

    try:
        q_img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.error(f"[pipeline] Failed to open query image: {exc}")
        q_img = None

    q_phash = compute_phash(q_img) if q_img is not None else None
    q_face = (
        extract_face_embedding(q_img)
        if q_img is not None
        else FaceEmbeddingResult(success=False, embedding=None, num_faces=0, error="query_img_load_fail")
    )

    if not q_face.success:
        logger.warning(
            f"[pipeline] Query face extraction failed ({q_face.error}); "
            "face verification skipped, only pHash will be used."
        )

    # ----------------------------------------------------------------
    # STEP 3 — VERIFY EACH CANDIDATE (parallel workers)
    # ----------------------------------------------------------------
    if progress_callback:
        progress_callback(f"Verifying {len(candidates)} candidates...", 2)

    def process(item: dict[str, Any]) -> dict[str, Any]:
        # Try thumbnail first (smaller/faster), fallback to full url
        c_img = _fetch_image(item.get("thumbnail") or "") or _fetch_image(item.get("url") or "")

        verification: dict[str, Any] = {
            "status": "unverifiable",
            "face_status": "not_checked",
            "face_similarity": None,
            "face_verdict": None,
            "phash_distance": None,
            "phash_verdict": None,
        }

        if c_img is None:
            return {**item, "verification": verification}

        verification["status"] = "checked"

        # pHash
        if q_phash is not None:
            c_hash = compute_phash(c_img)
            if c_hash is not None:
                d = hamming_distance(q_phash, c_hash)
                verification["phash_distance"] = d
                verification["phash_verdict"] = classify_phash(d)

        # Face
        if q_face.success and q_face.embedding is not None:
            c_face = extract_face_embedding(c_img)
            if c_face.success and c_face.embedding is not None:
                sim = cosine_similarity(q_face.embedding, c_face.embedding)
                verification["face_similarity"] = sim
                verification["face_verdict"] = classify_face_match(sim)
                verification["face_status"] = "ok"
            else:
                verification["face_status"] = f"no_face_in_candidate:{c_face.error}"
        else:
            verification["face_status"] = "no_face_in_query"

        return {**item, "verification": verification}

    verified: list[dict[str, Any]] = []
    if candidates:
        with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as pool:
            futures = [pool.submit(process, c) for c in candidates]
            for f in as_completed(futures):
                try:
                    r = f.result()
                    if r:
                        verified.append(r)
                except Exception as exc:
                    logger.debug(f"[pipeline] worker failed: {exc}")

    # ----------------------------------------------------------------
    # STEP 4 — RANK by confidence score (no filtering)
    # ----------------------------------------------------------------
    if progress_callback:
        progress_callback("Ranking results by confidence...", 3)

    def _rank_score(item: dict[str, Any]) -> float:
        """
        Composite confidence score in [0, 1].

        Exact pHash duplicate  → 1.0
        Same person (face)     → 0.70 + 0.30 * face_similarity
        Near pHash duplicate   → up to 0.80, decays with Hamming distance
        Similar face           → 0.20 + 0.40 * face_similarity
        Weak face similarity   → face_similarity * 0.18
        Unverifiable / no-face → 0.0
        """
        v = item.get("verification", {})
        pv = v.get("phash_verdict")
        fv = v.get("face_verdict")
        fs = v.get("face_similarity")   # float | None
        pd = v.get("phash_distance")    # int   | None

        score = 0.0

        # pHash contribution
        if pv == "exact_duplicate":
            score = 1.0
        elif pv == "near_duplicate":
            # phash_distance ∈ [1, ~60] for hash_size=16; score decays linearly
            score = max(score, max(0.0, 0.80 * (1.0 - (pd or 40) / 40.0)))

        # Face contribution
        if fv == "same_person" and fs is not None:
            score = max(score, 0.70 + 0.30 * float(fs))
        elif fv == "similar_face" and fs is not None:
            score = max(score, 0.20 + 0.40 * float(fs))
        elif fs is not None and float(fs) > 0:
            score = max(score, float(fs) * 0.18)

        return round(min(score, 1.0), 4)

    def _classify_tier(score: float) -> str:
        if score >= 0.90:
            return "critical"
        if score >= 0.70:
            return "high"
        if score >= 0.45:
            return "medium"
        if score >= 0.15:
            return "low"
        return "unlikely"

    # Annotate every verified candidate with rank_score + tier
    ranked: list[dict[str, Any]] = []
    for item in verified:
        rs = _rank_score(item)
        ranked.append({**item, "rank_score": rs, "tier": _classify_tier(rs)})

    # Sort highest confidence first, then assign 1-based rank index
    ranked.sort(key=lambda x: x["rank_score"], reverse=True)
    for i, item in enumerate(ranked):
        item["rank"] = i + 1

    # Aggregate tier counts
    tier_counts: dict[str, int] = {
        "critical": 0, "high": 0, "medium": 0, "low": 0, "unlikely": 0,
    }
    for item in ranked:
        tier_counts[item["tier"]] = tier_counts.get(item["tier"], 0) + 1

    notable_count = tier_counts["critical"] + tier_counts["high"] + tier_counts["medium"]

    logger.info(
        f"[pipeline] ranked {len(ranked)} results — "
        f"critical={tier_counts['critical']} high={tier_counts['high']} "
        f"medium={tier_counts['medium']} low={tier_counts['low']} "
        f"unlikely={tier_counts['unlikely']}"
    )

    if progress_callback:
        progress_callback("Done", 4)

    return {
        "total_candidates": len(candidates),
        "ranked_results": ranked,
        "notable_count": notable_count,
        "tier_counts": tier_counts,
        "query_had_face": q_face.success,
        "engine_stats": stats,
    }
