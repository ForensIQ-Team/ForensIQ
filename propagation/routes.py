# ================================================================
# ForensIQ — Propagation API Routes
#
# POST /api/propagation/search
#   Accepts an uploaded image, runs Google Lens via SerpApi,
#   and returns structured propagation results.
#
# This file does NOT import or touch anything from the detection
# pipeline (predict_v2_facecrop.py, api_server.py detection routes).
# ================================================================

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger("forensiq-propagation-routes")

router = APIRouter()

# Accepted image types for propagation search
PROP_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
PROP_MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


def _validate_image(filename: str, size: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in PROP_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format '{ext}'. "
                "Accepted formats: JPG, JPEG, PNG, WEBP."
            ),
        )
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size > PROP_MAX_SIZE_BYTES:
        mb = size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image too large ({mb:.1f} MB). Maximum is 20 MB.",
        )


@router.post("/api/propagation/search")
async def propagation_search(file: UploadFile = File(...)) -> JSONResponse:
    """
    Accepts an uploaded image and runs parallel discovery (Google Lens + Yandex)
    followed by pHash and facial embedding verification.

    Returns verified identity matches and filtered candidate statistics.
    """
    from propagation.verification_pipeline import run_verified_search

    filename = file.filename or "upload.jpg"
    file_bytes = await file.read()

    _validate_image(filename, len(file_bytes))

    ext = Path(filename).suffix.lower() or ".jpg"
    tmp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
            prefix=f"forensiq_prop_{uuid.uuid4().hex[:8]}_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        logger.info(
            f"[Propagation] Received {filename} "
            f"({len(file_bytes) / 1024:.1f} KB) — starting verified search pipeline"
        )

        pipeline_res = run_verified_search(tmp_path)
        engine_stats = pipeline_res.get("engine_stats", {
            "google": 0,
            "yandex": 0,
            "google_error": None,
            "yandex_error": None,
        })

        _empty_tiers: dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0, "unlikely": 0,
        }

        # When both engines fail and there are no candidates -> status "error" with HTTP 503
        if (
            pipeline_res.get("total_candidates", 0) == 0
            and engine_stats.get("google_error")
            and engine_stats.get("yandex_error")
        ):
            err_msg = (
                f"All search engines failed: Google: {engine_stats.get('google_error')}; "
                f"Yandex: {engine_stats.get('yandex_error')}"
            )
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "ranked_results": [],
                    "notable_count": 0,
                    "tier_counts": _empty_tiers,
                    "total_candidates": 0,
                    "query_had_face": pipeline_res.get("query_had_face", False),
                    "engine_stats": engine_stats,
                    "error": err_msg,
                },
            )

        ranked_results = pipeline_res.get("ranked_results", [])

        # When ranked_results is empty -> status "no_results"
        if not ranked_results:
            return JSONResponse(
                content={
                    "status": "no_results",
                    "ranked_results": [],
                    "notable_count": 0,
                    "tier_counts": _empty_tiers,
                    "total_candidates": pipeline_res.get("total_candidates", 0),
                    "query_had_face": pipeline_res.get("query_had_face", False),
                    "engine_stats": engine_stats,
                    "error": None,
                    "message": "No results could be retrieved from the public web.",
                }
            )

        return JSONResponse(
            content={
                "status": "success",
                "ranked_results": ranked_results,
                "notable_count": pipeline_res.get("notable_count", 0),
                "tier_counts": pipeline_res.get("tier_counts", _empty_tiers),
                "total_candidates": pipeline_res.get("total_candidates", 0),
                "query_had_face": pipeline_res.get("query_had_face", False),
                "engine_stats": engine_stats,
                "error": None,
            }
        )

    except HTTPException:
        raise

    except RuntimeError as rte:
        # Known domain errors (quota, auth, upload failure)
        msg = str(rte)
        logger.warning(f"[Propagation] Runtime error: {msg}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "ranked_results": [],
                "notable_count": 0,
                "tier_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "unlikely": 0},
                "total_candidates": 0,
                "query_had_face": False,
                "engine_stats": {
                    "google": 0,
                    "yandex": 0,
                    "google_error": msg,
                    "yandex_error": None,
                },
                "error": msg,
            },
        )

    except Exception as exc:
        logger.exception(f"[Propagation] Unexpected error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "ranked_results": [],
                "notable_count": 0,
                "tier_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "unlikely": 0},
                "total_candidates": 0,
                "query_had_face": False,
                "engine_stats": {
                    "google": 0,
                    "yandex": 0,
                    "google_error": str(exc),
                    "yandex_error": None,
                },
                "error": "Propagation search failed due to an unexpected error.",
            },
        )

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
