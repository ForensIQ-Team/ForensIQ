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
    Accepts an uploaded image and runs a Google Lens reverse image
    search via SerpApi to discover where the image appears publicly.

    Returns structured propagation results:
        {
            "status": "success" | "no_results" | "error",
            "total": int,
            "exact_matches": [...],
            "visual_matches": [...],
            "all_results": [...],
            "error": str | null,
        }

    IMPORTANT: No pHash or CLIP verification is performed here.
    Returned results are DISCOVERED CANDIDATES, not verified matches.
    """
    # ---- Import service here so startup doesn't fail if key is missing ----
    from propagation.serpapi_service import search_google_lens

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
            f"({len(file_bytes) / 1024:.1f} KB) — starting Google Lens search"
        )

        results = search_google_lens(tmp_path)

        if results["total"] == 0:
            return JSONResponse(content={
                "status": "no_results",
                "total": 0,
                "exact_matches": [],
                "visual_matches": [],
                "all_results": [],
                "error": None,
                "message": "No matching images were found on the public web.",
            })

        return JSONResponse(content={
            "status": "success",
            "total": results["total"],
            "exact_matches": results["exact_matches"],
            "visual_matches": results["visual_matches"],
            "all_results": results["all_results"],
            "error": None,
        })

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
                "total": 0,
                "exact_matches": [],
                "visual_matches": [],
                "all_results": [],
                "error": msg,
            },
        )

    except Exception as exc:
        logger.exception(f"[Propagation] Unexpected error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "total": 0,
                "exact_matches": [],
                "visual_matches": [],
                "all_results": [],
                "error": "Propagation search failed due to an unexpected error.",
            },
        )

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
