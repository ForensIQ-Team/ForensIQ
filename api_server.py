# ================================================================
# FORENSIQ V2.1 — FastAPI DETECTION SERVICE
#
# Wraps the existing predict_v2_facecrop.py pipeline.
#
# IMPORTANT:
#   This file is ONLY an API adapter.
#   It does NOT implement any ML logic.
#   ALL inference runs inside the existing inference functions
#   (analyze_image, analyze_video, create_face_crop,
#    predict_with_tta, confidence_gated_fusion, final_decision)
#   from predict_v2_facecrop.py.
#
# MODELS ARE LOADED ONCE AT STARTUP and reused across requests.
#
# Ensemble weights come directly from the predictor:
#   Xception     = XCEPTION_WEIGHT     (0.45)
#   EfficientNet = EFFICIENTNET_WEIGHT (0.25)
#   ViT          = VIT_WEIGHT          (0.30)
# ================================================================

import sys
import os
import uuid
import tempfile
import logging
from pathlib import Path

# ----------------------------------------------------------------
# Load .env FIRST — so SERPAPI_KEY and any other secrets are
# available before any module import that needs them.
# ----------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# ----------------------------------------------------------------
# Locate predict_v2_facecrop.py (lives in ForensIQ-ML/inference/).
# Tries several layouts so this works no matter which folder the
# script is launched from.
# ----------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

_INFERENCE_CANDIDATES = [
    _HERE / "ForensIQ-ML" / "inference",              # script at project root  ← primary
    _HERE / "inference",                              # script inside ForensIQ-ML/
    _HERE.parent / "inference",                       # fallback
    _HERE.parent / "ForensIQ-ML" / "inference",       # fallback
    Path(r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML\inference"),  # absolute fallback
]

_INFERENCE_DIR = None
for _cand in _INFERENCE_CANDIDATES:
    if _cand.is_dir() and (_cand / "predict_v2_facecrop.py").exists():
        _INFERENCE_DIR = _cand
        break

if _INFERENCE_DIR is None:
    raise RuntimeError(
        "Could not locate predict_v2_facecrop.py. "
        "Searched: " + ", ".join(str(c) for c in _INFERENCE_CANDIDATES)
    )

# Keep these for downstream compatibility
FORENSIQ_INFERENCE_DIR = _INFERENCE_DIR
FORENSIQ_ROOT_DIR      = _HERE

if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # for 'propagation' package

# Also expose ForensIQ-ML root so the module can be imported by package name
_FORENSIQ_ML_ROOT = _INFERENCE_DIR.parent
if str(_FORENSIQ_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_FORENSIQ_ML_ROOT))

# Report storage at workspace root
REPORTS_DIR = _HERE / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------
# Force UTF-8 stdout/stderr so that the pipeline's Unicode print
# statements (✓, ⚠) don't crash on Windows with charmap encoding.
# ----------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure") and not sys.stdout.closed:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and not sys.stderr.closed:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------
# FastAPI imports
# ----------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------------------------------------------
# Import the ENTIRE inference module from the existing pipeline.
# We do NOT copy any ML logic here. We call it directly.
# ----------------------------------------------------------------

try:
    import predict_v2_facecrop as pipeline
    print(f"[OK] Loaded predict_v2_facecrop from: {pipeline.__file__}")

    PIPELINE_LOADED = True
    PIPELINE_ERROR = None

    # --------------------------------------------------------------
    # V2.2 → V2.3 COMPATIBILITY SHIM
    #
    # V2.3 renamed / removed some module-level constants. The API
    # adapter still references the old names. Rather than rewrite the
    # whole adapter, we inject fallback attributes onto the module so
    # both versions work.
    # --------------------------------------------------------------

    # FACE_MARGIN (V2.2) → FACE_MARGIN_SIDE (V2.3)
    if not hasattr(pipeline, "FACE_MARGIN"):
        pipeline.FACE_MARGIN = getattr(pipeline, "FACE_MARGIN_SIDE", 0.30)

    # VIDEO_UPPER_WEIGHT (V2.2) → VIDEO_P75_WEIGHT (V2.3)
    if not hasattr(pipeline, "VIDEO_UPPER_WEIGHT"):
        pipeline.VIDEO_UPPER_WEIGHT = getattr(
            pipeline, "VIDEO_P75_WEIGHT",
            getattr(pipeline, "VIDEO_UPPER_WEIGHT", 0.20)
        )

    # VIDEO_UPPER_PERCENTILE — keep for JSON metadata even in V2.3
    if not hasattr(pipeline, "VIDEO_UPPER_PERCENTILE"):
        pipeline.VIDEO_UPPER_PERCENTILE = 75

except Exception as exc:
    PIPELINE_LOADED = False
    PIPELINE_ERROR = str(exc)
    pipeline = None

# ----------------------------------------------------------------
# Configure logging
# ----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("forensiq-api")


# ----------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------

app = FastAPI(
    title="ForensIQ V2.1 API",
    version="2.1.0",
    description="Wraps the existing predict_v2_facecrop.py 3-model pipeline (Xception + EfficientNet-B4 + ViT) and the Propagation / Find Misuse module.",
)

# Allow Vite dev server origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# Mount the Propagation router (Find Misuse → User Submission)
# This is entirely separate from the detection pipeline.
# ----------------------------------------------------------------
try:
    from propagation.routes import router as propagation_router
    app.include_router(propagation_router)
    logger.info("[Propagation] Router mounted at /api/propagation/*")
except Exception as _prop_exc:
    logger.warning(f"[Propagation] Router failed to mount: {_prop_exc}")

# ----------------------------------------------------------------
# Mount Backend Auth & Investigator Routers
# ----------------------------------------------------------------
try:
    from backend.database import Base, engine
    from backend.routes_auth import router as auth_router
    from backend.routes_investigator import router as investigator_router
    from backend.routes_analytics import router as analytics_router
    from backend.routes_monitoring import router as monitoring_router
    from backend.routes_team import router as team_router
    from backend.routes_evidence import router as evidence_router

    # Create all tables (idempotent — safe to call on every startup)
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Tables ensured / created via SQLAlchemy")

    app.include_router(auth_router,        prefix="/api/auth",       tags=["auth"])
    app.include_router(investigator_router, prefix="/api/investigator", tags=["investigator"])
    app.include_router(analytics_router,   prefix="/api/analytics",  tags=["analytics"])
    app.include_router(monitoring_router,  prefix="/api/monitoring",  tags=["monitoring"])
    app.include_router(team_router,        prefix="/api/team",        tags=["team"])
    app.include_router(evidence_router,    prefix="/api/evidence",    tags=["evidence"])

    logger.info("[Backend] All routers mounted: /api/auth, /api/investigator, /api/analytics, /api/monitoring, /api/team, /api/evidence")
except Exception as _backend_exc:
    logger.error(f"[Backend] Router mount failed: {_backend_exc}", exc_info=True)


# ================================================================
# HEALTH CHECK
# ================================================================

@app.get("/api/health")
def health():
    if not PIPELINE_LOADED:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "pipeline_loaded": False,
                "error": PIPELINE_ERROR,
            },
        )

    return {
        "status": "ok",
        "pipeline_loaded": True,
        "device": str(pipeline.DEVICE),
        "models": {
            "xception": str(pipeline.XCEPTION_V2_CHECKPOINT),
            "efficientnet_b4": str(pipeline.EFFICIENTNET_V2_CHECKPOINT),
            "vit": str(pipeline.VIT_V2_CHECKPOINT),
        },
        "weights": {
            "xception": pipeline.XCEPTION_WEIGHT,
            "efficientnet_b4": pipeline.EFFICIENTNET_WEIGHT,
            "vit": pipeline.VIT_WEIGHT,
        },
        "tta_enabled": pipeline.USE_TTA,
    }


# ================================================================
# UNIFIED DETECT ENDPOINT
# Accepts images AND videos.
# ================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
VIDEO_MIMES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo",
               "video/x-matroska", "video/webm", "video/mpeg"}

MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


def _determine_media_type(filename: str, content_type: str) -> str:
    """Return 'image', 'video', or raise HTTPException."""
    ext = Path(filename).suffix.lower()
    ct  = (content_type or "").lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    # Fall back to MIME if extension is ambiguous / missing
    if any(m in ct for m in ["image/"]):
        return "image"
    if any(m in ct for m in ["video/"]):
        return "video"

    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type '{ext}' (MIME: {content_type}). "
            "Supported images: JPG, JPEG, PNG, WEBP, BMP, TIFF. "
            "Supported videos: MP4, AVI, MOV, MKV, WEBM, MPEG."
        ),
    )


def _build_image_response(api_result: dict, filename: str, file_bytes: bytes) -> dict:
    """
    Map the output of pipeline.analyze_image() → structured JSON response.

    The predictor's analyze_image() returns a dict with keys:
        face_detection, xception_v2, efficientnet_b4_v2, vit_v2_1, fusion, final
    """
    final      = api_result.get("final", {})
    xv2        = api_result.get("xception_v2", {})
    ev2        = api_result.get("efficientnet_b4_v2", {})
    vv2        = api_result.get("vit_v2_1", {})
    fusion     = api_result.get("fusion", {})
    face       = api_result.get("face_detection", {})
    prepro     = api_result.get("preprocessing", {})

    fake_score = float(final.get("fake_score", 0.0))
    real_score = float(final.get("real_score", 1.0))
    prediction = str(final.get("prediction", "REAL"))
    confidence = float(final.get("confidence", 0.0))

    # --- No face detected: return inconclusive ---
    face_detected = bool(face.get("detected", False))
    fallback      = bool(face.get("fallback", False))

    if not face_detected:
        return {
            "status": "inconclusive",
            "media_type": "image",
            "reason": "No face detected",
            "face_detection": face,
            "data": None,
            "error": None,
        }

    # --- Full result ---
    return {
        "status": "success",
        "media_type": "image",
        "reason": None,
        "error": None,
        "data": {
            "media_type": "image",
            "filename": filename,
            "size_bytes": len(file_bytes),
            "label": prediction,
            "prediction": prediction,
            "fake_probability": fake_score,
            "real_probability": real_score,
            "confidence": confidence,
            "face_detection": {
                "detected": face_detected,
                "fallback": fallback,
                "num_faces": face.get("num_faces", 0),
                "face_confidence": face.get("confidence", 0.0),
                "crop_box": face.get("crop_box"),
                "margin": face.get("margin", pipeline.FACE_MARGIN),
            },
            "models": {
                "xception": {
                    "fake_probability": float(xv2.get("fake", 0.0)),
                    "real_probability": float(xv2.get("real", 1.0)),
                    "prediction": str(xv2.get("prediction", "?")),
                    "weight": pipeline.XCEPTION_WEIGHT,
                    "tta_original_fake": float(xv2.get("original_fake", 0.0)),
                    "tta_flipped_fake": xv2.get("flipped_fake"),
                },
                "efficientnet_b4": {
                    "fake_probability": float(ev2.get("fake", 0.0)),
                    "real_probability": float(ev2.get("real", 1.0)),
                    "prediction": str(ev2.get("prediction", "?")),
                    "weight": pipeline.EFFICIENTNET_WEIGHT,
                    "tta_original_fake": float(ev2.get("original_fake", 0.0)),
                    "tta_flipped_fake": ev2.get("flipped_fake"),
                },
                "vit": {
                    "fake_probability": float(vv2.get("fake", 0.0)),
                    "real_probability": float(vv2.get("real", 1.0)),
                    "prediction": str(vv2.get("prediction", "?")),
                    "weight": pipeline.VIT_WEIGHT,
                    "tta_original_fake": float(vv2.get("original_fake", 0.0)),
                    "tta_flipped_fake": vv2.get("flipped_fake"),
                },
            },
            "fusion": {
                "xception_weight": pipeline.XCEPTION_WEIGHT,
                "efficientnet_weight": pipeline.EFFICIENTNET_WEIGHT,
                "vit_weight": pipeline.VIT_WEIGHT,
                "normal_weighted_score": float(fusion.get("normal_weighted_score", 0.0)),
                "final_fake_score": fake_score,
                "final_real_score": real_score,
                "fusion_mode": str(fusion.get("fusion_mode", "STANDARD_WEIGHTED_FUSION")),
                "difference": float(fusion.get("difference", 0.0)),
                "disagreement_level": str(fusion.get("disagreement_level", "LOW")),
            },
            "preprocessing": {
                "image_size": pipeline.IMAGE_SIZE,
                "vit_image_size": pipeline.VIT_IMAGE_SIZE,
                "tta": pipeline.USE_TTA,
                "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            },
            "agreement": {
                "disagreement_level": str(fusion.get("disagreement_level", "LOW")),
                "score_difference": float(fusion.get("difference", 0.0)),
                "models_agree": (
                    xv2.get("prediction") == ev2.get("prediction") == vv2.get("prediction")
                ),
                "xception_prediction": str(xv2.get("prediction", "?")),
                "efficientnet_prediction": str(ev2.get("prediction", "?")),
                "vit_prediction": str(vv2.get("prediction", "?")),
            },
        },
    }


def _build_video_response(video_result: dict, filename: str, file_bytes: bytes) -> dict:
    """
    Map the output of pipeline.analyze_video() → structured JSON response.
    """
    final       = video_result.get("final", {})
    sampling    = video_result.get("video_sampling", {})
    aggregation = video_result.get("aggregation", {})
    frame_stats = video_result.get("frame_statistics", {})
    input_info  = video_result.get("input", {})

    fake_score  = float(final.get("fake_score", 0.0))
    real_score  = float(final.get("real_score", 1.0))
    prediction  = str(final.get("prediction", "REAL"))
    confidence  = float(final.get("confidence", 0.0))

    frames_analyzed = int(sampling.get("successfully_analyzed", 0))
    no_face_frames  = int(sampling.get("no_face_frames", 0))

    if frames_analyzed < pipeline.VIDEO_MIN_ANALYZED_FRAMES:
        return {
            "status": "inconclusive",
            "media_type": "video",
            "reason": f"Not enough valid frames analyzed ({frames_analyzed}/{pipeline.VIDEO_MIN_ANALYZED_FRAMES} minimum).",
            "data": None,
            "error": None,
        }

    # Serialize individual frame results (limit to 100 frames for response size)
    raw_frames = video_result.get("frames", [])
    frame_list = []
    for fr in raw_frames[:100]:
        frame_final = fr.get("final", {})
        frame_list.append({
            "frame_index": fr.get("frame_index", 0),
            "timestamp_seconds": fr.get("timestamp_seconds", 0.0),
            "face_detected": bool(fr.get("face_detected", False)),
            "fake_score": float(frame_final.get("fake_score", 0.0)),
            "real_score": float(frame_final.get("real_score", 1.0)),
            "prediction": str(frame_final.get("prediction", "?")),
            "confidence": float(frame_final.get("confidence", 0.0)),
            "xception_fake": float(fr.get("xception_v2", {}).get("fake", 0.0)),
            "efficientnet_fake": float(fr.get("efficientnet_b4_v2", {}).get("fake", 0.0)),
            "vit_fake": float(fr.get("vit_v2_1", {}).get("fake", 0.0)),
        })

    return {
        "status": "success",
        "media_type": "video",
        "reason": None,
        "error": None,
        "data": {
            "media_type": "video",
            "filename": filename,
            "size_bytes": len(file_bytes),
            "label": prediction,
            "prediction": prediction,
            "fake_probability": fake_score,
            "real_probability": real_score,
            "confidence": confidence,
            "video_info": {
                "width": input_info.get("width", 0),
                "height": input_info.get("height", 0),
                "fps": input_info.get("fps", 0.0),
                "total_frames": input_info.get("total_frames", 0),
                "duration_seconds": input_info.get("duration_seconds", 0.0),
            },
            "sampling": {
                "selected_frames": sampling.get("selected_frames", 0),
                "successfully_analyzed": frames_analyzed,
                "failed_frames": sampling.get("failed_frames", 0),
                "no_face_frames": no_face_frames,
                "usable_face_frames": frames_analyzed - no_face_frames,
                "sampling_method": sampling.get("sampling_method", "uniform_temporal_sampling"),
                "max_frames": pipeline.VIDEO_MAX_FRAMES,
            },
            "aggregation": {
                "mean_fake": float(aggregation.get("mean_fake", 0.0)),
                "median_fake": float(aggregation.get("median_fake", 0.0)),
                "upper_percentile": pipeline.VIDEO_UPPER_PERCENTILE,
                "upper_percentile_fake": float(aggregation.get("upper_percentile_fake", 0.0)),
                "mean_weight": pipeline.VIDEO_MEAN_WEIGHT,
                "median_weight": pipeline.VIDEO_MEDIAN_WEIGHT,
                "upper_weight": pipeline.VIDEO_UPPER_WEIGHT,
                "video_fake_score": fake_score,
                "video_real_score": real_score,
            },
            "frame_statistics": {
                "fake_frames": int(frame_stats.get("fake_frames", 0)),
                "real_frames": int(frame_stats.get("real_frames", 0)),
                "fake_frame_ratio": float(frame_stats.get("fake_frame_ratio", 0.0)),
                "real_frame_ratio": float(frame_stats.get("real_frame_ratio", 0.0)),
                "xception_mean_fake": float(frame_stats.get("xception_mean_fake", 0.0)),
                "efficientnet_mean_fake": float(frame_stats.get("efficientnet_mean_fake", 0.0)),
                "vit_mean_fake": float(frame_stats.get("vit_mean_fake", 0.0)),
            },
            "models": {
                "xception": {
                    "weight": pipeline.XCEPTION_WEIGHT,
                    "mean_fake": float(frame_stats.get("xception_mean_fake", 0.0)),
                },
                "efficientnet_b4": {
                    "weight": pipeline.EFFICIENTNET_WEIGHT,
                    "mean_fake": float(frame_stats.get("efficientnet_mean_fake", 0.0)),
                },
                "vit": {
                    "weight": pipeline.VIT_WEIGHT,
                    "mean_fake": float(frame_stats.get("vit_mean_fake", 0.0)),
                },
            },
            "frames": frame_list,
            "warnings": video_result.get("warnings", []),
        },
    }


@app.post("/api/v1/detect")
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    Unified detect endpoint for images and videos.

    Receives an uploaded image or video, routes to the correct
    predict_v2_facecrop.py pipeline function, and returns structured JSON.

    Models are NOT reloaded on each request. They were loaded once
    at server startup (via the import of predict_v2_facecrop).
    """

    # ----------------------------------------------------------------
    # Pipeline availability check
    # ----------------------------------------------------------------

    if not PIPELINE_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"ML pipeline failed to load: {PIPELINE_ERROR}",
        )

    # ----------------------------------------------------------------
    # Determine media type
    # ----------------------------------------------------------------

    filename     = file.filename or "upload"
    content_type = file.content_type or ""

    media_type = _determine_media_type(filename, content_type)

    # ----------------------------------------------------------------
    # Read file content and enforce size limit
    # ----------------------------------------------------------------

    file_bytes = await file.read()
    size_limit = MAX_IMAGE_SIZE_BYTES if media_type == "image" else MAX_VIDEO_SIZE_BYTES

    if len(file_bytes) > size_limit:
        size_mb = len(file_bytes) / (1024 * 1024)
        limit_mb = size_limit / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed is {limit_mb:.0f} MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ----------------------------------------------------------------
    # Write to a secure temporary file
    # ----------------------------------------------------------------

    ext      = Path(filename).suffix.lower() or (".jpg" if media_type == "image" else ".mp4")
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
            prefix=f"forensiq_{uuid.uuid4().hex[:8]}_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        logger.info(
            f"[{media_type.upper()}] {filename} ({len(file_bytes)/1024:.1f} KB) → {tmp_path.name}"
        )

        # ----------------------------------------------------------------
        # IMAGE INFERENCE
        # ----------------------------------------------------------------

        if media_type == "image":
            from PIL import Image

            try:
                image = Image.open(tmp_path).convert("RGB")
            except Exception as img_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read image. Is it a valid image file? ({img_err})",
                )

            logger.info("Running analyze_image (Xception V2 + EfficientNet-B4 V2 + ViT V2.1 + TTA)...")
            try:
                result = pipeline.analyze_image(image, verbose=False)
            except Exception as infer_err:
                logger.exception(f"Image inference error: {infer_err}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Image inference error: {type(infer_err).__name__}: {infer_err}",
                )

            response = _build_image_response(result, filename, file_bytes)
            final    = result.get("final", {})
            logger.info(
                f"Image result: {final.get('prediction','?')} "
                f"(fake={final.get('fake_score',0):.4f}, "
                f"confidence={final.get('confidence',0):.3f})"
            )
            return JSONResponse(content=response)

        # ----------------------------------------------------------------
        # VIDEO INFERENCE
        # ----------------------------------------------------------------

        if media_type == "video":
            logger.info("Running analyze_video (frame-sampling + 3-model ensemble)...")
            try:
                video_result = pipeline.analyze_video(tmp_path)
            except RuntimeError as rt_err:
                # Not enough valid frames, or could not open video
                error_msg = str(rt_err)
                if "not enough valid frames" in error_msg.lower():
                    return JSONResponse(
                        content={
                            "status": "inconclusive",
                            "media_type": "video",
                            "reason": "Not enough usable frames with faces could be analyzed.",
                            "data": None,
                            "error": None,
                        }
                    )
                raise HTTPException(
                    status_code=400,
                    detail=f"Video could not be processed: {error_msg}",
                )
            except Exception as vid_err:
                logger.exception(f"Video inference error: {vid_err}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Video inference error: {type(vid_err).__name__}: {vid_err}",
                )

            response = _build_video_response(video_result, filename, file_bytes)
            final    = video_result.get("final", {})
            logger.info(
                f"Video result: {final.get('prediction','?')} "
                f"(fake={final.get('fake_score',0):.4f}, "
                f"confidence={final.get('confidence',0):.3f})"
            )
            return JSONResponse(content=response)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"Unhandled inference error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference pipeline error: {type(exc).__name__}: {exc}",
        )

    finally:
        # Always clean up the temporary file
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ================================================================
# REPORT ENDPOINTS
# /api/v1/report          POST  — generate report
# /api/v1/report/{id}/html     GET   — serve inline
# /api/v1/report/{id}/download GET   — serve as attachment
# ================================================================

from fastapi import BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse


@app.post("/api/v1/report")
async def generate_report(file: UploadFile = File(...)):
    """
    Generate a Groq-powered ForensIQ forensic report for an image or video.
    Calls the REAL pipeline (analyze_image / analyze_video) and
    gradcam_advanced.py --model xception, then calls Groq for narrative analysis.
    """
    if not PIPELINE_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"ML pipeline failed to load: {PIPELINE_ERROR}",
        )

    filename     = file.filename or "upload"
    content_type = file.content_type or ""
    media_type   = _determine_media_type(filename, content_type)

    file_bytes = await file.read()
    size_limit = MAX_IMAGE_SIZE_BYTES if media_type == "image" else MAX_VIDEO_SIZE_BYTES

    if len(file_bytes) > size_limit:
        size_mb  = len(file_bytes) / (1024 * 1024)
        limit_mb = size_limit / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum {limit_mb:.0f} MB.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ext      = Path(filename).suffix.lower() or (".jpg" if media_type == "image" else ".mp4")
    tmp_path = None

    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext,
            prefix=f"forensiq_report_{uuid.uuid4().hex[:8]}_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        # Generate a unique report ID and output dir
        report_id  = f"FX-{__import__('datetime').datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
        output_dir = REPORTS_DIR / report_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-import report modules (they live in ForensIQ-ML/)
        _forensiq_ml = str(_FORENSIQ_ML_ROOT)
        if _forensiq_ml not in sys.path:
            sys.path.insert(0, _forensiq_ml)

        from report_generator import ForensicReportGenerator
        from report_renderer  import render_html

        gen = ForensicReportGenerator()

        if media_type == "image":
            report = gen.generate_image_report(
                tmp_path,
                output_dir,
                report_id=report_id,
                original_filename=file.filename,
            )
        else:
            report = gen.generate_video_report(
                tmp_path,
                output_dir,
                report_id=report_id,
                original_filename=file.filename,
            )

        render_html(report, output_dir)

        final = report.get("verdict", {})

        return JSONResponse(content={
            "status":       "success",
            "report_id":    report_id,
            "report_url":   f"/api/v1/report/{report_id}/html",
            "download_url": f"/api/v1/report/{report_id}/download",
            "verdict":      final.get("label", "UNKNOWN"),
            "confidence":   round(float(final.get("confidence", 0.0)), 4),
        })

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Report generation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.get("/api/v1/report/{report_id}/html", response_class=HTMLResponse)
async def serve_report_html(report_id: str):
    """Serve report.html inline."""
    html_path = REPORTS_DIR / report_id / "report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/v1/report/{report_id}/download")
async def download_report(report_id: str):
    """Serve report.html as a downloadable attachment."""
    html_path = REPORTS_DIR / report_id / "report.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return FileResponse(
        path=str(html_path),
        media_type="text/html",
        headers={"Content-Disposition": f'attachment; filename="forensiq_{report_id}.html"'},
    )


@app.api_route("/api/identity-protection", methods=["GET", "POST"])
def identity_protection_placeholder():
    return {"status": "coming_soon", "message": "Identity Protection module coming soon."}


# ================================================================
# MOUNT BACKEND ROUTERS & DB INITIALIZATION
# ================================================================
from backend.routes_auth            import router as auth_router
from backend.routes_investigator    import router as investigator_router
from backend.routes_evidence        import router as evidence_router
from backend.routes_team            import router as team_router
from backend.routes_monitoring      import router as monitoring_router
from backend.routes_analytics       import router as analytics_router

from backend.database import Base, engine
import backend.models  # noqa — register models

@app.on_event("startup")
def _create_tables():
    print("[db] Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("[db] tables created")

app.include_router(auth_router,        prefix="/api/auth",        tags=["auth"])
app.include_router(investigator_router,prefix="/api/investigator",tags=["investigator"])
app.include_router(evidence_router,    prefix="/api/evidence",    tags=["evidence"])
app.include_router(team_router,        prefix="/api/team",        tags=["team"])
app.include_router(monitoring_router,  prefix="/api/monitoring",  tags=["monitoring"])
app.include_router(analytics_router,   prefix="/api/analytics",   tags=["analytics"])


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

