# ================================================================
# FORENSIQ V2 — FastAPI DETECTION SERVICE
#
# Wraps the existing predict_v2_facecrop.py pipeline.
# 
# IMPORTANT:
#   This file is ONLY an API adapter.
#   It does NOT implement any ML logic.
#   ALL inference runs inside the existing inference functions
#   (create_face_crop, predict_with_tta, confidence_gated_fusion,
#    final_decision) from predict_v2_facecrop.py.
#
# MODELS ARE LOADED ONCE AT STARTUP and reused across requests.
# ================================================================

import sys
import os
import uuid
import tempfile
import logging
from pathlib import Path

# ----------------------------------------------------------------
# Add ForensIQ-ML/inference to Python path so we can import
# the existing inference module.
# ----------------------------------------------------------------

FORENSIQ_INFERENCE_DIR = Path(__file__).resolve().parent / "ForensIQ-ML" / "inference"

sys.path.insert(0, str(FORENSIQ_INFERENCE_DIR))

# ----------------------------------------------------------------
# Force UTF-8 stdout/stderr so that the pipeline's Unicode print
# statements (✓, ⚠) don't crash on Windows with charmap encoding.
# ----------------------------------------------------------------
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ----------------------------------------------------------------
# FastAPI imports
# ----------------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ----------------------------------------------------------------
# Import ONLY the inference functions from the existing pipeline.
# We do NOT copy any ML logic here. We call it directly.
# ----------------------------------------------------------------

try:
    import predict_v2_facecrop as pipeline

    PIPELINE_LOADED = True
    PIPELINE_ERROR = None

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
    title="ForensIQ V2 Detection API",
    version="2.0.0",
    description="Wraps the existing predict_v2_facecrop.py pipeline.",
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
        "xception_checkpoint": str(pipeline.XCEPTION_V2_CHECKPOINT),
        "efficientnet_checkpoint": str(pipeline.EFFICIENTNET_V2_CHECKPOINT),
        "tta_enabled": pipeline.USE_TTA,
    }


# ================================================================
# DETECT ENDPOINT
# ================================================================

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    Receive an uploaded image, run the existing predict_v2_facecrop
    pipeline (face detection → face crop → Xception V2 + TTA →
    EfficientNet-B4 V2 + TTA → confidence-gated fusion → decision),
    and return a structured JSON result.

    Models are NOT reloaded on each request. They were loaded once
    when this service started (via the import of predict_v2_facecrop).
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
    # Validate uploaded file
    # ----------------------------------------------------------------

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    ALLOWED_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    filename = file.filename or "upload"
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Supported: JPG, JPEG, PNG, WEBP, BMP, TIFF.",
        )

    content_type = file.content_type or ""
    if content_type and content_type not in ALLOWED_MIMES:
        logger.warning(f"Unusual MIME type received: {content_type}")

    # ----------------------------------------------------------------
    # Read file content and enforce size limit
    # ----------------------------------------------------------------

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(file_bytes) // (1024*1024):.1f} MB). Maximum allowed is 50 MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ----------------------------------------------------------------
    # Write to a secure temporary file
    # ----------------------------------------------------------------

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
            prefix=f"forensiq_{uuid.uuid4().hex[:8]}_",
        ) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = Path(tmp_file.name)

        logger.info(f"Analyzing: {filename} ({len(file_bytes) / 1024:.1f} KB) → {tmp_path.name}")

        # ----------------------------------------------------------------
        # Call the EXISTING pipeline functions directly.
        # This is identical to what python predict_v2_facecrop.py does.
        # ----------------------------------------------------------------

        from PIL import Image

        try:
            image = Image.open(tmp_path).convert("RGB")
        except Exception as img_err:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read the uploaded image. Is it a valid image file? ({img_err})",
            )

        image_width, image_height = image.size

        # --- FACE DETECTION & CROP (exact existing logic) ---
        face_result = pipeline.create_face_crop(image)
        face_crop = face_result["crop"]

        # --- XCEPTION V2 + TTA (exact existing logic) ---
        logger.info("Running Xception V2 + TTA...")
        xception_result = pipeline.predict_with_tta(pipeline.xception_model, face_crop)

        # --- EFFICIENTNET-B4 V2 + TTA (exact existing logic) ---
        logger.info("Running EfficientNet-B4 V2 + TTA...")
        efficientnet_result = pipeline.predict_with_tta(pipeline.efficientnet_model, face_crop)

        # --- ENSEMBLE FUSION (exact existing logic) ---
        xception_fake = xception_result["fake"]
        efficientnet_fake = efficientnet_result["fake"]

        normal_score = pipeline.weighted_fusion(xception_fake, efficientnet_fake)
        gated = pipeline.confidence_gated_fusion(xception_fake, efficientnet_fake)
        fused_score = gated["score"]
        disagreement = gated["difference"]
        disagreement_level = pipeline.classify_disagreement(disagreement)

        # --- FINAL DECISION (exact existing logic) ---
        decision = pipeline.final_decision(fused_score)

        # --- INDIVIDUAL MODEL PREDICTIONS (same threshold as script) ---
        x_prediction = "FAKE" if xception_fake >= 0.50 else "REAL"
        e_prediction = "FAKE" if efficientnet_fake >= 0.50 else "REAL"

        # ----------------------------------------------------------------
        # Build structured response matching the JSON output of the
        # existing predict_v2_facecrop.py script.
        # ----------------------------------------------------------------

        result = {
            "version": "ForensIQ-V2",

            "source": "real_ml_pipeline",

            "input": {
                "filename": filename,
                "width": image_width,
                "height": image_height,
                "size_bytes": len(file_bytes),
            },

            "face_detection": {
                "detected": face_result["face_detected"],
                "num_faces": face_result["num_faces"],
                "confidence": face_result["confidence"],
                "crop_box": face_result["box"],
                "raw_face_box": face_result.get("raw_box"),
                "fallback": face_result["fallback"],
                "margin": pipeline.FACE_MARGIN,
                "crop_size": [face_crop.width, face_crop.height],
            },

            "preprocessing": {
                "image_size": pipeline.IMAGE_SIZE,
                "normalization": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                },
                "tta": pipeline.USE_TTA,
            },

            "xception_v2": {
                "fake": xception_result["fake"],
                "real": xception_result["real"],
                "fake_probability": xception_result["fake"],
                "real_probability": xception_result["real"],
                "original_fake": xception_result["original"]["fake"],
                "original_real": xception_result["original"]["real"],
                "flipped_fake": (
                    xception_result["flipped"]["fake"]
                    if pipeline.USE_TTA and xception_result["flipped"]
                    else None
                ),
                "flipped_real": (
                    xception_result["flipped"]["real"]
                    if pipeline.USE_TTA and xception_result["flipped"]
                    else None
                ),
                "prediction": x_prediction,
            },

            "efficientnet_b4_v2": {
                "fake": efficientnet_result["fake"],
                "real": efficientnet_result["real"],
                "fake_probability": efficientnet_result["fake"],
                "real_probability": efficientnet_result["real"],
                "original_fake": efficientnet_result["original"]["fake"],
                "original_real": efficientnet_result["original"]["real"],
                "flipped_fake": (
                    efficientnet_result["flipped"]["fake"]
                    if pipeline.USE_TTA and efficientnet_result["flipped"]
                    else None
                ),
                "flipped_real": (
                    efficientnet_result["flipped"]["real"]
                    if pipeline.USE_TTA and efficientnet_result["flipped"]
                    else None
                ),
                "prediction": e_prediction,
            },

            "fusion": {
                "xception_weight": pipeline.XCEPTION_WEIGHT,
                "efficientnet_weight": pipeline.EFFICIENTNET_WEIGHT,
                "normal_weighted_score": normal_score,
                "final_fake_score": fused_score,
                "final_real_score": decision["real_score"],
                "fusion_mode": gated["mode"],
                "difference": disagreement,
                "disagreement_level": disagreement_level,
            },

            "agreement": {
                "models_agree": x_prediction == e_prediction,
                "xception_prediction": x_prediction,
                "efficientnet_prediction": e_prediction,
                "score_difference": disagreement,
                "disagreement_level": disagreement_level,
            },

            "final": {
                "prediction": decision["prediction"],
                "confidence": decision["confidence"],
                "fake_probability": fused_score,
                "real_probability": decision["real_score"],
            },

            # Top-level aliases expected by the frontend
            "prediction": decision["prediction"],
            "fake_probability": fused_score,
            "real_probability": decision["real_score"],
            "confidence": decision["confidence"],
        }

        logger.info(
            f"Result: {decision['prediction']} "
            f"(confidence={decision['confidence']:.3f}, "
            f"fake={fused_score:.4f}, "
            f"xception={xception_fake:.4f}, "
            f"efficientnet={efficientnet_fake:.4f}, "
            f"mode={gated['mode']})"
        )

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(f"Inference failed: {exc}")
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
