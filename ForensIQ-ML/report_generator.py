"""
ForensIQ Report Generator
=========================
Produces a structured forensic report by:
  1. Calling the REAL inference pipeline (predict_v2_facecrop.analyze_image /
     analyze_video)  — never mocking results.
  2. Running gradcam_advanced.py --model xception via subprocess and locating
     the produced 02_overlay_on_original.jpg for THAT SPECIFIC INPUT.
  3. Calling Groq with structured evidence text.
  4. Saving report.json + report.html into output_dir.

RULES:
  * NO hardcoded values anywhere.
  * Every path is derived at runtime from the CURRENT input.
  * If a real function fails, propagate or mark section unavailable.
  * Groq receives TEXT only (evidence dict -> formatted string).
  * Sanity-check both embedded images; raise if either is < 1000 bytes.
"""

import os
import sys
import io
import json
import uuid
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 on Windows so Unicode print symbols don't fail
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure") and not sys.stdout.closed:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure") and not sys.stderr.closed:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Locate the ForensIQ-ML root and inference dir (same robust search pattern)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # ForensIQ-ML/

_INFERENCE_CANDIDATES = [
    _HERE / "inference",
    _HERE / "ForensIQ-ML" / "inference",
    _HERE.parent / "inference",
    _HERE.parent / "ForensIQ-ML" / "inference",
    Path(r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML\inference"),
]

_INFERENCE_DIR = None
for _cand in _INFERENCE_CANDIDATES:
    if _cand.is_dir() and (_cand / "predict_v2_facecrop.py").exists():
        _INFERENCE_DIR = _cand
        break

if _INFERENCE_DIR is None:
    raise ModuleNotFoundError(
        "Could not locate predict_v2_facecrop.py. Searched: "
        + ", ".join(str(c) for c in _INFERENCE_CANDIDATES)
    )

if str(_INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_DIR))

_FORENSIQ_ML_ROOT = _INFERENCE_DIR.parent
if str(_FORENSIQ_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_FORENSIQ_ML_ROOT))

import predict_v2_facecrop as pipeline
print(f"[ReportGenerator] Loaded predict_v2_facecrop from: {pipeline.__file__}")

FORENSIQ_ML_DIR = _FORENSIQ_ML_ROOT
GRADCAM_SCRIPT  = FORENSIQ_ML_DIR / "gradcam_advanced.py"

# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    print("[ReportGenerator][warn] groq package not installed. "
          "pip install groq  -- AI analysis will be unavailable.")

from dotenv import load_dotenv
load_dotenv(dotenv_path=_FORENSIQ_ML_ROOT.parent / ".env", override=False)

# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are a digital forensics analyst writing a professional report for a
non-technical reviewer. Interpret evidence from an AI image/video forensics
pipeline and explain it clearly, calmly, and honestly.

You will receive:
  * A final verdict (FAKE, REAL, or INCONCLUSIVE)
  * A confidence score between 0.0 and 1.0
  * The location of a suspicious region identified by the model attention map
  * Basic metadata about the image

Write a report in EXACTLY this JSON structure (no extra keys, no missing keys):

{
  "headline": "one sentence, max 15 words, plain-language verdict summary",
  "summary": "2-3 sentences. Non-technical. What was found and how sure we are.",
  "analysis": {
    "overview": "3-5 sentences. Explain how the analysis worked (face detection, attention map, verdict). Do NOT name any model architectures.",
    "heatmap_interpretation": "2-4 sentences. Explain what the highlighted region likely represents in plain language. Do NOT invent details you cannot support.",
    "region_focus": "1-2 sentences about WHERE the analysis focused. Base this on the bounding box position -- do not hallucinate."
  },
  "observed_indicators": [
    "3-6 short bullet strings. If FAKE: specific artifact-like signals. If REAL: reassuring features. Each grounded in the region/bbox."
  ],
  "confidence_statement": "1-2 sentences. Honestly describe confidence.",
  "limitations": "2-3 sentences. What this analysis CANNOT determine.",
  "recommendation": "1-2 sentences. What the reviewer should do next."
}

Rules:
- Never claim 100% certainty.
- Never name model architectures (Xception, EfficientNet, ViT, CNN, Transformer).
- If verdict is REAL, describe the heatmap honestly.
- If confidence < 0.55, say the evidence is weak or borderline.
- If verdict is INCONCLUSIVE, say a firm determination could not be made.
- Output ONLY the JSON object. No text before or after.
"""


def _build_user_prompt(evidence: dict) -> str:
    verdict          = evidence.get("verdict", "UNKNOWN")
    confidence       = evidence.get("confidence", 0.0)
    media_type       = evidence.get("media_type", "image")
    width            = evidence.get("width", 0)
    height           = evidence.get("height", 0)
    detection_method = evidence.get("detection_method", "unknown")
    aligned          = evidence.get("aligned", False)
    face_detected    = evidence.get("face_detected", False)

    bbox = evidence.get("bbox")
    if bbox:
        bx, by, bw, bh = bbox
        bbox_fraction = (bw * bh) / max(width * height, 1)
        bbox_cx_pct   = (bx + bw / 2) / max(width, 1)
        bbox_cy_pct   = (by + bh / 2) / max(height, 1)
        bbox_section = (
            f"  Position: x={bx}, y={by}, width={bw}, height={bh}\n"
            f"  Region covers {bbox_fraction:.1%} of the image\n"
            f"  Region center: {bbox_cx_pct:.0%} horizontally, {bbox_cy_pct:.0%} vertically\n"
            f"  (0% = top/left, 100% = bottom/right)"
        )
    else:
        bbox_section = "  No suspicious region bounding box available."

    video_extra = ""
    if media_type == "video":
        frames_analyzed = evidence.get("frames_analyzed", "N/A")
        fake_frames     = evidence.get("fake_frames", "N/A")
        fake_ratio      = evidence.get("fake_frame_ratio", None)
        mean_fake       = evidence.get("mean_fake", None)
        median_fake     = evidence.get("median_fake", None)
        p75_fake        = evidence.get("p75_fake", None)
        p90_fake        = evidence.get("p90_fake", None)
        peak_escalated  = evidence.get("peak_escalated", None)
        segment_pattern = evidence.get("segment_pattern", "N/A")

        def _fmt(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        video_extra = (
            f"\nVideo-specific evidence:\n"
            f"  Frames analyzed: {frames_analyzed}\n"
            f"  Frames classified FAKE: {fake_frames} "
            f"({_fmt(fake_ratio, '.1%') if fake_ratio is not None else 'N/A'})\n"
            f"  Score mean: {_fmt(mean_fake)}, median: {_fmt(median_fake)}\n"
            f"  Score p75: {_fmt(p75_fake)}, p90: {_fmt(p90_fake)}\n"
            f"  Peak-frame escalation triggered: {peak_escalated}\n"
            f"  Temporal pattern: {segment_pattern}\n"
        )

    return (
        f"FORENSIC EVIDENCE:\n\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.3f}\n"
        f"Media type: {media_type}\n\n"
        f"Image dimensions: {width} x {height} pixels\n"
        f"Face detection: {detection_method}\n"
        f"Face alignment applied: {aligned}\n"
        f"Face crop used for analysis: {face_detected}\n\n"
        f"Suspicious region (from attention map):\n"
        f"{bbox_section}\n"
        f"{video_extra}\n"
        f"Write the report now, as a single JSON object."
    )


# ============================================================
# HELPERS
# ============================================================

def _copy_asset(src: Path, dst: Path) -> int:
    """Copy src -> dst and return file size in bytes. Raises on failure."""
    shutil.copy2(str(src), str(dst))
    size = dst.stat().st_size
    print(f"[ReportGenerator] Asset copied: {dst.name} ({size:,} bytes) <- {src}")
    return size


def _sanity_check_asset(path: Path, label: str, min_bytes: int = 1000) -> None:
    """Raise if the file doesn't exist or is smaller than min_bytes."""
    if not path.exists():
        raise FileNotFoundError(
            f"[ReportGenerator] SANITY FAIL: {label} not found at: {path}"
        )
    size = path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"[ReportGenerator] SANITY FAIL: {label} is suspiciously small "
            f"({size} bytes < {min_bytes} byte minimum). "
            f"Path: {path}"
        )
    print(f"[ReportGenerator] Sanity OK: {label} = {size:,} bytes")


# ============================================================
# REPORT GENERATOR CLASS
# ============================================================

class ForensicReportGenerator:

    def _call_groq(self, evidence: dict) -> dict:
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_key:
            print("[ReportGenerator][warn] GROQ_API_KEY not set.")
            return {"error": "GROQ_API_KEY not configured"}
        if not _GROQ_AVAILABLE:
            return {"error": "groq package not installed (pip install groq)"}

        client      = _GroqClient(api_key=groq_key)
        user_prompt = _build_user_prompt(evidence)
        candidate_models = [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ]

        last_err = None
        for model_name in candidate_models:
            try:
                print(f"[ReportGenerator] Calling Groq with model '{model_name}'...")
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=1200,
                    timeout=30,
                )
                raw_text = completion.choices[0].message.content.strip()
                parsed = json.loads(raw_text)
                print(f"[ReportGenerator] Groq OK from '{model_name}'.")
                return parsed
            except json.JSONDecodeError as e:
                last_err = f"Groq returned non-JSON: {e}"
            except Exception as e:
                last_err = f"Groq call failed ({model_name}): {type(e).__name__}: {e}"
                print(f"[ReportGenerator][warn] Model {model_name} failed: {e}. Trying next...")

        return {"error": last_err or "All Groq model attempts failed"}

    def _run_gradcam(self, image_path: Path) -> Path:
        """
        Run gradcam_advanced.py on image_path.
        Returns the absolute path to 02_overlay_on_original.jpg for that image.
        Raises RuntimeError / FileNotFoundError on any failure.
        """
        image_path = Path(image_path).resolve()   # always absolute

        if not GRADCAM_SCRIPT.exists():
            raise FileNotFoundError(
                f"gradcam_advanced.py not found at: {GRADCAM_SCRIPT}"
            )

        cmd = [
            sys.executable, str(GRADCAM_SCRIPT),
            "--image", str(image_path),
            "--model", "xception",
        ]
        print(f"[ReportGenerator] Running Grad-CAM: {' '.join(cmd)}")

        sub_env = os.environ.copy()
        sub_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(FORENSIQ_ML_DIR),
            env=sub_env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Grad-CAM failed (exit {proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}"
            )

        # Grad-CAM writes to:  gradcam_results/xception/<stem>/02_overlay_on_original.jpg
        # where <stem> is derived from the image filename (basename without extension).
        stem         = image_path.stem
        overlay_path = (
            FORENSIQ_ML_DIR / "gradcam_results" / "xception" / stem
            / "02_overlay_on_original.jpg"
        )
        if not overlay_path.exists():
            raise RuntimeError(
                f"Expected Grad-CAM output not found: {overlay_path}\n"
                f"Grad-CAM stdout (last 1000 chars):\n{proc.stdout[-1000:]}"
            )
        print(f"[ReportGenerator] Grad-CAM overlay located: {overlay_path}")
        return overlay_path

    # ------------------------------------------------------------------
    # IMAGE REPORT
    # ------------------------------------------------------------------

    def generate_image_report(
        self,
        image_path: Path,
        output_dir: Path,
        precomputed_overlay: Path | None = None,
        report_id: str | None = None,
        original_filename: str | None = None,
    ) -> dict:
        """
        Generate a forensic report for a single image.

        Args:
            image_path:          Path to the input image (any PIL-readable format).
            output_dir:          Where to write report.json and assets/.
            precomputed_overlay: If the caller already ran Grad-CAM and knows the
                                 overlay path, pass it here to skip re-running it.
                                 Pass None (default) to run Grad-CAM automatically.
            report_id:           Optional pre-assigned report ID (e.g. FX-YYYYMMDD-XXXX).
            original_filename:   Optional logical filename (e.g. from user upload).
        """
        from PIL import Image as _PILImage

        image_path = Path(image_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- 1. Load input image ----------------------------------------
        image_pil    = _PILImage.open(image_path).convert("RGB")
        img_w, img_h = image_pil.size
        print(f"[ReportGenerator] Input image: {image_path} ({img_w}x{img_h})")

        # ---- 2. Run real inference pipeline -----------------------------
        print(f"[ReportGenerator] Running analyze_image on: {image_path}")
        result = pipeline.analyze_image(image_pil, verbose=False)

        final = result["final"]
        face  = result["face_detection"]

        verdict    = str(final["prediction"])
        confidence = float(final["confidence"])
        fake_score = float(final["fake_score"])
        print(f"[ReportGenerator] verdict = {verdict} | confidence = {confidence:.4f}")

        if not verdict or verdict not in ("FAKE", "REAL", "SUSPICIOUS", "INCONCLUSIVE"):
            raise ValueError(
                f"[ReportGenerator] Pipeline returned unexpected verdict: {verdict!r}. "
                "Cannot produce a report with unreliable data."
            )

        face_detected    = bool(face.get("detected", False))
        detection_method = str(face.get("detection_method", "unknown"))
        aligned          = bool(face.get("aligned", False))
        crop_box_raw     = face.get("crop_box")

        bbox = None
        if crop_box_raw is not None:
            try:
                x1, y1, x2, y2 = [int(v) for v in crop_box_raw]
                bbox = [x1, y1, x2 - x1, y2 - y1]
            except Exception:
                bbox = None

        bbox_fraction   = None
        bbox_center_pct = None
        if bbox:
            bx, by, bw, bh = bbox
            bbox_fraction   = round((bw * bh) / max(img_w * img_h, 1), 6)
            bbox_center_pct = [
                round((bx + bw / 2) / max(img_w, 1), 4),
                round((by + bh / 2) / max(img_h, 1), 4),
            ]

        # ---- 3. Copy original image to assets/ --------------------------
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Preserve original extension so renderer picks the right MIME type
        orig_ext      = image_path.suffix.lower() or ".jpg"
        orig_asset    = assets_dir / f"original{orig_ext}"
        orig_size     = _copy_asset(image_path, orig_asset)
        _sanity_check_asset(orig_asset, "original image", min_bytes=1000)

        # ---- 4. Run Grad-CAM on THIS specific image ---------------------
        gradcam_error = None
        overlay_path  = None
        overlay_size  = 0

        try:
            if precomputed_overlay is not None:
                precomputed_overlay = Path(precomputed_overlay).resolve()
                if not precomputed_overlay.exists():
                    raise FileNotFoundError(
                        f"precomputed_overlay not found: {precomputed_overlay}"
                    )
                overlay_path = precomputed_overlay
                print(f"[ReportGenerator] Using precomputed Grad-CAM: {overlay_path}")
            else:
                overlay_path = self._run_gradcam(image_path)

            overlay_asset = assets_dir / "gradcam_overlay.jpg"
            overlay_size  = _copy_asset(overlay_path, overlay_asset)
            _sanity_check_asset(overlay_asset, "Grad-CAM overlay", min_bytes=1000)
        except Exception as e:
            gradcam_error = str(e)
            print(f"[ReportGenerator][error] Grad-CAM failed: {e}")

        # ---- 5. Call Groq for narrative analysis ------------------------
        evidence = {
            "verdict":          verdict,
            "confidence":       confidence,
            "media_type":       "image",
            "width":            img_w,
            "height":           img_h,
            "face_detected":    face_detected,
            "detection_method": detection_method,
            "aligned":          aligned,
            "bbox":             bbox,
        }

        ai_analysis       = self._call_groq(evidence)
        ai_analysis_error = ai_analysis.pop("error", None) if "error" in ai_analysis else None

        # ---- 6. Assemble report dict ------------------------------------
        report_id = report_id or f"FX-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

        report = {
            "report_id":    report_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version":      "ForensIQ-Report-V1",
            "input": {
                "media_type": "image",
                "filename":   original_filename or image_path.name,
                "width":      img_w,
                "height":     img_h,
            },
            "verdict": {
                "label":      verdict,
                "confidence": confidence,
                "fake_score": fake_score,
                "headline":   ai_analysis.get("headline", "") if not ai_analysis_error else "",
            },
            "visuals": {
                # Renderer reads these as relative paths inside assets_dir
                "original_image":       f"assets/original{orig_ext}",
                "original_ext":         orig_ext,
                "original_size_bytes":  orig_size,
                "gradcam_overlay":      "assets/gradcam_overlay.jpg" if overlay_path else None,
                "gradcam_overlay_size": overlay_size,
                "gradcam_error":        gradcam_error,
            },
            "ai_analysis":       ai_analysis if not ai_analysis_error else {},
            "ai_analysis_error": ai_analysis_error,
            "metadata": {
                "face_detected":    face_detected,
                "detection_method": detection_method,
                "aligned":          aligned,
                "bbox":             bbox,
                "bbox_fraction":    bbox_fraction,
                "bbox_center_pct":  bbox_center_pct,
            },
            "_pipeline_result": {
                "final":          result.get("final"),
                "face_detection": result.get("face_detection"),
                "fusion":         result.get("fusion"),
            },
        }

        report_json_path = output_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[ReportGenerator] report.json saved: {report_json_path}")
        return report

    # ------------------------------------------------------------------
    # VIDEO REPORT
    # ------------------------------------------------------------------

    def generate_video_report(
        self,
        video_path: Path,
        output_dir: Path,
        report_id: str | None = None,
        original_filename: str | None = None,
    ) -> dict:
        import cv2 as _cv2

        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ReportGenerator] Running analyze_video on: {video_path}")
        result = pipeline.analyze_video(video_path)

        final       = result["final"]
        aggregation = result.get("aggregation", {})
        frame_stats = result.get("frame_statistics", {})
        segments    = result.get("segments")
        input_info  = result.get("input", {})
        frames_list = result.get("frames", [])

        verdict    = str(final["prediction"])
        confidence = float(final["confidence"])
        fake_score = float(final["fake_score"])
        vid_w      = int(input_info.get("width", 0))
        vid_h      = int(input_info.get("height", 0))

        print(f"[ReportGenerator] verdict = {verdict} | confidence = {confidence:.4f}")

        if not verdict or verdict not in ("FAKE", "REAL", "SUSPICIOUS", "INCONCLUSIVE"):
            raise ValueError(
                f"[ReportGenerator] Pipeline returned unexpected verdict: {verdict!r}."
            )

        # ---- Identify peak-fake frame -----------------------------------
        peak_frame_result = None
        peak_fake_idx     = 0
        if frames_list:
            peak_frame_result = max(
                frames_list,
                key=lambda r: r.get("final", {}).get("fake_score", 0.0),
            )
            peak_fake_idx = int(peak_frame_result.get("frame_index", 0))

        # ---- Extract peak frame to a temp PNG ---------------------------
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(exist_ok=True)
        frame_png = output_dir / "peak_frame.png"

        cap = _cv2.VideoCapture(str(video_path))
        try:
            cap.set(_cv2.CAP_PROP_POS_FRAMES, peak_fake_idx)
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame_bgr = cap.read()
            if ok:
                _cv2.imwrite(str(frame_png), frame_bgr)
            else:
                raise RuntimeError(
                    f"Could not extract frame {peak_fake_idx} from video: {video_path}"
                )
        finally:
            cap.release()

        # ---- Copy the peak frame as original asset ----------------------
        _copy_asset(frame_png, assets_dir / "original.jpg")
        _sanity_check_asset(assets_dir / "original.jpg", "peak video frame", min_bytes=1000)

        # ---- Run Grad-CAM on the peak frame ----------------------------
        gradcam_error = None
        overlay_path  = None
        overlay_size  = 0
        try:
            overlay_path = self._run_gradcam(frame_png)
            overlay_asset = assets_dir / "gradcam_overlay.jpg"
            overlay_size  = _copy_asset(overlay_path, overlay_asset)
            _sanity_check_asset(overlay_asset, "Grad-CAM overlay", min_bytes=1000)
        except Exception as e:
            gradcam_error = str(e)
            print(f"[ReportGenerator][error] Grad-CAM failed: {e}")

        # ---- Face info from peak frame ----------------------------------
        bbox             = None
        detection_method = "unknown"
        aligned          = False
        face_detected    = False

        if peak_frame_result:
            face_info    = peak_frame_result.get("face_detection", {})
            crop_box_raw = face_info.get("crop_box")
            if crop_box_raw:
                try:
                    x1, y1, x2, y2 = [int(v) for v in crop_box_raw]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                except Exception:
                    bbox = None
            detection_method = str(face_info.get("detection_method", "unknown"))
            aligned          = bool(face_info.get("aligned", False))
            face_detected    = bool(face_info.get("detected", False))

        bbox_fraction   = None
        bbox_center_pct = None
        if bbox and vid_w and vid_h:
            bx, by, bw, bh = bbox
            bbox_fraction   = round((bw * bh) / max(vid_w * vid_h, 1), 6)
            bbox_center_pct = [
                round((bx + bw / 2) / max(vid_w, 1), 4),
                round((by + bh / 2) / max(vid_h, 1), 4),
            ]

        sampling_info   = result.get("video_sampling", {})
        frames_analyzed = int(sampling_info.get("successfully_analyzed", len(frames_list)))

        evidence = {
            "verdict":          verdict,
            "confidence":       confidence,
            "media_type":       "video",
            "width":            vid_w,
            "height":           vid_h,
            "face_detected":    face_detected,
            "detection_method": detection_method,
            "aligned":          aligned,
            "bbox":             bbox,
            "frames_analyzed":  frames_analyzed,
            "fake_frames":      frame_stats.get("fake_frames"),
            "fake_frame_ratio": frame_stats.get("fake_frame_ratio"),
            "mean_fake":        aggregation.get("mean_fake"),
            "median_fake":      aggregation.get("median_fake"),
            "p75_fake":         aggregation.get("p75_fake"),
            "p90_fake":         aggregation.get("p90_fake"),
            "peak_escalated":   aggregation.get("peak_escalated"),
            "segment_pattern":  segments.get("pattern") if segments else None,
        }

        ai_analysis       = self._call_groq(evidence)
        ai_analysis_error = ai_analysis.pop("error", None) if "error" in ai_analysis else None

        report_id = report_id or f"FX-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

        report = {
            "report_id":    report_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "version":      "ForensIQ-Report-V1",
            "input": {
                "media_type": "video",
                "filename":   original_filename or video_path.name,
                "width":      vid_w,
                "height":     vid_h,
            },
            "verdict": {
                "label":      verdict,
                "confidence": confidence,
                "fake_score": fake_score,
                "headline":   ai_analysis.get("headline", "") if not ai_analysis_error else "",
            },
            "visuals": {
                "original_image":       "assets/original.jpg",
                "original_ext":         ".jpg",
                "original_size_bytes":  (assets_dir / "original.jpg").stat().st_size if (assets_dir / "original.jpg").exists() else 0,
                "gradcam_overlay":      "assets/gradcam_overlay.jpg" if overlay_path else None,
                "gradcam_overlay_size": overlay_size,
                "gradcam_error":        gradcam_error,
            },
            "ai_analysis":       ai_analysis if not ai_analysis_error else {},
            "ai_analysis_error": ai_analysis_error,
            "metadata": {
                "face_detected":    face_detected,
                "detection_method": detection_method,
                "aligned":          aligned,
                "bbox":             bbox,
                "bbox_fraction":    bbox_fraction,
                "bbox_center_pct":  bbox_center_pct,
                "peak_frame_index": peak_fake_idx,
            },
            "video_stats": {
                "frames_analyzed":  frames_analyzed,
                "fake_frames":      frame_stats.get("fake_frames"),
                "fake_frame_ratio": frame_stats.get("fake_frame_ratio"),
                "mean_fake":        aggregation.get("mean_fake"),
                "median_fake":      aggregation.get("median_fake"),
                "p75_fake":         aggregation.get("p75_fake"),
                "p90_fake":         aggregation.get("p90_fake"),
                "peak_escalated":   aggregation.get("peak_escalated"),
                "segment_pattern":  segments.get("pattern") if segments else None,
            },
            "_pipeline_result": {
                "final":       result.get("final"),
                "aggregation": result.get("aggregation"),
            },
        }

        report_json_path = output_dir / "report.json"
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[ReportGenerator] report.json saved: {report_json_path}")
        return report
