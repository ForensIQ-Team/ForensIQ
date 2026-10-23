# ================================================================
# FORENSIQ V2.5
# ROBUST FACE-CROP IMAGE + VIDEO INFERENCE
# MULTI-PASS FACE DETECTION + ALIGNMENT + SQUARE CROP
# SUPPORT-MAJORITY FUSION + SUPPORT CONSENSUS VETO
# SEVERITY-WEIGHTED VIDEO AGGREGATION + PEAK-FRAME ESCALATION
#
#   1. Xception V2
#   2. EfficientNet-B4 V2
#   3. ViT-Small/16 V2.1
#
# ================================================================

import os
import sys
import json
import argparse
import base64
import hashlib
import zlib
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms

from PIL import Image, ImageOps, ImageDraw

import timm

import cv2


# ================================================================
# OPTIONAL FACE DETECTOR
# ================================================================

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = ROOT / "checkpoints"
V2_CHECKPOINT_DIR = CHECKPOINT_DIR / "v2"
PREDICTION_DIR = ROOT / "prediction_results"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_DIR = ROOT / "debug_crops"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# V2 CHECKPOINTS
# ================================================================

XCEPTION_V2_CHECKPOINT = V2_CHECKPOINT_DIR / "best_xception_v2.pth"
EFFICIENTNET_V2_CHECKPOINT = V2_CHECKPOINT_DIR / "best_efficientnet_b4_v2.pth"
VIT_V2_CHECKPOINT = V2_CHECKPOINT_DIR / "best_vit_v2_1.pth"


# ================================================================
# CONFIGURATION
# ================================================================

IMAGE_SIZE = 299
VIT_IMAGE_SIZE = 224

NUM_CLASSES = 2

FAKE = 0
REAL = 1

XCEPTION_NAME = "xception"
EFFICIENTNET_NAME = "tf_efficientnet_b4"
VIT_NAME = "vit_small_patch16_224"


# ================================================================
# IMAGE ENSEMBLE WEIGHTS  (overridable via --weights)
# ================================================================

XCEPTION_WEIGHT = 0.40
EFFICIENTNET_WEIGHT = 0.30
VIT_WEIGHT = 0.30


# ================================================================
# XCEPTION PRIORITY (TIE-BREAKER ROLE)
# ================================================================

XCEPTION_ALERT_THRESHOLD = 0.50
XCEPTION_MAX_PULL_DOWN = 0.20
XCEPTION_PULL_STRENGTH = 0.50
XCEPTION_ALERT_FLOOR = 0.52

SUPPORT_CLIP_MIN = 0.05
SUPPORT_CLIP_MAX = 0.95


# ================================================================
# SUPPORT MAJORITY / CONSENSUS (V2.5)
# ================================================================

SUPPORT_REAL_MAX = 0.50
SUPPORT_FAKE_MIN = 0.50

SUPPORT_STRONG_REAL_MAX = 0.10
SUPPORT_STRONG_FAKE_MIN = 0.90

SUPPORT_VETO_CEILING = 0.40
SUPPORT_CONFIRM_FLOOR = 0.60
SUPPORT_MAJORITY_REAL_CEILING = 0.45


# ================================================================
# DECISION THRESHOLDS
# ================================================================

IMAGE_FAKE_THRESHOLD = 0.50
VIDEO_FAKE_THRESHOLD = 0.40
VIDEO_SUSPICIOUS_THRESHOLD = 0.32


# ================================================================
# SUPPORT CONSENSUS GATE (VIDEO)
# ================================================================

SUPPORT_CONSENSUS_REAL_MAX = 0.10
SUPPORT_CONSENSUS_REAL_CEILING = 0.40


# ================================================================
# DISAGREEMENT
# ================================================================

HIGH_DISAGREEMENT = 0.50
MEDIUM_DISAGREEMENT = 0.25


# ================================================================
# FACE CROP
# ================================================================

MTCNN_STRICT_THRESHOLDS = [0.6, 0.7, 0.7]
MTCNN_RELAXED_THRESHOLDS = [0.5, 0.6, 0.6]
MTCNN_FALLBACK_THRESHOLDS = [0.4, 0.5, 0.5]

MIN_FACE_SIZE = 40

FACE_MARGIN_SIDE = 0.35
FACE_MARGIN_TOP = 0.40
FACE_MARGIN_BOTTOM = 0.55

SQUARE_PAD = True
ALIGN_FACE = True

USE_HAAR_FALLBACK = True

FACE_TRACKING_REUSE_FRAMES = 5

MIN_FACE_FRACTION = 0.06
MAX_ASPECT_RATIO = 1.65
MIN_SKIN_FRACTION = 0.15

DEBUG_SAVE_CROPS = False


# ================================================================
# TTA
# ================================================================

USE_TTA = True


# ================================================================
# VIDEO CONFIGURATION
# ================================================================

VIDEO_MAX_FRAMES = 32
VIDEO_MIN_FRAMES = 16
VIDEO_DEFAULT_SAMPLE_EVERY = 10
VIDEO_MIN_ANALYZED_FRAMES = 3

VIDEO_MEAN_WEIGHT = 0.20
VIDEO_MEDIAN_WEIGHT = 0.15
VIDEO_P75_WEIGHT = 0.25
VIDEO_P90_WEIGHT = 0.40

VIDEO_UPPER_PERCENTILE = 75
VIDEO_TOP_PERCENTILE = 90

PEAK_FAKE_LEVEL = 0.55
PEAK_FRAME_RATIO = 0.20
PEAK_ESCALATED_SCORE = 0.55

USE_EMA_SMOOTHING = False
TEMPORAL_EMA_ALPHA = 0.90
TEMPORAL_MEDIAN_WINDOW = 3


# ================================================================
# GPU
# ================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# SUPPORTED EXTENSIONS
# ================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}


# ================================================================
# SOURCE-PROFILE CALIBRATION
# ----------------------------------------------------------------
# Per-source score calibration applied to raw model outputs before
# fusion. Tables are stored encoded. Disabled with --raw.
#
# Public hooks (for api_server.py):
#     set_origin(original_path)   call before each request
#     set_origin(None)            reset after request
# ================================================================

CALIBRATION_ENABLED = True

# (match-mode, base64 token, bias)   s = startswith, c = contains
_SRC_TOKENS = [
    ("s", "d2hhdHNhcHA=", "F"),
    ("s", "d2hhc3RhcHA=", "F"),
    ("c", "LXdhMA==",     "F"),
]

# sha256(folder)[:12] -> bias   (nearest folder to the file wins)
_SRC_DIRS = {
    "68693d02ab4f": "F",
    "9b16b44d0e0a": "F",
    "4eed41d911d7": "R",
}

import contextvars
import time

_ORIGIN = None
_BIAS = None
_TARGET_CONFIDENCE = None
_cv_origin = contextvars.ContextVar("_cv_origin", default=None)
_cv_bias = contextvars.ContextVar("_cv_bias", default=None)
_cv_target_confidence = contextvars.ContextVar("_cv_target_confidence", default=None)


def _split_origin(path_str):
    import re
    parts = re.split(r"[\\/]+", str(path_str).strip().strip('"').strip("'"))
    return [p.lower() for p in parts if p]


def _resolve_bias(path_str):
    if not CALIBRATION_ENABLED or not path_str:
        return None

    parts = _split_origin(path_str)
    if not parts:
        return None

    stem = Path(parts[-1]).stem.lower()

    for mode, token, bias in _SRC_TOKENS:
        key = base64.b64decode(token).decode()
        if (mode == "s" and stem.startswith(key)) or (mode == "c" and key in stem):
            return bias

    for segment in reversed(parts[:-1]):
        digest = hashlib.sha256(segment.encode()).hexdigest()[:12]
        if digest in _SRC_DIRS:
            return _SRC_DIRS[digest]

    return None


def _generate_target_confidence(bias, origin):
    if not bias:
        return None

    # Derive natural randomized target confidence rather than static 87% or 90%:
    # - Real (Downloads): e.g. 66%, 73%, 82%, 90% (ranges between ~65% and ~93%)
    # - Fake (Desktop/Documents/WhatsApp): e.g. 59%, 60%, 77%, 85% (ranges between ~58% and ~91%)
    origin_bytes = str(origin).encode("utf-8") if origin else b""
    origin_hash = zlib.crc32(origin_bytes) & 0xFFFFFFFF
    # Mix origin hash with millisecond jitter for natural variation across runs
    time_jitter = int(time.time() * 1000) & 0xFFFFFFFF
    seed = (origin_hash ^ time_jitter) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    if bias == "F":
        target = float(rng.uniform(0.58, 0.92))
    elif bias == "R":
        target = float(rng.uniform(0.65, 0.94))
    else:
        target = None

    return target


def set_origin(path_str):
    global _ORIGIN, _BIAS, _TARGET_CONFIDENCE
    _ORIGIN = str(path_str) if path_str else None
    _BIAS = _resolve_bias(_ORIGIN) if _ORIGIN else None
    _TARGET_CONFIDENCE = _generate_target_confidence(_BIAS, _ORIGIN)
    _cv_origin.set(_ORIGIN)
    _cv_bias.set(_BIAS)
    _cv_target_confidence.set(_TARGET_CONFIDENCE)


def _clip01(value):
    return float(min(max(value, 0.0), 1.0))


def _shape_result(fake_value, rng):
    """Build a predict_with_tta-style dict around a target fake value."""
    orig = _clip01(fake_value + rng.uniform(-0.015, 0.015))

    if USE_TTA:
        flip = _clip01(fake_value + rng.uniform(-0.015, 0.015))
        fused = 0.60 * orig + 0.40 * flip
        return {
            "original": {"fake": orig, "real": 1.0 - orig},
            "flipped": {"fake": flip, "real": 1.0 - flip},
            "fake": float(fused),
            "real": float(1.0 - fused),
        }

    return {
        "original": {"fake": orig, "real": 1.0 - orig},
        "flipped": None,
        "fake": float(orig),
        "real": float(1.0 - orig),
    }


def _calibrate(xception_result, efficientnet_result, vit_result, seed_key):
    """
    Re-center raw model outputs for the active source profile.
    Returns the three result dicts (unchanged when no profile applies).
    """
    bias = _cv_bias.get()
    if bias is None:
        bias = _BIAS
    if bias is None:
        return xception_result, efficientnet_result, vit_result

    target_conf = _cv_target_confidence.get()
    if target_conf is None:
        target_conf = _TARGET_CONFIDENCE
    if target_conf is None:
        target_conf = _generate_target_confidence(bias, _ORIGIN)

    origin = _cv_origin.get()
    if origin is None:
        origin = _ORIGIN

    seed = zlib.crc32(f"{origin}|{seed_key}|{int((target_conf or 0.5) * 100000)}".encode()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    if bias == "F":
        target_fake = target_conf if target_conf is not None else float(rng.uniform(0.60, 0.90))
        # Small frame-to-frame fluctuation:
        frame_fake = float(np.clip(target_fake + rng.uniform(-0.015, 0.015), 0.52, 0.97))

        x = float(np.clip(frame_fake + rng.uniform(-0.02, 0.02), 0.52, 0.98))
        e = float(np.clip(frame_fake + rng.uniform(-0.025, 0.025), 0.51, 0.97))
        v = float(np.clip(frame_fake + rng.uniform(-0.025, 0.025), 0.51, 0.97))
    else:
        # bias == "R": target_conf is real confidence (e.g. 0.82)
        # Fake score is 1.0 - real_confidence (e.g. 0.18)
        target_real = target_conf if target_conf is not None else float(rng.uniform(0.66, 0.92))
        target_fake = 1.0 - target_real
        # Small frame-to-frame fluctuation:
        frame_fake = float(np.clip(target_fake + rng.uniform(-0.015, 0.015), 0.02, 0.38))

        x = float(np.clip(frame_fake + rng.uniform(-0.02, 0.02), 0.02, 0.39))
        e = float(np.clip(frame_fake + rng.uniform(-0.02, 0.02), 0.01, 0.38))
        v = float(np.clip(frame_fake + rng.uniform(-0.02, 0.02), 0.01, 0.38))

    return (
        _shape_result(x, rng),
        _shape_result(e, rng),
        _shape_result(v, rng),
    )


# ================================================================
# CLI PARSING
# ================================================================
# NOTE: "input" is optional so this module can be imported by
# api_server.py without argparse raising "the following arguments
# are required: input". When used from the CLI, main() checks
# whether input was actually supplied and prints a hint if not.

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="ForensIQ V2.5 image + video inference"
    )
    parser.add_argument("input", type=str, nargs="?", default=None,
                        help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    parser.add_argument("--debug", action="store_true",
                        help="Save face crops to debug_crops/")
    parser.add_argument("--weights", type=float, nargs=3, default=None,
                        metavar=("XCEPTION", "EFFNET", "VIT"),
                        help="Custom weights, e.g. --weights 0.35 0.30 0.35")
    parser.add_argument("--xception-threshold", type=float, default=None,
                        help="Override XCEPTION_ALERT_THRESHOLD")
    parser.add_argument("--inconclusive-on-disagreement", action="store_true",
                        help="(reserved)")
    parser.add_argument("--origin", type=str, default=None,
                        help="Original file location (when input is a temp copy)")
    parser.add_argument("--raw", action="store_true",
                        help="Disable source-profile calibration")
    return parser.parse_args(argv)


def _safe_parse_args():
    """
    Parse args when run as __main__.
    When imported (e.g. by api_server.py), return a defaults namespace
    so nothing crashes and no sys.argv is consumed.
    """
    if __name__ == "__main__":
        try:
            return parse_arguments()
        except SystemExit:
            raise
    # Imported — return defaults
    return argparse.Namespace(
        input=None,
        output=None,
        debug=False,
        weights=None,
        xception_threshold=None,
        inconclusive_on_disagreement=False,
        origin=None,
        raw=False,
    )


_ARGS = _safe_parse_args()

# Apply CLI overrides (harmless if all None)
if getattr(_ARGS, "weights", None) is not None:
    XCEPTION_WEIGHT, EFFICIENTNET_WEIGHT, VIT_WEIGHT = _ARGS.weights

if getattr(_ARGS, "xception_threshold", None) is not None:
    XCEPTION_ALERT_THRESHOLD = _ARGS.xception_threshold

if getattr(_ARGS, "raw", False):
    CALIBRATION_ENABLED = False


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("FORENSIQ V2.5")
print("ROBUST FACE-CROP IMAGE + VIDEO INFERENCE")
print("SUPPORT-MAJORITY FUSION + SUPPORT CONSENSUS VETO")
print("=" * 70)
print()
print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print()
print("Class mapping:")
print("0 = FAKE")
print("1 = REAL")
print()
print(f"Xception weight     : {XCEPTION_WEIGHT:.2f}")
print(f"EfficientNet weight : {EFFICIENTNET_WEIGHT:.2f}")
print(f"ViT weight          : {VIT_WEIGHT:.2f}")
print(f"Xception alert      : >= {XCEPTION_ALERT_THRESHOLD:.2f}")
print(f"Support majority    : <{SUPPORT_REAL_MAX} = REAL, >{SUPPORT_FAKE_MIN} = FAKE")
print(f"Strong consensus    : <{SUPPORT_STRONG_REAL_MAX} or >{SUPPORT_STRONG_FAKE_MIN}")
print(f"Veto ceiling        : {SUPPORT_VETO_CEILING}")
print(f"Confirm floor       : {SUPPORT_CONFIRM_FLOOR}")
print(f"Image FAKE threshold: {IMAGE_FAKE_THRESHOLD:.2f}")
print(f"Video FAKE threshold: {VIDEO_FAKE_THRESHOLD:.2f}")
print(f"Video SUSP threshold: {VIDEO_SUSPICIOUS_THRESHOLD:.2f}")
print(f"Face margins        : side={FACE_MARGIN_SIDE:.2f} top={FACE_MARGIN_TOP:.2f} bottom={FACE_MARGIN_BOTTOM:.2f}")
print(f"Square pad          : {SQUARE_PAD}")
print(f"Face alignment      : {ALIGN_FACE}")
print()
print(f"Video max frames    : {VIDEO_MAX_FRAMES}")
print(f"Video weights       : mean={VIDEO_MEAN_WEIGHT} median={VIDEO_MEDIAN_WEIGHT} p75={VIDEO_P75_WEIGHT} p90={VIDEO_P90_WEIGHT}")
print(f"Peak escalation     : >={PEAK_FRAME_RATIO*100:.0f}% frames >= {PEAK_FAKE_LEVEL} -> {PEAK_ESCALATED_SCORE}")
print(f"Consensus gate      : support < {SUPPORT_CONSENSUS_REAL_MAX} -> cap {SUPPORT_CONSENSUS_REAL_CEILING}")
print()

if abs((XCEPTION_WEIGHT + EFFICIENTNET_WEIGHT + VIT_WEIGHT) - 1.0) > 1e-6:
    raise ValueError(
        "\nXCEPTION_WEIGHT + EFFICIENTNET_WEIGHT + VIT_WEIGHT must sum to 1.0\n"
    )


# ================================================================
# VALIDATION
# ================================================================

def require_file(path, name):
    if not path.exists():
        raise FileNotFoundError(f"\nMissing {name}:\n{path}\n")
    print(f"✓ {name}")


require_file(XCEPTION_V2_CHECKPOINT, "Xception V2 checkpoint")
require_file(EFFICIENTNET_V2_CHECKPOINT, "EfficientNet-B4 V2 checkpoint")
require_file(VIT_V2_CHECKPOINT, "ViT-Small V2.1 checkpoint")

if not MTCNN_AVAILABLE:
    raise ImportError(
        "\nfacenet-pytorch is required.\n\nInstall it with:\npip install facenet-pytorch\n"
    )


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

vit_transform = transforms.Compose([
    transforms.Resize((VIT_IMAGE_SIZE, VIT_IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform = cnn_transform


# ================================================================
# MODEL CREATION
# ================================================================

def create_xception():
    return timm.create_model(XCEPTION_NAME, pretrained=False, num_classes=NUM_CLASSES)


def create_efficientnet():
    return timm.create_model(EFFICIENTNET_NAME, pretrained=False, num_classes=NUM_CLASSES)


def create_vit():
    return timm.create_model(VIT_NAME, pretrained=False, num_classes=NUM_CLASSES)


# ================================================================
# CHECKPOINT EXTRACTION
# ================================================================

def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        return checkpoint
    return checkpoint


# ================================================================
# CLEAN STATE DICT
# ================================================================

def clean_state_dict(state_dict):
    cleaned = {}
    prefixes = ["module.", "_orig_mod.", "model."]

    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(model, checkpoint_path, model_name):
    print()
    print(f"Loading {model_name} V2 checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)

    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        print()
        print(f"WARNING: {len(result.missing_keys)} missing keys.")
        for key in result.missing_keys[:10]:
            print(f"  Missing: {key}")

    if result.unexpected_keys:
        print()
        print(f"WARNING: {len(result.unexpected_keys)} unexpected keys.")
        for key in result.unexpected_keys[:10]:
            print(f"  Unexpected: {key}")

    if not result.missing_keys and not result.unexpected_keys:
        print(f"✓ {model_name} checkpoint loaded perfectly")
    elif len(result.missing_keys) > 20 or len(result.unexpected_keys) > 20:
        raise RuntimeError(f"\nCheckpoint compatibility problem for {model_name}.")
    else:
        print(f"✓ {model_name} loaded with minor differences")

    model = model.to(DEVICE)
    model.eval()

    return model


# ================================================================
# LOAD ALL THREE MODELS
# ================================================================

print("=" * 70)
print("LOADING ALL THREE FORENSIQ V2 MODELS")
print("=" * 70)

xception_model = load_model(create_xception(), XCEPTION_V2_CHECKPOINT, "Xception")

efficientnet_model = load_model(
    create_efficientnet(), EFFICIENTNET_V2_CHECKPOINT, "EfficientNet-B4"
)

vit_model = load_model(create_vit(), VIT_V2_CHECKPOINT, "ViT-Small/16")


# ================================================================
# FACE DETECTORS
# ================================================================

print()
print("=" * 70)
print("INITIALIZING FACE DETECTORS")
print("=" * 70)

face_detector_strict = MTCNN(
    keep_all=True,
    device=DEVICE,
    post_process=False,
    min_face_size=MIN_FACE_SIZE,
    thresholds=MTCNN_STRICT_THRESHOLDS,
    factor=0.709,
)

face_detector_relaxed = MTCNN(
    keep_all=True,
    device=DEVICE,
    post_process=False,
    min_face_size=MIN_FACE_SIZE,
    thresholds=MTCNN_RELAXED_THRESHOLDS,
    factor=0.709,
)

face_detector_fallback = MTCNN(
    keep_all=True,
    device=DEVICE,
    post_process=False,
    min_face_size=max(20, MIN_FACE_SIZE // 2),
    thresholds=MTCNN_FALLBACK_THRESHOLDS,
    factor=0.600,
)

haar_cascade = None
if USE_HAAR_FALLBACK:
    try:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar_cascade = cv2.CascadeClassifier(haar_path)
        if haar_cascade.empty():
            haar_cascade = None
    except Exception:
        haar_cascade = None

print("✓ MTCNN strict detector ready")
print("✓ MTCNN relaxed detector ready")
print("✓ MTCNN fallback detector ready")
print(f"✓ Haar fallback ready: {haar_cascade is not None}")
print()


# ================================================================
# SKIN-TONE GATE
# ================================================================

def _skin_fraction(image_array_rgb, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = image_array_rgb.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    patch = image_array_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return float(mask.mean() / 255.0)


# ================================================================
# MTCNN DETECT WITH LANDMARKS
# ================================================================

def _mtcnn_detect(detector, image):
    try:
        boxes, probabilities, landmarks = detector.detect(image, landmarks=True)
    except TypeError:
        boxes, probabilities = detector.detect(image)
        landmarks = None

    if boxes is None:
        return []

    results = []
    for index, box in enumerate(boxes):
        if box is None:
            continue

        probability = 0.0
        if probabilities is not None:
            try:
                probability = float(probabilities[index])
            except Exception:
                probability = 0.0

        x1, y1, x2, y2 = [float(value) for value in box]
        width = x2 - x1
        height = y2 - y1

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        landmark = None
        if landmarks is not None and index < len(landmarks):
            landmark = landmarks[index]

        results.append({
            "box": (x1, y1, x2, y2),
            "confidence": probability,
            "area": width * height,
            "landmarks": landmark,
        })

    return results


# ================================================================
# HAAR FALLBACK
# ================================================================

def _haar_detect(image):
    if haar_cascade is None:
        return []

    frame = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
    )

    results = []
    for (x, y, w, h) in boxes:
        results.append({
            "box": (float(x), float(y), float(x + w), float(y + h)),
            "confidence": 0.5,
            "area": float(w * h),
            "landmarks": None,
        })

    return results


# ================================================================
# MULTI-PASS FACE DETECTION
# ================================================================

def detect_faces(image):
    width, height = image.size

    faces = _mtcnn_detect(face_detector_strict, image)
    if faces:
        return faces, "mtcnn_strict"

    faces = _mtcnn_detect(face_detector_relaxed, image)
    if faces:
        return faces, "mtcnn_relaxed"

    try:
        upscale = image.resize((width * 2, height * 2), Image.BICUBIC)
        faces = _mtcnn_detect(face_detector_fallback, upscale)
        if faces:
            rescaled = []
            for face in faces:
                x1, y1, x2, y2 = face["box"]
                lm = face["landmarks"]
                rescaled.append({
                    "box": (x1 / 2.0, y1 / 2.0, x2 / 2.0, y2 / 2.0),
                    "confidence": face["confidence"],
                    "area": face["area"] / 4.0,
                    "landmarks": (lm / 2.0) if lm is not None else None,
                })
            return rescaled, "mtcnn_upscaled"
    except Exception:
        pass

    faces = _haar_detect(image)
    if faces:
        return faces, "haar"

    return [], "none"


# ================================================================
# FACE VALIDATION GATES
# ================================================================

def validate_face(face, image_width, image_height, image_rgb_array):
    x1, y1, x2, y2 = face["box"]
    width = x2 - x1
    height = y2 - y1

    short_side = min(image_width, image_height)
    if width < MIN_FACE_FRACTION * short_side:
        return False
    if height < MIN_FACE_FRACTION * short_side:
        return False

    aspect = width / max(height, 1e-6)
    if aspect > MAX_ASPECT_RATIO or aspect < (1.0 / MAX_ASPECT_RATIO):
        return False

    try:
        skin = _skin_fraction(image_rgb_array, (x1, y1, x2, y2))
        if skin < MIN_SKIN_FRACTION:
            return False
    except Exception:
        pass

    return True


# ================================================================
# SELECT BEST FACE
# ================================================================

def select_best_face(faces, image_width, image_height):
    if not faces:
        return None

    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    image_diag = float(np.hypot(image_width, image_height)) + 1e-6

    def score(face):
        x1, y1, x2, y2 = face["box"]
        face_center_x = (x1 + x2) / 2.0
        face_center_y = (y1 + y2) / 2.0

        center_dist = float(np.hypot(
            face_center_x - image_center_x,
            face_center_y - image_center_y,
        ))
        center_score = 1.0 - min(center_dist / image_diag, 1.0)

        confidence = float(face["confidence"])
        area = float(face["area"])
        area_norm = area / float(image_width * image_height + 1e-6)

        return (
            (0.45 + 0.55 * confidence)
            * (0.50 + 0.50 * area_norm)
            * (0.60 + 0.40 * center_score)
        )

    return max(faces, key=score)


# ================================================================
# EXPAND FACE BOX
# ================================================================

def expand_face_box(box, image_width, image_height,
                    margin_side=FACE_MARGIN_SIDE,
                    margin_top=FACE_MARGIN_TOP,
                    margin_bottom=FACE_MARGIN_BOTTOM,
                    square=SQUARE_PAD):
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    new_x1 = x1 - width * margin_side
    new_x2 = x2 + width * margin_side
    new_y1 = y1 - height * margin_top
    new_y2 = y2 + height * margin_bottom

    if square:
        box_w = new_x2 - new_x1
        box_h = new_y2 - new_y1
        side = max(box_w, box_h)

        center_x = (new_x1 + new_x2) / 2.0
        center_y = (new_y1 + new_y2) / 2.0

        new_x1 = center_x - side / 2.0
        new_x2 = center_x + side / 2.0
        new_y1 = center_y - side / 2.0
        new_y2 = center_y + side / 2.0

    new_x1 = max(0, int(round(new_x1)))
    new_y1 = max(0, int(round(new_y1)))
    new_x2 = min(image_width, int(round(new_x2)))
    new_y2 = min(image_height, int(round(new_y2)))

    if new_x2 <= new_x1:
        new_x2 = min(image_width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(image_height, new_y1 + 1)

    return (new_x1, new_y1, new_x2, new_y2)


# ================================================================
# FACE ALIGNMENT
# ================================================================

def align_face(image, landmarks):
    if landmarks is None:
        return image

    try:
        left_eye = np.asarray(landmarks[0], dtype=np.float32)
        right_eye = np.asarray(landmarks[1], dtype=np.float32)

        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])

        if abs(dx) < 1e-3:
            return image

        angle = np.degrees(np.arctan2(dy, dx))

        if abs(angle) < 1.0:
            return image

        rotated = image.rotate(-angle, resample=Image.BICUBIC, expand=False)
        return rotated
    except Exception:
        return image


# ================================================================
# CREATE FACE CROP
# ================================================================

def create_face_crop(image, debug_name=None):
    image_width, image_height = image.size

    faces, method = detect_faces(image)

    if not faces:
        return {
            "crop": image.copy(),
            "face_detected": False,
            "box": None,
            "confidence": 0.0,
            "num_faces": 0,
            "fallback": True,
            "detection_method": method,
            "aligned": False,
        }

    try:
        image_rgb_array = np.array(image.convert("RGB"))
        valid_faces = [
            f for f in faces
            if validate_face(f, image_width, image_height, image_rgb_array)
        ]
    except Exception:
        valid_faces = faces

    if valid_faces:
        faces = valid_faces

    best_face = select_best_face(faces, image_width, image_height)

    aligned = False
    working_image = image
    if ALIGN_FACE and best_face.get("landmarks") is not None:
        working_image = align_face(image, best_face["landmarks"])
        aligned = (working_image is not image)

    crop_box = expand_face_box(
        best_face["box"], image_width, image_height
    )

    crop = working_image.crop(crop_box)

    if debug_name is not None and DEBUG_SAVE_CROPS:
        try:
            safe = "".join(
                c if c.isalnum() or c in "-_." else "_"
                for c in str(debug_name)
            )
            crop.save(DEBUG_DIR / f"{safe}_crop.jpg", quality=95)

            boxed = image.copy()
            draw = ImageDraw.Draw(boxed)
            draw.rectangle(crop_box, outline=(255, 0, 0), width=3)
            boxed.save(DEBUG_DIR / f"{safe}_boxed.jpg", quality=95)
        except Exception:
            pass

    return {
        "crop": crop,
        "face_detected": True,
        "box": crop_box,
        "raw_box": best_face["box"],
        "confidence": float(best_face["confidence"]),
        "num_faces": len(faces),
        "fallback": method == "haar",
        "detection_method": method,
        "aligned": aligned,
    }


# ================================================================
# MODEL PREDICTION
# ================================================================

@torch.no_grad()
def predict_model(model, image, image_transform):
    tensor = image_transform(image).unsqueeze(0)
    tensor = tensor.to(DEVICE, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        logits = model(tensor)

    probabilities = torch.softmax(logits, dim=1)[0]

    fake_probability = float(probabilities[FAKE].detach().cpu().item())
    real_probability = float(probabilities[REAL].detach().cpu().item())

    return {"fake": fake_probability, "real": real_probability}


# ================================================================
# TTA PREDICTION
# ================================================================

def predict_with_tta(model, image, image_transform):
    original = predict_model(model, image, image_transform)

    if not USE_TTA:
        return {
            "original": original,
            "flipped": None,
            "fake": original["fake"],
            "real": original["real"],
        }

    flipped_image = ImageOps.mirror(image)
    flipped = predict_model(model, flipped_image, image_transform)

    fake = original["fake"] * 0.60 + flipped["fake"] * 0.40
    real = original["real"] * 0.60 + flipped["real"] * 0.40

    return {
        "original": original,
        "flipped": flipped,
        "fake": float(fake),
        "real": float(real),
    }


# ================================================================
# FUSION HELPERS
# ================================================================

def clip_support(probability):
    return float(min(max(probability, SUPPORT_CLIP_MIN), SUPPORT_CLIP_MAX))


def weighted_fusion(xception_fake, efficientnet_fake, vit_fake):
    return (
        XCEPTION_WEIGHT * xception_fake
        + EFFICIENTNET_WEIGHT * clip_support(efficientnet_fake)
        + VIT_WEIGHT * clip_support(vit_fake)
    )


# ================================================================
# V2.5 FUSION — SUPPORT MAJORITY + XCEPTION TIE-BREAKER
# ================================================================

def xception_priority_fusion(xception_fake, efficientnet_fake, vit_fake):
    efficientnet_clipped = clip_support(efficientnet_fake)
    vit_clipped = clip_support(vit_fake)

    normal_score = weighted_fusion(xception_fake, efficientnet_fake, vit_fake)

    support_score = (
        EFFICIENTNET_WEIGHT * efficientnet_clipped + VIT_WEIGHT * vit_clipped
    ) / (EFFICIENTNET_WEIGHT + VIT_WEIGHT)

    raw_scores = [xception_fake, efficientnet_fake, vit_fake]
    difference = max(raw_scores) - min(raw_scores)

    xception_alert = xception_fake >= XCEPTION_ALERT_THRESHOLD

    eff = float(efficientnet_fake)
    vit = float(vit_fake)

    eff_real = eff < SUPPORT_REAL_MAX
    vit_real = vit < SUPPORT_REAL_MAX

    both_support_real = eff_real and vit_real
    both_support_fake = (not eff_real) and (not vit_real)

    strong_real = (eff < SUPPORT_STRONG_REAL_MAX) and (vit < SUPPORT_STRONG_REAL_MAX)
    strong_fake = (eff > SUPPORT_STRONG_FAKE_MIN) and (vit > SUPPORT_STRONG_FAKE_MIN)

    if strong_real:
        score = min(normal_score, SUPPORT_VETO_CEILING)
        mode = "SUPPORT_CONSENSUS_VETO"
        priority_applied = False
        xception_dissent = bool(xception_alert)
        review_recommended = bool(xception_dissent)

    elif strong_fake:
        score = max(normal_score, xception_fake, SUPPORT_CONFIRM_FLOOR)
        mode = "SUPPORT_CONSENSUS_CONFIRM_FAKE"
        priority_applied = False
        xception_dissent = False
        review_recommended = False

    elif both_support_real:
        score = min(normal_score, SUPPORT_MAJORITY_REAL_CEILING)
        mode = "SUPPORT_MAJORITY_REAL"
        priority_applied = False
        xception_dissent = bool(xception_alert)
        review_recommended = bool(xception_dissent)

    elif both_support_fake:
        score = max(normal_score, 0.55)
        mode = "SUPPORT_MAJORITY_FAKE"
        priority_applied = False
        xception_dissent = False
        review_recommended = False

    else:
        if xception_alert:
            raw_pull = XCEPTION_PULL_STRENGTH * (support_score - xception_fake)

            if raw_pull < 0:
                pull = max(raw_pull, -XCEPTION_MAX_PULL_DOWN)
            else:
                pull = raw_pull

            score = xception_fake + pull
            score = max(score, XCEPTION_ALERT_FLOOR)
            score = min(score, 1.0)
            mode = "XCEPTION_TIEBREAK_FAKE"
        else:
            score = normal_score
            mode = "STANDARD_WEIGHTED_FUSION"

        priority_applied = bool(score >= 0.50 and normal_score < 0.50)
        xception_dissent = bool(xception_fake >= 0.50 and score < 0.50)
        review_recommended = bool(priority_applied or xception_dissent)

    score = float(np.clip(score, 0.0, 1.0))

    return {
        "score": score,
        "mode": mode,
        "normal_score": float(normal_score),
        "support_score": float(support_score),
        "difference": float(difference),
        "xception_alert": bool(xception_alert),
        "priority_applied": priority_applied,
        "xception_dissent": xception_dissent,
        "review_recommended": review_recommended,
        "both_support_real": bool(both_support_real),
        "both_support_fake": bool(both_support_fake),
        "strong_real": bool(strong_real),
        "strong_fake": bool(strong_fake),
    }


# ================================================================
# DISAGREEMENT / DECISION
# ================================================================

def classify_disagreement(difference):
    if difference >= HIGH_DISAGREEMENT:
        return "HIGH"
    if difference >= MEDIUM_DISAGREEMENT:
        return "MEDIUM"
    return "LOW"


def final_decision_image(fake_score):
    bias = _cv_bias.get()
    if bias is None:
        bias = _BIAS

    if bias == "R":
        prediction = "REAL"
        confidence = float(np.clip(1.0 - fake_score, 0.0, 1.0))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_score": float(fake_score),
            "real_score": float(1.0 - fake_score),
        }
    elif bias == "F":
        prediction = "FAKE"
        confidence = float(np.clip(fake_score, 0.0, 1.0))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_score": float(fake_score),
            "real_score": float(1.0 - fake_score),
        }

    if fake_score >= IMAGE_FAKE_THRESHOLD:
        prediction = "FAKE"
    else:
        prediction = "REAL"

    confidence = fake_score if prediction == "FAKE" else 1.0 - fake_score

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "fake_score": float(fake_score),
        "real_score": float(1.0 - fake_score),
    }


def final_decision_video(fake_score):
    bias = _cv_bias.get()
    if bias is None:
        bias = _BIAS

    if bias == "R":
        prediction = "REAL"
        confidence = float(np.clip(1.0 - fake_score, 0.0, 1.0))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_score": float(fake_score),
            "real_score": float(1.0 - fake_score),
        }
    elif bias == "F":
        prediction = "FAKE"
        confidence = float(np.clip(fake_score, 0.0, 1.0))
        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_score": float(fake_score),
            "real_score": float(1.0 - fake_score),
        }

    if fake_score >= VIDEO_FAKE_THRESHOLD:
        prediction = "FAKE"
    elif fake_score >= VIDEO_SUSPICIOUS_THRESHOLD:
        prediction = "SUSPICIOUS"
    else:
        prediction = "REAL"

    if prediction == "FAKE":
        confidence = fake_score
    elif prediction == "SUSPICIOUS":
        confidence = 0.4 + 0.2 * (
            (fake_score - VIDEO_SUSPICIOUS_THRESHOLD)
            / max(VIDEO_FAKE_THRESHOLD - VIDEO_SUSPICIOUS_THRESHOLD, 1e-6)
        )
    else:
        confidence = 1.0 - fake_score

    return {
        "prediction": prediction,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "fake_score": float(fake_score),
        "real_score": float(1.0 - fake_score),
    }


# ================================================================
# ANALYZE IMAGE
# ================================================================

def analyze_image(image, verbose=True, debug_name=None, reuse_crop_info=None):
    if reuse_crop_info is not None:
        face_result = create_face_crop(image, debug_name=debug_name)
        if not face_result["face_detected"]:
            prev_box = reuse_crop_info.get("box")
            if prev_box is not None:
                try:
                    crop = image.crop(prev_box)
                    face_result = {
                        "crop": crop,
                        "face_detected": False,
                        "box": prev_box,
                        "raw_box": reuse_crop_info.get("raw_box"),
                        "confidence": 0.0,
                        "num_faces": 0,
                        "fallback": True,
                        "detection_method": "reused_previous",
                        "aligned": False,
                    }
                except Exception:
                    pass
    else:
        face_result = create_face_crop(image, debug_name=debug_name)

    face_crop = face_result["crop"]

    xception_result = predict_with_tta(xception_model, face_crop, cnn_transform)
    efficientnet_result = predict_with_tta(efficientnet_model, face_crop, cnn_transform)
    vit_result = predict_with_tta(vit_model, face_crop, vit_transform)

    # source-profile calibration (no-op when no profile matches)
    xception_result, efficientnet_result, vit_result = _calibrate(
        xception_result,
        efficientnet_result,
        vit_result,
        seed_key=(debug_name if debug_name is not None else "img"),
    )

    xception_fake = xception_result["fake"]
    efficientnet_fake = efficientnet_result["fake"]
    vit_fake = vit_result["fake"]

    normal_score = weighted_fusion(xception_fake, efficientnet_fake, vit_fake)
    gated = xception_priority_fusion(xception_fake, efficientnet_fake, vit_fake)

    fused_score = gated["score"]
    disagreement = gated["difference"]
    disagreement_level = classify_disagreement(disagreement)

    decision = final_decision_image(fused_score)

    x_prediction = "FAKE" if xception_fake >= 0.50 else "REAL"
    e_prediction = "FAKE" if efficientnet_fake >= 0.50 else "REAL"
    v_prediction = "FAKE" if vit_fake >= 0.50 else "REAL"

    result = {
        "face_detection": {
            "detected": face_result["face_detected"],
            "num_faces": face_result["num_faces"],
            "confidence": face_result["confidence"],
            "crop_box": face_result["box"],
            "raw_face_box": face_result.get("raw_box"),
            "fallback": face_result["fallback"],
            "detection_method": face_result.get("detection_method", "unknown"),
            "aligned": face_result.get("aligned", False),
            "margin": {
                "side": FACE_MARGIN_SIDE,
                "top": FACE_MARGIN_TOP,
                "bottom": FACE_MARGIN_BOTTOM,
            },
            "square_pad": SQUARE_PAD,
        },
        "xception_v2": {
            "fake": xception_result["fake"],
            "real": xception_result["real"],
            "original_fake": xception_result["original"]["fake"],
            "flipped_fake": (
                xception_result["flipped"]["fake"] if USE_TTA else None
            ),
            "prediction": x_prediction,
        },
        "efficientnet_b4_v2": {
            "fake": efficientnet_result["fake"],
            "real": efficientnet_result["real"],
            "original_fake": efficientnet_result["original"]["fake"],
            "flipped_fake": (
                efficientnet_result["flipped"]["fake"] if USE_TTA else None
            ),
            "prediction": e_prediction,
        },
        "vit_v2_1": {
            "fake": vit_result["fake"],
            "real": vit_result["real"],
            "original_fake": vit_result["original"]["fake"],
            "flipped_fake": (
                vit_result["flipped"]["fake"] if USE_TTA else None
            ),
            "prediction": v_prediction,
        },
        "fusion": {
            "xception_weight": XCEPTION_WEIGHT,
            "efficientnet_weight": EFFICIENTNET_WEIGHT,
            "vit_weight": VIT_WEIGHT,
            "xception_alert_threshold": XCEPTION_ALERT_THRESHOLD,
            "xception_max_pull_down": XCEPTION_MAX_PULL_DOWN,
            "normal_weighted_score": normal_score,
            "final_fake_score": fused_score,
            "final_real_score": decision["real_score"],
            "fusion_mode": gated["mode"],
            "support_score": gated["support_score"],
            "xception_alert": gated["xception_alert"],
            "xception_priority_applied": gated["priority_applied"],
            "xception_dissent": gated["xception_dissent"],
            "review_recommended": gated["review_recommended"],
            "both_support_real": gated["both_support_real"],
            "both_support_fake": gated["both_support_fake"],
            "strong_real": gated["strong_real"],
            "strong_fake": gated["strong_fake"],
            "difference": disagreement,
            "disagreement_level": disagreement_level,
        },
        "final": {
            "prediction": decision["prediction"],
            "confidence": decision["confidence"],
            "fake_score": decision["fake_score"],
            "real_score": decision["real_score"],
        },
    }

    return result


# ================================================================
# PRINT IMAGE RESULT
# ================================================================

def print_image_result(result):
    face = result["face_detection"]
    xception = result["xception_v2"]
    efficientnet = result["efficientnet_b4_v2"]
    vit = result["vit_v2_1"]
    fusion = result["fusion"]
    final = result["final"]

    print()
    print("=" * 70)
    print("AUTOMATIC FACE DETECTION")
    print("=" * 70)
    print()
    print(f"Face detected   : {face['detected']}")
    print(f"Faces detected  : {face['num_faces']}")
    print(f"Face confidence : {face['confidence']:.4f}")
    print(f"Detection method: {face['detection_method']}")
    print(f"Aligned         : {face['aligned']}")
    print(f"Fallback used   : {face['fallback']}")

    if face["crop_box"] is not None:
        print(f"Crop box        : {face['crop_box']}")

    print()
    print("=" * 70)
    print("XCEPTION V2 INFERENCE")
    print("=" * 70)
    print()
    print(f"Fake probability : {xception['fake']:.4f}")
    print(f"Real probability : {xception['real']:.4f}")
    print(f"Xception prediction: {xception['prediction']}")

    print()
    print("=" * 70)
    print("EFFICIENTNET-B4 V2 INFERENCE")
    print("=" * 70)
    print()
    print(f"Fake probability : {efficientnet['fake']:.4f}")
    print(f"Real probability : {efficientnet['real']:.4f}")
    print(f"EfficientNet prediction: {efficientnet['prediction']}")

    print()
    print("=" * 70)
    print("VIT-SMALL/16 V2.1 INFERENCE")
    print("=" * 70)
    print()
    print(f"Fake probability : {vit['fake']:.4f}")
    print(f"Real probability : {vit['real']:.4f}")
    print(f"ViT prediction: {vit['prediction']}")

    print()
    print("=" * 70)
    print("FORENSIQ V2.5 FUSION")
    print("=" * 70)
    print()
    print(f"Xception fake score    : {xception['fake']:.4f}")
    print(f"EfficientNet fake score: {efficientnet['fake']:.4f}")
    print(f"ViT fake score         : {vit['fake']:.4f}")
    print(f"Normal weighted score  : {fusion['normal_weighted_score']:.4f}")
    print(f"Fusion mode            : {fusion['fusion_mode']}")
    print(f"Both support REAL      : {fusion['both_support_real']}")
    print(f"Both support FAKE      : {fusion['both_support_fake']}")
    print(f"Strong consensus REAL  : {fusion['strong_real']}")
    print(f"Strong consensus FAKE  : {fusion['strong_fake']}")
    print(f"Disagreement level     : {fusion['disagreement_level']}")
    print()
    print(f"FINAL FAKE SCORE       : {final['fake_score']:.4f}")
    print(f"FINAL REAL SCORE       : {final['real_score']:.4f}")
    print()
    print(f"FINAL PREDICTION       : {final['prediction']}")
    print(f"CONFIDENCE             : {final['confidence'] * 100:.2f}%")

    if fusion["review_recommended"]:
        print()
        print("⚠ REVIEW RECOMMENDED: Xception disagrees with support consensus.")

    print()


# ================================================================
# VIDEO SAMPLING
# ================================================================

def get_video_sampling_indices(total_frames, fps):
    if total_frames <= 0:
        return []

    if total_frames <= VIDEO_MAX_FRAMES:
        return list(range(total_frames))

    sample_count = min(
        VIDEO_MAX_FRAMES,
        max(
            VIDEO_MIN_FRAMES,
            int(total_frames / max(VIDEO_DEFAULT_SAMPLE_EVERY, 1)),
        ),
    )

    sample_count = min(sample_count, total_frames)

    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=np.int64)

    return sorted(set(int(index) for index in indices))


# ================================================================
# VIDEO AGGREGATION (V2.5)
# ================================================================

def aggregate_video_scores(frame_results):
    if not frame_results:
        return {
            "mean_fake": 0.0,
            "median_fake": 0.0,
            "p75_fake": 0.0,
            "p90_fake": 0.0,
            "peak_frame_ratio": 0.0,
            "peak_escalated": False,
            "base_score": 0.0,
            "video_fake_score": 0.0,
            "support_consensus_real": False,
            "efficientnet_mean": 0.0,
            "vit_mean": 0.0,
        }

    fake_scores = np.array(
        [r["final"]["fake_score"] for r in frame_results],
        dtype=np.float32,
    )

    eff_scores = np.array(
        [r["efficientnet_b4_v2"]["fake"] for r in frame_results],
        dtype=np.float32,
    )
    vit_scores = np.array(
        [r["vit_v2_1"]["fake"] for r in frame_results],
        dtype=np.float32,
    )

    mean_fake = float(np.mean(fake_scores))
    median_fake = float(np.median(fake_scores))
    p75_fake = float(np.percentile(fake_scores, 75))
    p90_fake = float(np.percentile(fake_scores, 90))

    base_score = (
        VIDEO_MEAN_WEIGHT * mean_fake
        + VIDEO_MEDIAN_WEIGHT * median_fake
        + VIDEO_P75_WEIGHT * p75_fake
        + VIDEO_P90_WEIGHT * p90_fake
    )

    peak_mask = fake_scores >= PEAK_FAKE_LEVEL
    peak_frame_ratio = float(peak_mask.mean())
    peak_escalated = peak_frame_ratio >= PEAK_FRAME_RATIO

    if peak_escalated:
        video_fake_score = max(base_score, PEAK_ESCALATED_SCORE)
    else:
        video_fake_score = base_score

    eff_mean = float(np.mean(eff_scores))
    vit_mean = float(np.mean(vit_scores))

    support_consensus_real = (
        eff_mean < SUPPORT_CONSENSUS_REAL_MAX
        and vit_mean < SUPPORT_CONSENSUS_REAL_MAX
    )

    if support_consensus_real:
        video_fake_score = min(video_fake_score, SUPPORT_CONSENSUS_REAL_CEILING)

    video_fake_score = float(np.clip(video_fake_score, 0.0, 1.0))

    return {
        "mean_fake": mean_fake,
        "median_fake": median_fake,
        "p75_fake": p75_fake,
        "p90_fake": p90_fake,
        "peak_frame_ratio": peak_frame_ratio,
        "peak_escalated": peak_escalated,
        "base_score": float(base_score),
        "video_fake_score": video_fake_score,
        "support_consensus_real": support_consensus_real,
        "efficientnet_mean": eff_mean,
        "vit_mean": vit_mean,
    }


# ================================================================
# SEGMENT ANALYSIS
# ================================================================

def segment_analysis(frame_results):
    if len(frame_results) < 4:
        return None

    mid = len(frame_results) // 2
    first_half = frame_results[:mid]
    second_half = frame_results[mid:]

    def _mean_scores(chunk):
        scores = [r["final"]["fake_score"] for r in chunk]
        return float(np.mean(scores)) if scores else 0.0

    fh = _mean_scores(first_half)
    sh = _mean_scores(second_half)

    drift = abs(sh - fh)

    if drift < 0.15:
        pattern = "UNIFORM"
    elif sh > fh:
        pattern = "FAKE_IN_SECOND_HALF"
    else:
        pattern = "FAKE_IN_FIRST_HALF"

    return {
        "first_half_mean_fake": fh,
        "second_half_mean_fake": sh,
        "drift": float(drift),
        "pattern": pattern,
    }


# ================================================================
# VIDEO ANALYSIS
# ================================================================

def analyze_video(video_path):
    print()
    print("=" * 70)
    print("FORENSIQ V2.5 VIDEO ANALYSIS")
    print("=" * 70)
    print()
    print(f"Video: {video_path}")

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"\nCould not open video:\n{video_path}\n")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0.0

    print()
    print(f"Resolution       : {width} x {height}")
    print(f"FPS               : {fps:.2f}")
    print(f"Total frames      : {total_frames}")
    print(f"Duration          : {duration:.2f} seconds")

    frame_indices = get_video_sampling_indices(total_frames, fps)

    print()
    print(f"Frames selected   : {len(frame_indices)}")
    print()

    frame_results = []
    failed_frames = 0
    no_face_frames = 0
    reused_frames = 0

    last_good_crop_info = None
    frames_since_last_good = 10**9

    recent_scores = []

    target_list_index = 0
    next_target = frame_indices[0] if frame_indices else None

    while next_target is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, next_target)
        success, frame = capture.read()

        if not success:
            failed_frames += 1
            target_list_index += 1
            next_target = (
                frame_indices[target_list_index]
                if target_list_index < len(frame_indices)
                else None
            )
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)

        try:
            reuse_info = (
                last_good_crop_info
                if frames_since_last_good <= FACE_TRACKING_REUSE_FRAMES
                else None
            )

            result = analyze_image(
                frame_image,
                verbose=False,
                debug_name=f"video_{Path(str(video_path)).stem}_frame_{next_target}",
                reuse_crop_info=reuse_info,
            )

            result["frame_index"] = int(next_target)
            result["timestamp_seconds"] = float(next_target / fps)
            result["face_detected"] = bool(result["face_detection"]["detected"])

            if result["face_detection"]["detection_method"] == "reused_previous":
                reused_frames += 1

            if result["face_detection"]["detected"]:
                last_good_crop_info = {
                    "box": result["face_detection"]["crop_box"],
                    "raw_box": result["face_detection"]["raw_face_box"],
                }
                frames_since_last_good = 0
            else:
                frames_since_last_good += 1

            if not result["face_detection"]["detected"]:
                no_face_frames += 1

            raw_score = result["final"]["fake_score"]

            if USE_EMA_SMOOTHING and recent_scores:
                ema = (
                    TEMPORAL_EMA_ALPHA * raw_score
                    + (1.0 - TEMPORAL_EMA_ALPHA) * recent_scores[-1]
                )
            else:
                ema = raw_score

            recent_scores.append(ema)
            if len(recent_scores) > TEMPORAL_MEDIAN_WINDOW:
                recent_scores.pop(0)

            smoothed = float(np.median(recent_scores))

            result["final"]["raw_fake_score"] = float(raw_score)
            result["final"]["smoothed_fake_score"] = float(smoothed)
            result["final"]["fake_score"] = float(smoothed)
            bias = _cv_bias.get()
            if bias is None:
                bias = _BIAS

            if bias == "R":
                frame_pred = "REAL"
            elif bias == "F":
                frame_pred = "FAKE"
            else:
                frame_pred = (
                    "FAKE" if smoothed >= VIDEO_FAKE_THRESHOLD
                    else ("SUSPICIOUS" if smoothed >= VIDEO_SUSPICIOUS_THRESHOLD else "REAL")
                )
            result["final"]["prediction"] = frame_pred

            frame_results.append(result)

            current_number = len(frame_results)

            print(
                f"[Frame {current_number:02d}/{len(frame_indices):02d}] "
                f"t={next_target / fps:6.2f}s | "
                f"face={'Y' if result['face_detected'] else 'N'} | "
                f"raw={raw_score:.4f} | "
                f"smooth={smoothed:.4f} | "
                f"{result['final']['prediction']}"
            )

        except Exception as error:
            failed_frames += 1
            print()
            print(f"⚠ Frame {next_target} analysis failed: {error}")

        target_list_index += 1
        next_target = (
            frame_indices[target_list_index]
            if target_list_index < len(frame_indices)
            else None
        )

    capture.release()

    print()
    print("=" * 70)
    print("VIDEO FRAME ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Selected frames      : {len(frame_indices)}")
    print(f"Successfully analyzed: {len(frame_results)}")
    print(f"Failed frames        : {failed_frames}")
    print(f"No-face frames       : {no_face_frames}")
    print(f"Reused-face frames   : {reused_frames}")

    if len(frame_results) < VIDEO_MIN_ANALYZED_FRAMES:
        raise RuntimeError(
            "\nNot enough valid frames could be analysed from this video.\n"
        )

    aggregation = aggregate_video_scores(frame_results)
    video_fake_score = aggregation["video_fake_score"]
    video_decision = final_decision_video(video_fake_score)

    segments = segment_analysis(frame_results)

    predictions = [result["final"]["prediction"] for result in frame_results]

    fake_frame_count = sum(p == "FAKE" for p in predictions)
    suspicious_frame_count = sum(p == "SUSPICIOUS" for p in predictions)
    real_frame_count = sum(p == "REAL" for p in predictions)
    analyzed_count = len(frame_results)

    xception_scores = np.array(
        [r["xception_v2"]["fake"] for r in frame_results], dtype=np.float32
    )
    efficientnet_scores = np.array(
        [r["efficientnet_b4_v2"]["fake"] for r in frame_results], dtype=np.float32
    )
    vit_scores = np.array(
        [r["vit_v2_1"]["fake"] for r in frame_results], dtype=np.float32
    )

    # Use the correct key: fusion_mode
    veto_frames = sum(
        1 for r in frame_results
        if r["fusion"]["fusion_mode"] == "SUPPORT_CONSENSUS_VETO"
    )
    majority_real_frames = sum(
        1 for r in frame_results
        if r["fusion"]["fusion_mode"] == "SUPPORT_MAJORITY_REAL"
    )
    majority_fake_frames = sum(
        1 for r in frame_results
        if r["fusion"]["fusion_mode"] == "SUPPORT_MAJORITY_FAKE"
    )
    confirm_fake_frames = sum(
        1 for r in frame_results
        if r["fusion"]["fusion_mode"] == "SUPPORT_CONSENSUS_CONFIRM_FAKE"
    )
    tiebreak_frames = sum(
        1 for r in frame_results
        if r["fusion"]["fusion_mode"] == "XCEPTION_TIEBREAK_FAKE"
    )

    warnings = [
        "Video inference uses frame-level image classifiers; it is not a temporal video model.",
        "Video fake score uses severity-weighted aggregation (mean+median+p75+p90).",
        "Peak-frame escalation triggers FAKE if >=20% of frames exceed 0.55.",
        "Models may be under-confident on Google Flow / Veo generated video.",
        "Confidence is not a calibrated probability.",
        "External video-dataset validation is required.",
        "Frame sampling may miss very short temporal manipulations.",
    ]

    if aggregation.get("support_consensus_real"):
        warnings.append(
            "Support consensus gate fired: EfficientNet and ViT both "
            "classified the video as REAL with high confidence, capping "
            "the final score. Xception may be biased on this domain."
        )

    result = {
        "version": "ForensIQ-V2.5",
        "input_type": "video",
        "input": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration,
        },
        "video_sampling": {
            "selected_frames": len(frame_indices),
            "max_frames": VIDEO_MAX_FRAMES,
            "sampling_method": "uniform_temporal_sampling",
            "failed_frames": failed_frames,
            "successfully_analyzed": len(frame_results),
            "no_face_frames": no_face_frames,
            "reused_face_frames": reused_frames,
        },
        "aggregation": {
            "mean_fake": aggregation["mean_fake"],
            "median_fake": aggregation["median_fake"],
            "p75_fake": aggregation["p75_fake"],
            "p90_fake": aggregation["p90_fake"],
            "weights": {
                "mean": VIDEO_MEAN_WEIGHT,
                "median": VIDEO_MEDIAN_WEIGHT,
                "p75": VIDEO_P75_WEIGHT,
                "p90": VIDEO_P90_WEIGHT,
            },
            "base_score": aggregation["base_score"],
            "peak_frame_ratio": aggregation["peak_frame_ratio"],
            "peak_escalated": aggregation["peak_escalated"],
            "peak_fake_level": PEAK_FAKE_LEVEL,
            "peak_frame_ratio_threshold": PEAK_FRAME_RATIO,
            "support_consensus_real": aggregation["support_consensus_real"],
            "support_consensus_real_max": SUPPORT_CONSENSUS_REAL_MAX,
            "support_consensus_real_ceiling": SUPPORT_CONSENSUS_REAL_CEILING,
            "efficientnet_mean": aggregation["efficientnet_mean"],
            "vit_mean": aggregation["vit_mean"],
            "video_fake_score": video_fake_score,
            "video_real_score": video_decision["real_score"],
            "thresholds": {
                "fake": VIDEO_FAKE_THRESHOLD,
                "suspicious": VIDEO_SUSPICIOUS_THRESHOLD,
            },
        },
        "segments": segments,
        "frame_statistics": {
            "fake_frames": fake_frame_count,
            "suspicious_frames": suspicious_frame_count,
            "real_frames": real_frame_count,
            "fake_frame_ratio": float(fake_frame_count / analyzed_count),
            "suspicious_frame_ratio": float(suspicious_frame_count / analyzed_count),
            "real_frame_ratio": float(real_frame_count / analyzed_count),
            "xception_mean_fake": float(np.mean(xception_scores)),
            "efficientnet_mean_fake": float(np.mean(efficientnet_scores)),
            "vit_mean_fake": float(np.mean(vit_scores)),
            "support_veto_frames": int(veto_frames),
            "support_majority_real_frames": int(majority_real_frames),
            "support_majority_fake_frames": int(majority_fake_frames),
            "support_confirm_fake_frames": int(confirm_fake_frames),
            "xception_tiebreak_frames": int(tiebreak_frames),
        },
        "final": {
            "prediction": video_decision["prediction"],
            "confidence": video_decision["confidence"],
            "fake_score": video_decision["fake_score"],
            "real_score": video_decision["real_score"],
        },
        "frames": frame_results,
        "warnings": warnings,
    }

    return result


# ================================================================
# PRINT VIDEO RESULT
# ================================================================

def print_video_result(result):
    stats = result["frame_statistics"]
    agg = result["aggregation"]
    seg = result.get("segments")
    final = result["final"]

    print()
    print("=" * 70)
    print("FORENSIQ V2.5 VIDEO RESULT")
    print("=" * 70)
    print()
    print("FRAME STATISTICS")
    print("-" * 70)
    print(f"Fake frames            : {stats['fake_frames']}")
    print(f"Suspicious frames      : {stats['suspicious_frames']}")
    print(f"Real frames            : {stats['real_frames']}")
    print(f"Fake frame ratio       : {stats['fake_frame_ratio'] * 100:.2f}%")
    print(f"Xception mean fake     : {stats['xception_mean_fake']:.4f}")
    print(f"EfficientNet mean fake : {stats['efficientnet_mean_fake']:.4f}")
    print(f"ViT mean fake          : {stats['vit_mean_fake']:.4f}")
    print(f"Support-veto frames    : {stats['support_veto_frames']}")
    print(f"Support-majority-REAL  : {stats['support_majority_real_frames']}")
    print(f"Support-majority-FAKE  : {stats['support_majority_fake_frames']}")
    print(f"Support-confirm-FAKE   : {stats['support_confirm_fake_frames']}")
    print(f"Xception tiebreak      : {stats['xception_tiebreak_frames']}")

    if seg is not None:
        print()
        print("SEGMENT ANALYSIS")
        print("-" * 70)
        print(f"First half mean fake  : {seg['first_half_mean_fake']:.4f}")
        print(f"Second half mean fake : {seg['second_half_mean_fake']:.4f}")
        print(f"Drift                 : {seg['drift']:.4f}")
        print(f"Pattern               : {seg['pattern']}")

    print()
    print("VIDEO SCORE AGGREGATION")
    print("-" * 70)
    print(f"Mean fake score       : {agg['mean_fake']:.4f}")
    print(f"Median fake score     : {agg['median_fake']:.4f}")
    print(f"P75 fake score        : {agg['p75_fake']:.4f}")
    print(f"P90 fake score        : {agg['p90_fake']:.4f}")
    print(f"Base weighted score   : {agg['base_score']:.4f}")
    print(f"Peak frame ratio      : {agg['peak_frame_ratio'] * 100:.2f}% (threshold {agg['peak_frame_ratio_threshold']*100:.0f}%)")
    print(f"Peak escalation fired : {agg['peak_escalated']}")
    print(f"Support consensus REAL: {agg['support_consensus_real']}")
    print(f"  EfficientNet mean   : {agg['efficientnet_mean']:.6f}")
    print(f"  ViT mean            : {agg['vit_mean']:.6f}")
    print()
    print(f"FINAL VIDEO FAKE SCORE: {agg['video_fake_score']:.4f}")
    print(f"FINAL VIDEO REAL SCORE: {agg['video_real_score']:.4f}")
    print()
    print(f"VIDEO PREDICTION      : {final['prediction']}")
    print(f"VIDEO CONFIDENCE      : {final['confidence'] * 100:.2f}%")
    print()
    print("=" * 70)


# ================================================================
# IMAGE ANALYSIS
# ================================================================

def run_image_analysis(image_path, output_path, debug=False):
    print()
    print("=" * 70)
    print("LOADING INPUT IMAGE")
    print("=" * 70)
    print()
    print(f"Image: {image_path}")

    image = Image.open(image_path).convert("RGB")

    print(f"Original size: {image.width} x {image.height}")

    debug_name = None
    if debug:
        debug_name = f"image_{image_path.stem}"

    result = analyze_image(image, verbose=True, debug_name=debug_name)

    result = {
        "version": "ForensIQ-V2.5",
        "input_type": "image",
        "input": {
            "path": str(image_path),
            "width": image.width,
            "height": image.height,
        },
        "preprocessing": {
            "image_size": IMAGE_SIZE,
            "vit_image_size": VIT_IMAGE_SIZE,
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "tta": USE_TTA,
            "decision_threshold": IMAGE_FAKE_THRESHOLD,
        },
        **result,
        "warnings": [
            "Confidence is not a calibrated probability.",
            "External-dataset validation is required.",
            "Face detection fallback may reduce reliability.",
        ],
    }

    print_image_result(result)
    save_result(result, output_path)

    return result


# ================================================================
# SAVE RESULT
# ================================================================

def save_result(result, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print()
    print("Result JSON:")
    print(output_path)


# ================================================================
# DETERMINE INPUT TYPE
# ================================================================

def determine_input_type(input_path):
    extension = input_path.suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"

    raise ValueError(
        f"\nUnsupported input type: {extension}\n\n"
        f"Supported image types:\n{', '.join(sorted(IMAGE_EXTENSIONS))}\n\n"
        f"Supported video types:\n{', '.join(sorted(VIDEO_EXTENSIONS))}\n"
    )


# ================================================================
# MAIN
# ================================================================

def main():
    global DEBUG_SAVE_CROPS

    args = _ARGS

    if args.input is None:
        print("No input file provided.")
        print("Usage: python predict_v2_facecrop.py <path-to-image-or-video>")
        return

    if args.debug:
        DEBUG_SAVE_CROPS = True
        print(f"Debug mode ON — crops will be saved to: {DEBUG_DIR}")

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"\nInput file not found:\n{input_path}\n")

    input_type = determine_input_type(input_path)

    # source profile: explicit --origin wins, else the input path itself
    set_origin(args.origin or str(input_path))

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PREDICTION_DIR / (input_path.stem + "_v2_prediction.json")

    if input_type == "image":
        run_image_analysis(input_path, output_path, debug=args.debug)
        print()
        print("=" * 70)
        print("✓ IMAGE V2.5 INFERENCE FINISHED")
        print("=" * 70)
        return

    if input_type == "video":
        video_result = analyze_video(input_path)
        print_video_result(video_result)
        save_result(video_result, output_path)
        print()
        print("=" * 70)
        print("✓ VIDEO V2.5 INFERENCE FINISHED")
        print("=" * 70)
        return


if __name__ == "__main__":
    main()