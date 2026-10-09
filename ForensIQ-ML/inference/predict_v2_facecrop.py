# ================================================================
# FORENSIQ V2.3
# ROBUST FACE-CROP IMAGE + VIDEO INFERENCE
# MULTI-PASS FACE DETECTION + ALIGNMENT + SQUARE CROP
# SEVERITY-WEIGHTED VIDEO AGGREGATION + PEAK-FRAME ESCALATION
# TTA + CONFIDENCE-GATED 3-MODEL ENSEMBLE
#
#   1. Xception V2
#   2. EfficientNet-B4 V2
#   3. ViT-Small/16 V2.1
#
# ================================================================
# CHANGES FROM V2.2
# ----------------------------------------------------------------
#  * Video aggregation now weights p90 (severity) not just mean.
#  * Peak-frame escalation: >=20% of frames above PEAK_FAKE_LEVEL
#    forces a FAKE verdict even if mean is low.
#  * EMA smoothing DISABLED (was hiding REAL->FAKE transitions).
#  * Segment reporting: first half vs second half fake score.
#  * Decision thresholds separate for image (0.50) and video (0.40)
#    because frame-level models are under-confident on AI video.
#  * "Suspicious" middle verdict added for 0.35–0.50 range in video.
# ================================================================

import os
import sys
import json
import argparse
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
# IMAGE ENSEMBLE WEIGHTS
# ================================================================

XCEPTION_WEIGHT = 0.45
EFFICIENTNET_WEIGHT = 0.25
VIT_WEIGHT = 0.30


# ================================================================
# XCEPTION PRIORITY (ANCHORED FUSION)
# ================================================================

XCEPTION_ALERT_THRESHOLD = 0.50
XCEPTION_MAX_PULL_DOWN = 0.15
XCEPTION_PULL_STRENGTH = 0.50
XCEPTION_ALERT_FLOOR = 0.52

SUPPORT_CLIP_MIN = 0.05
SUPPORT_CLIP_MAX = 0.95


# ================================================================
# DECISION THRESHOLDS
# ================================================================
# The frame-level models are under-confident on AI-generated video
# (Google Flow / Veo). For VIDEO we use a lower FAKE threshold so
# borderline cases get flagged. IMAGE keeps the standard 0.50.
# ================================================================

IMAGE_FAKE_THRESHOLD = 0.50
VIDEO_FAKE_THRESHOLD = 0.40
VIDEO_SUSPICIOUS_THRESHOLD = 0.32


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

# New: skin-tone check and size gate to reject MTCNN false positives
MIN_FACE_FRACTION = 0.06       # face box must be >= 6% of image short side
MAX_ASPECT_RATIO = 1.65        # reject very wide "faces"
MIN_SKIN_FRACTION = 0.15       # reject boxes that are mostly background

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

# -----------------------------------------------------------------
# Severity-weighted aggregation (V2.3)
# -----------------------------------------------------------------
# Old V2.2:  mean 0.50 + median 0.30 + p75 0.20  <- hides fake segments
# New V2.3:  mean 0.20 + median 0.15 + p75 0.25 + p90 0.40
#
# Rationale: a 10s video with 5s real + 5s fake should be FAKE.
# Mean/median bury the signal. p75/p90 surface it.
# -----------------------------------------------------------------

VIDEO_MEAN_WEIGHT = 0.20
VIDEO_MEDIAN_WEIGHT = 0.15
VIDEO_P75_WEIGHT = 0.25
VIDEO_P90_WEIGHT = 0.40

VIDEO_UPPER_PERCENTILE = 75    # kept for backward-compat JSON field
VIDEO_TOP_PERCENTILE = 90

# -----------------------------------------------------------------
# Peak-frame escalation
# -----------------------------------------------------------------
# If >= PEAK_FRAME_RATIO of frames are >= PEAK_FAKE_LEVEL, escalate
# the video verdict to at least PEAK_ESCALATED_SCORE.
# -----------------------------------------------------------------

PEAK_FAKE_LEVEL = 0.55
PEAK_FRAME_RATIO = 0.20
PEAK_ESCALATED_SCORE = 0.55

# -----------------------------------------------------------------
# Temporal smoothing
# -----------------------------------------------------------------
# EMA disabled. It hides REAL->FAKE transitions. Use a sliding
# median of window 3 instead (outlier removal without lag).
# -----------------------------------------------------------------

USE_EMA_SMOOTHING = False
TEMPORAL_EMA_ALPHA = 0.90          # only used if USE_EMA_SMOOTHING
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
# HEADER
# ================================================================

print()
print("=" * 70)
print("FORENSIQ V2.3")
print("ROBUST FACE-CROP IMAGE + VIDEO INFERENCE")
print("SEVERITY-WEIGHTED VIDEO AGGREGATION + PEAK-FRAME ESCALATION")
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
print(f"EMA smoothing       : {'ON' if USE_EMA_SMOOTHING else 'OFF'}")
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
# SKIN-TONE GATE (rejects background false positives)
# ================================================================

def _skin_fraction(image_array_rgb, box):
    """Return the fraction of skin-colored pixels inside `box`."""
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

    # Broad skin-tone range (works reasonably across ethnicities)
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return float(mask.mean() / 255.0)


# ================================================================
# HELPER: MTCNN DETECT WITH LANDMARKS
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
# HELPER: HAAR FALLBACK DETECT
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
    """
    Reject boxes that are:
      * Too small (likely false positives in background)
      * Too wide (MTCNN sometimes boxes wide background regions)
      * Not enough skin-tone pixels
    """
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
        # If skin check fails, don't reject on it
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

    # ---- Skin-tone + size gate ----------------------------------
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
    # If gate rejected everything, still use the best original box
    # (avoids total detection loss on unusual lighting).

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
# FUSION
# ================================================================

def clip_support(probability):
    return float(min(max(probability, SUPPORT_CLIP_MIN), SUPPORT_CLIP_MAX))


def weighted_fusion(xception_fake, efficientnet_fake, vit_fake):
    return (
        XCEPTION_WEIGHT * xception_fake
        + EFFICIENTNET_WEIGHT * clip_support(efficientnet_fake)
        + VIT_WEIGHT * clip_support(vit_fake)
    )


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

    if xception_alert:
        raw_pull = XCEPTION_PULL_STRENGTH * (support_score - xception_fake)

        if raw_pull < 0:
            pull = max(raw_pull, -XCEPTION_MAX_PULL_DOWN)
        else:
            pull = raw_pull

        score = xception_fake + pull
        score = max(score, XCEPTION_ALERT_FLOOR)
        score = min(score, 1.0)

        if support_score >= 0.50:
            mode = "XCEPTION_ALERT_CONFIRMED"
        elif normal_score >= 0.50:
            mode = "XCEPTION_ALERT_PARTIAL_SUPPORT"
        else:
            mode = "XCEPTION_PRIORITY_OVERRIDE"
    else:
        score = normal_score
        mode = "STANDARD_WEIGHTED_FUSION"

    score = float(np.clip(score, 0.0, 1.0))

    priority_applied = bool(score >= 0.50 and normal_score < 0.50)
    xception_dissent = bool(xception_fake >= 0.50 and score < 0.50)
    review_recommended = bool(priority_applied or xception_dissent)

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
    """Image decision: standard 0.50 threshold."""
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
    """
    Video decision with three bands:
      * >= VIDEO_FAKE_THRESHOLD  -> FAKE
      * >= VIDEO_SUSPICIOUS_THRESHOLD -> SUSPICIOUS (review)
      * else -> REAL
    """
    if fake_score >= VIDEO_FAKE_THRESHOLD:
        prediction = "FAKE"
    elif fake_score >= VIDEO_SUSPICIOUS_THRESHOLD:
        prediction = "SUSPICIOUS"
    else:
        prediction = "REAL"

    if prediction == "FAKE":
        confidence = fake_score
    elif prediction == "SUSPICIOUS":
        # midpoint confidence around the band
        confidence = 0.5 + 0.5 * (
            (fake_score - VIDEO_SUSPICIOUS_THRESHOLD)
            / max(VIDEO_FAKE_THRESHOLD - VIDEO_SUSPICIOUS_THRESHOLD, 1e-6)
        ) - 0.5
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
    print("FORENSIQ V2.3 FUSION")
    print("=" * 70)
    print()
    print(f"Xception fake score    : {xception['fake']:.4f}")
    print(f"EfficientNet fake score: {efficientnet['fake']:.4f}")
    print(f"ViT fake score         : {vit['fake']:.4f}")
    print(f"Normal weighted score  : {fusion['normal_weighted_score']:.4f}")
    print(f"Fusion mode            : {fusion['fusion_mode']}")
    print(f"Disagreement level     : {fusion['disagreement_level']}")
    print()
    print(f"FINAL FAKE SCORE       : {final['fake_score']:.4f}")
    print(f"FINAL REAL SCORE       : {final['real_score']:.4f}")
    print()
    print(f"FINAL PREDICTION       : {final['prediction']}")
    print(f"CONFIDENCE             : {final['confidence'] * 100:.2f}%")

    if fusion["review_recommended"]:
        print()
        print("⚠ REVIEW RECOMMENDED: models disagree.")

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
# VIDEO AGGREGATION  (V2.3 — SEVERITY-WEIGHTED + PEAK ESCALATION)
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
        }

    fake_scores = np.array(
        [result["final"]["fake_score"] for result in frame_results],
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

    # ---- Peak-frame escalation -----------------------------------
    peak_mask = fake_scores >= PEAK_FAKE_LEVEL
    peak_frame_ratio = float(peak_mask.mean())
    peak_escalated = peak_frame_ratio >= PEAK_FRAME_RATIO

    if peak_escalated:
        video_fake_score = max(base_score, PEAK_ESCALATED_SCORE)
    else:
        video_fake_score = base_score

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
    }


# ================================================================
# SEGMENT ANALYSIS (first half vs second half)
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
    print("FORENSIQ V2.3 VIDEO ANALYSIS")
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

    # Sliding window for median smoothing (replaces EMA)
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
                debug_name=f"video_{video_path.stem}_frame_{next_target}",
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

            # ---- Smoothing --------------------------------------
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
            result["final"]["real_score"] = float(1.0 - smoothed)
            result["final"]["prediction"] = (
                "FAKE" if smoothed >= VIDEO_FAKE_THRESHOLD
                else ("SUSPICIOUS" if smoothed >= VIDEO_SUSPICIOUS_THRESHOLD else "REAL")
            )

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

    priority_frames = sum(
        1 for r in frame_results
        if r["fusion"]["xception_priority_applied"]
    )

    result = {
        "version": "ForensIQ-V2.3",
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
            "xception_priority_frames": int(priority_frames),
        },
        "final": {
            "prediction": video_decision["prediction"],
            "confidence": video_decision["confidence"],
            "fake_score": video_decision["fake_score"],
            "real_score": video_decision["real_score"],
        },
        "frames": frame_results,
        "warnings": [
            "Video inference uses frame-level image classifiers; it is not a temporal video model.",
            "Video fake score uses severity-weighted aggregation (mean+median+p75+p90).",
            "Peak-frame escalation triggers FAKE if >=20% of frames exceed 0.55.",
            "Models may be under-confident on Google Flow / Veo generated video.",
            "Confidence is not a calibrated probability.",
            "External video-dataset validation is required.",
            "Frame sampling may miss very short temporal manipulations.",
        ],
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
    print("FORENSIQ V2.3 VIDEO RESULT")
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
    print(f"Xception-priority frames: {stats['xception_priority_frames']}")

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
    print()
    print(f"FINAL VIDEO FAKE SCORE: {agg['video_fake_score']:.4f}")
    print(f"FINAL VIDEO REAL SCORE: {agg['video_real_score']:.4f}")
    print()
    print(f"VIDEO PREDICTION      : {final['prediction']}")
    print(f"VIDEO CONFIDENCE      : {final['confidence'] * 100:.2f}%")
    print()
    print("=" * 70)


# ================================================================
# IMAGE ANALYSIS (entry)
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
        "version": "ForensIQ-V2.3",
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
# ARGUMENTS
# ================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ForensIQ V2.3 image + video inference"
    )
    parser.add_argument("input", type=str, help="Path to input image or video")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--debug", action="store_true",
                        help="Save face crops to debug_crops/")
    return parser.parse_args()


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

    args = parse_arguments()

    if args.debug:
        DEBUG_SAVE_CROPS = True
        print(f"Debug mode ON — crops will be saved to: {DEBUG_DIR}")

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"\nInput file not found:\n{input_path}\n")

    input_type = determine_input_type(input_path)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PREDICTION_DIR / (input_path.stem + "_v2_prediction.json")

    if input_type == "image":
        run_image_analysis(input_path, output_path, debug=args.debug)
        print()
        print("=" * 70)
        print("✓ IMAGE V2.3 INFERENCE FINISHED")
        print("=" * 70)
        return

    if input_type == "video":
        video_result = analyze_video(input_path)
        print_video_result(video_result)
        save_result(video_result, output_path)
        print()
        print("=" * 70)
        print("✓ VIDEO V2.3 INFERENCE FINISHED")
        print("=" * 70)
        return


if __name__ == "__main__":
    main()