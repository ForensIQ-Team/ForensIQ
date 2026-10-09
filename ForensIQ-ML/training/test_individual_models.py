# ================================================================
# ForensIQ - Individual Model External Image Tester V2.1
#
# PURPOSE
#   Test ONE external image through the three trained models
#   independently, then compare their predictions with the
#   validated 3-model fusion.
#
# MODELS
#   1. Xception V2
#   2. EfficientNet-B4 V2
#   3. ViT-Small V2.1
#
# CLASS MAPPING
#   0 = FAKE
#   1 = REAL
#
# PREPROCESSING
#   Xception       : 299 x 299
#   EfficientNet   : 299 x 299
#   ViT            : 224 x 224
#   Normalization  : mean=[0.5]*3, std=[0.5]*3
#
# INFERENCE PIPELINE
#   External image
#       -> MTCNN face detection
#       -> strongest face selection
#       -> margin-expanded face crop
#       -> each model independently
#       -> original + horizontal-flip TTA
#       -> 60/40 TTA score
#       -> 0.60 / 0.20 / 0.20 3-model fusion
#
# IMPORTANT
#   This script DOES NOT modify or retrain any checkpoint.
#   It is an external/generalization diagnostic script.
#
# USAGE
#   python test_individual_models.py "C:\\path\\to\\image.jpg"
#
# Optional:
#   python test_individual_models.py "image1.jpg" "image2.jpg"
# ================================================================

import sys
import json
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms

import timm

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "v2"
RESULTS_DIR = ROOT / "prediction_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

XCEPTION_CHECKPOINT = CHECKPOINT_DIR / "best_xception_v2.pth"
EFFICIENTNET_CHECKPOINT = CHECKPOINT_DIR / "best_efficientnet_b4_v2.pth"
VIT_CHECKPOINT = CHECKPOINT_DIR / "best_vit_v2_1.pth"


# ================================================================
# MODEL CONFIG
# ================================================================

XCEPTION_NAME = "xception"
EFFICIENTNET_NAME = "tf_efficientnet_b4"
VIT_NAME = "vit_small_patch16_224"

NUM_CLASSES = 2
FAKE = 0
REAL = 1

XCEPTION_WEIGHT = 0.60
EFFICIENTNET_WEIGHT = 0.20
VIT_WEIGHT = 0.20

TTA_ORIGINAL_WEIGHT = 0.60
TTA_FLIPPED_WEIGHT = 0.40

IMAGE_SIZE_CNN = 299
IMAGE_SIZE_VIT = 224

FACE_MARGIN = 0.30
MIN_FACE_SIZE = 40

MTCNN_THRESHOLDS = [0.6, 0.7, 0.7]
MTCNN_FACTOR = 0.709

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# REPRODUCIBILITY / PERFORMANCE
# ================================================================

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ================================================================
# HEADER
# ================================================================

def print_header():
    print()
    print("=" * 78)
    print("ForensIQ - INDIVIDUAL MODEL EXTERNAL IMAGE TEST V2.1")
    print("=" * 78)
    print(f"Device : {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print()
    print("Models:")
    print("  Xception V2        : 299x299")
    print("  EfficientNet-B4 V2 : 299x299")
    print("  ViT V2.1           : 224x224")
    print()
    print("Class mapping: 0 = FAKE, 1 = REAL")
    print("TTA: original 60% + horizontal flip 40%")
    print("Fusion: Xception 60% + EfficientNet 20% + ViT 20%")
    print("=" * 78)
    print()


# ================================================================
# CHECKPOINT VALIDATION
# ================================================================

def check_checkpoints():
    print("Checking checkpoints...")

    checkpoints = [
        ("Xception V2", XCEPTION_CHECKPOINT),
        ("EfficientNet-B4 V2", EFFICIENTNET_CHECKPOINT),
        ("ViT V2.1", VIT_CHECKPOINT),
    ]

    for name, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(
                f"\n{name} checkpoint not found:\n{path}\n"
            )
        print(f"  ✓ {path.name}")

    print()


# ================================================================
# TRANSFORMS
# ================================================================

cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_CNN, IMAGE_SIZE_CNN)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])

vit_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE_VIT, IMAGE_SIZE_VIT)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ================================================================
# MODEL CREATION
# ================================================================

def create_models():
    print("Loading models...")

    xception = timm.create_model(
        XCEPTION_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    efficientnet = timm.create_model(
        EFFICIENTNET_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    vit = timm.create_model(
        VIT_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    xception = load_checkpoint(xception, XCEPTION_CHECKPOINT)
    efficientnet = load_checkpoint(efficientnet, EFFICIENTNET_CHECKPOINT)
    vit = load_checkpoint(vit, VIT_CHECKPOINT)

    xception.eval()
    efficientnet.eval()
    vit.eval()

    print()
    return xception, efficientnet, vit


# ================================================================
# CHECKPOINT LOADER
# ================================================================

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}

    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
    )

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

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(f"  ✓ Loaded {checkpoint_path.name}")
    print(f"    Missing keys    : {len(missing)}")
    print(f"    Unexpected keys : {len(unexpected)}")

    if missing:
        print(f"    WARNING missing: {missing[:5]}")

    if unexpected:
        print(f"    WARNING unexpected: {unexpected[:5]}")

    return model


# ================================================================
# AMP COMPATIBILITY
# ================================================================

def autocast_context():
    """Return an AMP context compatible with different PyTorch versions."""
    if DEVICE.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)

    return torch.cuda.amp.autocast(enabled=True)


# ================================================================
# FACE DETECTOR
# ================================================================

def create_face_detector():
    if MTCNN is None:
        raise ImportError(
            "facenet-pytorch is not installed. "
            "Install it with: pip install facenet-pytorch"
        )

    print("Initializing MTCNN face detector...")

    detector = MTCNN(
        keep_all=True,
        device=DEVICE,
        post_process=False,
        min_face_size=MIN_FACE_SIZE,
        thresholds=MTCNN_THRESHOLDS,
        factor=MTCNN_FACTOR,
    )

    print("  ✓ MTCNN ready")
    print()
    return detector


# ================================================================
# FACE DETECTION
# ================================================================

def detect_faces(image, detector):
    boxes, probabilities = detector.detect(image)

    if boxes is None:
        return []

    faces = []

    for index, box in enumerate(boxes):
        if box is None:
            continue

        probability = 0.0
        if probabilities is not None:
            probability = float(probabilities[index])

        x1, y1, x2, y2 = [float(v) for v in box]

        width = x2 - x1
        height = y2 - y1

        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue

        faces.append({
            "box": (x1, y1, x2, y2),
            "confidence": probability,
            "area": width * height,
        })

    return faces


# ================================================================
# BEST FACE SELECTION
# ================================================================

def select_best_face(faces):
    if not faces:
        return None

    def score(face):
        return face["area"] * (0.5 + 0.5 * face["confidence"])

    return max(faces, key=score)


# ================================================================
# FACE BOX EXPANSION
# ================================================================

def expand_face_box(box, image_width, image_height):
    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    horizontal_margin = width * FACE_MARGIN
    vertical_margin = height * FACE_MARGIN * 1.10

    new_x1 = max(0, int(round(x1 - horizontal_margin)))
    new_y1 = max(0, int(round(y1 - vertical_margin)))
    new_x2 = min(image_width, int(round(x2 + horizontal_margin)))
    new_y2 = min(image_height, int(round(y2 + vertical_margin)))

    return new_x1, new_y1, new_x2, new_y2


# ================================================================
# CREATE FACE CROP
# ================================================================

def create_face_crop(image, detector):
    width, height = image.size
    faces = detect_faces(image, detector)

    if not faces:
        return {
            "crop": image.copy(),
            "face_detected": False,
            "box": None,
            "raw_box": None,
            "confidence": 0.0,
            "num_faces": 0,
            "fallback": True,
        }

    best = select_best_face(faces)

    crop_box = expand_face_box(
        best["box"],
        width,
        height,
    )

    crop = image.crop(crop_box)

    return {
        "crop": crop,
        "face_detected": True,
        "box": crop_box,
        "raw_box": best["box"],
        "confidence": float(best["confidence"]),
        "num_faces": len(faces),
        "fallback": False,
    }


# ================================================================
# SINGLE MODEL PREDICTION
# ================================================================

def predict_model(model, image, transform):
    tensor = transform(image).unsqueeze(0).to(
        DEVICE,
        non_blocking=True,
    )

    with torch.inference_mode():
        with autocast_context():
            logits = model(tensor)

        probabilities = F.softmax(logits, dim=1)[0]

    fake = float(probabilities[FAKE].cpu().item())
    real = float(probabilities[REAL].cpu().item())

    return {
        "fake": fake,
        "real": real,
        "label": "FAKE" if fake >= 0.50 else "REAL",
    }


# ================================================================
# TTA PREDICTION
# ================================================================

def predict_with_tta(model, image, transform):
    original = predict_model(
        model,
        image,
        transform,
    )

    flipped_image = ImageOps.mirror(image)

    flipped = predict_model(
        model,
        flipped_image,
        transform,
    )

    fake = (
        TTA_ORIGINAL_WEIGHT * original["fake"]
        + TTA_FLIPPED_WEIGHT * flipped["fake"]
    )

    real = (
        TTA_ORIGINAL_WEIGHT * original["real"]
        + TTA_FLIPPED_WEIGHT * flipped["real"]
    )

    return {
        "original": original,
        "flipped": flipped,
        "fake": float(fake),
        "real": float(real),
        "label": "FAKE" if fake >= 0.50 else "REAL",
    }


# ================================================================
# FUSION
# ================================================================

def fuse_predictions(xception_fake, efficientnet_fake, vit_fake):
    fake = (
        XCEPTION_WEIGHT * xception_fake
        + EFFICIENTNET_WEIGHT * efficientnet_fake
        + VIT_WEIGHT * vit_fake
    )

    fake = float(fake)
    real = float(1.0 - fake)

    return {
        "fake": fake,
        "real": real,
        "label": "FAKE" if fake >= 0.50 else "REAL",
        "weights": {
            "xception": XCEPTION_WEIGHT,
            "efficientnet": EFFICIENTNET_WEIGHT,
            "vit": VIT_WEIGHT,
        },
    }


# ================================================================
# DISPLAY HELPERS
# ================================================================

def pct(value):
    return f"{value * 100:7.2f}%"


def print_model_result(name, result):
    print("-" * 78)
    print(name)
    print("-" * 78)
    print(f"  FAKE probability : {pct(result['fake'])}")
    print(f"  REAL probability : {pct(result['real'])}")
    print(f"  Decision         : {result['label']}")
    print()
    print("  TTA details:")
    print(f"    Original FAKE  : {pct(result['original']['fake'])}")
    print(f"    Flipped FAKE   : {pct(result['flipped']['fake'])}")
    print(f"    TTA FAKE       : {pct(result['fake'])}")


def print_comparison(xception, efficientnet, vit, fusion):
    print()
    print("=" * 78)
    print("INDIVIDUAL MODEL COMPARISON")
    print("=" * 78)

    print()
    print(f"{'Model':<24}{'FAKE':>12}{'REAL':>12}{'Decision':>15}")
    print("-" * 63)

    rows = [
        ("Xception V2", xception),
        ("EfficientNet-B4 V2", efficientnet),
        ("ViT V2.1", vit),
        ("3-Model Fusion", fusion),
    ]

    for name, result in rows:
        print(
            f"{name:<24}"
            f"{result['fake'] * 100:>11.2f}%"
            f"{result['real'] * 100:>11.2f}%"
            f"{result['label']:>15}"
        )

    print()
    print("Fusion weights:")
    print("  Xception     = 0.60")
    print("  EfficientNet = 0.20")
    print("  ViT          = 0.20")

    print()
    print("Model disagreement:")
    fake_scores = {
        "Xception": xception["fake"],
        "EfficientNet": efficientnet["fake"],
        "ViT": vit["fake"],
    }

    highest = max(fake_scores.items(), key=lambda x: x[1])
    lowest = min(fake_scores.items(), key=lambda x: x[1])
    spread = highest[1] - lowest[1]

    print(f"  Highest FAKE score : {highest[0]} ({highest[1] * 100:.2f}%)")
    print(f"  Lowest FAKE score  : {lowest[0]} ({lowest[1] * 100:.2f}%)")
    print(f"  Score spread       : {spread * 100:.2f} percentage points")

    if spread >= 0.50:
        level = "HIGH"
    elif spread >= 0.25:
        level = "MEDIUM"
    else:
        level = "LOW"

    print(f"  Disagreement level : {level}")


# ================================================================
# JSON SERIALIZATION
# ================================================================

def clean_for_json(value):
    if isinstance(value, dict):
        return {k: clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


# ================================================================
# TEST ONE IMAGE
# ================================================================

def test_image(
    image_path,
    detector,
    xception,
    efficientnet,
    vit,
):
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print()
    print("=" * 78)
    print(f"TEST IMAGE: {image_path.name}")
    print("=" * 78)

    image = Image.open(image_path).convert("RGB")

    print(f"Original size : {image.width} x {image.height}")

    face_info = create_face_crop(image, detector)
    face_crop = face_info["crop"]

    print(f"Faces detected: {face_info['num_faces']}")
    print(f"Face confidence: {face_info['confidence']:.4f}")
    print(f"Face fallback  : {face_info['fallback']}")

    if face_info["box"] is not None:
        print(f"Crop box       : {face_info['box']}")

    print()
    print("Running Xception V2...")
    xception_result = predict_with_tta(
        xception,
        face_crop,
        cnn_transform,
    )

    print("Running EfficientNet-B4 V2...")
    efficientnet_result = predict_with_tta(
        efficientnet,
        face_crop,
        cnn_transform,
    )

    print("Running ViT V2.1...")
    vit_result = predict_with_tta(
        vit,
        face_crop,
        vit_transform,
    )

    fusion_result = fuse_predictions(
        xception_result["fake"],
        efficientnet_result["fake"],
        vit_result["fake"],
    )

    print()
    print_model_result("XCEPTION V2", xception_result)
    print_model_result("EFFICIENTNET-B4 V2", efficientnet_result)
    print_model_result("VIT V2.1", vit_result)

    print()
    print("=" * 78)
    print("3-MODEL FUSION")
    print("=" * 78)
    print(f"  Xception contribution     : {xception_result['fake'] * XCEPTION_WEIGHT * 100:.2f} percentage points")
    print(f"  EfficientNet contribution : {efficientnet_result['fake'] * EFFICIENTNET_WEIGHT * 100:.2f} percentage points")
    print(f"  ViT contribution          : {vit_result['fake'] * VIT_WEIGHT * 100:.2f} percentage points")
    print()
    print(f"  FINAL FAKE probability : {fusion_result['fake'] * 100:.2f}%")
    print(f"  FINAL REAL probability : {fusion_result['real'] * 100:.2f}%")
    print(f"  FINAL DECISION         : {fusion_result['label']}")

    print_comparison(
        xception_result,
        efficientnet_result,
        vit_result,
        fusion_result,
    )

    result = {
        "image": str(image_path),
        "image_name": image_path.name,
        "image_size": [image.width, image.height],
        "face_detection": {
            "face_detected": face_info["face_detected"],
            "num_faces": face_info["num_faces"],
            "confidence": face_info["confidence"],
            "box": face_info["box"],
            "raw_box": face_info["raw_box"],
            "fallback": face_info["fallback"],
        },
        "models": {
            "xception_v2": xception_result,
            "efficientnet_b4_v2": efficientnet_result,
            "vit_v2_1": vit_result,
        },
        "ensemble": fusion_result,
        "configuration": {
            "class_mapping": {
                "0": "FAKE",
                "1": "REAL",
            },
            "cnn_input_size": IMAGE_SIZE_CNN,
            "vit_input_size": IMAGE_SIZE_VIT,
            "face_margin": FACE_MARGIN,
            "tta_original_weight": TTA_ORIGINAL_WEIGHT,
            "tta_flipped_weight": TTA_FLIPPED_WEIGHT,
            "xception_weight": XCEPTION_WEIGHT,
            "efficientnet_weight": EFFICIENTNET_WEIGHT,
            "vit_weight": VIT_WEIGHT,
            "decision_threshold": 0.50,
        },
    }

    output_path = RESULTS_DIR / f"individual_test_{image_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            clean_for_json(result),
            f,
            indent=2,
        )

    print()
    print(f"Saved detailed JSON: {output_path}")
    print("=" * 78)
    print()

    return result


# ================================================================
# ARGUMENT PARSER
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test external images independently with "
            "Xception V2, EfficientNet-B4 V2, ViT V2.1 "
            "and the 3-model fusion."
        )
    )

    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths",
    )

    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():
    print_header()
    check_checkpoints()

    xception, efficientnet, vit = create_models()
    detector = create_face_detector()

    results = []

    for image_path in sys.argv[1:]:
        try:
            results.append(
                test_image(
                    image_path,
                    detector,
                    xception,
                    efficientnet,
                    vit,
                )
            )

        except Exception as exc:
            print()
            print("! ERROR")
            print(f"  Image: {image_path}")
            print(f"  {type(exc).__name__}: {exc}")
            print()

    if not results:
        print("No image was successfully processed.")
        return 1

    # ------------------------------------------------------------
    # Final multi-image summary
    # ------------------------------------------------------------
    if len(results) > 1:
        print()
        print("=" * 78)
        print("MULTI-IMAGE SUMMARY")
        print("=" * 78)
        print()
        print(
            f"{'Image':<30}"
            f"{'Xception':>12}"
            f"{'EffNet':>12}"
            f"{'ViT':>12}"
            f"{'Fusion':>12}"
        )
        print("-" * 78)

        for result in results:
            name = result["image_name"][:29]
            print(
                f"{name:<30}"
                f"{result['models']['xception_v2']['fake'] * 100:>11.2f}%"
                f"{result['models']['efficientnet_b4_v2']['fake'] * 100:>11.2f}%"
                f"{result['models']['vit_v2_1']['fake'] * 100:>11.2f}%"
                f"{result['ensemble']['fake'] * 100:>11.2f}%"
            )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
