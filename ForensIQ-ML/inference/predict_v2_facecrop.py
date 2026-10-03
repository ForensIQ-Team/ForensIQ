# ================================================================
# FORENSIQ V2
# ROBUST FACE-CROP INFERENCE + TTA + CONFIDENCE-GATED ENSEMBLE
#
# Purpose:
#   Solve the full-image vs face-crop distribution mismatch.
#
# V2 TRAINING:
#   - Xception V2
#   - EfficientNet-B4 V2
#   - Synthetic AI-generated faces
#   - Real synthetic faces
#   - FF++ replay
#
# INFERENCE:
#   Full image
#       |
#       v
#   Face detection
#       |
#       v
#   Largest / strongest face
#       |
#       v
#   Margin-expanded face crop
#       |
#       v
#   299 x 299
#       |
#       +------------------+
#       |                  |
#       v                  v
#   Xception V2       EfficientNet V2
#       |                  |
#       +--------+---------+
#                |
#                v
#          TTA score fusion
#                |
#                v
#       Confidence-gated fusion
#                |
#                v
#          FINAL PREDICTION
#
# CLASS MAPPING:
#   0 = FAKE
#   1 = REAL
#
# IMPORTANT:
#   This script does NOT retrain the models.
#   It only improves inference preprocessing and ensemble logic.
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

from PIL import Image, ImageOps

import timm

from sklearn.metrics import confusion_matrix


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

V2_CHECKPOINT_DIR = (
    CHECKPOINT_DIR / "v2"
)

PREDICTION_DIR = (
    ROOT / "prediction_results"
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# V2 CHECKPOINTS
# ================================================================

XCEPTION_V2_CHECKPOINT = (
    V2_CHECKPOINT_DIR /
    "best_xception_v2.pth"
)

EFFICIENTNET_V2_CHECKPOINT = (
    V2_CHECKPOINT_DIR /
    "best_efficientnet_b4_v2.pth"
)


# ================================================================
# CONFIGURATION
# ================================================================

IMAGE_SIZE = 299

NUM_CLASSES = 2

FAKE = 0

REAL = 1


# ------------------------------------------------
# Ensemble weights
# ------------------------------------------------
#
# Xception is currently performing better on your
# external tests, so we intentionally give it more
# influence.
#
# This is NOT max() and NOT a blind override.
# ------------------------------------------------

XCEPTION_WEIGHT = 0.70

EFFICIENTNET_WEIGHT = 0.30


# ------------------------------------------------
# Confidence gates
# ------------------------------------------------
#
# Strong Xception fake:
#       >= 0.70
#
# Strong EfficientNet real:
#       <= 0.30
#
# If one model is very confident and the other is
# strongly opposite, use a gated score rather than
# allowing the weaker model to erase the stronger
# signal completely.
# ------------------------------------------------

STRONG_FAKE_THRESHOLD = 0.70

STRONG_REAL_THRESHOLD = 0.30

EXTREME_FAKE_THRESHOLD = 0.85

EXTREME_REAL_THRESHOLD = 0.15


# ------------------------------------------------
# Disagreement
# ------------------------------------------------

HIGH_DISAGREEMENT = 0.50

MEDIUM_DISAGREEMENT = 0.25


# ------------------------------------------------
# Face crop
# ------------------------------------------------
#
# Extra context around the detected face.
#
# 0.30 means approximately 30% margin relative
# to the detected face dimensions.
# ------------------------------------------------

FACE_MARGIN = 0.30


# ------------------------------------------------
# Minimum acceptable face size
# ------------------------------------------------

MIN_FACE_SIZE = 40


# ------------------------------------------------
# TTA
# ------------------------------------------------

USE_TTA = True


# ------------------------------------------------
# GPU
# ------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# HEADER
# ================================================================

print()

print("=" * 70)

print(
    "FORENSIQ V2"
)

print(
    "ROBUST FACE-CROP INFERENCE"
)

print(
    "FACE DETECTION + TTA + CONFIDENCE-GATED ENSEMBLE"
)

print("=" * 70)

print()

print(
    f"Device : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()

print(
    "Class mapping:"
)

print(
    "0 = FAKE"
)

print(
    "1 = REAL"
)

print()

print(
    f"Xception weight     : "
    f"{XCEPTION_WEIGHT:.2f}"
)

print(
    f"EfficientNet weight : "
    f"{EFFICIENTNET_WEIGHT:.2f}"
)

print(
    f"Face margin         : "
    f"{FACE_MARGIN:.2f}"
)

print()


# ================================================================
# VALIDATION
# ================================================================

def require_file(
    path,
    name
):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}\n"
        )

    print(
        f"✓ {name}"
    )


require_file(
    XCEPTION_V2_CHECKPOINT,
    "Xception V2 checkpoint"
)

require_file(
    EFFICIENTNET_V2_CHECKPOINT,
    "EfficientNet-B4 V2 checkpoint"
)


if not MTCNN_AVAILABLE:

    raise ImportError(
        "\nfacenet-pytorch is required.\n\n"
        "Install it with:\n"
        "pip install facenet-pytorch\n"
    )


# ================================================================
# IMAGE PREPROCESSING
# ================================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.5,
            0.5,
            0.5
        ],

        std=[
            0.5,
            0.5,
            0.5
        ]

    )

])


# ================================================================
# MODEL CREATION
# ================================================================

def create_xception():

    model = timm.create_model(

        "xception",

        pretrained=False,

        num_classes=NUM_CLASSES

    )

    return model


def create_efficientnet():

    model = timm.create_model(

        "tf_efficientnet_b4",

        pretrained=False,

        num_classes=NUM_CLASSES

    )

    return model


# ================================================================
# CHECKPOINT EXTRACTION
# ================================================================

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            return checkpoint[
                "model_state_dict"
            ]

        if (
            "state_dict"
            in checkpoint
        ):

            return checkpoint[
                "state_dict"
            ]

        if (
            "model"
            in checkpoint
            and
            isinstance(
                checkpoint["model"],
                dict
            )
        ):

            return checkpoint[
                "model"
            ]

        return checkpoint

    return checkpoint


# ================================================================
# CLEAN STATE DICT
# ================================================================

def clean_state_dict(
    state_dict
):

    cleaned = {}

    prefixes = [

        "module.",

        "_orig_mod.",

        "model.",

    ]

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(
                    prefix
                ):

                    new_key = (
                        new_key[
                            len(prefix):
                        ]
                    )

                    changed = True

        cleaned[
            new_key
        ] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(
    model,
    checkpoint_path,
    model_name
):

    print()

    print(
        f"Loading {model_name} "
        f"V2 checkpoint..."
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location="cpu"

    )

    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )

    state_dict = (
        clean_state_dict(
            state_dict
        )
    )

    result = model.load_state_dict(

        state_dict,

        strict=False

    )

    if result.missing_keys:

        print()

        print(
            f"WARNING: "
            f"{len(result.missing_keys)} "
            f"missing keys."
        )

        for key in result.missing_keys[:10]:

            print(
                f"  Missing: {key}"
            )

    if result.unexpected_keys:

        print()

        print(
            f"WARNING: "
            f"{len(result.unexpected_keys)} "
            f"unexpected keys."
        )

        for key in result.unexpected_keys[:10]:

            print(
                f"  Unexpected: {key}"
            )

    if (
        not result.missing_keys
        and
        not result.unexpected_keys
    ):

        print(
            f"✓ {model_name} checkpoint "
            f"loaded perfectly"
        )

    elif (
        len(result.missing_keys) > 20
        or
        len(result.unexpected_keys) > 20
    ):

        raise RuntimeError(
            f"\nCheckpoint compatibility "
            f"problem for {model_name}."
        )

    else:

        print(
            f"✓ {model_name} loaded "
            f"with minor differences"
        )

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


# ================================================================
# LOAD BOTH MODELS
# ================================================================

print(
    "=" * 70
)

print(
    "LOADING BOTH FORENSIQ V2 MODELS"
)

print(
    "=" * 70
)

xception_model = load_model(

    create_xception(),

    XCEPTION_V2_CHECKPOINT,

    "Xception"

)


efficientnet_model = load_model(

    create_efficientnet(),

    EFFICIENTNET_V2_CHECKPOINT,

    "EfficientNet-B4"

)


# ================================================================
# FACE DETECTOR
# ================================================================

print()

print(
    "=" * 70
)

print(
    "INITIALIZING FACE DETECTOR"
)

print(
    "=" * 70
)

face_detector = MTCNN(

    keep_all=True,

    device=DEVICE,

    post_process=False,

    min_face_size=MIN_FACE_SIZE,

    thresholds=[
        0.6,
        0.7,
        0.7
    ],

    factor=0.709

)

print(
    "✓ MTCNN face detector ready"
)

print()


# ================================================================
# FACE DETECTION
# ================================================================

def detect_faces(
    image
):

    boxes, probabilities = (
        face_detector.detect(
            image
        )
    )

    if boxes is None:

        return []

    faces = []

    for index, box in enumerate(
        boxes
    ):

        if box is None:

            continue

        probability = 0.0

        if probabilities is not None:

            probability = float(
                probabilities[index]
            )

        x1, y1, x2, y2 = [
            float(value)
            for value in box
        ]

        width = x2 - x1

        height = y2 - y1

        if (
            width < MIN_FACE_SIZE
            or
            height < MIN_FACE_SIZE
        ):

            continue

        faces.append({

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "confidence":
                probability,

            "area":
                width * height

        })

    return faces


# ================================================================
# SELECT BEST FACE
# ================================================================

def select_best_face(
    faces
):

    if not faces:

        return None

    # ------------------------------------------------------------
    # Prefer a large, confident face.
    #
    # This is better than simply selecting the largest face
    # because a huge low-confidence detection should not always
    # win.
    # ------------------------------------------------------------

    def score(face):

        confidence = (
            face["confidence"]
        )

        area = (
            face["area"]
        )

        return (
            area *
            (0.5 + 0.5 * confidence)
        )

    return max(
        faces,
        key=score
    )


# ================================================================
# EXPAND FACE BOX
# ================================================================

def expand_face_box(
    box,
    image_width,
    image_height,
    margin=FACE_MARGIN
):

    x1, y1, x2, y2 = box

    width = x2 - x1

    height = y2 - y1

    # Use slightly more vertical context.
    #
    # This helps retain forehead/chin and some surrounding
    # facial structure.
    horizontal_margin = (
        width * margin
    )

    vertical_margin = (
        height * margin * 1.10
    )

    new_x1 = (
        x1 -
        horizontal_margin
    )

    new_y1 = (
        y1 -
        vertical_margin
    )

    new_x2 = (
        x2 +
        horizontal_margin
    )

    new_y2 = (
        y2 +
        vertical_margin
    )

    new_x1 = max(
        0,
        int(round(new_x1))
    )

    new_y1 = max(
        0,
        int(round(new_y1))
    )

    new_x2 = min(
        image_width,
        int(round(new_x2))
    )

    new_y2 = min(
        image_height,
        int(round(new_y2))
    )

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


# ================================================================
# CROP FACE
# ================================================================

def create_face_crop(
    image
):

    image_width, image_height = (
        image.size
    )

    faces = detect_faces(
        image
    )

    if not faces:

        return {

            "crop":
                image.copy(),

            "face_detected":
                False,

            "box":
                None,

            "confidence":
                0.0,

            "num_faces":
                0,

            "fallback":
                True

        }

    best_face = select_best_face(
        faces
    )

    crop_box = expand_face_box(

        best_face["box"],

        image_width,

        image_height,

        FACE_MARGIN

    )

    crop = image.crop(
        crop_box
    )

    return {

        "crop":
            crop,

        "face_detected":
            True,

        "box":
            crop_box,

        "raw_box":
            best_face["box"],

        "confidence":
            float(
                best_face["confidence"]
            ),

        "num_faces":
            len(faces),

        "fallback":
            False

    }


# ================================================================
# MODEL PREDICTION
# ================================================================

@torch.no_grad()
def predict_model(
    model,
    image
):

    tensor = transform(
        image
    ).unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE,
        non_blocking=True
    )

    with torch.cuda.amp.autocast(

        enabled=(
            DEVICE.type == "cuda"
        )

    ):

        logits = model(
            tensor
        )

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    fake_probability = float(
        probabilities[FAKE]
        .detach()
        .cpu()
        .item()
    )

    real_probability = float(
        probabilities[REAL]
        .detach()
        .cpu()
        .item()
    )

    return {

        "fake":
            fake_probability,

        "real":
            real_probability

    }


# ================================================================
# TTA PREDICTION
# ================================================================

def predict_with_tta(
    model,
    image
):

    original = predict_model(

        model,

        image

    )

    if not USE_TTA:

        return {

            "original":
                original,

            "flipped":
                None,

            "fake":
                original["fake"],

            "real":
                original["real"]

        }

    flipped_image = ImageOps.mirror(
        image
    )

    flipped = predict_model(

        model,

        flipped_image

    )

    fake = (
        original["fake"] * 0.60
        +
        flipped["fake"] * 0.40
    )

    real = (
        original["real"] * 0.60
        +
        flipped["real"] * 0.40
    )

    return {

        "original":
            original,

        "flipped":
            flipped,

        "fake":
            float(fake),

        "real":
            float(real)

    }


# ================================================================
# NORMAL WEIGHTED FUSION
# ================================================================

def weighted_fusion(
    xception_fake,
    efficientnet_fake
):

    return (

        XCEPTION_WEIGHT *
        xception_fake

        +

        EFFICIENTNET_WEIGHT *
        efficientnet_fake

    )


# ================================================================
# CONFIDENCE-GATED FUSION
# ================================================================
#
# This is the important part.
#
# We DON'T simply do:
#
#     max(xception, efficientnet)
#
# because that would be too aggressive.
#
# Instead:
#
# Normal case:
#       70% Xception
#       30% EfficientNet
#
# Strong Xception fake:
#       X >= 0.70
#       EfficientNet <= 0.30
#
# We allow Xception to dominate but retain a small
# contribution from EfficientNet.
#
# Extreme Xception:
#       X >= 0.85
#       Efficient <= 0.15
#
# We use a stronger Xception gate.
# ================================================================

def confidence_gated_fusion(
    xception_fake,
    efficientnet_fake
):

    normal_score = weighted_fusion(

        xception_fake,

        efficientnet_fake

    )

    difference = abs(

        xception_fake
        -
        efficientnet_fake

    )

    # ------------------------------------------------------------
    # CASE 1:
    # Xception strongly says FAKE.
    # ------------------------------------------------------------

    if (

        xception_fake
        >=
        STRONG_FAKE_THRESHOLD

        and

        efficientnet_fake
        <=
        STRONG_REAL_THRESHOLD

    ):

        # Strong Xception signal.
        #
        # Do NOT completely ignore EfficientNet.
        #
        # Give Xception 85% influence here.
        score = (

            0.85 *
            xception_fake

            +

            0.15 *
            efficientnet_fake

        )

        return {

            "score":
                float(score),

            "mode":
                "XCEPTION_STRONG_FAKE_GATE",

            "normal_score":
                float(normal_score),

            "difference":
                float(difference)

        }

    # ------------------------------------------------------------
    # CASE 2:
    # EfficientNet strongly says FAKE.
    #
    # We don't want Xception to completely suppress it.
    # ------------------------------------------------------------

    if (

        efficientnet_fake
        >=
        EXTREME_FAKE_THRESHOLD

        and

        xception_fake
        <=
        STRONG_REAL_THRESHOLD

    ):

        score = (

            0.60 *
            efficientnet_fake

            +

            0.40 *
            xception_fake

        )

        return {

            "score":
                float(score),

            "mode":
                "EFFICIENTNET_STRONG_FAKE_GATE",

            "normal_score":
                float(normal_score),

            "difference":
                float(difference)

        }

    # ------------------------------------------------------------
    # CASE 3:
    # Xception strongly says REAL.
    # ------------------------------------------------------------

    if (

        xception_fake
        <=
        STRONG_REAL_THRESHOLD

        and

        efficientnet_fake
        >=
        STRONG_FAKE_THRESHOLD

    ):

        score = (

            0.70 *
            xception_fake

            +

            0.30 *
            efficientnet_fake

        )

        return {

            "score":
                float(score),

            "mode":
                "STANDARD_XCEPTION_WEIGHT",

            "normal_score":
                float(normal_score),

            "difference":
                float(difference)

        }

    # ------------------------------------------------------------
    # CASE 4:
    # Normal disagreement / agreement.
    # ------------------------------------------------------------

    return {

        "score":
            float(normal_score),

        "mode":
            "STANDARD_WEIGHTED_FUSION",

        "normal_score":
            float(normal_score),

        "difference":
            float(difference)

    }


# ================================================================
# DISAGREEMENT CLASSIFICATION
# ================================================================

def classify_disagreement(
    difference
):

    if difference >= HIGH_DISAGREEMENT:

        return "HIGH"

    if difference >= MEDIUM_DISAGREEMENT:

        return "MEDIUM"

    return "LOW"


# ================================================================
# FINAL DECISION
# ================================================================

def final_decision(
    fake_score
):

    if fake_score >= 0.50:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    confidence = (

        fake_score
        if prediction == "FAKE"
        else
        1.0 - fake_score

    )

    return {

        "prediction":
            prediction,

        "confidence":
            float(confidence),

        "fake_score":
            float(fake_score),

        "real_score":
            float(
                1.0 - fake_score
            )

    }


# ================================================================
# ARGUMENTS
# ================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=
        "ForensIQ V2 robust face-crop inference"

    )

    parser.add_argument(

        "image",

        type=str,

        help=
        "Path to input image"

    )

    parser.add_argument(

        "--output",

        type=str,

        default=None,

        help=
        "Optional JSON output path"

    )

    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_arguments()

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        raise FileNotFoundError(

            f"\nInput image not found:\n"
            f"{image_path}\n"

        )

    print()

    print(
        "=" * 70
    )

    print(
        "LOADING INPUT IMAGE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Image: {image_path}"
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    print(
        f"Original size: "
        f"{image.width} x {image.height}"
    )

    print()

    # ------------------------------------------------------------
    # FACE DETECTION
    # ------------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "AUTOMATIC FACE DETECTION"
    )

    print(
        "=" * 70
    )

    face_result = create_face_crop(
        image
    )

    print()

    if face_result[
        "face_detected"
    ]:

        print(
            "✓ Face detected"
        )

        print(
            f"Faces detected : "
            f"{face_result['num_faces']}"
        )

        print(
            f"Face confidence: "
            f"{face_result['confidence']:.4f}"
        )

        print(
            f"Crop box       : "
            f"{face_result['box']}"
        )

        print(
            f"Raw face box   : "
            f"{tuple(round(v, 2) for v in face_result['raw_box'])}"
        )

    else:

        print(
            "⚠ No face detected."
        )

        print(
            "Using full-image fallback."
        )

    face_crop = face_result[
        "crop"
    ]

    print()

    print(
        f"Model input crop: "
        f"{face_crop.width} x "
        f"{face_crop.height}"
    )

    print()

    # ------------------------------------------------------------
    # XCEPTION
    # ------------------------------------------------------------

    print(
        "=" * 70
    )

    print(
        "XCEPTION V2 INFERENCE"
    )

    print(
        "=" * 70
    )

    xception_result = (
        predict_with_tta(

            xception_model,

            face_crop

        )
    )

    print()

    print(
        f"Fake probability : "
        f"{xception_result['fake']:.4f}"
    )

    print(
        f"Real probability : "
        f"{xception_result['real']:.4f}"
    )

    print()

    print(
        f"TTA original fake: "
        f"{xception_result['original']['fake']:.4f}"
    )

    if USE_TTA:

        print(
            f"TTA flipped fake : "
            f"{xception_result['flipped']['fake']:.4f}"
        )

    print()

    print(
        "Xception prediction: "
        +
        (
            "FAKE"
            if xception_result["fake"]
            >= 0.50
            else
            "REAL"
        )
    )

    # ------------------------------------------------------------
    # EFFICIENTNET
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "EFFICIENTNET-B4 V2 INFERENCE"
    )

    print(
        "=" * 70
    )

    efficientnet_result = (
        predict_with_tta(

            efficientnet_model,

            face_crop

        )
    )

    print()

    print(
        f"Fake probability : "
        f"{efficientnet_result['fake']:.4f}"
    )

    print(
        f"Real probability : "
        f"{efficientnet_result['real']:.4f}"
    )

    print()

    print(
        f"TTA original fake: "
        f"{efficientnet_result['original']['fake']:.4f}"
    )

    if USE_TTA:

        print(
            f"TTA flipped fake : "
            f"{efficientnet_result['flipped']['fake']:.4f}"
        )

    print()

    print(
        "EfficientNet prediction: "
        +
        (
            "FAKE"
            if efficientnet_result["fake"]
            >= 0.50
            else
            "REAL"
        )
    )

    # ------------------------------------------------------------
    # FUSION
    # ------------------------------------------------------------

    xception_fake = (
        xception_result["fake"]
    )

    efficientnet_fake = (
        efficientnet_result["fake"]
    )

    normal_score = weighted_fusion(

        xception_fake,

        efficientnet_fake

    )

    gated = confidence_gated_fusion(

        xception_fake,

        efficientnet_fake

    )

    fused_score = (
        gated["score"]
    )

    disagreement = (
        gated["difference"]
    )

    disagreement_level = (
        classify_disagreement(
            disagreement
        )
    )

    decision = final_decision(
        fused_score
    )

    # ------------------------------------------------------------
    # FUSION OUTPUT
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FORENSIQ V2 CONFIDENCE-GATED FUSION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Xception weight       : "
        f"{XCEPTION_WEIGHT:.2f}"
    )

    print(
        f"EfficientNet weight   : "
        f"{EFFICIENTNET_WEIGHT:.2f}"
    )

    print()

    print(
        f"Xception fake score   : "
        f"{xception_fake:.4f}"
    )

    print(
        f"EfficientNet fake score: "
        f"{efficientnet_fake:.4f}"
    )

    print()

    print(
        f"Normal weighted score : "
        f"{normal_score:.4f}"
    )

    print()

    print(
        f"Fusion mode           : "
        f"{gated['mode']}"
    )

    print(
        f"Model disagreement    : "
        f"{disagreement:.4f}"
    )

    print(
        f"Disagreement level    : "
        f"{disagreement_level}"
    )

    print()

    print(
        f"FINAL FAKE SCORE      : "
        f"{fused_score:.4f}"
    )

    print(
        f"FINAL REAL SCORE      : "
        f"{decision['real_score']:.4f}"
    )

    print()

    print(
        f"FINAL PREDICTION      : "
        f"{decision['prediction']}"
    )

    print(
        f"CONFIDENCE            : "
        f"{decision['confidence'] * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # MODEL AGREEMENT
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "MODEL AGREEMENT"
    )

    print(
        "=" * 70
    )

    x_prediction = (
        "FAKE"
        if xception_fake >= 0.50
        else "REAL"
    )

    e_prediction = (
        "FAKE"
        if efficientnet_fake >= 0.50
        else "REAL"
    )

    print()

    print(
        f"Xception V2       : "
        f"{x_prediction}"
    )

    print(
        f"EfficientNet-B4 V2: "
        f"{e_prediction}"
    )

    if x_prediction == e_prediction:

        print()

        print(
            "✓ MODELS AGREE"
        )

    else:

        print()

        print(
            "⚠ MODELS DISAGREE"
        )

        print(
            f"Difference: "
            f"{disagreement:.4f}"
        )

        print(
            f"Level: "
            f"{disagreement_level}"
        )

    # ------------------------------------------------------------
    # FACE INFORMATION
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FACE PREPROCESSING INFORMATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Face detected : "
        f"{face_result['face_detected']}"
    )

    print(
        f"Number faces  : "
        f"{face_result['num_faces']}"
    )

    print(
        f"Face confidence: "
        f"{face_result['confidence']:.4f}"
    )

    print(
        f"Fallback used : "
        f"{face_result['fallback']}"
    )

    print(
        f"Crop size     : "
        f"{face_crop.width} x "
        f"{face_crop.height}"
    )

    print(
        f"Model size    : "
        f"{IMAGE_SIZE} x "
        f"{IMAGE_SIZE}"
    )

    # ------------------------------------------------------------
    # JSON RESULT
    # ------------------------------------------------------------

    result = {

        "version":
            "ForensIQ-V2",

        "input":
            {
                "path":
                    str(image_path),

                "width":
                    image.width,

                "height":
                    image.height,
            },

        "face_detection":
            {

                "detected":
                    face_result[
                        "face_detected"
                    ],

                "num_faces":
                    face_result[
                        "num_faces"
                    ],

                "confidence":
                    face_result[
                        "confidence"
                    ],

                "crop_box":
                    face_result[
                        "box"
                    ],

                "raw_face_box":
                    face_result.get(
                        "raw_box"
                    ),

                "fallback":
                    face_result[
                        "fallback"
                    ],

                "margin":
                    FACE_MARGIN,

            },

        "preprocessing":
            {

                "image_size":
                    IMAGE_SIZE,

                "normalization":
                    {

                        "mean":
                            [0.5, 0.5, 0.5],

                        "std":
                            [0.5, 0.5, 0.5],

                    },

                "tta":
                    USE_TTA,

            },

        "xception_v2":
            {

                "fake":
                    xception_result[
                        "fake"
                    ],

                "real":
                    xception_result[
                        "real"
                    ],

                "original_fake":
                    xception_result[
                        "original"
                    ]["fake"],

                "flipped_fake":
                    (
                        xception_result[
                            "flipped"
                        ]["fake"]
                        if USE_TTA
                        else None
                    ),

                "prediction":
                    x_prediction,

            },

        "efficientnet_b4_v2":
            {

                "fake":
                    efficientnet_result[
                        "fake"
                    ],

                "real":
                    efficientnet_result[
                        "real"
                    ],

                "original_fake":
                    efficientnet_result[
                        "original"
                    ]["fake"],

                "flipped_fake":
                    (
                        efficientnet_result[
                            "flipped"
                        ]["fake"]
                        if USE_TTA
                        else None
                    ),

                "prediction":
                    e_prediction,

            },

        "fusion":
            {

                "xception_weight":
                    XCEPTION_WEIGHT,

                "efficientnet_weight":
                    EFFICIENTNET_WEIGHT,

                "normal_weighted_score":
                    normal_score,

                "final_fake_score":
                    fused_score,

                "final_real_score":
                    decision[
                        "real_score"
                    ],

                "fusion_mode":
                    gated["mode"],

                "difference":
                    disagreement,

                "disagreement_level":
                    disagreement_level,

            },

        "final":
            {

                "prediction":
                    decision[
                        "prediction"
                    ],

                "confidence":
                    decision[
                        "confidence"
                    ],

            },

        "warnings":
            [

                "Confidence is not a calibrated probability.",

                "External-dataset validation is required.",

                "Face detection fallback may reduce reliability.",

            ]

    }

    # ------------------------------------------------------------
    # OUTPUT PATH
    # ------------------------------------------------------------

    if args.output:

        output_path = Path(
            args.output
        )

    else:

        output_path = (

            PREDICTION_DIR /

            (
                image_path.stem
                +
                "_v2_prediction.json"
            )

        )

    with open(

        output_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            result,

            file,

            indent=4

        )

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "FORENSIQ V2 ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Result JSON:"
    )

    print(
        output_path
    )

    print()

    print(
        "=" * 70
    )

    print(
        "✓ FACE-CROP V2 INFERENCE FINISHED"
    )

    print(
        "=" * 70
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()