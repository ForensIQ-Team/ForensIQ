# ================================================================
# FORENSIQ V2
# EXTERNAL IMAGE PREDICTOR
#
# V2 MODELS:
#   1. Xception V2
#   2. EfficientNet-B4 V2
#
# V2 TRAINING:
#   Synthetic AI-generated faces
#   +
#   Synthetic real faces
#   +
#   Balanced FF++ replay
#
# IMPORTANT:
#   The datasets are NOT searched during inference.
#   Inference uses only the trained V2 checkpoints.
#
# CLASS MAPPING:
#   0 = FAKE
#   1 = REAL
#
# PREPROCESSING MUST MATCH V2:
#   299 x 299
#   mean = [0.5, 0.5, 0.5]
#   std  = [0.5, 0.5, 0.5]
#
# MODEL ARCHITECTURES MUST MATCH V2 TRAINING:
#   Xception          -> timm "xception"
#   EfficientNet-B4   -> timm "tf_efficientnet_b4"
#
# ================================================================

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "v2"

XCEPTION_PATH = (
    CHECKPOINT_DIR / "best_xception_v2.pth"
)

EFFICIENTNET_PATH = (
    CHECKPOINT_DIR / "best_efficientnet_b4_v2.pth"
)

RESULT_DIR = BASE_DIR / "prediction_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

NUM_CLASSES = 2

FAKE = 0
REAL = 1

IMAGE_SIZE = 299

# ------------------------------------------------
# V2 fusion
# ------------------------------------------------
#
# IMPORTANT:
# These are provisional V2 inference weights.
# They should be recalibrated after the external
# evaluation set is tested.
#
# Xception V2 is currently the stronger V2 model
# according to the reported synthetic + FF++ metrics.
# ------------------------------------------------

XCEPTION_WEIGHT = 0.60
EFFICIENTNET_WEIGHT = 0.40

# Do NOT reuse the old 0.45 threshold blindly.
# Until external calibration is completed, use the
# natural 0.50 probability threshold.
DECISION_THRESHOLD = 0.50

# ------------------------------------------------
# Test-time augmentation
# ------------------------------------------------
#
# Original image + horizontal flip.
#
# This reduces sensitivity to a single orientation
# without changing the training architecture.
# ------------------------------------------------

USE_TTA = True


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# PREPROCESSING
# ================================================================

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])


# ================================================================
# OUTPUT HELPERS
# ================================================================

def header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def require_file(path, name):
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )


# ================================================================
# CHECKPOINT UTILITIES
# ================================================================

def extract_state_dict(checkpoint):
    """
    Supports the V2 checkpoint format and common
    wrapped checkpoint formats.
    """

    if isinstance(checkpoint, dict):

        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights",
        ]:

            value = checkpoint.get(key)

            if isinstance(value, dict):
                return value

        # Raw state dictionary
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not locate a model state_dict "
        "inside the checkpoint."
    )


def clean_state_dict(state_dict):
    """
    Removes wrappers commonly introduced by
    DataParallel / compilation / model wrappers.
    """

    cleaned = {}

    prefixes = [
        "module.",
        "_orig_mod.",
        "model.",
    ]

    for key, value in state_dict.items():

        new_key = str(key)

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(prefix):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(model, checkpoint_path, model_name):

    print(
        f"\nLoading {model_name} V2 checkpoint..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    if result.missing_keys:

        print(
            f"WARNING: {len(result.missing_keys)} "
            f"missing keys."
        )

        for key in result.missing_keys[:10]:
            print(f"  Missing: {key}")

    if result.unexpected_keys:

        print(
            f"WARNING: {len(result.unexpected_keys)} "
            f"unexpected keys."
        )

        for key in result.unexpected_keys[:10]:
            print(f"  Unexpected: {key}")

    # For a V2 checkpoint we expect exact compatibility.
    if (
        result.missing_keys
        or result.unexpected_keys
    ):

        raise RuntimeError(
            f"\n{model_name} V2 checkpoint is "
            f"NOT fully compatible with the "
            f"architecture used by predict_image.py.\n"
            f"Missing keys: {len(result.missing_keys)}\n"
            f"Unexpected keys: {len(result.unexpected_keys)}\n"
            f"Check that the predictor uses the "
            f"same timm architecture as training."
        )

    print(
        f"✓ {model_name} V2 checkpoint loaded perfectly"
    )

    return checkpoint


# ================================================================
# MODEL CREATION
# ================================================================

def create_xception():

    # MUST match V2 training exactly.
    return timm.create_model(
        "xception",
        pretrained=False,
        num_classes=NUM_CLASSES
    )


def create_efficientnet():

    # MUST match V2 training exactly.
    #
    # The V2 training script uses:
    #   timm.create_model("tf_efficientnet_b4", ...)
    #
    # Do NOT replace this with "efficientnet_b4".
    return timm.create_model(
        "tf_efficientnet_b4",
        pretrained=False,
        num_classes=NUM_CLASSES
    )


# ================================================================
# LOAD MODELS
# ================================================================

def load_xception():

    model = create_xception()

    checkpoint = load_checkpoint(
        model,
        XCEPTION_PATH,
        "Xception"
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


def load_efficientnet():

    model = create_efficientnet()

    checkpoint = load_checkpoint(
        model,
        EFFICIENTNET_PATH,
        "EfficientNet-B4"
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


# ================================================================
# SINGLE MODEL INFERENCE
# ================================================================

@torch.no_grad()
def infer_tensor(model, image_tensor):

    logits = model(image_tensor)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    fake_probability = float(
        probabilities[FAKE].item()
    )

    real_probability = float(
        probabilities[REAL].item()
    )

    prediction = (
        "FAKE"
        if fake_probability >= real_probability
        else "REAL"
    )

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "prediction": prediction,
    }


# ================================================================
# TTA INFERENCE
# ================================================================

@torch.no_grad()
def infer_with_tta(model, image, model_name):

    original_tensor = (
        TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    original = infer_tensor(
        model,
        original_tensor
    )

    if not USE_TTA:

        return {
            **original,
            "tta_used": False,
            "tta_fake_probability": None,
            "tta_real_probability": None,
        }

    # Horizontal flip.
    flipped_image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    flipped_tensor = (
        TRANSFORM(flipped_image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    flipped = infer_tensor(
        model,
        flipped_tensor
    )

    # Average probabilities, not hard predictions.
    fake_probability = (
        original["fake_probability"]
        +
        flipped["fake_probability"]
    ) / 2.0

    real_probability = (
        original["real_probability"]
        +
        flipped["real_probability"]
    ) / 2.0

    prediction = (
        "FAKE"
        if fake_probability >= real_probability
        else "REAL"
    )

    return {
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "prediction": prediction,
        "tta_used": True,
        "original_fake_probability":
            original["fake_probability"],
        "flipped_fake_probability":
            flipped["fake_probability"],
        "original_prediction":
            original["prediction"],
        "flipped_prediction":
            flipped["prediction"],
    }


# ================================================================
# MAIN ANALYSIS
# ================================================================

def analyze_image(image_path):

    image_path = Path(image_path)

    header(
        "FORENSIQ V2 EXTERNAL IMAGE ANALYSIS"
    )

    # ------------------------------------------------------------
    # Validate image
    # ------------------------------------------------------------

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print(
        f"Image: {image_path}"
    )

    # ------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        raise RuntimeError(
            f"Could not open image:\n"
            f"{image_path}\n\n"
            f"Error: {error}"
        )

    print(
        f"\nOriginal size: "
        f"{image.width} x {image.height}"
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        props = torch.cuda.get_device_properties(0)

        print(
            f"VRAM: "
            f"{props.total_memory / (1024 ** 3):.2f} GB"
        )

    # ------------------------------------------------------------
    # Validate checkpoints
    # ------------------------------------------------------------

    header(
        "CHECKING FORENSIQ V2 CHECKPOINTS"
    )

    require_file(
        XCEPTION_PATH,
        "Xception V2 checkpoint"
    )

    require_file(
        EFFICIENTNET_PATH,
        "EfficientNet-B4 V2 checkpoint"
    )

    print(
        f"✓ Xception V2\n  {XCEPTION_PATH}"
    )

    print(
        f"✓ EfficientNet-B4 V2\n  {EFFICIENTNET_PATH}"
    )

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    header(
        "LOADING BOTH FORENSIQ V2 MODELS"
    )

    xception, x_checkpoint = (
        load_xception()
    )

    efficientnet, e_checkpoint = (
        load_efficientnet()
    )

    # ------------------------------------------------------------
    # XCEPTION
    # ------------------------------------------------------------

    header(
        "XCEPTION V2 INFERENCE"
    )

    x_result = infer_with_tta(
        xception,
        image,
        "Xception V2"
    )

    print(
        f"Fake probability : "
        f"{x_result['fake_probability']:.4f}"
    )

    print(
        f"Real probability : "
        f"{x_result['real_probability']:.4f}"
    )

    print(
        f"Prediction       : "
        f"{x_result['prediction']}"
    )

    if USE_TTA:

        print(
            f"\nTTA original fake : "
            f"{x_result['original_fake_probability']:.4f}"
        )

        print(
            f"TTA flipped fake  : "
            f"{x_result['flipped_fake_probability']:.4f}"
        )

    # ------------------------------------------------------------
    # EFFICIENTNET
    # ------------------------------------------------------------

    header(
        "EFFICIENTNET-B4 V2 INFERENCE"
    )

    e_result = infer_with_tta(
        efficientnet,
        image,
        "EfficientNet-B4 V2"
    )

    print(
        f"Fake probability : "
        f"{e_result['fake_probability']:.4f}"
    )

    print(
        f"Real probability : "
        f"{e_result['real_probability']:.4f}"
    )

    print(
        f"Prediction       : "
        f"{e_result['prediction']}"
    )

    if USE_TTA:

        print(
            f"\nTTA original fake : "
            f"{e_result['original_fake_probability']:.4f}"
        )

        print(
            f"TTA flipped fake  : "
            f"{e_result['flipped_fake_probability']:.4f}"
        )

    # ------------------------------------------------------------
    # FUSION
    # ------------------------------------------------------------

    header(
        "FORENSIQ V2 TWO-MODEL SCORE FUSION"
    )

    fused_fake = (
        XCEPTION_WEIGHT
        * x_result["fake_probability"]
        +
        EFFICIENTNET_WEIGHT
        * e_result["fake_probability"]
    )

    fused_real = (
        XCEPTION_WEIGHT
        * x_result["real_probability"]
        +
        EFFICIENTNET_WEIGHT
        * e_result["real_probability"]
    )

    # Numerical safety.
    total = fused_fake + fused_real

    if total > 0:

        fused_fake /= total
        fused_real /= total

    final_prediction = (
        "FAKE"
        if fused_fake >= DECISION_THRESHOLD
        else "REAL"
    )

    confidence = (
        fused_fake
        if final_prediction == "FAKE"
        else fused_real
    )

    # ------------------------------------------------------------
    # Model agreement
    # ------------------------------------------------------------

    agreement = (
        x_result["prediction"]
        ==
        e_result["prediction"]
    )

    score_difference = abs(
        x_result["fake_probability"]
        -
        e_result["fake_probability"]
    )

    # This is a disagreement magnitude, NOT a calibrated
    # probability of correctness.
    disagreement_level = (
        "LOW"
        if score_difference < 0.15
        else
        "MODERATE"
        if score_difference < 0.30
        else
        "HIGH"
    )

    print(
        f"Xception V2 weight       : "
        f"{XCEPTION_WEIGHT:.2f}"
    )

    print(
        f"EfficientNet V2 weight   : "
        f"{EFFICIENTNET_WEIGHT:.2f}"
    )

    print(
        f"Decision threshold       : "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"\nXception V2 fake score   : "
        f"{x_result['fake_probability']:.4f}"
    )

    print(
        f"EfficientNet V2 fake score: "
        f"{e_result['fake_probability']:.4f}"
    )

    print(
        f"\nFUSED FAKE SCORE          : "
        f"{fused_fake:.4f}"
    )

    print(
        f"FUSED REAL SCORE          : "
        f"{fused_real:.4f}"
    )

    print(
        f"\nFINAL PREDICTION           : "
        f"{final_prediction}"
    )

    print(
        f"CONFIDENCE                 : "
        f"{confidence * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Agreement
    # ------------------------------------------------------------

    header(
        "MODEL AGREEMENT"
    )

    if agreement:

        print(
            f"✓ BOTH V2 MODELS AGREE: "
            f"{x_result['prediction']}"
        )

    else:

        print(
            "⚠ V2 MODELS DISAGREE"
        )

        print(
            f"Xception V2       : "
            f"{x_result['prediction']}"
        )

        print(
            f"EfficientNet-B4 V2: "
            f"{e_result['prediction']}"
        )

    print(
        f"\nFake-score difference : "
        f"{score_difference:.4f}"
    )

    print(
        f"Disagreement level    : "
        f"{disagreement_level}"
    )

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------

    header(
        "FORENSIQ V2 SUMMARY"
    )

    if final_prediction == "FAKE":

        print(
            "The V2 ensemble produced a fake score "
            "above the current decision threshold."
        )

        print(
            "The result is consistent with "
            "manipulation or synthetic-generation "
            "indicators learned by the V2 models."
        )

    else:

        print(
            "The V2 ensemble produced a fake score "
            "below the current decision threshold."
        )

        print(
            "The image is classified as REAL by "
            "the current V2 ensemble."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The displayed confidence is the ensemble "
        "score, not a mathematically calibrated "
        "probability that the image is truly fake "
        "or real."
    )

    print(
        "External-dataset evaluation is required "
        "before treating this score as a reliable "
        "real-world confidence."
    )

    # ------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------

    filename = image_path.stem

    result = {

        "version":
            "ForensIQ-V2",

        "image":
            str(image_path),

        "device":
            str(DEVICE),

        "models": {

            "xception_v2": {

                "checkpoint":
                    str(XCEPTION_PATH),

                "fake_probability":
                    x_result[
                        "fake_probability"
                    ],

                "real_probability":
                    x_result[
                        "real_probability"
                    ],

                "prediction":
                    x_result[
                        "prediction"
                    ],

                "weight":
                    XCEPTION_WEIGHT,

                "tta_used":
                    USE_TTA,

            },

            "efficientnet_b4_v2": {

                "checkpoint":
                    str(EFFICIENTNET_PATH),

                "fake_probability":
                    e_result[
                        "fake_probability"
                    ],

                "real_probability":
                    e_result[
                        "real_probability"
                    ],

                "prediction":
                    e_result[
                        "prediction"
                    ],

                "weight":
                    EFFICIENTNET_WEIGHT,

                "tta_used":
                    USE_TTA,

            },

        },

        "fusion": {

            "xception_weight":
                XCEPTION_WEIGHT,

            "efficientnet_weight":
                EFFICIENTNET_WEIGHT,

            "fake_probability":
                fused_fake,

            "real_probability":
                fused_real,

            "prediction":
                final_prediction,

            "confidence":
                confidence,

            "decision_threshold":
                DECISION_THRESHOLD,

        },

        "model_agreement":
            agreement,

        "fake_score_difference":
            score_difference,

        "disagreement_level":
            disagreement_level,

        "preprocessing": {

            "image_size":
                IMAGE_SIZE,

            "mean":
                [0.5, 0.5, 0.5],

            "std":
                [0.5, 0.5, 0.5],

        },

        "tta": {

            "enabled":
                USE_TTA,

            "method":
                "original + horizontal flip "
                "probability averaging",

        },

        "note":
            "V2 inference uses the trained V2 "
            "checkpoints. Training datasets are "
            "not searched during inference. "
            "Fusion weights and threshold are "
            "provisional until external "
            "evaluation/calibration.",

    }

    result_path = (
        RESULT_DIR
        /
        f"{filename}_v2_prediction.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        f"\nResult JSON saved to:\n"
        f"{result_path}"
    )

    header(
        "FORENSIQ V2 ANALYSIS COMPLETE"
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python inference\\predict_image_v2.py '
            '"C:\\path\\to\\image.jpg"'
        )

        sys.exit(1)

    input_image = sys.argv[1]

    try:

        analyze_image(
            input_image
        )

    except KeyboardInterrupt:

        print(
            "\n\nAnalysis interrupted."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print("=" * 70)
        print("FORENSIQ V2 PREDICTION ERROR")
        print("=" * 70)

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        sys.exit(1)
