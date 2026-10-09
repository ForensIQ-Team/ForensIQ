# ================================================================
# FORENSIQ V2
# GRAD-CAM FACE-CROP EXPLAINABILITY
#
# File:
#   inference/gradcam_v2_facecrop.py
#
# PURPOSE:
#   Generate Grad-CAM explanations for the SAME V2 inference
#   pipeline used by predict_v2_facecrop.py.
#
# IMPORTANT:
#   - DOES NOT retrain models
#   - DOES NOT modify checkpoints
#   - DOES NOT modify ensemble weights
#   - DOES NOT modify confidence gates
#   - DOES NOT modify TTA
#   - DOES NOT modify face detection
#   - DOES NOT modify final prediction
#
# PIPELINE:
#
#   Input Image
#       |
#       v
#   MTCNN Face Detection
#       |
#       v
#   Same face selection
#       |
#       v
#   Same margin-expanded face crop
#       |
#       v
#   299 x 299
#       |
#       +----------------------+
#       |                      |
#       v                      v
#   Xception V2          EfficientNet-B4 V2
#       |                      |
#       v                      v
#   Grad-CAM               Grad-CAM
#       |                      |
#       +----------+-----------+
#                  |
#                  v
#          Visualization
#
# CLASS MAPPING:
#   0 = FAKE
#   1 = REAL
#
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

import cv2

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False


# ================================================================
# PROJECT PATHS
# ================================================================

INFERENCE_DIR = Path(__file__).resolve().parent

ROOT = INFERENCE_DIR.parent

CHECKPOINT_DIR = ROOT / "checkpoints"

V2_CHECKPOINT_DIR = CHECKPOINT_DIR / "v2"

GRADCAM_OUTPUT_DIR = ROOT / "gradcam_results"

GRADCAM_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# CHECKPOINTS
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
# CONFIG
# ================================================================

IMAGE_SIZE = 299

NUM_CLASSES = 2

FAKE = 0

REAL = 1

FACE_MARGIN = 0.30

MIN_FACE_SIZE = 40

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# DISPLAY
# ================================================================

print()
print("=" * 70)
print("FORENSIQ V2")
print("GRAD-CAM FACE-CROP EXPLAINABILITY")
print("=" * 70)
print()

print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print()

print("Class mapping:")
print("0 = FAKE")
print("1 = REAL")

print()


# ================================================================
# VALIDATION
# ================================================================

def require_file(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}\n"
        )

    print(f"✓ {name}")


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
        "Install with:\n"
        "pip install facenet-pytorch\n"
    )


# ================================================================
# PREPROCESSING
#
# MUST MATCH predict_v2_facecrop.py
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

    return timm.create_model(

        "xception",

        pretrained=False,

        num_classes=NUM_CLASSES

    )


def create_efficientnet():

    return timm.create_model(

        "tf_efficientnet_b4",

        pretrained=False,

        num_classes=NUM_CLASSES

    )


# ================================================================
# CHECKPOINT EXTRACTION
# ================================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            return checkpoint[
                "model_state_dict"
            ]

        if "state_dict" in checkpoint:

            return checkpoint[
                "state_dict"
            ]

        if (
            "model" in checkpoint
            and
            isinstance(
                checkpoint["model"],
                dict
            )
        ):

            return checkpoint["model"]

        return checkpoint

    return checkpoint


# ================================================================
# CLEAN STATE DICT
# ================================================================

def clean_state_dict(state_dict):

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

                if new_key.startswith(prefix):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned[new_key] = value

    return cleaned


# ================================================================
# LOAD MODEL
# ================================================================

def load_model(
    model,
    checkpoint_path,
    model_name
):

    print(
        f"Loading {model_name} V2..."
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
            f"WARNING: "
            f"{len(result.missing_keys)} "
            f"missing keys"
        )

        for key in result.missing_keys[:10]:
            print(f"  {key}")

    if result.unexpected_keys:

        print(
            f"WARNING: "
            f"{len(result.unexpected_keys)} "
            f"unexpected keys"
        )

        for key in result.unexpected_keys[:10]:
            print(f"  {key}")

    if (
        not result.missing_keys
        and
        not result.unexpected_keys
    ):

        print(
            f"✓ {model_name} checkpoint "
            f"loaded perfectly"
        )

    model = model.to(DEVICE)

    model.eval()

    return model


# ================================================================
# LOAD MODELS
# ================================================================

print()
print("=" * 70)
print("LOADING FORENSIQ V2 MODELS")
print("=" * 70)
print()

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
print("=" * 70)
print("INITIALIZING FACE DETECTOR")
print("=" * 70)
print()

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

print("✓ MTCNN ready")
print()


# ================================================================
# FACE DETECTION
# ================================================================

def detect_faces(image):

    boxes, probabilities = (
        face_detector.detect(image)
    )

    if boxes is None:
        return []

    faces = []

    for index, box in enumerate(boxes):

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
#
# EXACT SAME LOGIC AS V2 INFERENCE
# ================================================================

def select_best_face(faces):

    if not faces:
        return None

    def score(face):

        return (
            face["area"] *
            (
                0.5 +
                0.5 *
                face["confidence"]
            )
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

    horizontal_margin = (
        width * margin
    )

    vertical_margin = (
        height *
        margin *
        1.10
    )

    new_x1 = max(
        0,
        int(round(
            x1 - horizontal_margin
        ))
    )

    new_y1 = max(
        0,
        int(round(
            y1 - vertical_margin
        ))
    )

    new_x2 = min(
        image_width,
        int(round(
            x2 + horizontal_margin
        ))
    )

    new_y2 = min(
        image_height,
        int(round(
            y2 + vertical_margin
        ))
    )

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


# ================================================================
# CREATE SAME FACE CROP
# ================================================================

def create_face_crop(image):

    image_width, image_height = image.size

    faces = detect_faces(image)

    if not faces:

        return {

            "crop":
                image.copy(),

            "face_detected":
                False,

            "box":
                None,

            "raw_box":
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
# FIND GRAD-CAM TARGET LAYER
# ================================================================
#
# We automatically locate the final suitable convolutional
# feature layer.
#
# This avoids hard-coding fragile internal layer names.
# ================================================================

def find_last_conv_layer(model):

    candidate = None

    for module in model.modules():

        if isinstance(
            module,
            (
                nn.Conv2d,
                nn.Conv3d
            )
        ):

            candidate = module

    if candidate is None:

        raise RuntimeError(
            "Could not find a convolutional "
            "layer for Grad-CAM."
        )

    return candidate


# ================================================================
# GRAD-CAM CLASS
# ================================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )


    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output


    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]


    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        self.activations = None

        self.gradients = None

        logits = self.model(
            input_tensor
        )

        score = logits[
            0,
            target_class
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation "
                "was not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients "
                "were not captured."
            )

        activations = self.activations

        gradients = self.gradients

        # --------------------------------------------------------
        # Handle standard CNN tensor:
        #
        # [B, C, H, W]
        # --------------------------------------------------------

        if activations.ndim != 4:

            raise RuntimeError(
                "Grad-CAM expected a "
                "4D convolutional activation "
                f"but received shape "
                f"{tuple(activations.shape)}"
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = torch.relu(
            cam
        )

        cam = torch.nn.functional.interpolate(

            cam,

            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),

            mode="bilinear",

            align_corners=False

        )

        cam = cam[
            0,
            0
        ]

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 1e-8:

            cam /= maximum

        else:

            cam[:] = 0

        return {

            "cam":
                cam,

            "logits":
                logits.detach().cpu(),

            "target_class":
                int(target_class),

            "target_score":
                float(
                    score.detach()
                    .cpu()
                    .item()
                )

        }


# ================================================================
# MODEL PROBABILITIES
# ================================================================

def get_probabilities(
    model,
    image
):

    tensor = transform(
        image
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1
            )[0]
        )

    fake = float(
        probabilities[FAKE]
        .cpu()
        .item()
    )

    real = float(
        probabilities[REAL]
        .cpu()
        .item()
    )

    prediction = (
        FAKE
        if fake >= real
        else REAL
    )

    confidence = max(
        fake,
        real
    )

    return {

        "tensor":
            tensor,

        "fake":
            fake,

        "real":
            real,

        "prediction":
            prediction,

        "confidence":
            confidence

    }


# ================================================================
# CREATE HEATMAP
# ================================================================

def create_heatmap(
    cam,
    width,
    height
):

    cam_uint8 = np.uint8(
        cam * 255
    )

    cam_resized = cv2.resize(
        cam_uint8,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    heatmap = cv2.applyColorMap(
        cam_resized,
        cv2.COLORMAP_JET
    )

    return heatmap


# ================================================================
# OVERLAY HEATMAP
# ================================================================

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.45
):

    image_np = np.array(
        image
    )

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(

        image_bgr,

        1.0 - alpha,

        heatmap,

        alpha,

        0

    )

    return overlay


# ================================================================
# SAVE IMAGE
# ================================================================

def save_image(
    path,
    image
):

    cv2.imwrite(
        str(path),
        image
    )

    print(
        f"✓ Saved: {path}"
    )


# ================================================================
# DRAW FACE BOX
# ================================================================

def save_face_detection_image(
    original,
    face_result,
    output_path
):

    image = np.array(
        original
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    if face_result["box"] is not None:

        x1, y1, x2, y2 = (
            face_result["box"]
        )

        cv2.rectangle(

            image,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            3

        )

    save_image(
        output_path,
        image
    )


# ================================================================
# PROCESS ONE MODEL
# ================================================================

def process_model(
    model,
    model_name,
    face_crop,
    output_prefix
):

    print()
    print("-" * 70)
    print(
        f"{model_name} GRAD-CAM"
    )
    print("-" * 70)

    probabilities = get_probabilities(
        model,
        face_crop
    )

    prediction = probabilities[
        "prediction"
    ]

    prediction_name = (
        "FAKE"
        if prediction == FAKE
        else "REAL"
    )

    print(
        f"Fake probability : "
        f"{probabilities['fake']:.4f}"
    )

    print(
        f"Real probability : "
        f"{probabilities['real']:.4f}"
    )

    print(
        f"Prediction        : "
        f"{prediction_name}"
    )

    print(
        f"Confidence        : "
        f"{probabilities['confidence']:.4f}"
    )

    target_layer = find_last_conv_layer(
        model
    )

    print(
        f"Target layer      : "
        f"{target_layer}"
    )

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:

        # --------------------------------------------------------
        # Explain the model's OWN predicted class.
        #
        # This answers:
        # "What regions caused the model to make this prediction?"
        # --------------------------------------------------------

        result = gradcam.generate(

            probabilities["tensor"],

            prediction

        )

    finally:

        gradcam.remove_hooks()

    cam = result["cam"]

    heatmap = create_heatmap(

        cam,

        face_crop.width,

        face_crop.height

    )

    overlay = overlay_heatmap(

        face_crop,

        heatmap,

        alpha=0.45

    )

    # ------------------------------------------------------------
    # Raw heatmap
    # ------------------------------------------------------------

    heatmap_path = (
        output_prefix
        + "_heatmap.jpg"
    )

    save_image(
        heatmap_path,
        heatmap
    )

    # ------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------

    overlay_path = (
        output_prefix
        + "_overlay.jpg"
    )

    save_image(
        overlay_path,
        overlay
    )

    # ------------------------------------------------------------
    # CAM grayscale
    # ------------------------------------------------------------

    cam_path = (
        output_prefix
        + "_cam.jpg"
    )

    cam_uint8 = np.uint8(
        cam * 255
    )

    save_image(
        cam_path,
        cam_uint8
    )

    return {

        "model":
            model_name,

        "fake":
            probabilities["fake"],

        "real":
            probabilities["real"],

        "prediction":
            prediction_name,

        "confidence":
            probabilities["confidence"],

        "target_class":
            int(prediction),

        "target_layer":
            str(target_layer),

        "heatmap":
            heatmap_path,

        "overlay":
            overlay_path,

        "cam":
            cam_path

    }


# ================================================================
# ARGUMENTS
# ================================================================

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=
        "ForensIQ V2 face-crop Grad-CAM"

    )

    parser.add_argument(

        "image",

        type=str,

        help=
        "Path to input image"

    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default=None,

        help=
        "Optional Grad-CAM output directory"

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

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    else:

        output_dir = (
            GRADCAM_OUTPUT_DIR /
            image_path.stem
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    print()
    print("=" * 70)
    print("LOADING INPUT IMAGE")
    print("=" * 70)
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
        f"{image.width} x "
        f"{image.height}"
    )

    # ============================================================
    # SAME FACE-CROP PIPELINE
    # ============================================================

    print()
    print("=" * 70)
    print("FACE DETECTION + FACE CROP")
    print("=" * 70)
    print()

    face_result = create_face_crop(
        image
    )

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

    print(
        f"Model crop size: "
        f"{face_crop.width} x "
        f"{face_crop.height}"
    )

    # ============================================================
    # SAVE FACE CROP
    # ============================================================

    face_crop_path = (
        output_dir /
        "face_crop.jpg"
    )

    face_crop.save(
        face_crop_path,
        quality=95
    )

    print(
        f"✓ Saved: {face_crop_path}"
    )

    # ============================================================
    # SAVE DETECTION BOX
    # ============================================================

    detection_path = (
        output_dir /
        "face_detection.jpg"
    )

    save_face_detection_image(

        image,

        face_result,

        detection_path

    )

    # ============================================================
    # XCEPTION
    # ============================================================

    xception_output = (
        output_dir /
        "xception"
    )

    xception_output.mkdir(
        exist_ok=True
    )

    xception_result = process_model(

        xception_model,

        "Xception V2",

        face_crop,

        str(
            xception_output /
            "xception_v2"
        )

    )

    # ============================================================
    # EFFICIENTNET
    # ============================================================

    efficientnet_output = (
        output_dir /
        "efficientnet"
    )

    efficientnet_output.mkdir(
        exist_ok=True
    )

    efficientnet_result = process_model(

        efficientnet_model,

        "EfficientNet-B4 V2",

        face_crop,

        str(
            efficientnet_output /
            "efficientnet_b4_v2"
        )

    )

    # ============================================================
    # SAVE METADATA
    # ============================================================

    metadata = {

        "version":
            "ForensIQ-V2-GradCAM",

        "input":
            {

                "path":
                    str(image_path),

                "width":
                    image.width,

                "height":
                    image.height

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
                    FACE_MARGIN

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
                            [0.5, 0.5, 0.5]

                    }

            },

        "xception_v2":
            xception_result,

        "efficientnet_b4_v2":
            efficientnet_result,

        "notes":
            [

                "Grad-CAM is an explainability visualization.",

                "Grad-CAM does not modify model prediction.",

                "The visualization uses the model's predicted class.",

                "The face-crop preprocessing matches ForensIQ V2 inference.",

                "The heatmap should not be interpreted as a pixel-perfect segmentation mask."

            ]

    }

    metadata_path = (
        output_dir /
        "gradcam_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print("FORENSIQ V2 GRAD-CAM COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Output directory:\n"
        f"{output_dir}"
    )

    print()

    print(
        "Generated:"
    )

    print(
        "  ✓ Same face crop"
    )

    print(
        "  ✓ Face detection visualization"
    )

    print(
        "  ✓ Xception V2 heatmap"
    )

    print(
        "  ✓ Xception V2 overlay"
    )

    print(
        "  ✓ EfficientNet-B4 V2 heatmap"
    )

    print(
        "  ✓ EfficientNet-B4 V2 overlay"
    )

    print(
        "  ✓ Grad-CAM metadata JSON"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Grad-CAM does NOT alter the V2 prediction pipeline."
    )

    print(
        "It only explains the model decision."
    )

    print()


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()