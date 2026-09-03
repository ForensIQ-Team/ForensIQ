# ================================================================
# FORENSIQ - STRONG TWO-MODEL GRAD-CAM EXPLAINABILITY
# XCEPTION + EFFICIENTNET-B4
#
# Class mapping:
#   0 = FAKE
#   1 = REAL
#
# Fusion:
#   Xception      = 0.55
#   EfficientNet  = 0.45
#
# Features:
#   - Both models always participate
#   - Automatic target-class Grad-CAM
#   - Separate Xception CAM
#   - Separate EfficientNet CAM
#   - Fused CAM
#   - Face-aware visualization when a face is detected
#   - Original image preserved
#   - No retraining required
#   - Robust model/checkpoint loading
# ================================================================

import os
import sys
import json
import warnings

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm

warnings.filterwarnings("ignore", category=UserWarning)

# Optional face detector
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except Exception:
    MTCNN_AVAILABLE = False


# ================================================================
# CONFIGURATION
# ================================================================

XCEPTION_WEIGHT = 0.55
EFFICIENTNET_WEIGHT = 0.45

FAKE_CLASS = 0
REAL_CLASS = 1

IMAGE_SIZE = 299

CHECKPOINT_XCEPTION = (
    r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML"
    r"\checkpoints\best_xception.pth"
)

CHECKPOINT_EFFICIENTNET = (
    r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML"
    r"\checkpoints\best_efficientnet_b4.pth"
)

OUTPUT_DIR = (
    r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML"
    r"\\gradcam_results"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# PRINT HELPERS
# ================================================================

def line():
    print("=" * 64)


def section(title):
    print()
    line()
    print(title)
    line()


# ================================================================
# CHECK FILES
# ================================================================

def check_files(image_path):

    required = [
        ("Input image", image_path),
        ("Xception checkpoint", CHECKPOINT_XCEPTION),
        ("EfficientNet checkpoint", CHECKPOINT_EFFICIENTNET),
    ]

    for name, path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )
        print(f"✓ {name}")


# ================================================================
# CHECKPOINT LOADING
# ================================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        prefixes = [
            "module.",
            "model.",
            "network.",
        ]

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(model, checkpoint_path, model_name):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"], dict
        ):
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    if missing:
        print(
            f"⚠ {model_name}: {len(missing)} missing parameters"
        )

    if unexpected:
        print(
            f"⚠ {model_name}: {len(unexpected)} unexpected parameters"
        )

    model.to(DEVICE)
    model.eval()

    print(f"✓ {model_name} checkpoint loaded")

    return model


# ================================================================
# MODEL CREATION
# ================================================================

def create_models():

    section("LOADING BOTH FORENSIQ MODELS")

    print("Loading Xception...")

    xception = timm.create_model(
        "xception",
        pretrained=False,
        num_classes=2
    )

    xception = load_checkpoint(
        xception,
        CHECKPOINT_XCEPTION,
        "Xception"
    )

    print()
    print("Loading EfficientNet-B4...")

    efficientnet = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=2
    )

    efficientnet = load_checkpoint(
        efficientnet,
        CHECKPOINT_EFFICIENTNET,
        "EfficientNet-B4"
    )

    return xception, efficientnet


# ================================================================
# PREPROCESSING
#
# IMPORTANT:
# This matches the training setup used by the models.
# ================================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    tensor = transform(image).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    return image, original, tensor


# ================================================================
# MODEL PROBABILITY
# ================================================================

@torch.no_grad()
def get_probabilities(model, tensor):

    logits = model(tensor)

    probabilities = torch.softmax(logits, dim=1)

    return probabilities[0].detach().cpu().numpy()


# ================================================================
# TARGET LAYER DISCOVERY
#
# Automatically finds the last Conv2D layer.
# This avoids hard-coding architecture-specific layer names.
# ================================================================

def find_last_conv_layer(model):

    candidates = []

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            candidates.append(
                (name, module)
            )

    if not candidates:

        raise RuntimeError(
            "Could not find a Conv2D layer for Grad-CAM."
        )

    name, layer = candidates[-1]

    print(
        f"Grad-CAM target layer: {name}"
    )

    return layer


# ================================================================
# GRAD-CAM ENGINE
# ================================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output
    ):

        if isinstance(output, tuple):
            output = output[0]

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        if grad_output is None:
            return

        gradient = grad_output[0]

        self.gradients = gradient

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        score = output[:, target_class]

        score.sum().backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        # Expected:
        # [batch, channels, height, width]

        if activations.ndim != 4:

            raise RuntimeError(
                "Grad-CAM target layer does not produce "
                "a 4D feature map."
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize safely
        cam -= cam.min()

        max_value = cam.max()

        if max_value > 1e-8:
            cam /= max_value

        else:
            cam[:] = 0.0

        return cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ================================================================
# CAM SMOOTHING
#
# Reduces extremely noisy single-pixel activations.
# ================================================================

def smooth_cam(cam):

    cam_uint8 = np.uint8(
        np.clip(cam, 0, 1) * 255
    )

    cam_uint8 = cv2.GaussianBlur(
        cam_uint8,
        (0, 0),
        sigmaX=5
    )

    cam = cam_uint8.astype(np.float32) / 255.0

    max_value = cam.max()

    if max_value > 1e-8:
        cam /= max_value

    return cam


# ================================================================
# CAM THRESHOLDING
#
# Keeps only meaningful high-activation regions.
# ================================================================

def create_attention_mask(cam):

    threshold = np.percentile(
        cam,
        70
    )

    mask = np.where(
        cam >= threshold,
        cam,
        0
    )

    mask = cv2.GaussianBlur(
        mask.astype(np.float32),
        (0, 0),
        3
    )

    max_value = mask.max()

    if max_value > 1e-8:
        mask /= max_value

    return mask


# ================================================================
# HEATMAP
# ================================================================

def create_heatmap(cam):

    cam_uint8 = np.uint8(
        np.clip(cam, 0, 1) * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    return heatmap


# ================================================================
# OVERLAY
# ================================================================

def overlay_cam(
    original_rgb,
    cam,
    alpha=0.42
):

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    heatmap = create_heatmap(cam)

    heatmap = cv2.resize(
        heatmap,
        (
            original_bgr.shape[1],
            original_bgr.shape[0]
        )
    )

    result = cv2.addWeighted(
        original_bgr,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return result


# ================================================================
# FACE DETECTION
# ================================================================

def detect_faces(original_pil):

    if not MTCNN_AVAILABLE:

        print(
            "⚠ facenet-pytorch not available. "
            "Face-aware CAM skipped."
        )

        return []

    try:

        detector = MTCNN(
            keep_all=True,
            device=DEVICE
        )

        boxes, probabilities = detector.detect(
            original_pil
        )

        if boxes is None:
            return []

        faces = []

        for box, probability in zip(
            boxes,
            probabilities
        ):

            if probability is None:
                continue

            if probability < 0.85:
                continue

            x1, y1, x2, y2 = box

            faces.append({
                "box": [
                    int(max(0, x1)),
                    int(max(0, y1)),
                    int(x2),
                    int(y2)
                ],
                "confidence": float(probability)
            })

        return faces

    except Exception as e:

        print(
            f"⚠ Face detection skipped: {e}"
        )

        return []


# ================================================================
# FACE-AWARE CAM
#
# We do NOT artificially claim that the face is fake.
#
# Instead we report how much CAM activation lies inside
# detected face regions.
# ================================================================

def analyze_face_attention(
    cam,
    faces,
    original_shape
):

    if not faces:
        return {
            "faces_detected": 0,
            "face_attention_ratio": None
        }

    height, width = original_shape[:2]

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    face_mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    for face in faces:

        x1, y1, x2, y2 = face["box"]

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))

        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))

        if x2 > x1 and y2 > y1:

            face_mask[
                y1:y2,
                x1:x2
            ] = 1.0

    total_attention = np.sum(
        np.maximum(cam_resized, 0)
    )

    face_attention = np.sum(
        np.maximum(cam_resized, 0) * face_mask
    )

    if total_attention <= 1e-8:

        ratio = 0.0

    else:

        ratio = (
            face_attention /
            total_attention
        )

    return {
        "faces_detected": len(faces),
        "face_attention_ratio": float(ratio)
    }


# ================================================================
# DRAW FACE BOXES
# ================================================================

def draw_faces(
    original_rgb,
    faces
):

    image = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    for face in faces:

        x1, y1, x2, y2 = face["box"]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            4
        )

        label = (
            f"Face {face['confidence'] * 100:.1f}%"
        )

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return image


# ================================================================
# SAVE IMAGE
# ================================================================

def save_image(
    path,
    image
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    cv2.imwrite(
        path,
        image
    )

    print(
        f"✓ Saved: {path}"
    )


# ================================================================
# MAIN ANALYSIS
# ================================================================

def analyze(image_path):

    section(
        "FORENSIQ STRONG TWO-MODEL GRAD-CAM"
    )

    print(
        f"Input image:\n{image_path}"
    )

    print()
    print(
        f"Device: {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    check_files(image_path)

    # ------------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------------

    section("LOADING IMAGE")

    pil_image, original_rgb, input_tensor = load_image(
        image_path
    )

    height, width = original_rgb.shape[:2]

    print(
        f"Original size: {width} x {height}"
    )

    # ------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------

    xception, efficientnet = create_models()

    # ------------------------------------------------------------
    # MODEL SCORES
    # ------------------------------------------------------------

    section("MODEL PREDICTIONS")

    x_probs = get_probabilities(
        xception,
        input_tensor
    )

    e_probs = get_probabilities(
        efficientnet,
        input_tensor
    )

    x_fake = float(x_probs[FAKE_CLASS])
    x_real = float(x_probs[REAL_CLASS])

    e_fake = float(e_probs[FAKE_CLASS])
    e_real = float(e_probs[REAL_CLASS])

    fused_fake = (
        XCEPTION_WEIGHT * x_fake +
        EFFICIENTNET_WEIGHT * e_fake
    )

    fused_real = (
        XCEPTION_WEIGHT * x_real +
        EFFICIENTNET_WEIGHT * e_real
    )

    predicted_class = (
        FAKE_CLASS
        if fused_fake >= fused_real
        else REAL_CLASS
    )

    confidence = max(
        fused_fake,
        fused_real
    )

    prediction_name = (
        "FAKE"
        if predicted_class == FAKE_CLASS
        else "REAL"
    )

    print()
    print("XCEPTION")
    print(
        f"Fake : {x_fake:.4f}"
    )
    print(
        f"Real : {x_real:.4f}"
    )

    print()
    print("EFFICIENTNET-B4")
    print(
        f"Fake : {e_fake:.4f}"
    )
    print(
        f"Real : {e_real:.4f}"
    )

    print()
    print("FUSED MODEL")
    print(
        f"Xception contribution     : "
        f"{XCEPTION_WEIGHT:.2f}"
    )

    print(
        f"EfficientNet contribution : "
        f"{EFFICIENTNET_WEIGHT:.2f}"
    )

    print(
        f"Fused Fake score          : "
        f"{fused_fake:.4f}"
    )

    print(
        f"Fused Real score          : "
        f"{fused_real:.4f}"
    )

    print(
        f"Final prediction          : "
        f"{prediction_name}"
    )

    print(
        f"Confidence                : "
        f"{confidence * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # TARGET CLASS
    #
    # VERY IMPORTANT:
    # CAM explains the class selected by the fused classifier.
    # ------------------------------------------------------------

    section(
        f"EXPLAINING PREDICTED CLASS: {prediction_name}"
    )

    target_class = predicted_class

    # ------------------------------------------------------------
    # TARGET LAYERS
    # ------------------------------------------------------------

    section(
        "SELECTING GRAD-CAM TARGET LAYERS"
    )

    x_layer = find_last_conv_layer(
        xception
    )

    e_layer = find_last_conv_layer(
        efficientnet
    )

    # ------------------------------------------------------------
    # CAM
    # ------------------------------------------------------------

    print()
    print("Generating Xception Grad-CAM...")

    x_cam_engine = GradCAM(
        xception,
        x_layer
    )

    x_cam = x_cam_engine.generate(
        input_tensor,
        target_class
    )

    x_cam_engine.close()

    print("✓ Xception CAM generated")

    print()
    print(
        "Generating EfficientNet-B4 Grad-CAM..."
    )

    e_cam_engine = GradCAM(
        efficientnet,
        e_layer
    )

    e_cam = e_cam_engine.generate(
        input_tensor,
        target_class
    )

    e_cam_engine.close()

    print(
        "✓ EfficientNet-B4 CAM generated"
    )

    # ------------------------------------------------------------
    # SMOOTH
    # ------------------------------------------------------------

    x_cam = smooth_cam(x_cam)
    e_cam = smooth_cam(e_cam)

    # ------------------------------------------------------------
    # NORMALIZED FUSION
    #
    # Both CAMs are normalized first.
    # Then the same 55/45 model fusion is applied.
    # ------------------------------------------------------------

    fused_cam = (
        XCEPTION_WEIGHT * x_cam +
        EFFICIENTNET_WEIGHT * e_cam
    )

    fused_cam -= fused_cam.min()

    max_value = fused_cam.max()

    if max_value > 1e-8:
        fused_cam /= max_value

    # ------------------------------------------------------------
    # FACE DETECTION
    # ------------------------------------------------------------

    section(
        "FACE-AWARE EXPLANATION"
    )

    faces = detect_faces(
        pil_image
    )

    if faces:

        print(
            f"✓ Detected {len(faces)} face(s)"
        )

        for i, face in enumerate(faces, 1):

            print(
                f"Face {i}: "
                f"{face['box']} | "
                f"confidence="
                f"{face['confidence']:.3f}"
            )

    else:

        print(
            "No high-confidence face detected."
        )

    x_face_stats = analyze_face_attention(
        x_cam,
        faces,
        original_rgb.shape
    )

    e_face_stats = analyze_face_attention(
        e_cam,
        faces,
        original_rgb.shape
    )

    fused_face_stats = analyze_face_attention(
        fused_cam,
        faces,
        original_rgb.shape
    )

    print()
    print(
        "Face attention:"
    )

    if faces:

        print(
            f"Xception      : "
            f"{x_face_stats['face_attention_ratio'] * 100:.2f}%"
        )

        print(
            f"EfficientNet  : "
            f"{e_face_stats['face_attention_ratio'] * 100:.2f}%"
        )

        print(
            f"Fused         : "
            f"{fused_face_stats['face_attention_ratio'] * 100:.2f}%"
        )

    else:

        print(
            "Not available because no face was detected."
        )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # ------------------------------------------------------------
    # CREATE VISUALIZATIONS
    # ------------------------------------------------------------

    x_overlay = overlay_cam(
        original_rgb,
        x_cam
    )

    e_overlay = overlay_cam(
        original_rgb,
        e_cam
    )

    fused_overlay = overlay_cam(
        original_rgb,
        fused_cam
    )

    face_image = draw_faces(
        original_rgb,
        faces
    )

    # ------------------------------------------------------------
    # FACE-FOCUSED FUSED CAM
    #
    # This does NOT alter the prediction.
    # It is only an explanation view.
    # ------------------------------------------------------------

    face_focused = fused_overlay.copy()

    if faces:

        face_mask = np.zeros(
            face_focused.shape[:2],
            dtype=np.uint8
        )

        for face in faces:

            x1, y1, x2, y2 = face["box"]

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                face_focused.shape[1],
                x2
            )

            y2 = min(
                face_focused.shape[0],
                y2
            )

            if x2 > x1 and y2 > y1:

                face_mask[
                    y1:y2,
                    x1:x2
                ] = 255

        # Slightly expand face region
        kernel = np.ones(
            (31, 31),
            np.uint8
        )

        face_mask = cv2.dilate(
            face_mask,
            kernel,
            iterations=1
        )

        darkened = (
            face_focused.astype(np.float32)
            * 0.35
        ).astype(np.uint8)

        face_focused = np.where(
            face_mask[..., None] > 0,
            face_focused,
            darkened
        )

    # ------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------

    section(
        "SAVING GRAD-CAM RESULTS"
    )

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_original.jpg"
        ),
        original_bgr
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_xception_gradcam.jpg"
        ),
        x_overlay
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_efficientnet_gradcam.jpg"
        ),
        e_overlay
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_fused_gradcam.jpg"
        ),
        fused_overlay
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_face_focused_gradcam.jpg"
        ),
        face_focused
    )

    save_image(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_face_detection.jpg"
        ),
        face_image
    )

    # ------------------------------------------------------------
    # RAW CAM ARRAYS
    # ------------------------------------------------------------

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_xception_cam.npy"
        ),
        x_cam
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_efficientnet_cam.npy"
        ),
        e_cam
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{base_name}_fused_cam.npy"
        ),
        fused_cam
    )

    # ------------------------------------------------------------
    # JSON REPORT
    # ------------------------------------------------------------

    report = {

        "image": os.path.abspath(image_path),

        "image_size": {
            "width": width,
            "height": height
        },

        "device": str(DEVICE),

        "class_mapping": {
            "0": "FAKE",
            "1": "REAL"
        },

        "fusion": {
            "xception_weight": XCEPTION_WEIGHT,
            "efficientnet_weight": EFFICIENTNET_WEIGHT
        },

        "models": {

            "xception": {
                "fake_probability": x_fake,
                "real_probability": x_real,
                "prediction": (
                    "FAKE"
                    if x_fake >= x_real
                    else "REAL"
                )
            },

            "efficientnet_b4": {
                "fake_probability": e_fake,
                "real_probability": e_real,
                "prediction": (
                    "FAKE"
                    if e_fake >= e_real
                    else "REAL"
                )
            }
        },

        "fusion_result": {

            "fake_score": fused_fake,
            "real_score": fused_real,
            "prediction": prediction_name,
            "confidence": confidence
        },

        "gradcam": {

            "target_class": prediction_name,

            "xception_target_layer": (
                x_layer.__class__.__name__
            ),

            "efficientnet_target_layer": (
                e_layer.__class__.__name__
            ),

            "face_analysis": {

                "faces_detected":
                    len(faces),

                "xception_face_attention":
                    x_face_stats[
                        "face_attention_ratio"
                    ],

                "efficientnet_face_attention":
                    e_face_stats[
                        "face_attention_ratio"
                    ],

                "fused_face_attention":
                    fused_face_stats[
                        "face_attention_ratio"
                    ]
            }
        },

        "outputs": {

            "original":
                f"{base_name}_original.jpg",

            "xception":
                f"{base_name}_xception_gradcam.jpg",

            "efficientnet":
                f"{base_name}_efficientnet_gradcam.jpg",

            "fused":
                f"{base_name}_fused_gradcam.jpg",

            "face_focused":
                f"{base_name}_face_focused_gradcam.jpg",

            "face_detection":
                f"{base_name}_face_detection.jpg"
        }
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_gradcam_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print(
        f"✓ Report saved: {json_path}"
    )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    section(
        "FORENSIQ EXPLAINABILITY SUMMARY"
    )

    print(
        f"Xception fake score     : {x_fake:.4f}"
    )

    print(
        f"EfficientNet fake score : {e_fake:.4f}"
    )

    print(
        f"Fused fake score        : {fused_fake:.4f}"
    )

    print(
        f"Fused real score        : {fused_real:.4f}"
    )

    print()
    print(
        f"FINAL PREDICTION        : {prediction_name}"
    )

    print(
        f"CONFIDENCE              : "
        f"{confidence * 100:.2f}%"
    )

    print()
    print(
        "Grad-CAM target class   : "
        f"{prediction_name}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Grad-CAM highlights regions used by the model "
        "for the selected class."
    )

    print(
        "It does NOT directly label a highlighted region "
        "as fake or authentic."
    )

    print()
    print(
        f"Results directory:\n{OUTPUT_DIR}"
    )

    section(
        "ANALYSIS COMPLETE"
    )


# ================================================================
# COMMAND-LINE ENTRY
# ================================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            'python inference\\gradcam_fusion.py '
            '"C:\\path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        analyze(image_path)

    except KeyboardInterrupt:

        print()
        print(
            "Analysis interrupted by user."
        )

        sys.exit(1)

    except Exception as e:

        print()
        line()
        print("ERROR")
        line()
        print(str(e))
        line()

        import traceback

        traceback.print_exc()

        sys.exit(1)