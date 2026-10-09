"""
ForensIQ - Grad-CAM V3 (Face-crop-aware)

Key change vs V2:
    * Applies the SAME face detection + crop pipeline as inference
    * Runs Grad-CAM on the FACE CROP, not the full image
    * Maps the heatmap back to original-image coordinates for display
    * Also saves a crop-only heatmap (what the model actually sees)

This makes the heatmap explain the model's actual input.
If the heatmap is still wrong after this, the model itself
is the problem — not the visualization.
"""

import os
import sys
import io

# Force UTF-8 on Windows so Unicode print symbols don't fail
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure") and not sys.stdout.closed:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure") and not sys.stderr.closed:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import transforms

import timm

from pytorch_grad_cam import (
    GradCAMPlusPlus, GradCAM, HiResCAM, EigenGradCAM, LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ----------------------------------------------------------------
# Import the face crop pipeline from the predictor (do NOT duplicate)
# Locates predict_v2_facecrop.py robustly regardless of launch dir.
# ----------------------------------------------------------------
try:
    from pathlib import Path as _Path

    _HERE = _Path(__file__).resolve().parent

    _INFERENCE_CANDIDATES = [
        _HERE / "inference",                              # script inside ForensIQ-ML/  ← primary
        _HERE / "ForensIQ-ML" / "inference",              # script at project root
        _HERE.parent / "inference",                       # fallback
        _HERE.parent / "ForensIQ-ML" / "inference",       # fallback
        _Path(r"C:\Users\HARSHIL\Downloads\forensiq n\ForensIQ-ML\inference"),  # absolute fallback
    ]

    _INFERENCE_DIR = None
    for _cand in _INFERENCE_CANDIDATES:
        if _cand.is_dir() and (_cand / "predict_v2_facecrop.py").exists():
            _INFERENCE_DIR = _cand
            break

    if _INFERENCE_DIR is None:
        raise ModuleNotFoundError(
            "Could not locate predict_v2_facecrop.py. "
            "Searched: " + ", ".join(str(c) for c in _INFERENCE_CANDIDATES)
        )

    if str(_INFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(_INFERENCE_DIR))

    # Also expose the ForensIQ-ML root so the module can be imported by name
    _FORENSIQ_ML_ROOT = _INFERENCE_DIR.parent
    if str(_FORENSIQ_ML_ROOT) not in sys.path:
        sys.path.insert(0, str(_FORENSIQ_ML_ROOT))

    import predict_v2_facecrop as pipeline
    print(f"[OK] Loaded predict_v2_facecrop from: {pipeline.__file__}")
    HAS_PIPELINE = True
except Exception as e:
    HAS_PIPELINE = False
    pipeline = None
    print(f"[warn] Could not import predict_v2_facecrop: {e}")
    print("[warn] Falling back to full-image Grad-CAM.")


# ================================================================
# CONFIGURATION
# ================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "gradcam_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAKE_CLASS = 0
REAL_CLASS = 1

# Which CAM method to use. "gradcam++" is a good default.
# Options: gradcam, gradcam++, hirescam, eigengradcam, layercam
CAM_METHOD = "gradcam++"

# Always explain FAKE. (Your forensic use case.)
EXPLAIN_CLASS = "FAKE"

# Colormap — inferno reads better on faces than jet.
FORENSIC_COLORMAP = cv2.COLORMAP_INFERNO

# Overlay blend strength.
CAM_ALPHA = 0.45

# Target layer heuristics — skip classifier head convs.
SKIP_TOKENS = ("head", "classifier", "fc", "output", "proj", "pool")


CHECKPOINTS = {
    "xception":     os.path.join(CHECKPOINT_DIR, "best_xception_v2.pth"),
    "efficientnet": os.path.join(CHECKPOINT_DIR, "best_efficientnet_b4_v2.pth"),
    "vit":          os.path.join(CHECKPOINT_DIR, "best_vit_v2_1.pth"),
}

# Input sizes MUST match the predictor.
IMAGE_SIZES = {
    "xception":     299,
    "efficientnet": 299,
    "vit":          224,
}


# ================================================================
# MODEL CREATION
# ================================================================

def create_model(model_name):
    print()
    print("=" * 70)
    print(f"Loading {model_name.upper()} model")
    print("=" * 70)

    if model_name == "xception":
        model = timm.create_model("legacy_xception", pretrained=False, num_classes=2)
    elif model_name == "efficientnet":
        model = timm.create_model("tf_efficientnet_b4", pretrained=False, num_classes=2)
    elif model_name == "vit":
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=2)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt_path = CHECKPOINTS[model_name]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        for prefix in ("module.", "_orig_mod.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"Checkpoint : {os.path.basename(ckpt_path)}")
    print(f"Missing keys    : {len(missing)}")
    print(f"Unexpected keys : {len(unexpected)}")

    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")
    return model


# ================================================================
# TARGET LAYER SELECTION
# ================================================================

def get_target_layer_cnn(model):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            candidates.append((name, module))

    if not candidates:
        raise RuntimeError("No Conv2d found.")

    filtered = [
        (n, m) for (n, m) in candidates
        if not any(tok in n.lower() for tok in SKIP_TOKENS)
    ] or candidates

    name, layer = filtered[-1]
    print(f"Target layer (CNN) : {name}")
    return layer


def get_target_layer_vit(model):
    layer = model.blocks[-1].norm1
    print("Target layer (ViT) : blocks[-1].norm1")
    return layer


def make_vit_reshape(input_size):
    grid = input_size // 16

    def _reshape(t):
        # t: (B, 1+grid*grid, C)
        cls = t[:, 0, :]
        patches = t[:, 1:, :]
        patches = patches.reshape(t.size(0), grid, grid, t.size(-1))
        # Keep only patch tokens as spatial map
        return patches.permute(0, 3, 1, 2)
    return _reshape


# ================================================================
# PREPROCESSING
# ================================================================

def get_transform(model_name):
    size = IMAGE_SIZES[model_name]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ================================================================
# FACE CROP  (uses the project pipeline)
# ================================================================

def get_face_crop(image_pil):
    """
    Return (crop_pil, crop_box) where crop_box is in ORIGINAL image coords.
    If pipeline isn't available or no face found, returns (full image, (0,0,W,H)).
    """
    if not HAS_PIPELINE:
        w, h = image_pil.size
        return image_pil.copy(), (0, 0, w, h)

    try:
        face_result = pipeline.create_face_crop(image_pil)
        crop = face_result.get("crop", image_pil)
        box = face_result.get("box", None)
        if box is None:
            w, h = image_pil.size
            box = (0, 0, w, h)
        return crop, tuple(int(v) for v in box)
    except Exception as e:
        print(f"[warn] Face crop failed, using full image: {e}")
        w, h = image_pil.size
        return image_pil.copy(), (0, 0, w, h)


# ================================================================
# PREDICTION
# ================================================================

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    fake_p = probs[0, FAKE_CLASS].item()
    real_p = probs[0, REAL_CLASS].item()
    pred = int(torch.argmax(probs, dim=1).item())
    label = "FAKE" if pred == FAKE_CLASS else "REAL"
    return label, fake_p, real_p


# ================================================================
# CAM BUILDERS
# ================================================================

def get_cam_builder(name):
    mapping = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "hirescam": HiResCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
    }
    if name not in mapping:
        raise ValueError(f"Unknown CAM method: {name}")
    return mapping[name]


# ================================================================
# OVERLAY  (crop → original)
# ================================================================

def build_overlay_on_original(original_pil, grayscale_cam, crop_box,
                              alpha=CAM_ALPHA):
    """
    Takes a CAM in CROP coordinates and pastes it back onto the original
    image, so the heatmap is in original-image coordinates.
    """
    orig_w, orig_h = original_pil.size
    x1, y1, x2, y2 = crop_box
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)

    # Normalize CAM to [0, 1]
    cam = grayscale_cam.astype(np.float32)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Upsample CAM to crop size
    cam_crop = cv2.resize(cam, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    cam_crop = cv2.GaussianBlur(cam_crop, (0, 0),
                                sigmaX=max(cw, ch) * 0.01)

    # Colorize
    cam_uint8 = np.uint8(255 * np.clip(cam_crop, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, FORENSIC_COLORMAP)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Paste heatmap into a full-size canvas
    full_heatmap = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    full_heatmap[y1:y2, x1:x2] = heatmap_rgb
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = 255

    orig_np = np.array(original_pil).astype(np.float32) / 255.0
    heat_np = full_heatmap.astype(np.float32) / 255.0
    mask_f = (full_mask.astype(np.float32) / 255.0)[..., None]

    # Blend only inside the crop area
    blended = orig_np * (1 - alpha * mask_f) + heat_np * (alpha * mask_f)
    blended = np.clip(blended, 0, 1)

    return np.uint8(255 * blended), heatmap_bgr, full_mask


# ================================================================
# SUSPICIOUS REGION  (in crop coords, then map to original)
# ================================================================

def get_suspicious_region_on_original(grayscale_cam, crop_box, orig_w, orig_h):
    cam = grayscale_cam.astype(np.float32)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    threshold = np.percentile(cam, 88)
    mask = (cam >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    cam_h, cam_w = cam.shape
    x1, y1, x2, y2 = crop_box
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)

    # Crop coords → original coords
    ox = x1 + int(x * cw / cam_w)
    oy = y1 + int(y * ch / cam_h)
    ow = int(w * cw / cam_w)
    oh = int(h * ch / cam_h)

    ox = max(0, ox); oy = max(0, oy)
    ow = min(ow, orig_w - ox); oh = min(oh, orig_h - oy)
    return (ox, oy, ow, oh)


# ================================================================
# MAIN RUNNER
# ================================================================

def run(model_name, image_path):
    print()
    print("=" * 70)
    print(f"GRAD-CAM V3 — {model_name.upper()}")
    print("=" * 70)

    model = create_model(model_name)

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # ---- 1. FACE CROP (same pipeline as inference) ----------------
    crop_pil, crop_box = get_face_crop(image)
    print(f"Face crop box : {crop_box}")
    print(f"Crop size     : {crop_pil.size}")

    # ---- 2. TRANSFORM + PREDICT (on crop) -------------------------
    transform = get_transform(model_name)
    crop_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

    label, fake_p, real_p = predict(model, crop_tensor)
    print(f"Prediction    : {label} (fake={fake_p:.4f}, real={real_p:.4f})")

    # ---- 3. TARGET LAYER -----------------------------------------
    if model_name == "vit":
        target_layer = get_target_layer_vit(model)
        reshape = make_vit_reshape(IMAGE_SIZES["vit"])
    else:
        target_layer = get_target_layer_cnn(model)
        reshape = None

    target_class = FAKE_CLASS if EXPLAIN_CLASS == "FAKE" else REAL_CLASS
    targets = [ClassifierOutputTarget(target_class)]

    # ---- 4. CAM on crop ------------------------------------------
    cam_builder = get_cam_builder(CAM_METHOD)
    print(f"CAM method    : {CAM_METHOD}")

    with cam_builder(model=model, target_layers=[target_layer],
                     reshape_transform=reshape) as cam:
        grayscale_cam = cam(
            input_tensor=crop_tensor,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=False,
        )[0]

    print(f"CAM variance  : {float(np.var(grayscale_cam)):.6f}")

    # ---- 5. OVERLAY BACK ON ORIGINAL -----------------------------
    overlay_rgb, heatmap_crop_bgr, full_mask = build_overlay_on_original(
        image, grayscale_cam, crop_box, alpha=CAM_ALPHA
    )

    # Draw bounding box on the overlay
    bbox = get_suspicious_region_on_original(
        grayscale_cam, crop_box, orig_w, orig_h
    )
    if bbox:
        bx, by, bw, bh = bbox
        cv2.rectangle(overlay_rgb, (bx, by), (bx + bw, by + bh),
                      (255, 255, 0), 3)
        cv2.putText(overlay_rgb, "SUSPICIOUS REGION",
                    (bx, max(30, by - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(overlay_rgb, (crop_box[0], crop_box[1]),
                  (crop_box[2], crop_box[3]), (0, 255, 0), 2)

    # ---- 6. SAVE --------------------------------------------------
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, model_name, image_name)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "01_original.jpg"),
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(out_dir, "02_overlay_on_original.jpg"),
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    # Heatmap-only (crop coords) — this is what you send to the LLM
    heatmap_only_bgr = cv2.resize(
        heatmap_crop_bgr,
        (crop_pil.size[0], crop_pil.size[1]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    cv2.imwrite(os.path.join(out_dir, "03_heatmap_only_crop.jpg"),
                heatmap_only_bgr)

    # The crop itself
    crop_pil.save(os.path.join(out_dir, "04_face_crop.jpg"), quality=95)

    # Crop with heatmap blended (in crop coords)
    crop_np = np.array(crop_pil).astype(np.float32) / 255.0
    cam_crop = cv2.resize(grayscale_cam, crop_pil.size,
                          interpolation=cv2.INTER_LANCZOS4)
    cam_crop_n = cam_crop - cam_crop.min()
    if cam_crop_n.max() > 0:
        cam_crop_n /= cam_crop_n.max()
    hm = cv2.applyColorMap(np.uint8(255 * cam_crop_n), FORENSIC_COLORMAP)
    hm_rgb = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crop_overlay = np.clip(crop_np * (1 - CAM_ALPHA) + hm_rgb * CAM_ALPHA, 0, 1)
    cv2.imwrite(os.path.join(out_dir, "05_crop_with_heatmap.jpg"),
                cv2.cvtColor(np.uint8(255 * crop_overlay), cv2.COLOR_RGB2BGR))

    print()
    print(f"Saved to: {out_dir}")

    return {
        "model": model_name,
        "prediction": label,
        "fake_probability": fake_p,
        "real_probability": real_p,
        "crop_box": crop_box,
        "bbox": bbox,
        "cam_variance": float(np.var(grayscale_cam)),
        "output_dir": out_dir,
    }


# ================================================================
# CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True,
                        choices=["xception", "efficientnet", "vit", "all"])
    parser.add_argument("--report", action="store_true",
                        help="Generate a Groq-powered ForensIQ HTML report after Grad-CAM "
                             "(requires --model xception or all)")
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("ForensIQ Grad-CAM V3 (Face-crop-aware)")
    print("=" * 70)
    print(f"Device : {DEVICE}")
    print(f"Image  : {args.image}")
    print(f"Model  : {args.model}")
    print(f"Report : {args.report}")

    if args.report and args.model in ("vit", "efficientnet"):
        print()
        print("[warn] --report requires --model xception or --model all.")
        print("[warn] Skipping report generation. Re-run with --model xception.")
        args.report = False

    models = ["xception", "efficientnet", "vit"] if args.model == "all" else [args.model]

    results = []
    for m in models:
        try:
            results.append(run(m, args.image))
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {m}: {e}")
            traceback.print_exc()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    for r in results:
        print(f"{r['model']:14s} | {r['prediction']:5s} | "
              f"fake={r['fake_probability']:.4f} | "
              f"bbox={r['bbox']}")

    # ---- Report generation ----------------------------------------
    if args.report:
        xception_result = next((r for r in results if r["model"] == "xception"), None)
        if xception_result is None:
            print("[warn] No Xception result found — skipping report.")
            return

        try:
            from pathlib import Path as _Path
            import sys as _sys

            _report_gen_dir = _Path(__file__).resolve().parent
            if str(_report_gen_dir) not in _sys.path:
                _sys.path.insert(0, str(_report_gen_dir))

            from report_generator import ForensicReportGenerator
            from report_renderer  import render_html

            image_path = _Path(args.image).resolve()
            output_dir = (
                _Path(__file__).resolve().parent
                / "gradcam_results" / "xception" / image_path.stem
                / "report"
            )

            # The overlay was already produced by this very run.
            # Pass it as precomputed_overlay so the generator does NOT launch
            # a redundant second Grad-CAM subprocess.
            _xception_out_dir = _Path(xception_result["output_dir"])
            _precomp_overlay  = _xception_out_dir / "02_overlay_on_original.jpg"
            if not _precomp_overlay.exists():
                _precomp_overlay = None   # fall back to re-running it
                print("[warn] Grad-CAM overlay file not found; generator will re-run Grad-CAM.")

            gen    = ForensicReportGenerator()
            report = gen.generate_image_report(
                image_path, output_dir,
                precomputed_overlay=_precomp_overlay,
            )
            html_path = render_html(report, output_dir)

            print()
            print("=" * 70)
            print("REPORT GENERATED")
            print("=" * 70)
            print(f"report.json : {output_dir / 'report.json'}")
            print(f"report.html : {html_path}")

        except Exception as _rep_err:
            import traceback as _tb
            print(f"\n[ERROR] Report generation failed: {_rep_err}")
            _tb.print_exc()


if __name__ == "__main__":
    main()