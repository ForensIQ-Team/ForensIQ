import os
import sys
import json
import torch
import timm
from PIL import Image
from torchvision import transforms


# ============================================================
# FORENSIQ EXTERNAL IMAGE PREDICTOR
# XCEPTION + EFFICIENTNET-B4 FUSION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

XCEPTION_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_xception.pth"
)

EFFICIENTNET_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "best_efficientnet_b4.pth"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "prediction_results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CURRENT VALIDATED FUSION SETTINGS
# ============================================================

XCEPTION_WEIGHT = 0.55
EFFICIENTNET_WEIGHT = 0.45

DECISION_THRESHOLD = 0.45


# ============================================================
# PREPROCESSING
# MUST MATCH TRAINING
# ============================================================

IMAGE_SIZE = 299

TRANSFORM = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# HEADER
# ============================================================

def header(text):

    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # Handle different checkpoint formats.

    if isinstance(checkpoint, dict):

        for key in [
            "model_state_dict",
            "state_dict",
            "model",
            "weights"
        ]:

            if key in checkpoint:

                if isinstance(
                    checkpoint[key],
                    dict
                ):

                    checkpoint = checkpoint[key]

                break

    # Remove DataParallel prefix if present.

    cleaned_state_dict = {}

    for key, value in checkpoint.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:

        print(
            f"Warning: {len(missing)} missing keys"
        )

    if unexpected:

        print(
            f"Warning: {len(unexpected)} unexpected keys"
        )

    return model


# ============================================================
# LOAD XCEPTION
# ============================================================

def load_xception():

    print("Loading Xception...")

    model = timm.create_model(
        "xception",
        pretrained=False,
        num_classes=2
    )

    model = load_checkpoint(
        model,
        XCEPTION_PATH
    )

    model.to(DEVICE)
    model.eval()

    print("✓ Xception loaded")

    return model


# ============================================================
# LOAD EFFICIENTNET
# ============================================================

def load_efficientnet():

    print("Loading EfficientNet-B4...")

    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=2
    )

    model = load_checkpoint(
        model,
        EFFICIENTNET_PATH
    )

    model.to(DEVICE)
    model.eval()

    print("✓ EfficientNet-B4 loaded")

    return model


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model, image_tensor):

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # IMPORTANT:
    # 0 = FAKE
    # 1 = REAL

    fake_probability = float(
        probabilities[0].item()
    )

    real_probability = float(
        probabilities[1].item()
    )

    if fake_probability >= real_probability:

        prediction = "FAKE"

    else:

        prediction = "REAL"

    return (
        fake_probability,
        real_probability,
        prediction
    )


# ============================================================
# MAIN
# ============================================================

def analyze_image(image_path):

    header(
        "FORENSIQ EXTERNAL IMAGE ANALYSIS"
    )

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if not os.path.isfile(image_path):

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    print(
        f"Image: {image_path}"
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    print(
        f"\nOriginal size: "
        f"{image.width} x {image.height}"
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # LOAD BOTH MODELS
    # --------------------------------------------------------

    header(
        "LOADING BOTH FORENSIQ MODELS"
    )

    xception = load_xception()

    efficientnet = load_efficientnet()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image_tensor = TRANSFORM(
        image
    ).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # XCEPTION
    # --------------------------------------------------------

    header(
        "XCEPTION INFERENCE"
    )

    x_fake, x_real, x_prediction = predict(
        xception,
        image_tensor
    )

    print(
        f"Fake probability : "
        f"{x_fake:.4f}"
    )

    print(
        f"Real probability : "
        f"{x_real:.4f}"
    )

    print(
        f"Prediction       : "
        f"{x_prediction}"
    )

    # --------------------------------------------------------
    # EFFICIENTNET
    # --------------------------------------------------------

    header(
        "EFFICIENTNET-B4 INFERENCE"
    )

    e_fake, e_real, e_prediction = predict(
        efficientnet,
        image_tensor
    )

    print(
        f"Fake probability : "
        f"{e_fake:.4f}"
    )

    print(
        f"Real probability : "
        f"{e_real:.4f}"
    )

    print(
        f"Prediction       : "
        f"{e_prediction}"
    )

    # --------------------------------------------------------
    # TWO-MODEL FUSION
    # --------------------------------------------------------

    header(
        "TWO-MODEL SCORE FUSION"
    )

    fused_fake = (
        XCEPTION_WEIGHT * x_fake
        +
        EFFICIENTNET_WEIGHT * e_fake
    )

    fused_real = (
        XCEPTION_WEIGHT * x_real
        +
        EFFICIENTNET_WEIGHT * e_real
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    final_prediction = (
        "FAKE"
        if fused_fake >= DECISION_THRESHOLD
        else "REAL"
    )

    if final_prediction == "FAKE":

        confidence = fused_fake

    else:

        confidence = fused_real

    # --------------------------------------------------------
    # DISPLAY FUSION
    # --------------------------------------------------------

    print(
        f"Xception contribution     : "
        f"{XCEPTION_WEIGHT:.2f}"
    )

    print(
        f"EfficientNet contribution : "
        f"{EFFICIENTNET_WEIGHT:.2f}"
    )

    print(
        f"Decision threshold        : "
        f"{DECISION_THRESHOLD:.2f}"
    )

    print(
        f"\nXception fake score       : "
        f"{x_fake:.4f}"
    )

    print(
        f"EfficientNet fake score   : "
        f"{e_fake:.4f}"
    )

    print(
        f"\nFUSED FAKE SCORE           : "
        f"{fused_fake:.4f}"
    )

    print(
        f"FUSED REAL SCORE           : "
        f"{fused_real:.4f}"
    )

    print(
        f"\nFINAL PREDICTION            : "
        f"{final_prediction}"
    )

    print(
        f"CONFIDENCE                  : "
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # MODEL AGREEMENT
    # --------------------------------------------------------

    header(
        "MODEL AGREEMENT"
    )

    if x_prediction == e_prediction:

        agreement = True

        print(
            f"✓ BOTH MODELS AGREE: "
            f"{x_prediction}"
        )

    else:

        agreement = False

        print(
            "⚠ MODELS DISAGREE"
        )

        print(
            f"Xception     : {x_prediction}"
        )

        print(
            f"EfficientNet : {e_prediction}"
        )

    # --------------------------------------------------------
    # FORENSIQ SUMMARY
    # --------------------------------------------------------

    header(
        "FORENSIQ SUMMARY"
    )

    if final_prediction == "FAKE":

        print(
            "The fused ForensIQ model detected "
            "manipulation indicators consistent "
            "with a FAKE image."
        )

    else:

        print(
            "The fused ForensIQ model classified "
            "the image as REAL."
        )

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    result = {

        "image": image_path,

        "models": {

            "xception": {

                "fake_probability": x_fake,

                "real_probability": x_real,

                "prediction": x_prediction,

                "weight": XCEPTION_WEIGHT
            },

            "efficientnet_b4": {

                "fake_probability": e_fake,

                "real_probability": e_real,

                "prediction": e_prediction,

                "weight": EFFICIENTNET_WEIGHT
            }
        },

        "fusion": {

            "fake_probability": fused_fake,

            "real_probability": fused_real,

            "prediction": final_prediction,

            "confidence": confidence,

            "threshold": DECISION_THRESHOLD
        },

        "model_agreement": agreement
    }

    result_path = os.path.join(
        RESULT_DIR,
        f"{filename}_prediction.json"
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
        "ANALYSIS COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            'python inference\\predict_image.py '
            '"C:\\path\\to\\image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    try:

        analyze_image(
            image_path
        )

    except KeyboardInterrupt:

        print(
            "\n\nAnalysis interrupted."
        )

        sys.exit(1)

    except Exception as error:

        print(
            "\n" + "=" * 60
        )

        print(
            "FORENSIQ PREDICTION ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )

        sys.exit(1)