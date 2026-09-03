# training/ensemble_xception_efficientnet.py

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 0

IMAGE_SIZE = 299

# IMPORTANT:
# Neither model is allowed to disappear from the ensemble.
MIN_MODEL_WEIGHT = 0.20
MAX_MODEL_WEIGHT = 0.80

WEIGHT_STEP = 0.05

# Threshold search
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = ROOT / "dataset" / "split" / "train.csv"
VAL_CSV = ROOT / "dataset" / "split" / "val.csv"
TEST_CSV = ROOT / "dataset" / "split" / "test.csv"

XCEPTION_CHECKPOINT = (
    ROOT / "checkpoints" / "best_xception.pth"
)

EFFICIENTNET_CHECKPOINT = (
    ROOT / "checkpoints" / "best_efficientnet_b4.pth"
)

OUTPUT_DIR = ROOT / "ensemble_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = (
    OUTPUT_DIR /
    "xception_efficientnet_fusion_results.json"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("\n" + "=" * 56)
print("FORENSIQ TWO-MODEL SCORE FUSION")
print("XCEPTION + EFFICIENTNET-B4")
print("=" * 56)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Batch size: {BATCH_SIZE}")
print(f"DataLoader workers: {NUM_WORKERS}")

print("\nClass mapping:")
print("0 = FAKE")
print("1 = REAL")


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label(value):

    value = str(value).strip().lower()

    if value in {
        "fake",
        "0",
        "false",
        "manipulated",
        "deepfake",
    }:
        return 0

    if value in {
        "real",
        "1",
        "true",
        "original",
        "genuine",
    }:
        return 1

    raise ValueError(
        f"Unknown label encountered: {value}"
    )


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        if "face_path" not in self.data.columns:
            raise RuntimeError(
                f"{csv_file} does not contain face_path column."
            )

        if "label" not in self.data.columns:
            raise RuntimeError(
                f"{csv_file} does not contain label column."
            )

        self.labels = np.array(
            [
                convert_label(x)
                for x in self.data["label"]
            ],
            dtype=np.int64,
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = ROOT / str(
            row["face_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = int(self.labels[index])

        return image, label


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 56)
print("CHECKING REQUIRED FILES")
print("=" * 56)

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    XCEPTION_CHECKPOINT,
    EFFICIENTNET_CHECKPOINT,
]

for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file missing:\n{file_path}"
        )

    print(f"✓ {file_path.name}")


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 56)
print("LOADING DATASETS")
print("=" * 56)

val_dataset = FaceDataset(VAL_CSV)
test_dataset = FaceDataset(TEST_CSV)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print(
    f"VAL  dataset: {len(val_dataset)} images"
)

print(
    f"TEST dataset: {len(test_dataset)} images"
)


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(name, dataset):

    labels = dataset.labels

    fake_count = int(
        np.sum(labels == 0)
    )

    real_count = int(
        np.sum(labels == 1)
    )

    print(
        f"{name:<5} | "
        f"FAKE: {fake_count:5d} | "
        f"REAL: {real_count:5d} | "
        f"TOTAL: {len(labels):5d}"
    )


print("\n" + "=" * 56)
print("DATASET DISTRIBUTION")
print("=" * 56)

print_distribution("VAL", val_dataset)
print_distribution("TEST", test_dataset)


# ============================================================
# MODEL CREATION
# ============================================================

def create_xception():

    model = timm.create_model(
        "xception",
        pretrained=False,
        num_classes=2,
    )

    return model


def create_efficientnet():

    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=2,
    )

    return model


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        possible_keys = [
            "model_state_dict",
            "state_dict",
            "model",
        ]

        for key in possible_keys:

            if key in checkpoint:

                state = checkpoint[key]

                if isinstance(state, dict):
                    return state

        # Direct state dictionary
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            return checkpoint

    raise RuntimeError(
        "Unable to locate model state dictionary."
    )


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key.startswith("model."):
            new_key = new_key[6:]

        cleaned[new_key] = value

    return cleaned


def load_model(model, checkpoint_path, model_name):

    print(
        f"\nLoading {model_name}..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    state_dict = clean_state_dict(
        state_dict
    )

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if len(result.missing_keys) > 0:

        print(
            f"WARNING: {model_name} has "
            f"{len(result.missing_keys)} missing keys."
        )

        print(
            "First missing keys:",
            result.missing_keys[:5]
        )

    if len(result.unexpected_keys) > 0:

        print(
            f"WARNING: {model_name} has "
            f"{len(result.unexpected_keys)} unexpected keys."
        )

        print(
            "First unexpected keys:",
            result.unexpected_keys[:5]
        )

    model.to(device)
    model.eval()

    print(
        f"✓ {model_name} loaded"
    )

    return model


# ============================================================
# LOAD BOTH MODELS
# ============================================================

print("\n" + "=" * 56)
print("LOADING XCEPTION")
print("=" * 56)

xception = load_model(
    create_xception(),
    XCEPTION_CHECKPOINT,
    "Xception",
)


print("\n" + "=" * 56)
print("LOADING EFFICIENTNET-B4")
print("=" * 56)

efficientnet = load_model(
    create_efficientnet(),
    EFFICIENTNET_CHECKPOINT,
    "EfficientNet-B4",
)


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def predict_model(model, loader, model_name):

    probabilities = []
    labels = []

    total_batches = len(loader)

    print(
        f"\nRunning {model_name} inference..."
    )

    for batch_index, (images, batch_labels) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1,
        )

        # Class 0 = FAKE
        fake_probability = probs[:, 0]

        probabilities.extend(
            fake_probability.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        labels.extend(
            batch_labels.numpy().tolist()
        )

        if (
            batch_index == 1
            or batch_index % 50 == 0
            or batch_index == total_batches
        ):

            print(
                f"  Batch "
                f"{batch_index}/{total_batches}"
            )

    return (
        np.asarray(probabilities),
        np.asarray(labels),
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    fake_probability,
    labels,
    threshold,
):

    # Fake = probability >= threshold
    predictions = (
        fake_probability >= threshold
    ).astype(int)

    # Our labels are:
    # 0 = FAKE
    # 1 = REAL

    # Convert prediction format:
    # 1 = FAKE
    # 0 = REAL

    # This makes metric reporting easier.

    predicted_fake = (
        fake_probability >= threshold
    ).astype(int)

    actual_fake = (
        labels == 0
    ).astype(int)

    accuracy = accuracy_score(
        actual_fake,
        predicted_fake,
    )

    precision = precision_score(
        actual_fake,
        predicted_fake,
        zero_division=0,
    )

    recall = recall_score(
        actual_fake,
        predicted_fake,
        zero_division=0,
    )

    f1 = f1_score(
        actual_fake,
        predicted_fake,
        zero_division=0,
    )

    auc = roc_auc_score(
        actual_fake,
        fake_probability,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 56)
print("VALIDATION INFERENCE")
print("=" * 56)

x_val, y_val = predict_model(
    xception,
    val_loader,
    "Xception",
)

e_val, y_val_2 = predict_model(
    efficientnet,
    val_loader,
    "EfficientNet-B4",
)

if not np.array_equal(
    y_val,
    y_val_2,
):

    raise RuntimeError(
        "Validation label ordering mismatch "
        "between models."
    )


# ============================================================
# SANITY CHECK
# ============================================================

if np.isnan(x_val).any():
    raise RuntimeError(
        "NaN detected in Xception predictions."
    )

if np.isnan(e_val).any():
    raise RuntimeError(
        "NaN detected in EfficientNet predictions."
    )


# ============================================================
# INDIVIDUAL PERFORMANCE
# ============================================================

print("\n" + "=" * 56)
print("INDIVIDUAL VALIDATION PERFORMANCE")
print("=" * 56)

x_val_metrics = calculate_metrics(
    x_val,
    y_val,
    0.50,
)

e_val_metrics = calculate_metrics(
    e_val,
    y_val,
    0.50,
)

print("\nXCEPTION")

print(
    f"Accuracy  : "
    f"{x_val_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{x_val_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{x_val_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{x_val_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{x_val_metrics['auc']:.4f}"
)


print("\nEFFICIENTNET-B4")

print(
    f"Accuracy  : "
    f"{e_val_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{e_val_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{e_val_metrics['recall']:.4f}"
)

print(
    f"F1        : "
    f"{e_val_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{e_val_metrics['auc']:.4f}"
)


# ============================================================
# CORRELATION / DIVERSITY
# ============================================================

correlation = float(
    np.corrcoef(
        x_val,
        e_val,
    )[0, 1]
)

print("\n" + "=" * 56)
print("MODEL SCORE RELATIONSHIP")
print("=" * 56)

print(
    f"Prediction correlation : "
    f"{correlation:.4f}"
)

if correlation < 0.95:

    print(
        "✓ Models provide useful prediction diversity."
    )

else:

    print(
        "NOTE: Models produce highly similar scores."
    )


# ============================================================
# REAL SCORE FUSION
# ============================================================

print("\n" + "=" * 56)
print("TWO-MODEL SCORE FUSION SEARCH")
print("=" * 56)

print(
    f"Xception weight range     : "
    f"{MIN_MODEL_WEIGHT:.2f} → "
    f"{MAX_MODEL_WEIGHT:.2f}"
)

print(
    f"EfficientNet weight range: "
    f"{1 - MAX_MODEL_WEIGHT:.2f} → "
    f"{1 - MIN_MODEL_WEIGHT:.2f}"
)

print(
    "\nIMPORTANT: "
    "Neither model can receive 0% weight."
)

best_fusion = None

weights = np.arange(
    MIN_MODEL_WEIGHT,
    MAX_MODEL_WEIGHT + 0.0001,
    WEIGHT_STEP,
)

for x_weight in weights:

    x_weight = round(
        float(x_weight),
        2,
    )

    e_weight = round(
        1.0 - x_weight,
        2,
    )

    fused_val = (
        x_weight * x_val
        +
        e_weight * e_val
    )

    auc = roc_auc_score(
        y_val == 0,
        fused_val,
    )

    # Optimize threshold for F1
    best_threshold_for_weight = 0.50
    best_f1_for_weight = -1.0

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + 0.0001,
        THRESHOLD_STEP,
    )

    for threshold in thresholds:

        predicted_fake = (
            fused_val >= threshold
        ).astype(int)

        actual_fake = (
            y_val == 0
        ).astype(int)

        f1 = f1_score(
            actual_fake,
            predicted_fake,
            zero_division=0,
        )

        if f1 > best_f1_for_weight:

            best_f1_for_weight = f1

            best_threshold_for_weight = (
                float(
                    round(
                        float(threshold),
                        2,
                    )
                )
            )

    print(
        f"Xception={x_weight:.2f} | "
        f"EfficientNet={e_weight:.2f} | "
        f"AUC={auc:.4f} | "
        f"Best F1={best_f1_for_weight:.4f} | "
        f"Threshold={best_threshold_for_weight:.2f}"
    )

    # PRIMARY OBJECTIVE = AUC
    # SECONDARY OBJECTIVE = F1

    candidate = (
        auc,
        best_f1_for_weight,
        x_weight,
        e_weight,
        best_threshold_for_weight,
    )

    if best_fusion is None:

        best_fusion = candidate

    else:

        if (
            candidate[0] > best_fusion[0]
            or (
                abs(
                    candidate[0]
                    - best_fusion[0]
                ) < 1e-10
                and candidate[1]
                > best_fusion[1]
            )
        ):

            best_fusion = candidate


# ============================================================
# BEST FUSION
# ============================================================

(
    best_val_auc,
    best_val_f1,
    best_x_weight,
    best_e_weight,
    best_threshold,
) = best_fusion


print("\n" + "=" * 56)
print("BEST TWO-MODEL FUSION")
print("=" * 56)

print(
    f"Xception weight     : "
    f"{best_x_weight:.2f}"
)

print(
    f"EfficientNet weight : "
    f"{best_e_weight:.2f}"
)

print(
    f"Validation AUC      : "
    f"{best_val_auc:.4f}"
)

print(
    f"Validation F1       : "
    f"{best_val_f1:.4f}"
)

print(
    f"Decision threshold  : "
    f"{best_threshold:.2f}"
)


# ============================================================
# VERIFY BOTH MODELS CONTRIBUTE
# ============================================================

if (
    best_x_weight <= 0.0
    or best_e_weight <= 0.0
):

    raise RuntimeError(
        "Invalid fusion: one model received zero weight."
    )

print(
    "\n✓ BOTH MODELS ARE CONTRIBUTING "
    "TO THE FINAL SCORE."
)


# ============================================================
# FINAL VALIDATION FUSION
# ============================================================

fused_val = (
    best_x_weight * x_val
    +
    best_e_weight * e_val
)

final_val_metrics = calculate_metrics(
    fused_val,
    y_val,
    best_threshold,
)


print("\n" + "=" * 56)
print("FINAL VALIDATION FUSION PERFORMANCE")
print("=" * 56)

print(
    f"Accuracy  : "
    f"{final_val_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{final_val_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{final_val_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{final_val_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{final_val_metrics['auc']:.4f}"
)


# ============================================================
# TEST INFERENCE
# ============================================================

print("\n" + "=" * 56)
print("FINAL TEST INFERENCE")
print("=" * 56)

x_test, y_test = predict_model(
    xception,
    test_loader,
    "Xception",
)

e_test, y_test_2 = predict_model(
    efficientnet,
    test_loader,
    "EfficientNet-B4",
)

if not np.array_equal(
    y_test,
    y_test_2,
):

    raise RuntimeError(
        "Test label ordering mismatch "
        "between models."
    )


# ============================================================
# TEST FUSION
# ============================================================

fused_test = (
    best_x_weight * x_test
    +
    best_e_weight * e_test
)


# ============================================================
# TEST METRICS
# ============================================================

x_test_metrics = calculate_metrics(
    x_test,
    y_test,
    0.50,
)

e_test_metrics = calculate_metrics(
    e_test,
    y_test,
    0.50,
)

ensemble_test_metrics = calculate_metrics(
    fused_test,
    y_test,
    best_threshold,
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 56)
print("FINAL MODEL COMPARISON")
print("=" * 56)

print(
    "\nMODEL"
    + " " * 18
    + "AUC"
    + " " * 8
    + "F1"
    + " " * 8
    + "ACC"
)

print(
    f"Xception"
    f"{x_test_metrics['auc']:>18.4f}"
    f"{x_test_metrics['f1']:>10.4f}"
    f"{x_test_metrics['accuracy']:>10.4f}"
)

print(
    f"EfficientNet-B4"
    f"{e_test_metrics['auc']:>11.4f}"
    f"{e_test_metrics['f1']:>10.4f}"
    f"{e_test_metrics['accuracy']:>10.4f}"
)

print(
    f"FUSED ENSEMBLE"
    f"{ensemble_test_metrics['auc']:>12.4f}"
    f"{ensemble_test_metrics['f1']:>10.4f}"
    f"{ensemble_test_metrics['accuracy']:>10.4f}"
)


# ============================================================
# FINAL TEST RESULT
# ============================================================

print("\n" + "=" * 56)
print("FINAL TWO-MODEL FUSION TEST PERFORMANCE")
print("=" * 56)

print(
    f"Accuracy  : "
    f"{ensemble_test_metrics['accuracy']:.4f}"
)

print(
    f"Precision : "
    f"{ensemble_test_metrics['precision']:.4f}"
)

print(
    f"Recall    : "
    f"{ensemble_test_metrics['recall']:.4f}"
)

print(
    f"F1 Score  : "
    f"{ensemble_test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{ensemble_test_metrics['auc']:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

actual_fake_test = (
    y_test == 0
).astype(int)

predicted_fake_test = (
    fused_test >= best_threshold
).astype(int)

cm = confusion_matrix(
    actual_fake_test,
    predicted_fake_test,
)


print("\n" + "=" * 56)
print("CONFUSION MATRIX")
print("=" * 56)

print(
    "\nRows = Actual\n"
    "Columns = Predicted"
)

print(
    "             FAKE    REAL"
)

print(
    f"FAKE       "
    f"{cm[1,1] if cm.shape == (2,2) else 0:6d}"
    f"{cm[1,0] if cm.shape == (2,2) else 0:9d}"
)

print(
    f"REAL       "
    f"{cm[0,1] if cm.shape == (2,2) else 0:6d}"
    f"{cm[0,0] if cm.shape == (2,2) else 0:9d}"
)


# ============================================================
# MODEL AGREEMENT
# ============================================================

x_pred_test = (
    x_test >= 0.50
)

e_pred_test = (
    e_test >= 0.50
)

agreement = np.mean(
    x_pred_test == e_pred_test
)

print("\n" + "=" * 56)
print("MODEL AGREEMENT")
print("=" * 56)

print(
    f"Agreement rate : "
    f"{agreement * 100:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": (
        "Xception + EfficientNet-B4 "
        "two-model score fusion"
    ),

    "class_mapping": {
        "0": "FAKE",
        "1": "REAL",
    },

    "fusion": {

        "method": (
            "weighted probability fusion"
        ),

        "xception_weight": (
            float(best_x_weight)
        ),

        "efficientnet_weight": (
            float(best_e_weight)
        ),

        "minimum_model_weight": (
            float(MIN_MODEL_WEIGHT)
        ),

        "decision_threshold": (
            float(best_threshold)
        ),
    },

    "validation": {

        "xception_auc": (
            float(x_val_metrics["auc"])
        ),

        "efficientnet_auc": (
            float(e_val_metrics["auc"])
        ),

        "fusion_auc": (
            float(final_val_metrics["auc"])
        ),

        "fusion_f1": (
            float(final_val_metrics["f1"])
        ),
    },

    "test": {

        "xception": x_test_metrics,

        "efficientnet_b4": (
            e_test_metrics
        ),

        "fusion": (
            ensemble_test_metrics
        ),

        "confusion_matrix": (
            cm.tolist()
        ),

        "model_agreement": (
            float(agreement)
        ),
    },

    "checkpoints": {

        "xception": str(
            XCEPTION_CHECKPOINT
        ),

        "efficientnet_b4": str(
            EFFICIENTNET_CHECKPOINT
        ),
    },
}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 56)
print("TWO-MODEL FUSION COMPLETE")
print("=" * 56)

print(
    f"Xception weight     : "
    f"{best_x_weight:.2f}"
)

print(
    f"EfficientNet weight : "
    f"{best_e_weight:.2f}"
)

print(
    f"Decision threshold  : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation AUC      : "
    f"{final_val_metrics['auc']:.4f}"
)

print(
    f"Final Test AUC      : "
    f"{ensemble_test_metrics['auc']:.4f}"
)

print(
    f"Final Test F1       : "
    f"{ensemble_test_metrics['f1']:.4f}"
)

print("\nBoth models contributed to the fusion.")

print("\nResults saved to:")
print(RESULT_FILE)

print("\n" + "=" * 56)
print("FUSION FINISHED SUCCESSFULLY")
print("=" * 56)