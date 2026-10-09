# ================================================================
# ForensIQ - 3 MODEL ENSEMBLE V2.1
#
# Models:
#   1. Xception V2
#   2. EfficientNet-B4 V2
#   3. ViT-Small V2.1
#
# Mapping:
#   0 = FAKE
#   1 = REAL
#
# CNN input  : 299x299
# ViT input  : 224x224
#
# Purpose:
#   Evaluate individual models and find a suitable
#   3-model score-fusion configuration.
# ================================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm

from PIL import Image

from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ================================================================
# CONFIG
# ================================================================

SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 0

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
SPLIT_DIR = DATASET_DIR / "split"

CHECKPOINT_DIR = ROOT / "checkpoints" / "v2"

RESULTS_DIR = ROOT / "v2_results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------
# Checkpoints
# ------------------------------------------------

XCEPTION_CHECKPOINT = (
    CHECKPOINT_DIR /
    "best_xception_v2.pth"
)

EFFICIENTNET_CHECKPOINT = (
    CHECKPOINT_DIR /
    "best_efficientnet_b4_v2.pth"
)

VIT_CHECKPOINT = (
    CHECKPOINT_DIR /
    "best_vit_v2_1.pth"
)

# ------------------------------------------------
# Model names
# ------------------------------------------------

XCEPTION_NAME = "xception"

EFFICIENTNET_NAME = "tf_efficientnet_b4"

VIT_NAME = "vit_small_patch16_224"


# ================================================================
# REPRODUCIBILITY
# ================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("ForensIQ - 3 MODEL ENSEMBLE V2.1")
print("=" * 70)

print(
    f"Device : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 70)


# ================================================================
# CHECKPOINT CHECK
# ================================================================

print("\nChecking checkpoints...")

for checkpoint in [
    XCEPTION_CHECKPOINT,
    EFFICIENTNET_CHECKPOINT,
    VIT_CHECKPOINT
]:

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"\nCheckpoint not found:\n"
            f"{checkpoint}"
        )

    print(
        f"✓ {checkpoint.name}"
    )


# ================================================================
# TRANSFORMS
# ================================================================

cnn_transform = transforms.Compose([

    transforms.Resize(
        (299, 299)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


vit_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ================================================================
# DATASET
# ================================================================

class FaceDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]

        image_path = str(
            row["face_path"]
        )

        label = int(
            row["label"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        return image, label


# ================================================================
# LOAD CSV
# ================================================================

TEST_CSV = SPLIT_DIR / "test.csv"

VAL_CSV = SPLIT_DIR / "val.csv"


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "face_path" not in df.columns:

        raise ValueError(
            f"{path} does not contain "
            f"'face_path'"
        )

    if "label" not in df.columns:

        raise ValueError(
            f"{path} does not contain "
            f"'label'"
        )

    df = df.copy()

    # ------------------------------------------------------------
    # LABEL MAPPING
    #
    # FAKE = 0
    # REAL = 1
    # ------------------------------------------------------------

    if df["label"].dtype == object:

        df["label"] = (

            df["label"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "fake": 0,
                "real": 1,
                "0": 0,
                "1": 1
            })
        )

    else:

        df["label"] = pd.to_numeric(
            df["label"],
            errors="coerce"
        )

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    df["label"] = df[
        "label"
    ].astype(int)

    # ------------------------------------------------------------
    # PATHS
    # ------------------------------------------------------------

    paths = []

    for path_string in df[
        "face_path"
    ]:

        path = Path(
            str(path_string)
        )

        if not path.is_absolute():

            path = ROOT / path

        paths.append(
            str(path)
        )

    df["face_path"] = paths

    # ------------------------------------------------------------
    # REMOVE MISSING
    # ------------------------------------------------------------

    exists = df[
        "face_path"
    ].apply(
        lambda x: Path(x).exists()
    )

    missing = (
        ~exists
    ).sum()

    if missing:

        print(
            f"WARNING: removing "
            f"{missing} missing images"
        )

    df = df[
        exists
    ].reset_index(
        drop=True
    )

    return df


# ================================================================
# LOAD DATA
# ================================================================

print("\nLoading FF++ validation/test data...")

val_df = load_csv(
    VAL_CSV
)

test_df = load_csv(
    TEST_CSV
)

print(
    f"Validation images : "
    f"{len(val_df)}"
)

print(
    f"Test images       : "
    f"{len(test_df)}"
)


# ================================================================
# CUSTOM COLLATE FUNCTION
# ================================================================
# Keep PIL images as a list because the three models use different
# input sizes: CNNs = 299x299, ViT = 224x224.
# ================================================================

def collate_pil_batch(batch):

    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    return (
        images,
        torch.tensor(labels, dtype=torch.long)
    )


# ================================================================
# DATA LOADERS
# ================================================================

val_dataset = FaceDataset(
    val_df
)

test_dataset = FaceDataset(
    test_df
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_pil_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_pil_batch
)


# ================================================================
# MODEL CREATION
# ================================================================

print("\nLoading models...")


def create_xception():

    return timm.create_model(
        XCEPTION_NAME,
        pretrained=False,
        num_classes=2
    )


def create_efficientnet():

    return timm.create_model(
        EFFICIENTNET_NAME,
        pretrained=False,
        num_classes=2
    )


def create_vit():

    return timm.create_model(
        VIT_NAME,
        pretrained=False,
        num_classes=2
    )


xception = create_xception().to(
    DEVICE
)

efficientnet = create_efficientnet().to(
    DEVICE
)

vit = create_vit().to(
    DEVICE
)


# ================================================================
# CHECKPOINT LOADER
# ================================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # ------------------------------------------------------------
    # Handle common checkpoint formats
    # ------------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # ------------------------------------------------------------
    # Clean prefixes
    # ------------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        prefixes = [
            "module.",
            "_orig_mod.",
            "model."
        ]

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(
                    prefix
                ):

                    new_key = new_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned_state_dict[
            new_key
        ] = value


    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    print(
        f"\nLoaded: "
        f"{checkpoint_path.name}"
    )

    print(
        f"Missing keys    : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys : "
        f"{len(unexpected)}"
    )

    if len(missing) > 0:

        print(
            "WARNING: missing keys detected."
        )

    if len(unexpected) > 0:

        print(
            "WARNING: unexpected keys detected."
        )

    model.eval()

    return model


xception = load_checkpoint(
    xception,
    XCEPTION_CHECKPOINT
)

efficientnet = load_checkpoint(
    efficientnet,
    EFFICIENTNET_CHECKPOINT
)

vit = load_checkpoint(
    vit,
    VIT_CHECKPOINT
)


# ================================================================
# PREPROCESS SINGLE BATCH
# ================================================================

def prepare_batch(
    images,
    transform
):

    tensors = []

    for image in images:

        tensors.append(
            transform(image)
        )

    return torch.stack(
        tensors
    ).to(
        DEVICE,
        non_blocking=True
    )


# ================================================================
# GET MODEL PROBABILITIES
# ================================================================

@torch.no_grad()
def get_model_predictions(
    loader
):

    all_labels = []

    xception_probs = []
    efficientnet_probs = []
    vit_probs = []


    for batch_idx, (
        images,
        labels
    ) in enumerate(
        loader,
        1
    ):

        # --------------------------------------------------------
        # Prepare independently
        # --------------------------------------------------------

        cnn_images = prepare_batch(
            images,
            cnn_transform
        )

        vit_images = prepare_batch(
            images,
            vit_transform
        )


        # --------------------------------------------------------
        # XCEPTION
        # --------------------------------------------------------

        xception_logits = xception(
            cnn_images
        )

        xception_probability = (
            F.softmax(
                xception_logits,
                dim=1
            )[:, 0]
        )


        # --------------------------------------------------------
        # EFFICIENTNET
        # --------------------------------------------------------

        efficientnet_logits = (
            efficientnet(
                cnn_images
            )
        )

        efficientnet_probability = (
            F.softmax(
                efficientnet_logits,
                dim=1
            )[:, 0]
        )


        # --------------------------------------------------------
        # ViT
        # --------------------------------------------------------

        vit_logits = vit(
            vit_images
        )

        vit_probability = (
            F.softmax(
                vit_logits,
                dim=1
            )[:, 0]
        )


        all_labels.extend(
            labels.numpy()
        )

        xception_probs.extend(
            xception_probability
            .cpu()
            .numpy()
        )

        efficientnet_probs.extend(
            efficientnet_probability
            .cpu()
            .numpy()
        )

        vit_probs.extend(
            vit_probability
            .cpu()
            .numpy()
        )


        if (
            batch_idx == 1
            or batch_idx % 100 == 0
            or batch_idx == len(loader)
        ):

            print(
                f"Batch "
                f"{batch_idx}/"
                f"{len(loader)}"
            )


    return (
        np.array(all_labels),
        np.array(xception_probs),
        np.array(efficientnet_probs),
        np.array(vit_probs)
    )


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    labels,
    fake_probability
):

    # Original:
    # 0 = FAKE
    # 1 = REAL
    #
    # Convert to:
    # 1 = FAKE
    # 0 = REAL

    binary_labels = (
        labels == 0
    ).astype(int)

    predictions = (
        fake_probability >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        binary_labels,
        predictions
    )

    precision = precision_score(
        binary_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        binary_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        binary_labels,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        binary_labels,
        fake_probability
    )

    cm = confusion_matrix(
        binary_labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }


# ================================================================
# PRINT METRICS
# ================================================================

def print_metrics(
    name,
    metrics
):

    print("\n" + "-" * 70)

    print(name)

    print("-" * 70)

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : "
        f"{metrics['auc']:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )


# ================================================================
# GET PREDICTIONS
# ================================================================

print("\n" + "=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

(
    val_labels,
    val_xception,
    val_efficientnet,
    val_vit
) = get_model_predictions(
    val_loader
)


print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

(
    test_labels,
    test_xception,
    test_efficientnet,
    test_vit
) = get_model_predictions(
    test_loader
)


# ================================================================
# INDIVIDUAL MODEL RESULTS
# ================================================================

print("\n" + "=" * 70)
print("INDIVIDUAL MODEL RESULTS - FF++ VALIDATION")
print("=" * 70)


val_x_metrics = calculate_metrics(
    val_labels,
    val_xception
)

val_e_metrics = calculate_metrics(
    val_labels,
    val_efficientnet
)

val_v_metrics = calculate_metrics(
    val_labels,
    val_vit
)


print_metrics(
    "XCEPTION V2",
    val_x_metrics
)

print_metrics(
    "EFFICIENTNET-B4 V2",
    val_e_metrics
)

print_metrics(
    "ViT V2.1",
    val_v_metrics
)


# ================================================================
# WEIGHT SEARCH
# ================================================================
#
# We test sensible combinations rather than assuming
# equal weighting.
#
# weights must sum to 1.0
# ================================================================

weight_combinations = [

    # Xception dominant
    (0.60, 0.20, 0.20),
    (0.55, 0.25, 0.20),
    (0.55, 0.20, 0.25),
    (0.50, 0.30, 0.20),
    (0.50, 0.25, 0.25),

    # Balanced
    (0.45, 0.30, 0.25),
    (0.45, 0.25, 0.30),
    (0.40, 0.35, 0.25),
    (0.40, 0.30, 0.30),
    (0.35, 0.35, 0.30),

    # ViT stronger
    (0.40, 0.25, 0.35),
    (0.35, 0.30, 0.35),
    (0.30, 0.30, 0.40),

    # EfficientNet stronger
    (0.45, 0.35, 0.20),
    (0.40, 0.40, 0.20),
    (0.35, 0.40, 0.25)
]


# ================================================================
# SEARCH FUSION
# ================================================================

results = []

print("\n" + "=" * 70)
print("3-MODEL WEIGHT SEARCH")
print("=" * 70)

for (
    x_weight,
    e_weight,
    v_weight
) in weight_combinations:

    fused_probability = (

        x_weight * val_xception

        +

        e_weight * val_efficientnet

        +

        v_weight * val_vit
    )

    metrics = calculate_metrics(
        val_labels,
        fused_probability
    )

    results.append({

        "xception_weight":
            x_weight,

        "efficientnet_weight":
            e_weight,

        "vit_weight":
            v_weight,

        "accuracy":
            metrics["accuracy"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "auc":
            metrics["auc"]
    })


results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="auc",
    ascending=False
).reset_index(
    drop=True
)


print(
    "\nWeight combinations ranked "
    "by validation ROC-AUC:"
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# BEST WEIGHTS
# ================================================================

best = results_df.iloc[0]

BEST_X_WEIGHT = float(
    best["xception_weight"]
)

BEST_E_WEIGHT = float(
    best["efficientnet_weight"]
)

BEST_V_WEIGHT = float(
    best["vit_weight"]
)


print("\n" + "=" * 70)
print("BEST 3-MODEL FUSION")
print("=" * 70)

print(
    f"Xception     : "
    f"{BEST_X_WEIGHT:.2f}"
)

print(
    f"EfficientNet : "
    f"{BEST_E_WEIGHT:.2f}"
)

print(
    f"ViT          : "
    f"{BEST_V_WEIGHT:.2f}"
)

print(
    f"Validation AUC : "
    f"{best['auc']:.4f}"
)


# ================================================================
# FINAL VALIDATION ENSEMBLE
# ================================================================

val_fused = (

    BEST_X_WEIGHT * val_xception

    +

    BEST_E_WEIGHT * val_efficientnet

    +

    BEST_V_WEIGHT * val_vit
)


val_ensemble_metrics = calculate_metrics(
    val_labels,
    val_fused
)


print_metrics(
    "BEST 3-MODEL ENSEMBLE - FF++ VALIDATION",
    val_ensemble_metrics
)


# ================================================================
# TEST USING SELECTED VALIDATION WEIGHTS
# ================================================================

test_fused = (

    BEST_X_WEIGHT * test_xception

    +

    BEST_E_WEIGHT * test_efficientnet

    +

    BEST_V_WEIGHT * test_vit
)


test_ensemble_metrics = calculate_metrics(
    test_labels,
    test_fused
)


print_metrics(
    "BEST 3-MODEL ENSEMBLE - FF++ TEST",
    test_ensemble_metrics
)


# ================================================================
# XCEPTION-PRIORITY ANCHORED FUSION SEARCH
# ================================================================
#
# Same logic as predict_v2_facecrop.py:
#
#   * EfficientNet / ViT probabilities are clipped to
#     [SUPPORT_CLIP_MIN, SUPPORT_CLIP_MAX] (stops a saturated
#     model from dominating).
#   * If Xception fake >= ALERT, the score is anchored on
#     Xception; the other two can pull it DOWN by at most
#     MAX_PULL_DOWN, but can push it UP freely.
#   * Below ALERT, plain weighted average of all three.
#
# We search ALERT and MAX_PULL_DOWN on VALIDATION, then report
# the cost on TEST (especially the false-positive rate: REAL
# images wrongly called FAKE).
# ================================================================

SUPPORT_CLIP_MIN = 0.05
SUPPORT_CLIP_MAX = 0.95
PULL_STRENGTH = 0.50
ALERT_FLOOR = 0.52


def xception_priority_fusion_vec(
    xception,
    efficientnet,
    vit,
    alert,
    max_pull_down
):

    e_clipped = np.clip(
        efficientnet,
        SUPPORT_CLIP_MIN,
        SUPPORT_CLIP_MAX
    )

    v_clipped = np.clip(
        vit,
        SUPPORT_CLIP_MIN,
        SUPPORT_CLIP_MAX
    )

    normal = (
        BEST_X_WEIGHT * xception
        + BEST_E_WEIGHT * e_clipped
        + BEST_V_WEIGHT * v_clipped
    )

    support = (
        BEST_E_WEIGHT * e_clipped
        + BEST_V_WEIGHT * v_clipped
    ) / (
        BEST_E_WEIGHT + BEST_V_WEIGHT
    )

    raw_pull = PULL_STRENGTH * (
        support - xception
    )

    pull = np.where(
        raw_pull < 0,
        np.maximum(raw_pull, -max_pull_down),
        raw_pull
    )

    anchored = np.clip(
        np.maximum(
            xception + pull,
            ALERT_FLOOR
        ),
        0.0,
        1.0
    )

    return np.where(
        xception >= alert,
        anchored,
        normal
    )


def error_rates(
    labels,
    fake_probability
):

    # labels: 0 = FAKE, 1 = REAL
    is_fake = (labels == 0)

    predicted_fake = (
        fake_probability >= 0.5
    )

    false_positives = int(
        np.sum(~is_fake & predicted_fake)
    )

    false_negatives = int(
        np.sum(is_fake & ~predicted_fake)
    )

    real_total = max(int(np.sum(~is_fake)), 1)
    fake_total = max(int(np.sum(is_fake)), 1)

    return (
        false_positives / real_total,
        false_negatives / fake_total
    )


alert_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

pull_down_grid = [0.05, 0.10, 0.15, 0.20]

priority_results = []

print("\n" + "=" * 70)
print("XCEPTION-PRIORITY SEARCH (VALIDATION)")
print("=" * 70)

for alert in alert_grid:

    for pull_down in pull_down_grid:

        priority_val = xception_priority_fusion_vec(
            val_xception,
            val_efficientnet,
            val_vit,
            alert,
            pull_down
        )

        metrics = calculate_metrics(
            val_labels,
            priority_val
        )

        fpr, fnr = error_rates(
            val_labels,
            priority_val
        )

        priority_results.append({
            "alert_threshold": alert,
            "max_pull_down": pull_down,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        })


priority_df = pd.DataFrame(
    priority_results
).sort_values(
    by=["f1", "false_positive_rate"],
    ascending=[False, True]
).reset_index(
    drop=True
)

print(
    priority_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

best_priority = priority_df.iloc[0]

BEST_ALERT = float(
    best_priority["alert_threshold"]
)

BEST_PULL_DOWN = float(
    best_priority["max_pull_down"]
)


# ================================================================
# COMPARE STRATEGIES (VALIDATION + TEST)
# ================================================================

def strategy_row(
    name,
    labels,
    fake_probability
):

    metrics = calculate_metrics(
        labels,
        fake_probability
    )

    fpr, fnr = error_rates(
        labels,
        fake_probability
    )

    return {
        "strategy": name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_positive_rate": fpr,
        "false_negative_rate": fnr
    }


val_priority = xception_priority_fusion_vec(
    val_xception,
    val_efficientnet,
    val_vit,
    BEST_ALERT,
    BEST_PULL_DOWN
)

test_priority = xception_priority_fusion_vec(
    test_xception,
    test_efficientnet,
    test_vit,
    BEST_ALERT,
    BEST_PULL_DOWN
)

for split_name, split_labels, split_x, split_fused, split_priority in [
    ("VALIDATION", val_labels, val_xception, val_fused, val_priority),
    ("TEST", test_labels, test_xception, test_fused, test_priority)
]:

    comparison_df = pd.DataFrame([
        strategy_row(
            "Xception alone",
            split_labels,
            split_x
        ),
        strategy_row(
            "Plain weighted ensemble",
            split_labels,
            split_fused
        ),
        strategy_row(
            f"Xception priority "
            f"(alert={BEST_ALERT:.2f}, "
            f"pull_down={BEST_PULL_DOWN:.2f})",
            split_labels,
            split_priority
        )
    ])

    print("\n" + "=" * 70)
    print(f"STRATEGY COMPARISON - FF++ {split_name}")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

print(
    "\nNOTE: false_positive_rate = REAL images wrongly flagged FAKE."
)

print(
    "Giving Xception priority trades some false positives for "
    "fewer missed fakes."
)

print(
    "FF++ is in-distribution; also check your own AI-generated / "
    "real images before trusting the thresholds."
)

print(
    f"\nSelected: XCEPTION_ALERT_THRESHOLD = {BEST_ALERT:.2f}, "
    f"XCEPTION_MAX_PULL_DOWN = {BEST_PULL_DOWN:.2f}"
)


# ================================================================
# SAVE RESULTS
# ================================================================

results_csv = (
    RESULTS_DIR /
    "ensemble_v2_1_weight_search.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)

priority_csv = (
    RESULTS_DIR /
    "ensemble_v2_1_xception_priority_search.csv"
)

priority_df.to_csv(
    priority_csv,
    index=False
)


# ================================================================
# SAVE SELECTED CONFIG
# ================================================================

config = {

    "xception_checkpoint":
        str(XCEPTION_CHECKPOINT),

    "efficientnet_checkpoint":
        str(EFFICIENTNET_CHECKPOINT),

    "vit_checkpoint":
        str(VIT_CHECKPOINT),

    "xception_weight":
        BEST_X_WEIGHT,

    "efficientnet_weight":
        BEST_E_WEIGHT,

    "vit_weight":
        BEST_V_WEIGHT,

    "final_threshold":
        0.50,

    "xception_alert_threshold":
        BEST_ALERT,

    "xception_max_pull_down":
        BEST_PULL_DOWN,

    "xception_pull_strength":
        PULL_STRENGTH,

    "xception_alert_floor":
        ALERT_FLOOR,

    "support_clip_min":
        SUPPORT_CLIP_MIN,

    "support_clip_max":
        SUPPORT_CLIP_MAX,

    "label_mapping": {
        "0": "FAKE",
        "1": "REAL"
    },

    "cnn_input_size":
        299,

    "vit_input_size":
        224,

    "validation_metrics": {
        k: (
            v.tolist()
            if isinstance(v, np.ndarray)
            else float(v)
        )
        for k, v in val_ensemble_metrics.items()
        if k != "confusion_matrix"
    },

    "test_metrics": {
        k: (
            v.tolist()
            if isinstance(v, np.ndarray)
            else float(v)
        )
        for k, v in test_ensemble_metrics.items()
        if k != "confusion_matrix"
    }
}


config_path = (
    RESULTS_DIR /
    "ensemble_v2_1_config.npy"
)

np.save(
    config_path,
    config,
    allow_pickle=True
)


# ================================================================
# SAVE RAW PREDICTIONS
# ================================================================

predictions_df = pd.DataFrame({

    "face_path":
        test_df["face_path"].values,

    "label":
        test_labels,

    "xception_fake_probability":
        test_xception,

    "efficientnet_fake_probability":
        test_efficientnet,

    "vit_fake_probability":
        test_vit,

    "ensemble_fake_probability":
        test_fused,

    "ensemble_prediction":
        (
            test_fused >= 0.5
        ).astype(int),

    "xception_priority_fake_probability":
        test_priority,

    "xception_priority_prediction":
        (
            test_priority >= 0.5
        ).astype(int)
})


predictions_path = (
    RESULTS_DIR /
    "ensemble_v2_1_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 70)
print("3-MODEL ENSEMBLE EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nSelected weights:"
)

print(
    f"Xception     = "
    f"{BEST_X_WEIGHT:.2f}"
)

print(
    f"EfficientNet = "
    f"{BEST_E_WEIGHT:.2f}"
)

print(
    f"ViT          = "
    f"{BEST_V_WEIGHT:.2f}"
)

print(
    "\nResults:"
)

print(
    results_csv
)

print(
    config_path
)

print(
    predictions_path
)

print("\nNext step:")

print(
    "Use these selected weights in "
    "predict_v2_facecrop.py"
)

print(
    f"Also set XCEPTION_ALERT_THRESHOLD = {BEST_ALERT:.2f} and "
    f"XCEPTION_MAX_PULL_DOWN = {BEST_PULL_DOWN:.2f}"
)

print(
    priority_csv
)

print("=" * 70)