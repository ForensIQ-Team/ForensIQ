# ================================================================
# ForensIQ - ViT V2.1 Training
# Synthetic Dataset + Larger FF++ Replay
#
# Model:
#   ViT-Small / Patch16 / 224
#
# Dataset:
#   - All available synthetic images
#   - 7,500 FF++ replay images
#       3,750 FAKE
#       3,750 REAL
#
# Validation:
#   - Synthetic validation
#   - Separate FF++ validation
#   - Combined validation AUC for model selection
#
# Labels:
#   0 = FAKE
#   1 = REAL
# ================================================================

import os
import random
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from PIL import Image

from torchvision import transforms

from sklearn.model_selection import train_test_split
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

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

EPOCHS = 12
PATIENCE = 4

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4

# ------------------------------------------------
# FF++ replay
# ------------------------------------------------

FFPP_REPLAY_TRAIN_TOTAL = 7500
FFPP_REPLAY_VAL_TOTAL = 2500

FFPP_REPLAY_TRAIN_PER_CLASS = FFPP_REPLAY_TRAIN_TOTAL // 2
FFPP_REPLAY_VAL_PER_CLASS = FFPP_REPLAY_VAL_TOTAL // 2

# ------------------------------------------------
# Synthetic split
# ------------------------------------------------

SYNTHETIC_VAL_RATIO = 0.10
SYNTHETIC_TEST_RATIO = 0.10

# ------------------------------------------------
# Model
# ------------------------------------------------

MODEL_NAME = "vit_small_patch16_224"

# ------------------------------------------------
# Paths
# ------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
SPLIT_DIR = DATASET_DIR / "split"

SYNTHETIC_DIR = ROOT / "synthetic_dataset"

CHECKPOINT_DIR = ROOT / "checkpoints" / "v2"
RESULTS_DIR = ROOT / "v2_results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_vit_v2_1.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_vit_v2_1.pth"

HISTORY_FILE = RESULTS_DIR / "vit_v2_1_training_history.npy"


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ForensIQ - ViT V2.1 TRAINING")
print("=" * 70)

print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print(f"Image Size : {IMAGE_SIZE}")
print(f"Batch Size : {BATCH_SIZE}")

print("=" * 70)


# ================================================================
# AMP COMPATIBILITY
# ================================================================
# ================================================================
# AMP COMPATIBILITY
# ================================================================

USE_AMP = torch.cuda.is_available()

# GradScaler compatibility across PyTorch versions
if hasattr(torch.amp, "GradScaler"):

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

else:

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )


# Autocast compatibility
if hasattr(torch, "amp") and hasattr(
    torch.amp,
    "autocast"
):

    def autocast_context():

        return torch.amp.autocast(
            device_type="cuda",
            enabled=USE_AMP
        )

else:

    def autocast_context():

        return torch.cuda.amp.autocast(
            enabled=USE_AMP
        )

# ================================================================
# TRANSFORMS
# ================================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = str(row["face_path"])
        label = int(row["label"])

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ================================================================
# CSV LOADER
# ================================================================
def load_csv(csv_path):

    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)

    required = {
        "face_path",
        "label"
    }

    if not required.issubset(df.columns):

        print(
            f"WARNING: {csv_path} does not contain "
            f"face_path + label"
        )

        return None

    df = df.copy()

    # ------------------------------------------------------------
    # LABEL NORMALIZATION
    #
    # ForensIQ mapping:
    #   FAKE = 0
    #   REAL = 1
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

    # Remove invalid labels
    invalid_labels = df["label"].isna().sum()

    if invalid_labels > 0:

        print(
            f"WARNING: Removing "
            f"{invalid_labels} rows with invalid labels"
        )

        df = df[
            df["label"].notna()
        ].copy()

    df["label"] = df["label"].astype(int)

    # ------------------------------------------------------------
    # PATH NORMALIZATION
    # ------------------------------------------------------------

    fixed_paths = []

    for p in df["face_path"]:

        path = Path(str(p))

        if not path.is_absolute():
            path = ROOT / path

        fixed_paths.append(
            str(path)
        )

    df["face_path"] = fixed_paths

    # ------------------------------------------------------------
    # REMOVE MISSING IMAGES
    # ------------------------------------------------------------

    exists_mask = df["face_path"].apply(
        lambda x: Path(x).exists()
    )

    missing = (~exists_mask).sum()

    if missing > 0:

        print(
            f"WARNING: Removing "
            f"{missing} missing image paths "
            f"from {csv_path}"
        )

    df = df[
        exists_mask
    ].reset_index(drop=True)

    # ------------------------------------------------------------
    # FINAL CHECK
    # ------------------------------------------------------------

    invalid_final = ~df["label"].isin([0, 1])

    if invalid_final.any():

        print(
            "WARNING: Removing rows with "
            "labels other than 0/1"
        )

        df = df[
            ~invalid_final
        ].reset_index(drop=True)

    return df


# ================================================================
# FIND FF++ CSVs
# ================================================================

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

print("\nLoading FF++ CSV files...")

ffpp_train_full = load_csv(TRAIN_CSV)

if ffpp_train_full is None:

    raise FileNotFoundError(
        f"Could not load {TRAIN_CSV}"
    )

ffpp_val_full = load_csv(VAL_CSV)

if ffpp_val_full is not None:

    print(
        f"FF++ validation CSV found: "
        f"{len(ffpp_val_full)} images"
    )

else:

    print(
        "No separate FF++ val.csv found."
    )


# ================================================================
# NORMALIZE LABELS
# ================================================================

def normalize_labels(df):

    df = df.copy()

    # Expected:
    # FAKE = 0
    # REAL = 1

    if df["label"].dtype == object:

        df["label"] = (
            df["label"]
            .astype(str)
            .str.upper()
            .map({
                "FAKE": 0,
                "REAL": 1
            })
        )

    df["label"] = df["label"].astype(int)

    return df


ffpp_train_full = normalize_labels(
    ffpp_train_full
)

if ffpp_val_full is not None:

    ffpp_val_full = normalize_labels(
        ffpp_val_full
    )


# ================================================================
# SYNTHETIC DATASET
# ================================================================

FAKE_DIR = (
    SYNTHETIC_DIR /
    "AI-Generated Images"
)

REAL_DIR = (
    SYNTHETIC_DIR /
    "Real Images"
)


def collect_images(directory):

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    files = []

    if not directory.exists():

        print(
            f"WARNING: Directory does not exist: "
            f"{directory}"
        )

        return files

    for file in directory.rglob("*"):

        if file.is_file() and file.suffix.lower() in extensions:

            files.append(
                str(file)
            )

    return files


synthetic_fake = collect_images(
    FAKE_DIR
)

synthetic_real = collect_images(
    REAL_DIR
)

print("\n" + "=" * 70)
print("SYNTHETIC DATASET")
print("=" * 70)

print(
    f"AI Generated : {len(synthetic_fake)}"
)

print(
    f"Real         : {len(synthetic_real)}"
)


# ================================================================
# CREATE SYNTHETIC DATAFRAME
# ================================================================

synthetic_rows = []

for path in synthetic_fake:

    synthetic_rows.append({
        "face_path": path,
        "label": 0
    })

for path in synthetic_real:

    synthetic_rows.append({
        "face_path": path,
        "label": 1
    })


synthetic_df = pd.DataFrame(
    synthetic_rows
)


# ================================================================
# STRATIFIED SYNTHETIC SPLIT
# ================================================================

synthetic_train, synthetic_temp = train_test_split(
    synthetic_df,
    test_size=(
        SYNTHETIC_VAL_RATIO +
        SYNTHETIC_TEST_RATIO
    ),
    stratify=synthetic_df["label"],
    random_state=SEED
)


relative_test_size = (
    SYNTHETIC_TEST_RATIO /
    (
        SYNTHETIC_VAL_RATIO +
        SYNTHETIC_TEST_RATIO
    )
)


synthetic_val, synthetic_test = train_test_split(
    synthetic_temp,
    test_size=relative_test_size,
    stratify=synthetic_temp["label"],
    random_state=SEED
)


print("\nSynthetic split:")

print(
    f"Train : {len(synthetic_train)}"
)

print(
    f"Val   : {len(synthetic_val)}"
)

print(
    f"Test  : {len(synthetic_test)}"
)


# ================================================================
# FF++ REPLAY SAMPLING
# ================================================================

def balanced_sample(
    df,
    per_class,
    seed
):

    fake = df[
        df["label"] == 0
    ].sample(
        n=min(
            per_class,
            len(df[df["label"] == 0])
        ),
        random_state=seed
    )

    real = df[
        df["label"] == 1
    ].sample(
        n=min(
            per_class,
            len(df[df["label"] == 1])
        ),
        random_state=seed
    )

    result = pd.concat(
        [fake, real]
    )

    return result.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(
        drop=True
    )


# ================================================================
# CREATE DISJOINT FF++ REPLAY
# ================================================================

if ffpp_val_full is not None:

    # -----------------------------------------
    # Preferred:
    # train replay from train.csv
    # validation replay from val.csv
    # -----------------------------------------

    ffpp_replay_train = balanced_sample(
        ffpp_train_full,
        FFPP_REPLAY_TRAIN_PER_CLASS,
        SEED
    )

    ffpp_replay_val = balanced_sample(
        ffpp_val_full,
        FFPP_REPLAY_VAL_PER_CLASS,
        SEED + 1
    )

else:

    # -----------------------------------------
    # Fallback:
    # create a disjoint holdout from train.csv
    # -----------------------------------------

    ffpp_train_work = ffpp_train_full.copy()

    ffpp_replay_train = balanced_sample(
        ffpp_train_work,
        FFPP_REPLAY_TRAIN_PER_CLASS,
        SEED
    )

    remaining = ffpp_train_work.drop(
        ffpp_replay_train.index,
        errors="ignore"
    )

    # Safer fallback by removing selected paths
    remaining = ffpp_train_work[
        ~ffpp_train_work["face_path"].isin(
            ffpp_replay_train["face_path"]
        )
    ]

    ffpp_replay_val = balanced_sample(
        remaining,
        FFPP_REPLAY_VAL_PER_CLASS,
        SEED + 1
    )


print("\n" + "=" * 70)
print("FF++ REPLAY")
print("=" * 70)

print(
    f"Replay Train : {len(ffpp_replay_train)}"
)

print(
    f"Replay Val   : {len(ffpp_replay_val)}"
)

print(
    "\nReplay train distribution:"
)

print(
    ffpp_replay_train["label"]
    .value_counts()
    .sort_index()
)

print(
    "\nReplay validation distribution:"
)

print(
    ffpp_replay_val["label"]
    .value_counts()
    .sort_index()
)


# ================================================================
# COMBINE TRAINING DATA
# ================================================================

train_df = pd.concat(
    [
        synthetic_train,
        ffpp_replay_train
    ],
    ignore_index=True
)


print("\n" + "=" * 70)
print("FINAL TRAINING DATA")
print("=" * 70)

print(
    f"Synthetic train : {len(synthetic_train)}"
)

print(
    f"FF++ replay     : {len(ffpp_replay_train)}"
)

print(
    f"TOTAL           : {len(train_df)}"
)

print(
    "\nClass distribution:"
)

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


# ================================================================
# DATASETS
# ================================================================

train_dataset = FaceDataset(
    train_df,
    train_transform
)

synthetic_val_dataset = FaceDataset(
    synthetic_val,
    eval_transform
)

synthetic_test_dataset = FaceDataset(
    synthetic_test,
    eval_transform
)

ffpp_val_dataset = FaceDataset(
    ffpp_replay_val,
    eval_transform
)


# ================================================================
# DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

synthetic_val_loader = DataLoader(
    synthetic_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

synthetic_test_loader = DataLoader(
    synthetic_test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

ffpp_val_loader = DataLoader(
    ffpp_val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\n" + "=" * 70)
print("DATALOADERS")
print("=" * 70)

print(
    f"Train batches : {len(train_loader)}"
)

print(
    f"Synthetic val : {len(synthetic_val_loader)}"
)

print(
    f"FF++ val      : {len(ffpp_val_loader)}"
)


# ================================================================
# CLASS WEIGHTS
# ================================================================

class_counts = train_df["label"].value_counts()

fake_count = class_counts.get(0, 1)
real_count = class_counts.get(1, 1)

total = fake_count + real_count

fake_weight = total / (
    2 * fake_count
)

real_weight = total / (
    2 * real_count
)

class_weights = torch.tensor(
    [
        fake_weight,
        real_weight
    ],
    dtype=torch.float32,
    device=DEVICE
)

print("\n" + "=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print(
    f"FAKE : {fake_weight:.4f}"
)

print(
    f"REAL : {real_weight:.4f}"
)


# ================================================================
# MODEL
# ================================================================

print("\n" + "=" * 70)
print("LOADING ViT")
print("=" * 70)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=2
)

model = model.to(DEVICE)

print(
    f"Model : {MODEL_NAME}"
)

print(
    "Pretrained : True"
)

print(
    "Classes : 2"
)

print(
    "Mapping : 0=FAKE, 1=REAL"
)


# ================================================================
# LOSS
# ================================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ================================================================
# SCHEDULER
# ================================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    labels,
    fake_probabilities
):

    labels = np.asarray(labels)

    fake_probabilities = np.asarray(
        fake_probabilities
    )

    predictions = (
        fake_probabilities >= 0.5
    ).astype(int)

    # IMPORTANT:
    #
    # Original labels:
    # 0 = FAKE
    # 1 = REAL
    #
    # Convert labels so:
    # 1 = FAKE
    # 0 = REAL
    #
    # because fake_probability is the score
    # for the FAKE class.

    binary_labels = (
        labels == 0
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

    try:

        auc = roc_auc_score(
            binary_labels,
            fake_probabilities
        )

    except ValueError:

        auc = 0.5

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
# EVALUATION
# ================================================================

def evaluate(
    model,
    loader,
    name
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_fake_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with autocast_context():

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fake_probability = (
                probabilities[:, 0]
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_fake_probs.extend(
                fake_probability.cpu().numpy()
            )

    total = len(loader.dataset)

    avg_loss = (
        running_loss / total
    )

    metrics = calculate_metrics(
        all_labels,
        all_fake_probs
    )

    print("\n" + name)

    print(
        f"Loss      : {avg_loss:.4f}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1        : {metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['auc']:.4f}"
    )

    print(
        "Confusion Matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )

    return {
        "loss": avg_loss,
        **metrics
    }


# ================================================================
# TRAINING
# ================================================================

history = []

best_combined_auc = -1.0
best_epoch = 0

epochs_without_improvement = 0


print("\n" + "=" * 70)
print("STARTING ViT V2.1 TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    all_train_labels = []
    all_train_fake_probs = []

    current_lr = optimizer.param_groups[0]["lr"]

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Learning Rate: {current_lr:.2e}"
    )

    print("-" * 70)


    for batch_idx, (
        images,
        labels
    ) in enumerate(train_loader, 1):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast_context():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item() *
            images.size(0)
        )

        probabilities = torch.softmax(
            outputs.detach(),
            dim=1
        )

        fake_probability = (
            probabilities[:, 0]
        )

        all_train_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

        all_train_fake_probs.extend(
            fake_probability.cpu()
            .numpy()
        )


        if (
            batch_idx == 1
            or batch_idx % 100 == 0
            or batch_idx == len(train_loader)
        ):

            print(
                f"Batch "
                f"{batch_idx}/"
                f"{len(train_loader)}"
            )


    train_loss = (
        running_loss /
        len(train_loader.dataset)
    )


    train_metrics = calculate_metrics(
        all_train_labels,
        all_train_fake_probs
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    synthetic_metrics = evaluate(
        model,
        synthetic_val_loader,
        "SYNTHETIC VALIDATION"
    )

    ffpp_metrics = evaluate(
        model,
        ffpp_val_loader,
        "FF++ VALIDATION"
    )


    # ------------------------------------------------------------
    # COMBINED VALIDATION
    # ------------------------------------------------------------

    combined_auc = (
        0.5 * synthetic_metrics["auc"]
        +
        0.5 * ffpp_metrics["auc"]
    )

    combined_f1 = (
        0.5 * synthetic_metrics["f1"]
        +
        0.5 * ffpp_metrics["f1"]
    )


    print("\nCOMBINED VALIDATION")

    print(
        f"AUC : {combined_auc:.4f}"
    )

    print(
        f"F1  : {combined_f1:.4f}"
    )


    print(
        f"\nTRAIN | "
        f"Loss {train_loss:.4f} | "
        f"Acc {train_metrics['accuracy']:.4f} | "
        f"F1 {train_metrics['f1']:.4f} | "
        f"AUC {train_metrics['auc']:.4f}"
    )


    # ------------------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------------------

    scheduler.step(
        combined_auc
    )


    # ------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------

    epoch_record = {

        "epoch": epoch,

        "train_loss": train_loss,

        "train_accuracy":
            train_metrics["accuracy"],

        "train_precision":
            train_metrics["precision"],

        "train_recall":
            train_metrics["recall"],

        "train_f1":
            train_metrics["f1"],

        "train_auc":
            train_metrics["auc"],

        "synthetic_val_loss":
            synthetic_metrics["loss"],

        "synthetic_val_accuracy":
            synthetic_metrics["accuracy"],

        "synthetic_val_f1":
            synthetic_metrics["f1"],

        "synthetic_val_auc":
            synthetic_metrics["auc"],

        "ffpp_val_loss":
            ffpp_metrics["loss"],

        "ffpp_val_accuracy":
            ffpp_metrics["accuracy"],

        "ffpp_val_f1":
            ffpp_metrics["f1"],

        "ffpp_val_auc":
            ffpp_metrics["auc"],

        "combined_val_auc":
            combined_auc,

        "combined_val_f1":
            combined_f1,

        "learning_rate":
            optimizer.param_groups[0]["lr"]
    }


    history.append(
        epoch_record
    )


    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "model_name":
            MODEL_NAME,

        "version":
            "v2.1",

        "class_mapping": {
            "0": "FAKE",
            "1": "REAL"
        },

        "image_size":
            IMAGE_SIZE,

        "metrics":
            epoch_record,

        "best_combined_auc":
            best_combined_auc
    }


    torch.save(
        checkpoint,
        LAST_CHECKPOINT
    )


    if combined_auc > best_combined_auc:

        best_combined_auc = combined_auc

        best_epoch = epoch

        checkpoint[
            "best_combined_auc"
        ] = best_combined_auc

        torch.save(
            checkpoint,
            BEST_CHECKPOINT
        )

        epochs_without_improvement = 0

        print(
            "\n✓ NEW BEST ViT V2.1 CHECKPOINT"
        )

        print(
            f"  Combined AUC: "
            f"{combined_auc:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )


    # ------------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ================================================================
# FINAL TEST
# ================================================================

print("\n" + "=" * 70)
print("FINAL ViT V2.1 EVALUATION")
print("=" * 70)


# Load best checkpoint

best_checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)


# ------------------------------------------------
# Synthetic test
# ------------------------------------------------

synthetic_test_metrics = evaluate(
    model,
    synthetic_test_loader,
    "SYNTHETIC TEST"
)


# ------------------------------------------------
# FF++ validation diagnostic
# ------------------------------------------------

ffpp_final_metrics = evaluate(
    model,
    ffpp_val_loader,
    "FF++ VALIDATION FINAL"
)


# ================================================================
# SAVE HISTORY
# ================================================================

np.save(
    HISTORY_FILE,
    np.array(
        history,
        dtype=object
    ),
    allow_pickle=True
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("ViT V2.1 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Epoch : {best_epoch}"
)

print(
    f"Best Combined Val AUC : "
    f"{best_combined_auc:.4f}"
)

print(
    "\nSynthetic Test:"
)

print(
    f"Accuracy : "
    f"{synthetic_test_metrics['accuracy']:.4f}"
)

print(
    f"F1       : "
    f"{synthetic_test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{synthetic_test_metrics['auc']:.4f}"
)

print(
    "\nFF++ Validation:"
)

print(
    f"Accuracy : "
    f"{ffpp_final_metrics['accuracy']:.4f}"
)

print(
    f"F1       : "
    f"{ffpp_final_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{ffpp_final_metrics['auc']:.4f}"
)

print("\nBest Model:")

print(
    BEST_CHECKPOINT
)

print("\nLast Model:")

print(
    LAST_CHECKPOINT
)

print("\nHistory:")

print(
    HISTORY_FILE
)

print("=" * 70)