import os
import csv
import random
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# WINDOWS / IMAGE SAFETY
# ============================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "split"
TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
TEST_CSV = DATASET_DIR / "test.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = CHECKPOINT_DIR / "best_efficientnet_b4.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_efficientnet_b4.pth"

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 380

# RTX 2050 4GB
BATCH_SIZE = 8

# VERY IMPORTANT FOR WINDOWS
NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

EPOCHS = 30

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 7

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility without forcing deterministic CUDA kernels
    torch.backends.cudnn.benchmark = True


set_seed(SEED)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================================")
print("FORENSIQ EFFICIENTNET-B4 TRAINING")
print("========================================")

print(f"Device: {device}")

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

print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"DataLoader workers: {NUM_WORKERS}")

# ============================================================
# VERIFY DATASET
# ============================================================

for required_file in [
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
]:

    if not required_file.exists():

        raise FileNotFoundError(
            f"\nMissing dataset file:\n"
            f"{required_file}"
        )

# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None,
    ):

        self.transform = transform
        self.samples = []

        with open(
            csv_file,
            "r",
            encoding="utf-8",
        ) as f:

            reader = csv.DictReader(f)

            required_columns = {
                "face_path",
                "label",
            }

            missing = (
                required_columns
                - set(reader.fieldnames or [])
            )

            if missing:

                raise RuntimeError(
                    f"{csv_file} missing columns: "
                    f"{missing}"
                )

            for row in reader:

                face_path = (
                    ROOT / row["face_path"]
                )

                label = row["label"].strip().lower()

                if label == "fake":
                    target = 0

                elif label == "real":
                    target = 1

                else:
                    continue

                self.samples.append(
                    (
                        face_path,
                        target,
                    )
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"{e}"
            )

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n========================================")
print("LOADING DATASETS")
print("========================================")

train_dataset = FaceDataset(
    TRAIN_CSV,
    train_transform,
)

val_dataset = FaceDataset(
    VAL_CSV,
    eval_transform,
)

test_dataset = FaceDataset(
    TEST_CSV,
    eval_transform,
)

print(
    f"TRAIN dataset: "
    f"{len(train_dataset)} images"
)

print(
    f"VAL   dataset: "
    f"{len(val_dataset)} images"
)

print(
    f"TEST  dataset: "
    f"{len(test_dataset)} images"
)


# ============================================================
# DISTRIBUTION
# ============================================================

def get_distribution(dataset):

    fake = 0
    real = 0

    for _, label in dataset.samples:

        if label == 0:
            fake += 1
        else:
            real += 1

    return fake, real


train_fake, train_real = (
    get_distribution(train_dataset)
)

val_fake, val_real = (
    get_distribution(val_dataset)
)

test_fake, test_real = (
    get_distribution(test_dataset)
)


print("\n========================================")
print("DATASET DISTRIBUTION")
print("========================================")

print(
    f"TRAIN | FAKE: {train_fake:5d} "
    f"| REAL: {train_real:5d}"
)

print(
    f"VAL   | FAKE: {val_fake:5d} "
    f"| REAL: {val_real:5d}"
)

print(
    f"TEST  | FAKE: {test_fake:5d} "
    f"| REAL: {test_real:5d}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


# ============================================================
# LOAD EFFICIENTNET-B4
# ============================================================

print("\n========================================")
print("LOADING EFFICIENTNET-B4")
print("========================================")

model = timm.create_model(
    "efficientnet_b4",
    pretrained=True,
    num_classes=2,
)

model = model.to(device)


# ============================================================
# CLASS WEIGHTS
# ============================================================

total = train_fake + train_real

fake_weight = (
    total /
    (2.0 * train_fake)
)

real_weight = (
    total /
    (2.0 * train_real)
)

class_weights = torch.tensor(
    [
        fake_weight,
        real_weight,
    ],
    dtype=torch.float32,
    device=device,
)

print("\n========================================")
print("CLASS WEIGHTS")
print("========================================")

print(
    f"FAKE weight : "
    f"{fake_weight:.4f}"
)

print(
    f"REAL weight : "
    f"{real_weight:.4f}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
)


# ============================================================
# AMP
#
# Compatible with older and newer PyTorch versions.
# Avoids the GradScaler error encountered previously.
# ============================================================

AMP_ENABLED = torch.cuda.is_available()

try:

    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler

    scaler = GradScaler(
        enabled=AMP_ENABLED
    )

    USE_NEW_AMP = False

except Exception:

    scaler = None
    USE_NEW_AMP = False


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []
    targets = []

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            enabled=AMP_ENABLED
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

        if scaler is not None:

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predicted = (
            torch.argmax(
                outputs,
                dim=1,
            )
        )

        predictions.extend(
            predicted.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    epoch_acc = accuracy_score(
        targets,
        predictions,
    )

    return epoch_loss, epoch_acc


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader, dataset):

    model.eval()

    running_loss = 0.0

    predictions = []
    targets = []
    probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with autocast(
                enabled=AMP_ENABLED
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            probs = torch.softmax(
                outputs,
                dim=1,
            )

            predicted = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.cpu().numpy()
            )

            # Probability of FAKE
            probabilities.extend(
                probs[:, 0]
                .cpu()
                .numpy()
            )

    epoch_loss = (
        running_loss
        / len(dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    recall = recall_score(
        targets,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    f1 = f1_score(
        targets,
        predictions,
        pos_label=0,
        zero_division=0,
    )

    try:

        auc = roc_auc_score(
            targets,
            1.0 - np.array(probabilities),
        )

    except Exception:

        auc = 0.0

    return (
        epoch_loss,
        accuracy,
        precision,
        recall,
        f1,
        auc,
    )


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    path,
    epoch,
    best_auc,
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "best_auc": best_auc,
            "class_weights":
                class_weights.detach()
                .cpu(),
        },
        path,
    )


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

best_auc = -1.0
best_epoch = 0
patience_counter = 0

for epoch in range(
    1,
    EPOCHS + 1,
):

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"\nEpoch "
        f"{epoch:02d}/{EPOCHS}"
    )

    print(
        f"LR         : "
        f"{current_lr:.2e}"
    )

    train_loss, train_acc = (
        train_one_epoch()
    )

    (
        val_loss,
        val_acc,
        val_precision,
        val_recall,
        val_f1,
        val_auc,
    ) = evaluate(
        val_loader,
        val_dataset,
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc:.4f}"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc:.4f}"
    )

    print(
        f"Val Prec   : "
        f"{val_precision:.4f}"
    )

    print(
        f"Val Recall : "
        f"{val_recall:.4f}"
    )

    print(
        f"Val F1     : "
        f"{val_f1:.4f}"
    )

    print(
        f"Val AUC    : "
        f"{val_auc:.4f}"
    )

    # Scheduler follows validation AUC
    scheduler.step(val_auc)

    # Save last checkpoint
    save_checkpoint(
        LAST_MODEL,
        epoch,
        best_auc,
    )

    # Best model
    if val_auc > best_auc:

        best_auc = val_auc

        best_epoch = epoch

        patience_counter = 0

        save_checkpoint(
            BEST_MODEL,
            epoch,
            best_auc,
        )

        print(
            f"✓ BEST MODEL SAVED "
            f"(Val AUC={best_auc:.4f})"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/"
            f"{EARLY_STOPPING_PATIENCE})"
        )

    # Early stopping
    if (
        patience_counter
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\n========================================"
        )

        print(
            "EARLY STOPPING TRIGGERED"
        )

        print(
            "========================================"
        )

        break


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("\n========================================")
print("LOADING BEST CHECKPOINT")
print("========================================")

if not BEST_MODEL.exists():

    raise RuntimeError(
        "Best EfficientNet checkpoint "
        "was not created."
    )

checkpoint = torch.load(
    BEST_MODEL,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Val AUC : "
    f"{checkpoint['best_auc']:.4f}"
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")

(
    test_loss,
    test_acc,
    test_precision,
    test_recall,
    test_f1,
    test_auc,
) = evaluate(
    test_loader,
    test_dataset,
)

print(
    f"Test Loss : "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy  : "
    f"{test_acc:.4f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1 Score  : "
    f"{test_f1:.4f}"
)

print(
    f"ROC-AUC   : "
    f"{test_auc:.4f}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("EFFICIENTNET-B4 TRAINING COMPLETE")
print("========================================")

print(
    f"Best epoch    : "
    f"{checkpoint['epoch']}"
)

print(
    f"Best Val AUC  : "
    f"{checkpoint['best_auc']:.4f}"
)

print(
    f"Final Test AUC: "
    f"{test_auc:.4f}"
)

print("\nBest model:")
print(BEST_MODEL)

print("\nLast checkpoint:")
print(LAST_MODEL)

print("\n========================================")