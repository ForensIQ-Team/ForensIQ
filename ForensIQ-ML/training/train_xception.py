import os
import csv
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# FORENSIQ XCEPTION TRAINING
# WINDOWS + RTX 2050 SAFE VERSION
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset" / "split"
CHECKPOINT_DIR = ROOT / "checkpoints"

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV = DATASET_DIR / "val.csv"
TEST_CSV = DATASET_DIR / "test.csv"

BEST_MODEL = CHECKPOINT_DIR / "best_xception.pth"
LAST_MODEL = CHECKPOINT_DIR / "last_xception.pth"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

NUM_CLASSES = 2

IMAGE_SIZE = 299

# RTX 2050 4GB SAFE
BATCH_SIZE = 16

# IMPORTANT:
# Windows multiprocessing was causing your crash.
NUM_WORKERS = 0

EPOCHS = 30

INITIAL_LR = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 7

MIN_DELTA = 0.001

USE_AMP = True

GRADIENT_CLIP = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class FaceDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.transform = transform

        self.samples = []

        with open(
            csv_file,
            "r",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            required = {
                "face_path",
                "label"
            }

            missing = required - set(
                reader.fieldnames or []
            )

            if missing:

                raise RuntimeError(
                    f"{csv_file} missing columns: {missing}"
                )

            for row in reader:

                path = ROOT / row["face_path"]

                label = row["label"].lower().strip()

                if label == "fake":
                    target = 0

                elif label == "real":
                    target = 1

                else:
                    continue

                self.samples.append(
                    (path, target)
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
                f"Could not load image: {image_path}\n{e}"
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
        degrees=8
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


eval_transform = transforms.Compose([

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
# LOAD DATA
# ============================================================

def load_datasets():

    print("\n========================================")
    print("LOADING DATASETS")
    print("========================================")

    train_dataset = FaceDataset(
        TRAIN_CSV,
        train_transform
    )

    val_dataset = FaceDataset(
        VAL_CSV,
        eval_transform
    )

    test_dataset = FaceDataset(
        TEST_CSV,
        eval_transform
    )

    print(
        f"TRAIN dataset: {len(train_dataset)} images"
    )

    print(
        f"VAL   dataset: {len(val_dataset)} images"
    )

    print(
        f"TEST  dataset: {len(test_dataset)} images"
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def print_distribution(
    train_dataset,
    val_dataset,
    test_dataset
):

    def counts(dataset):

        fake = sum(
            1
            for _, label in dataset.samples
            if label == 0
        )

        real = sum(
            1
            for _, label in dataset.samples
            if label == 1
        )

        return fake, real

    train_fake, train_real = counts(
        train_dataset
    )

    val_fake, val_real = counts(
        val_dataset
    )

    test_fake, test_real = counts(
        test_dataset
    )

    print("\n========================================")
    print("DATASET DISTRIBUTION")
    print("========================================")

    print(
        f"TRAIN | FAKE: {train_fake:5} | "
        f"REAL: {train_real:5}"
    )

    print(
        f"VAL   | FAKE: {val_fake:5} | "
        f"REAL: {val_real:5}"
    )

    print(
        f"TEST  | FAKE: {test_fake:5} | "
        f"REAL: {test_real:5}"
    )

    return train_fake, train_real


# ============================================================
# MODEL
# ============================================================

def create_model():

    print("\n========================================")
    print("LOADING XCEPTION")
    print("========================================")

    model = timm.create_model(
        "xception",
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def create_class_weights(
    fake_count,
    real_count
):

    total = fake_count + real_count

    fake_weight = total / (
        2.0 * fake_count
    )

    real_weight = total / (
        2.0 * real_count
    )

    weights = torch.tensor(
        [
            fake_weight,
            real_weight
        ],
        dtype=torch.float32,
        device=device
    )

    print("\n========================================")
    print("CLASS WEIGHTS")
    print("========================================")

    print(
        f"FAKE weight : {fake_weight:.4f}"
    )

    print(
        f"REAL weight : {real_weight:.4f}"
    )

    return weights


# ============================================================

# ============================================================
# PYTORCH AMP COMPATIBILITY
# ============================================================
# Works with both newer torch.amp and older torch.cuda.amp APIs.

def create_grad_scaler():
    enabled = USE_AMP and device.type == "cuda"

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=enabled)
            except TypeError:
                pass

    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context():
    enabled = USE_AMP and device.type == "cuda"

    if not enabled:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except TypeError:
            pass

    return torch.cuda.amp.autocast(enabled=True)


# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    predictions = []

    targets = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
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

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    return epoch_loss, accuracy


# ============================================================
# VALIDATION / TEST
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_targets = []

    all_predictions = []

    all_probabilities = []

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with autocast_context():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of REAL class
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

    loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    precision = precision_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_targets,
        all_predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            all_targets,
            all_probabilities
        )

    except ValueError:

        auc = 0.0

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        auc
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_auc,
    path
):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_auc":
            best_auc,

        "class_mapping": {
            "fake": 0,
            "real": 1
        },

        "image_size":
            IMAGE_SIZE,

    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("\n========================================")
    print("FORENSIQ XCEPTION TRAINING")
    print("========================================")

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"VRAM: {gpu_memory:.2f} GB"
        )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"DataLoader workers: {NUM_WORKERS}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets()

    (
        fake_count,
        real_count
    ) = print_distribution(
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = create_class_weights(
        fake_count,
        real_count
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = create_grad_scaler()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_auc = -1.0

    best_epoch = 0

    epochs_without_improvement = 0

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"LR         : {current_lr:.2e}"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            val_loss,
            val_acc,
            val_precision,
            val_recall,
            val_f1,
            val_auc
        ) = evaluate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Train Acc  : {train_acc:.4f}"
        )

        print(
            f"Val Loss   : {val_loss:.4f}"
        )

        print(
            f"Val Acc    : {val_acc:.4f}"
        )

        print(
            f"Val Prec   : {val_precision:.4f}"
        )

        print(
            f"Val Recall : {val_recall:.4f}"
        )

        print(
            f"Val F1     : {val_f1:.4f}"
        )

        print(
            f"Val AUC    : {val_auc:.4f}"
        )

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_auc
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_auc > (
            best_auc + MIN_DELTA
        ):

            best_auc = val_auc

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_auc,
                BEST_MODEL
            )

            print(
                f"✓ BEST MODEL SAVED "
                f"(Val AUC={best_auc:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # LAST CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_auc,
            LAST_MODEL
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

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

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n========================================")
    print("LOADING BEST CHECKPOINT")
    print("========================================")

    if not BEST_MODEL.exists():

        raise RuntimeError(
            "Best checkpoint was not created."
        )

    try:
        checkpoint = torch.load(
            BEST_MODEL,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            BEST_MODEL,
            map_location=device
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch : {checkpoint['epoch']}"
    )

    print(
        f"Best Val AUC : "
        f"{checkpoint['best_auc']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n========================================")
    print("FINAL TEST EVALUATION")
    print("========================================")

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_auc
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    print(
        f"Test Loss : {test_loss:.4f}"
    )

    print(
        f"Accuracy  : {test_accuracy:.4f}"
    )

    print(
        f"Precision : {test_precision:.4f}"
    )

    print(
        f"Recall    : {test_recall:.4f}"
    )

    print(
        f"F1 Score  : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {test_auc:.4f}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(
        f"Best epoch : {best_epoch}"
    )

    print(
        f"Best Val AUC : {best_auc:.4f}"
    )

    print(
        f"Final Test AUC : {test_auc:.4f}"
    )

    print("\nBest model:")
    print(BEST_MODEL)

    print("\nLast checkpoint:")
    print(LAST_MODEL)

    print("\n========================================")


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    main()