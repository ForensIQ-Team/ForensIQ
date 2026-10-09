# ================================================================
# FORENSIQ V2
# ViT MODEL 3 TRAINING
#
# DATA:
#   Synthetic AI-generated faces
#   +
#   Real synthetic faces
#   +
#   Small FF++ replay buffer
#
# MODEL:
#   ViT-Small
#
# CLASS MAPPING:
#   0 = FAKE
#   1 = REAL
#
# RTX 2050 4GB SAFE VERSION
# ================================================================

import os
import csv
import random
from pathlib import Path
from contextlib import nullcontext

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image, ImageFile

import timm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ================================================================
# WINDOWS / IMAGE SAFETY
# ================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ================================================================
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
SPLIT_DIR = DATASET_DIR / "split"

SYNTHETIC_DIR = ROOT / "synthetic_dataset"

CHECKPOINT_DIR = ROOT / "checkpoints"
V2_CHECKPOINT_DIR = CHECKPOINT_DIR / "v2"

RESULT_DIR = ROOT / "v2_results"

V2_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# DATASET FILES
# ================================================================

FFPP_TRAIN_CSV = (
    SPLIT_DIR / "train.csv"
)


# ================================================================
# OUTPUT CHECKPOINTS
# ================================================================

BEST_MODEL = (
    V2_CHECKPOINT_DIR /
    "best_vit_v2.pth"
)

LAST_MODEL = (
    V2_CHECKPOINT_DIR /
    "last_vit_v2.pth"
)


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

NUM_CLASSES = 2

FAKE = 0
REAL = 1


# ================================================================
# ViT
# ================================================================

MODEL_NAME = "vit_small_patch16_224"

IMAGE_SIZE = 224


# ================================================================
# RTX 2050 4GB
# ================================================================

BATCH_SIZE = 8

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()


# ================================================================
# TRAINING
# ================================================================

EPOCHS = 10

PATIENCE = 3

MIN_DELTA = 0.0005

USE_AMP = True

GRADIENT_CLIP = 1.0


# ================================================================
# LEARNING RATE
# ================================================================

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 1e-4


# ================================================================
# FF++ REPLAY
# ================================================================

FFPP_REPLAY_TOTAL = 2500

FFPP_REPLAY_PER_CLASS = (
    FFPP_REPLAY_TOTAL // 2
)


# ================================================================
# SYNTHETIC SPLIT
# ================================================================

SYNTHETIC_VAL_RATIO = 0.10

SYNTHETIC_TEST_RATIO = 0.10


# ================================================================
# DEVICE
# ================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# SEED
# ================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = False


set_seed(SEED)


# ================================================================
# HEADER
# ================================================================

print()
print("=" * 70)
print("FORENSIQ V2")
print("MODEL 3 - ViT-SMALL")
print("SYNTHETIC DATA + FF++ REPLAY")
print("=" * 70)

print()

print(f"Device       : {DEVICE}")
print(f"Model        : {MODEL_NAME}")
print(f"Image size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    props = torch.cuda.get_device_properties(0)

    print(
        f"VRAM         : "
        f"{props.total_memory / (1024 ** 3):.2f} GB"
    )

print()

print("Class mapping:")
print("0 = FAKE")
print("1 = REAL")

print()


# ================================================================
# FILE CHECKS
# ================================================================

def require_file(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )

    print(f"✓ {name}")


def require_directory(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )

    if not path.is_dir():

        raise NotADirectoryError(
            f"\n{name} is not a directory:\n{path}"
        )

    print(f"✓ {name}")


print("=" * 70)
print("CHECKING DATA")
print("=" * 70)

require_directory(
    SYNTHETIC_DIR,
    "Synthetic dataset"
)

require_directory(
    SYNTHETIC_DIR / "AI-Generated Images",
    "AI-generated folder"
)

require_directory(
    SYNTHETIC_DIR / "Real Images",
    "Real folder"
)

require_file(
    FFPP_TRAIN_CSV,
    "FF++ train CSV"
)

print()


# ================================================================
# IMAGE EXTENSIONS
# ================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ================================================================
# LABELS
# ================================================================

def normalize_label(value):

    value = str(value).strip().lower()

    if value in {
        "fake",
        "f",
        "forged",
        "manipulated",
        "synthetic",
        "0",
    }:
        return FAKE

    if value in {
        "real",
        "r",
        "original",
        "authentic",
        "genuine",
        "1",
    }:
        return REAL

    raise ValueError(
        f"Unknown label: {value}"
    )


# ================================================================
# SYNTHETIC DATA COLLECTION
# ================================================================

def collect_synthetic_images():

    fake_dir = (
        SYNTHETIC_DIR /
        "AI-Generated Images"
    )

    real_dir = (
        SYNTHETIC_DIR /
        "Real Images"
    )

    samples = []

    # FAKE
    for path in fake_dir.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            samples.append(
                (
                    path,
                    FAKE
                )
            )

    # REAL
    for path in real_dir.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            samples.append(
                (
                    path,
                    REAL
                )
            )

    return samples


# ================================================================
# STRATIFIED SYNTHETIC SPLIT
# ================================================================

def stratified_split(
    samples,
    val_ratio,
    test_ratio,
    seed=42
):

    rng = random.Random(seed)

    fake = [
        sample
        for sample in samples
        if sample[1] == FAKE
    ]

    real = [
        sample
        for sample in samples
        if sample[1] == REAL
    ]

    rng.shuffle(fake)
    rng.shuffle(real)

    def split_class(items):

        n = len(items)

        test_count = int(
            n * test_ratio
        )

        val_count = int(
            n * val_ratio
        )

        test_items = items[
            :test_count
        ]

        val_items = items[
            test_count:
            test_count + val_count
        ]

        train_items = items[
            test_count + val_count:
        ]

        return (
            train_items,
            val_items,
            test_items
        )

    (
        fake_train,
        fake_val,
        fake_test
    ) = split_class(fake)

    (
        real_train,
        real_val,
        real_test
    ) = split_class(real)

    train = (
        fake_train +
        real_train
    )

    val = (
        fake_val +
        real_val
    )

    test = (
        fake_test +
        real_test
    )

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return (
        train,
        val,
        test
    )


# ================================================================
# FF++ PATH RESOLUTION
# ================================================================

def resolve_ffpp_path(value):

    value = str(value).strip()

    path = Path(value)

    # Absolute
    if path.is_absolute():

        if path.exists():

            return path

    # ROOT relative
    candidate = ROOT / value

    if candidate.exists():

        return candidate

    # dataset relative
    candidate = DATASET_DIR / value

    if candidate.exists():

        return candidate

    # split relative
    candidate = SPLIT_DIR / value

    if candidate.exists():

        return candidate

    # normalized Windows path
    normalized = (
        value
        .replace("\\", os.sep)
        .replace("/", os.sep)
    )

    candidate = ROOT / normalized

    if candidate.exists():

        return candidate

    candidate = (
        DATASET_DIR /
        normalized
    )

    if candidate.exists():

        return candidate

    return Path(value)


# ================================================================
# LOAD FF++ REPLAY
# ================================================================

def load_ffpp_replay():

    print("=" * 70)
    print("LOADING FF++ REPLAY BUFFER")
    print("=" * 70)

    fake_samples = []
    real_samples = []

    with open(
        FFPP_TRAIN_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            raise RuntimeError(
                "FF++ CSV has no header."
            )

        if (
            "face_path"
            not in reader.fieldnames
        ):

            raise RuntimeError(
                "FF++ CSV must contain "
                "'face_path'."
            )

        if (
            "label"
            not in reader.fieldnames
        ):

            raise RuntimeError(
                "FF++ CSV must contain "
                "'label'."
            )

        for row in reader:

            path = resolve_ffpp_path(
                row["face_path"]
            )

            if not path.exists():

                continue

            label = normalize_label(
                row["label"]
            )

            if label == FAKE:

                fake_samples.append(
                    (
                        path,
                        FAKE
                    )
                )

            else:

                real_samples.append(
                    (
                        path,
                        REAL
                    )
                )

    random.shuffle(fake_samples)
    random.shuffle(real_samples)

    if (
        len(fake_samples)
        <
        FFPP_REPLAY_PER_CLASS
    ):

        raise RuntimeError(
            "Not enough FF++ FAKE "
            "samples for replay."
        )

    if (
        len(real_samples)
        <
        FFPP_REPLAY_PER_CLASS
    ):

        raise RuntimeError(
            "Not enough FF++ REAL "
            "samples for replay."
        )

    fake_samples = fake_samples[
        :FFPP_REPLAY_PER_CLASS
    ]

    real_samples = real_samples[
        :FFPP_REPLAY_PER_CLASS
    ]

    replay = (
        fake_samples +
        real_samples
    )

    random.shuffle(replay)

    print(
        f"FF++ FAKE replay : "
        f"{len(fake_samples)}"
    )

    print(
        f"FF++ REAL replay : "
        f"{len(real_samples)}"
    )

    print(
        f"FF++ TOTAL       : "
        f"{len(replay)}"
    )

    print()

    return replay


# ================================================================
# DATASET CLASS
# ================================================================

class FaceDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples

        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = (
            self.samples[index]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"\nCould not load:\n"
                f"{image_path}\n"
                f"{e}"
            )

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# ================================================================
# TRANSFORMS
# ================================================================

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
    ),
])


eval_transform = transforms.Compose([

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
# LOAD SYNTHETIC DATA
# ================================================================

print("=" * 70)
print("LOADING SYNTHETIC DATASET")
print("=" * 70)

synthetic_samples = (
    collect_synthetic_images()
)

if not synthetic_samples:

    raise RuntimeError(
        "No synthetic images found."
    )

synthetic_fake = sum(
    1
    for _, label
    in synthetic_samples
    if label == FAKE
)

synthetic_real = sum(
    1
    for _, label
    in synthetic_samples
    if label == REAL
)

print(
    f"AI-Generated / FAKE : "
    f"{synthetic_fake}"
)

print(
    f"Real                 : "
    f"{synthetic_real}"
)

print(
    f"Total                : "
    f"{len(synthetic_samples)}"
)

print()


# ================================================================
# SPLIT SYNTHETIC DATA
# ================================================================

(
    synthetic_train,
    synthetic_val,
    synthetic_test
) = stratified_split(
    synthetic_samples,
    SYNTHETIC_VAL_RATIO,
    SYNTHETIC_TEST_RATIO,
    SEED
)

print("=" * 70)
print("SYNTHETIC SPLIT")
print("=" * 70)

print(
    f"TRAIN : {len(synthetic_train)}"
)

print(
    f"VAL   : {len(synthetic_val)}"
)

print(
    f"TEST  : {len(synthetic_test)}"
)

print()


# ================================================================
# FF++ REPLAY
# ================================================================

ffpp_replay = load_ffpp_replay()


# ================================================================
# COMBINED TRAINING DATA
# ================================================================

training_samples = (
    synthetic_train +
    ffpp_replay
)

random.shuffle(
    training_samples
)

print("=" * 70)
print("FINAL TRAINING DATA")
print("=" * 70)

train_fake = sum(
    1
    for _, label
    in training_samples
    if label == FAKE
)

train_real = sum(
    1
    for _, label
    in training_samples
    if label == REAL
)

print(
    f"TRAIN FAKE : {train_fake}"
)

print(
    f"TRAIN REAL : {train_real}"
)

print(
    f"TRAIN TOTAL: "
    f"{len(training_samples)}"
)

print()


# ================================================================
# DATASETS
# ================================================================

train_dataset = FaceDataset(
    training_samples,
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

ffpp_replay_dataset = FaceDataset(
    ffpp_replay,
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

    pin_memory=PIN_MEMORY,

    persistent_workers=False,
)


synthetic_val_loader = DataLoader(

    synthetic_val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,
)


synthetic_test_loader = DataLoader(

    synthetic_test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,
)


ffpp_replay_loader = DataLoader(

    ffpp_replay_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,
)


# ================================================================
# CREATE ViT
# ================================================================

print("=" * 70)
print("LOADING ViT-SMALL")
print("=" * 70)

model = timm.create_model(

    MODEL_NAME,

    pretrained=True,

    num_classes=NUM_CLASSES,
)

model = model.to(DEVICE)

print(
    f"✓ {MODEL_NAME} loaded"
)

print()


# ================================================================
# CLASS WEIGHTS
# ================================================================

total = (
    train_fake +
    train_real
)

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
        real_weight
    ],

    dtype=torch.float32,

    device=DEVICE
)

print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print(
    f"FAKE : {fake_weight:.4f}"
)

print(
    f"REAL : {real_weight:.4f}"
)

print()


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

    weight_decay=WEIGHT_DECAY,
)


# ================================================================
# SCHEDULER
# ================================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=1,

        min_lr=1e-7,
    )
)


# ================================================================
# AMP COMPATIBILITY
# ================================================================

def create_grad_scaler():

    enabled = (
        USE_AMP
        and
        DEVICE.type == "cuda"
    )

    if (
        hasattr(torch, "amp")
        and
        hasattr(torch.amp, "GradScaler")
    ):

        try:

            return torch.amp.GradScaler(
                "cuda",
                enabled=enabled
            )

        except TypeError:

            try:

                return torch.amp.GradScaler(
                    enabled=enabled
                )

            except TypeError:

                pass

    return torch.cuda.amp.GradScaler(
        enabled=enabled
    )


def autocast_context():

    enabled = (
        USE_AMP
        and
        DEVICE.type == "cuda"
    )

    if not enabled:

        return nullcontext()

    if (
        hasattr(torch, "amp")
        and
        hasattr(torch.amp, "autocast")
    ):

        try:

            return torch.amp.autocast(
                device_type="cuda",
                enabled=True
            )

        except TypeError:

            pass

    return torch.cuda.amp.autocast(
        enabled=True
    )


scaler = create_grad_scaler()


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(
    labels,
    fake_probabilities,
    threshold=0.5
):

    labels = np.asarray(labels)

    fake_probabilities = np.asarray(
        fake_probabilities
    )

    predictions = np.where(
        fake_probabilities >= threshold,
        FAKE,
        REAL
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        pos_label=FAKE,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        pos_label=FAKE,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        pos_label=FAKE,
        zero_division=0
    )

    try:

        # IMPORTANT:
        # positive class = FAKE
        #
        # labels are:
        # 0 = FAKE
        # 1 = REAL
        #
        # therefore convert labels so:
        # FAKE = 1
        # REAL = 0

        auc_labels = (
            labels == FAKE
        ).astype(int)

        auc = roc_auc_score(
            auc_labels,
            fake_probabilities
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[FAKE, REAL]
    )

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "auc":
            float(auc),

        "confusion_matrix":
            cm.tolist(),
    }


# ================================================================
# TRAIN
# ================================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    labels_all = []

    fake_probs_all = []

    total = len(
        train_loader.dataset
    )

    for batch_index, (
        images,
        labels
    ) in enumerate(train_loader):

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

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

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
            *
            images.size(0)
        )

        fake_probability = (
            torch.softmax(
                outputs,
                dim=1
            )[:, FAKE]
        )

        labels_all.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        fake_probs_all.extend(
            fake_probability.detach()
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            batch_index == 0
            or
            (batch_index + 1) % 100 == 0
            or
            (batch_index + 1)
            == len(train_loader)
        ):

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )

    epoch_loss = (
        running_loss /
        total
    )

    metrics = calculate_metrics(
        labels_all,
        fake_probs_all,
        threshold=0.5
    )

    metrics["loss"] = float(
        epoch_loss
    )

    return metrics


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(
    loader,
    dataset_name
):

    model.eval()

    running_loss = 0.0

    labels_all = []

    fake_probs_all = []

    eval_criterion = (
        nn.CrossEntropyLoss()
    )

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

            outputs = model(
                images
            )

            loss = eval_criterion(
                outputs,
                labels
            )

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        fake_probability = (
            torch.softmax(
                outputs,
                dim=1
            )[:, FAKE]
        )

        labels_all.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        fake_probs_all.extend(
            fake_probability.float()
            .cpu()
            .numpy()
            .tolist()
        )

    total = len(
        loader.dataset
    )

    metrics = calculate_metrics(
        labels_all,
        fake_probs_all,
        threshold=0.5
    )

    metrics["loss"] = float(
        running_loss /
        total
    )

    print()
    print(
        f"{dataset_name}"
    )

    print(
        f"Loss      : "
        f"{metrics['loss']:.4f}"
    )

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

    if np.isfinite(
        metrics["auc"]
    ):

        print(
            f"ROC-AUC   : "
            f"{metrics['auc']:.4f}"
        )

    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    return metrics


# ================================================================
# CHECKPOINT
# ================================================================

def save_checkpoint(
    path,
    epoch,
    metrics,
    best_auc
):

    torch.save(

        {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "epoch":
                epoch,

            "metrics":
                metrics,

            "best_auc":
                best_auc,

            "model_name":
                MODEL_NAME,

            "version":
                "ForensIQ-V2",

            "class_mapping":
                {
                    "0": "FAKE",
                    "1": "REAL"
                },

            "image_size":
                IMAGE_SIZE,

        },

        path
    )


# ================================================================
# TRAINING LOOP
# ================================================================

print("=" * 70)
print("STARTING ViT TRAINING")
print("=" * 70)

best_auc = -float("inf")

best_epoch = 0

no_improvement = 0

history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    print()
    print("-" * 70)

    print(
        f"Epoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.2e}"
    )

    print("-" * 70)

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_metrics = (
        train_one_epoch()
    )

    # ------------------------------------------------------------
    # SYNTHETIC VALIDATION
    # ------------------------------------------------------------

    synthetic_val_metrics = (
        evaluate(
            synthetic_val_loader,
            "SYNTHETIC VALIDATION"
        )
    )

    # ------------------------------------------------------------
    # FF++ REPLAY
    # ------------------------------------------------------------

    replay_metrics = (
        evaluate(
            ffpp_replay_loader,
            "FF++ REPLAY"
        )
    )

    # ------------------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------------------

    if np.isfinite(
        synthetic_val_metrics["auc"]
    ):

        scheduler.step(
            synthetic_val_metrics["auc"]
        )

    # ------------------------------------------------------------
    # TRAIN SUMMARY
    # ------------------------------------------------------------

    print()
    print(
        f"TRAIN | "
        f"Loss {train_metrics['loss']:.4f} | "
        f"Acc {train_metrics['accuracy']:.4f} | "
        f"F1 {train_metrics['f1']:.4f} | "
        f"AUC {train_metrics['auc']:.4f}"
    )

    # ------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------

    record = {

        "epoch": epoch,

        "train":
            train_metrics,

        "synthetic_val":
            synthetic_val_metrics,

        "ffpp_replay":
            replay_metrics,

        "learning_rate":
            optimizer.param_groups[0]["lr"],
    }

    history.append(record)

    # ------------------------------------------------------------
    # SAVE LAST
    # ------------------------------------------------------------

    save_checkpoint(

        LAST_MODEL,

        epoch,

        synthetic_val_metrics,

        best_auc
    )

    # ------------------------------------------------------------
    # BEST MODEL
    # ------------------------------------------------------------

    current_auc = (
        synthetic_val_metrics["auc"]
    )

    if (
        np.isfinite(current_auc)
        and
        current_auc
        >
        best_auc + MIN_DELTA
    ):

        best_auc = current_auc

        best_epoch = epoch

        no_improvement = 0

        save_checkpoint(

            BEST_MODEL,

            epoch,

            synthetic_val_metrics,

            best_auc
        )

        print()
        print(
            "✓ NEW BEST ViT CHECKPOINT"
        )

        print(
            f"  AUC: {best_auc:.4f}"
        )

    else:

        no_improvement += 1

        print()
        print(
            f"No improvement "
            f"({no_improvement}/{PATIENCE})"
        )

    # ------------------------------------------------------------
    # EARLY STOPPING
    # ------------------------------------------------------------

    if no_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ================================================================
# FINAL TEST
# ================================================================

print()
print("=" * 70)
print("FINAL ViT EVALUATION")
print("=" * 70)

synthetic_test_metrics = (
    evaluate(
        synthetic_test_loader,
        "SYNTHETIC TEST"
    )
)

ffpp_final_metrics = (
    evaluate(
        ffpp_replay_loader,
        "FF++ REPLAY FINAL"
    )
)


# ================================================================
# SAVE HISTORY
# ================================================================

history_path = (
    RESULT_DIR /
    "vit_v2_training_history.npy"
)

np.save(
    history_path,
    np.array(
        history,
        dtype=object
    ),
    allow_pickle=True
)


# ================================================================
# SUMMARY
# ================================================================

print()
print("=" * 70)
print("ViT TRAINING COMPLETE")
print("=" * 70)

print()

print(
    f"Best Epoch : {best_epoch}"
)

print(
    f"Best Val AUC : "
    f"{best_auc:.4f}"
)

print()

print(
    f"Best Model:"
)

print(
    BEST_MODEL
)

print()

print(
    f"Last Model:"
)

print(
    LAST_MODEL
)

print()

print(
    f"History:"
)

print(
    history_path
)

print()
print("=" * 70)