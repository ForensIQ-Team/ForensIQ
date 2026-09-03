# ================================================================
# FORENSIQ V2
# MULTI-SOURCE FINE-TUNING
#
# Synthetic AI-generated faces
# +
# Real synthetic faces
# +
# Small FF++ replay buffer
#
# V1 models:
#   Xception
#   EfficientNet-B4
#
# IMPORTANT:
#   0 = FAKE
#   1 = REAL
#
# V2 starts from V1 checkpoints.
#
# NO:
#   - frame extraction
#   - face detection
#   - ImageNet retraining
#
# FF++ uses existing face_path images.
#
# RTX 2050 / 4GB SAFE
# ================================================================

import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
)

from torchvision import transforms

from PIL import Image

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
# PROJECT PATHS
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
SPLIT_DIR = DATASET_DIR / "split"

SYNTHETIC_DIR = ROOT / "synthetic_dataset"

CHECKPOINT_DIR = ROOT / "checkpoints"

V2_CHECKPOINT_DIR = CHECKPOINT_DIR / "v2"

V2_RESULT_DIR = ROOT / "v2_results"

V2_CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

V2_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# FF++ CSV
# ================================================================

FFPP_TRAIN_CSV = SPLIT_DIR / "train.csv"


# ================================================================
# V1 CHECKPOINTS
# ================================================================

XCEPTION_V1 = (
    CHECKPOINT_DIR /
    "best_xception.pth"
)

EFFICIENTNET_V1 = (
    CHECKPOINT_DIR /
    "best_efficientnet_b4.pth"
)


# ================================================================
# V2 CHECKPOINTS
# ================================================================

XCEPTION_V2_BEST = (
    V2_CHECKPOINT_DIR /
    "best_xception_v2.pth"
)

XCEPTION_V2_LAST = (
    V2_CHECKPOINT_DIR /
    "last_xception_v2.pth"
)

EFFICIENTNET_V2_BEST = (
    V2_CHECKPOINT_DIR /
    "best_efficientnet_b4_v2.pth"
)

EFFICIENTNET_V2_LAST = (
    V2_CHECKPOINT_DIR /
    "last_efficientnet_b4_v2.pth"
)


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

NUM_CLASSES = 2

# IMPORTANT
FAKE = 0
REAL = 1

IMAGE_SIZE = 299

# RTX 2050 4GB
BATCH_SIZE = 8

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

# Keep V2 relatively short.
# We are fine-tuning, not training from scratch.
EPOCHS = 8

PATIENCE = 3

MIN_DELTA = 0.0005

USE_AMP = True

GRADIENT_CLIP = 1.0


# ================================================================
# LEARNING RATES
# ================================================================

# Very small backbone LR preserves V1 knowledge.
BACKBONE_LR = 2e-6

# Classifier adapts faster.
CLASSIFIER_LR = 1e-5

WEIGHT_DECAY = 1e-4


# ================================================================
# FF++ REPLAY
# ================================================================

FFPP_REPLAY_TOTAL = 2500

FFPP_REPLAY_PER_CLASS = (
    FFPP_REPLAY_TOTAL // 2
)


# ================================================================
# SYNTHETIC SPLITS
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
print("MULTI-SOURCE FINE-TUNING")
print("SYNTHETIC AI FACES + FF++ REPLAY")
print("=" * 70)

print(
    f"Device : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU    : "
        f"{torch.cuda.get_device_name(0)}"
    )

    props = torch.cuda.get_device_properties(0)

    vram = (
        props.total_memory /
        (1024 ** 3)
    )

    print(
        f"VRAM   : {vram:.2f} GB"
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

    print(
        f"✓ {name}"
    )


def require_directory(path, name):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n{path}"
        )

    if not path.is_dir():

        raise NotADirectoryError(
            f"\n{name} is not a directory:\n{path}"
        )

    print(
        f"✓ {name}"
    )


print("=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)

require_directory(
    SYNTHETIC_DIR,
    "Synthetic dataset"
)

require_directory(
    SYNTHETIC_DIR /
    "AI-Generated Images",
    "AI-generated folder"
)

require_directory(
    SYNTHETIC_DIR /
    "Real Images",
    "Real folder"
)

require_file(
    FFPP_TRAIN_CSV,
    "FF++ train CSV"
)

require_file(
    XCEPTION_V1,
    "Xception V1 checkpoint"
)

require_file(
    EFFICIENTNET_V1,
    "EfficientNet-B4 V1 checkpoint"
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
# LABEL NORMALIZATION
# ================================================================

def normalize_label(value):

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "fake",
            "f",
            "forged",
            "manipulated",
            "synthetic",
        }:

            return FAKE

        if value in {
            "real",
            "r",
            "original",
            "authentic",
            "genuine",
        }:

            return REAL

        if value == "0":

            return FAKE

        if value == "1":

            return REAL

    elif isinstance(
        value,
        (int, np.integer)
    ):

        if int(value) == 0:

            return FAKE

        if int(value) == 1:

            return REAL

    elif isinstance(
        value,
        (float, np.floating)
    ):

        if int(value) == 0:

            return FAKE

        if int(value) == 1:

            return REAL

    raise ValueError(
        f"Unknown label: {value!r}"
    )


# ================================================================
# COLLECT SYNTHETIC DATA
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
# STRATIFIED SPLIT
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
# LOAD SYNTHETIC DATASET
# ================================================================

print("=" * 70)
print("LOADING NEW SYNTHETIC DATASET")
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
    f"Real                  : "
    f"{synthetic_real}"
)

print(
    f"Total                 : "
    f"{len(synthetic_samples)}"
)

print()


# ================================================================
# SYNTHETIC SPLIT
# ================================================================

print("=" * 70)
print("CREATING SYNTHETIC TRAIN / VAL / TEST")
print("=" * 70)

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

print(
    f"NEW TRAIN : "
    f"{len(synthetic_train)}"
)

print(
    f"NEW VAL   : "
    f"{len(synthetic_val)}"
)

print(
    f"NEW TEST  : "
    f"{len(synthetic_test)}"
)

print()


# ================================================================
# FF++ PATH RESOLUTION
# ================================================================

def resolve_ffpp_path(value):

    value = str(value).strip()

    path = Path(value)

    # Absolute path
    if path.is_absolute():

        if path.exists():

            return path

    # ROOT relative
    candidate = ROOT / value

    if candidate.exists():

        return candidate

    # DATASET relative
    candidate = DATASET_DIR / value

    if candidate.exists():

        return candidate

    # SPLIT relative
    candidate = SPLIT_DIR / value

    if candidate.exists():

        return candidate

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
# LOAD FF++
# ================================================================

def load_ffpp_dataframe():

    print("=" * 70)
    print("LOADING FF++ FOR REPLAY")
    print("=" * 70)

    print(
        f"FF++ train CSV:\n"
        f"{FFPP_TRAIN_CSV}"
    )

    df = pd.read_csv(
        FFPP_TRAIN_CSV
    )

    lower_columns = {
        str(column)
        .strip()
        .lower():
        column
        for column
        in df.columns
    }

    # face_path gets FIRST priority
    path_candidates = [
        "face_path",
        "image_path",
        "img_path",
        "image",
        "filepath",
        "file_path",
        "path",
        "frame_path",
    ]

    path_col = None

    for candidate in path_candidates:

        if candidate in lower_columns:

            path_col = (
                lower_columns[candidate]
            )

            break

    if path_col is None:

        raise ValueError(
            "\nCould not find image/path "
            "column.\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    label_candidates = [
        "label",
        "class",
        "target",
        "y",
    ]

    label_col = None

    for candidate in label_candidates:

        if candidate in lower_columns:

            label_col = (
                lower_columns[candidate]
            )

            break

    if label_col is None:

        raise ValueError(
            "\nCould not find label column.\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    print()
    print(
        f"Detected image column : "
        f"{path_col}"
    )

    print(
        f"Detected label column : "
        f"{label_col}"
    )

    df["_normalized_label"] = (
        df[label_col]
        .apply(normalize_label)
    )

    df["_resolved_path"] = (
        df[path_col]
        .apply(resolve_ffpp_path)
    )

    exists_mask = (
        df["_resolved_path"]
        .apply(
            lambda path:
            Path(path).exists()
        )
    )

    missing_count = (
        (~exists_mask).sum()
    )

    if missing_count:

        print(
            f"\nWARNING: "
            f"{missing_count} FF++ paths "
            f"were not found."
        )

    df = df[
        exists_mask
    ].copy()

    if len(df) == 0:

        raise RuntimeError(
            "No valid FF++ images found."
        )

    return df


ffpp_df = load_ffpp_dataframe()

print()
print(
    f"Valid FF++ samples available: "
    f"{len(ffpp_df)}"
)

print()


# ================================================================
# CREATE BALANCED REPLAY
# ================================================================

def create_ffpp_replay(df):

    fake_df = df[
        df["_normalized_label"] == FAKE
    ].copy()

    real_df = df[
        df["_normalized_label"] == REAL
    ].copy()

    print("=" * 70)
    print("CREATING BALANCED FF++ REPLAY BUFFER")
    print("=" * 70)

    print(
        f"FF++ available FAKE : "
        f"{len(fake_df)}"
    )

    print(
        f"FF++ available REAL : "
        f"{len(real_df)}"
    )

    fake_count = min(
        FFPP_REPLAY_PER_CLASS,
        len(fake_df)
    )

    real_count = min(
        FFPP_REPLAY_PER_CLASS,
        len(real_df)
    )

    rng = np.random.default_rng(
        SEED
    )

    fake_indices = rng.choice(
        len(fake_df),
        size=fake_count,
        replace=False
    )

    real_indices = rng.choice(
        len(real_df),
        size=real_count,
        replace=False
    )

    fake_selected = (
        fake_df
        .iloc[fake_indices]
        .copy()
    )

    real_selected = (
        real_df
        .iloc[real_indices]
        .copy()
    )

    replay_df = pd.concat(
        [
            fake_selected,
            real_selected
        ],
        ignore_index=True
    )

    replay_samples = []

    for _, row in replay_df.iterrows():

        replay_samples.append(
            (
                Path(
                    row["_resolved_path"]
                ),
                int(
                    row["_normalized_label"]
                )
            )
        )

    random.Random(
        SEED
    ).shuffle(
        replay_samples
    )

    print()
    print("Selected FF++ replay:")

    print(
        f"FAKE : "
        f"{sum(1 for _, y in replay_samples if y == FAKE)}"
    )

    print(
        f"REAL : "
        f"{sum(1 for _, y in replay_samples if y == REAL)}"
    )

    print(
        f"TOTAL: "
        f"{len(replay_samples)}"
    )

    print()

    return replay_samples


ffpp_replay = (
    create_ffpp_replay(
        ffpp_df
    )
)


# ================================================================
# FINAL TRAINING DATA
# ================================================================

combined_train = (
    synthetic_train +
    ffpp_replay
)

print("=" * 70)
print("FINAL V2 TRAINING DATA")
print("=" * 70)

print(
    f"Synthetic training : "
    f"{len(synthetic_train)}"
)

print(
    f"FF++ replay         : "
    f"{len(ffpp_replay)}"
)

print(
    f"TOTAL               : "
    f"{len(combined_train)}"
)

print()


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
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.08,
        hue=0.01
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
# DATASET
# ================================================================

class ImageListDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = list(
            samples
        )

        self.transform = transform

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(self, index):

        path, label = (
            self.samples[index]
        )

        try:

            image = (
                Image.open(path)
                .convert("RGB")
            )

        except Exception as error:

            raise RuntimeError(
                f"\nCould not open:\n"
                f"{path}\n\n"
                f"Error: {error}"
            )

        if self.transform:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ================================================================
# DATASETS
# ================================================================

train_dataset = ImageListDataset(
    combined_train,
    train_transform
)

synthetic_val_dataset = ImageListDataset(
    synthetic_val,
    eval_transform
)

synthetic_test_dataset = ImageListDataset(
    synthetic_test,
    eval_transform
)


# ================================================================
# FF++ REPLAY VALIDATION
#
# Small diagnostic set only.
# It is NOT used for model selection.
# ================================================================

replay_val_dataset = ImageListDataset(
    ffpp_replay,
    eval_transform
)


# ================================================================
# BALANCED SAMPLER
# ================================================================

train_labels = [
    label
    for _, label
    in combined_train
]

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("=" * 70)
print("V2 CLASS DISTRIBUTION")
print("=" * 70)

print(
    f"FAKE : {class_counts[FAKE]}"
)

print(
    f"REAL : {class_counts[REAL]}"
)

class_weights = np.zeros(
    NUM_CLASSES,
    dtype=np.float64
)

for class_id in range(
    NUM_CLASSES
):

    if class_counts[class_id] > 0:

        class_weights[class_id] = (
            1.0 /
            class_counts[class_id]
        )

sample_weights = np.array(
    [
        class_weights[label]
        for label
        in train_labels
    ],
    dtype=np.float64
)

sampler = WeightedRandomSampler(
    weights=torch.as_tensor(
        sample_weights,
        dtype=torch.double
    ),
    num_samples=len(
        combined_train
    ),
    replacement=True
)


# ================================================================
# DATALOADERS
# ================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    sampler=sampler,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)

val_loader = DataLoader(

    synthetic_val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)

test_loader = DataLoader(

    synthetic_test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)

replay_val_loader = DataLoader(

    replay_val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=PIN_MEMORY,

)


# ================================================================
# DATALOADER INFO
# ================================================================

print()
print("=" * 70)
print("DATALOADERS READY")
print("=" * 70)

print(
    f"Train batches : "
    f"{len(train_loader)}"
)

print(
    f"Val batches   : "
    f"{len(val_loader)}"
)

print(
    f"Test batches  : "
    f"{len(test_loader)}"
)

print(
    f"Replay eval   : "
    f"{len(replay_val_loader)}"
)

print()


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

def extract_state_dict(
    checkpoint
):

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            return checkpoint[
                "model_state_dict"
            ]

        if (
            "state_dict"
            in checkpoint
        ):

            return checkpoint[
                "state_dict"
            ]

        if (
            "model"
            in checkpoint
            and
            isinstance(
                checkpoint["model"],
                dict
            )
        ):

            return checkpoint[
                "model"
            ]

        if all(
            isinstance(key, str)
            for key in checkpoint.keys()
        ):

            return checkpoint

    return checkpoint


# ================================================================
# CLEAN STATE DICT
# ================================================================

def clean_state_dict(
    state_dict
):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        prefixes = [
            "module.",
            "_orig_mod.",
            "model.",
        ]

        changed = True

        while changed:

            changed = False

            for prefix in prefixes:

                if new_key.startswith(
                    prefix
                ):

                    new_key = (
                        new_key[
                            len(prefix):
                        ]
                    )

                    changed = True

        cleaned[
            new_key
        ] = value

    return cleaned


# ================================================================
# LOAD V1
# ================================================================

def load_v1_checkpoint(
    model,
    checkpoint_path,
    model_name
):

    print()
    print(
        f"Loading {model_name} V1..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )

    state_dict = (
        clean_state_dict(
            state_dict
        )
    )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    if result.missing_keys:

        print(
            f"WARNING: "
            f"{len(result.missing_keys)} "
            f"missing keys."
        )

        for key in result.missing_keys[:10]:

            print(
                f"  Missing: {key}"
            )

    if result.unexpected_keys:

        print(
            f"WARNING: "
            f"{len(result.unexpected_keys)} "
            f"unexpected keys."
        )

        for key in result.unexpected_keys[:10]:

            print(
                f"  Unexpected: {key}"
            )

    if (
        not result.missing_keys
        and
        not result.unexpected_keys
    ):

        print(
            f"✓ {model_name} checkpoint "
            f"loaded perfectly"
        )

    elif (
        len(result.missing_keys) > 20
        or
        len(result.unexpected_keys) > 20
    ):

        raise RuntimeError(
            f"\nCheckpoint compatibility "
            f"problem for {model_name}."
        )

    else:

        print(
            f"✓ {model_name} loaded "
            f"with minor key differences"
        )

    return model


# ================================================================
# CLASSIFIER PARAMETERS
# ================================================================

def get_classifier_parameters(
    model
):

    classifier = None

    if hasattr(
        model,
        "get_classifier"
    ):

        try:

            classifier = (
                model.get_classifier()
            )

        except Exception:

            classifier = None

    if classifier is None:

        for name in [
            "classifier",
            "fc",
            "head",
        ]:

            if hasattr(
                model,
                name
            ):

                classifier = getattr(
                    model,
                    name
                )

                break

    if classifier is None:

        return []

    return list(
        classifier.parameters()
    )


# ================================================================
# OPTIMIZER
# ================================================================

def create_optimizer(
    model
):

    classifier_params = (
        get_classifier_parameters(
            model
        )
    )

    classifier_ids = {
        id(p)
        for p
        in classifier_params
    }

    backbone_params = [
        p
        for p in model.parameters()
        if id(p)
        not in classifier_ids
    ]

    groups = []

    if backbone_params:

        groups.append(
            {
                "params":
                    backbone_params,

                "lr":
                    BACKBONE_LR,
            }
        )

    if classifier_params:

        groups.append(
            {
                "params":
                    classifier_params,

                "lr":
                    CLASSIFIER_LR,
            }
        )

    return torch.optim.AdamW(
        groups,
        weight_decay=WEIGHT_DECAY
    )


# ================================================================
# LOSS
# ================================================================

def create_loss():

    return nn.CrossEntropyLoss()


# ================================================================
# CORRECT METRICS
#
# IMPORTANT:
#
# Internal labels:
#
#   FAKE = 0
#   REAL = 1
#
# probabilities:
#
#   fake_probability = P(class 0)
#
# Therefore:
#
#   fake_probability >= threshold
#       -> FAKE (0)
#
#   fake_probability < threshold
#       -> REAL (1)
#
# For sklearn AUC/F1:
#
# We create:
#
#   fake_target = 1 for FAKE
#   fake_target = 0 for REAL
#
# ================================================================

def calculate_metrics(
    labels,
    fake_probabilities,
    threshold=0.5
):

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    fake_probabilities = np.asarray(
        fake_probabilities,
        dtype=np.float64
    )

    # ------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------

    if not np.all(
        np.isfinite(
            fake_probabilities
        )
    ):

        raise ValueError(
            "Non-finite probabilities detected."
        )

    # ------------------------------------------------------------
    # Convert model labels:
    #
    # 0 = FAKE
    # 1 = REAL
    #
    # to metric target:
    #
    # 1 = FAKE
    # 0 = REAL
    # ------------------------------------------------------------

    fake_targets = (
        labels == FAKE
    ).astype(np.int64)

    # ------------------------------------------------------------
    # CORRECT prediction
    # ------------------------------------------------------------

    predictions = np.where(
        fake_probabilities >= threshold,
        FAKE,
        REAL
    ).astype(np.int64)

    fake_predictions = (
        predictions == FAKE
    ).astype(np.int64)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        fake_targets,
        fake_predictions,
        zero_division=0
    )

    recall = recall_score(
        fake_targets,
        fake_predictions,
        zero_division=0
    )

    f1 = f1_score(
        fake_targets,
        fake_predictions,
        zero_division=0
    )

    # ------------------------------------------------------------
    # AUC
    #
    # VERY IMPORTANT:
    #
    # fake_targets:
    #   1 = FAKE
    #   0 = REAL
    #
    # fake_probability:
    #   higher = more FAKE
    #
    # Therefore they align correctly.
    # ------------------------------------------------------------

    try:

        auc = roc_auc_score(
            fake_targets,
            fake_probabilities
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[
            FAKE,
            REAL
        ]
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

        "threshold":
            float(threshold),

    }


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    epoch,
    model_name
):

    model.train()

    running_loss = 0.0

    all_labels = []

    all_fake_probabilities = []

    total = len(
        loader.dataset
    )

    for batch_index, (
        images,
        labels
    ) in enumerate(loader):

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

        with torch.cuda.amp.autocast(
            enabled=(
                USE_AMP
                and
                DEVICE.type == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss detected "
                f"during {model_name}."
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

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_fake_probabilities.extend(
            fake_probability.detach()
            .float()
            .cpu()
            .numpy()
            .tolist()
        )

        if (
            batch_index == 0
            or
            (
                batch_index + 1
            ) % 100 == 0
            or
            (
                batch_index + 1
            ) == len(loader)
        ):

            print(
                f"  {model_name} | "
                f"Epoch {epoch:02d} | "
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    epoch_loss = (
        running_loss / total
    )

    metrics = calculate_metrics(
        all_labels,
        all_fake_probabilities,
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
    model,
    loader,
    model_name,
    split_name
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    all_labels = []

    all_fake_probabilities = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.cuda.amp.autocast(
            enabled=(
                USE_AMP
                and
                DEVICE.type == "cuda"
            )
        ):

            outputs = model(
                images
            )

            loss = criterion(
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

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_fake_probabilities.extend(
            fake_probability.float()
            .cpu()
            .numpy()
            .tolist()
        )

    total = len(
        loader.dataset
    )

    metrics = calculate_metrics(
        all_labels,
        all_fake_probabilities,
        threshold=0.5
    )

    metrics["loss"] = float(
        running_loss / total
    )

    return metrics


# ================================================================
# SAVE CHECKPOINT
# ================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    metrics,
    model_name,
    best_auc
):

    torch.save(

        {

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "metrics":
                metrics,

            "best_auc":
                best_auc,

            "model_name":
                model_name,

            "version":
                "ForensIQ-V2",

            "class_mapping":
                {
                    "FAKE": 0,
                    "REAL": 1
                },

            "image_size":
                IMAGE_SIZE,

            "normalization":
                {
                    "mean":
                        [0.5, 0.5, 0.5],

                    "std":
                        [0.5, 0.5, 0.5],
                },

        },

        path
    )


# ================================================================
# PRINT METRICS
# ================================================================

def print_metrics(
    title,
    metrics
):

    print()
    print(title)

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

    print(
        f"ROC-AUC   : "
        f"{metrics['auc']:.4f}"
    )


# ================================================================
# TRAIN ONE MODEL
# ================================================================

def train_model(
    model,
    model_name,
    v1_checkpoint,
    best_output,
    last_output
):

    print()
    print("=" * 70)
    print(
        f"STARTING V2 FINE-TUNING: "
        f"{model_name}"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # LOAD V1
    # ------------------------------------------------------------

    model = load_v1_checkpoint(
        model,
        v1_checkpoint,
        model_name
    )

    model = model.to(
        DEVICE
    )

    # ------------------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    criterion = create_loss()

    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            USE_AMP
            and
            DEVICE.type == "cuda"
        )
    )

    # ------------------------------------------------------------
    # LR SCHEDULER
    # ------------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=1,

            min_lr=1e-7,
        )
    )

    # ------------------------------------------------------------
    # TRACKING
    # ------------------------------------------------------------

    best_auc = -float("inf")

    best_epoch = 0

    no_improvement = 0

    history = []

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        backbone_lr = (
            optimizer.param_groups[0]["lr"]
        )

        classifier_lr = (
            optimizer.param_groups[-1]["lr"]
        )

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Backbone LR   : "
            f"{backbone_lr:.2e}"
        )

        print(
            f"Classifier LR : "
            f"{classifier_lr:.2e}"
        )

        print("-" * 70)

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        train_metrics = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler,

            epoch,

            model_name

        )

        # --------------------------------------------------------
        # SYNTHETIC VALIDATION
        # --------------------------------------------------------

        synthetic_val_metrics = evaluate(

            model,

            val_loader,

            model_name,

            "SYNTHETIC VAL"

        )

        # --------------------------------------------------------
        # FF++ REPLAY DIAGNOSTIC
        # --------------------------------------------------------

        replay_metrics = evaluate(

            model,

            replay_val_loader,

            model_name,

            "FF++ REPLAY"

        )

        # --------------------------------------------------------
        # SCHEDULER
        # --------------------------------------------------------

        if np.isfinite(
            synthetic_val_metrics["auc"]
        ):

            scheduler.step(
                synthetic_val_metrics["auc"]
            )

        # --------------------------------------------------------
        # PRINT
        # --------------------------------------------------------

        print_metrics(
            f"{model_name} TRAIN",
            train_metrics
        )

        print_metrics(
            f"{model_name} SYNTHETIC VAL",
            synthetic_val_metrics
        )

        print_metrics(
            f"{model_name} FF++ REPLAY",
            replay_metrics
        )

        # --------------------------------------------------------
        # HISTORY
        # --------------------------------------------------------

        epoch_record = {

            "epoch":
                epoch,

            "train":
                train_metrics,

            "synthetic_val":
                synthetic_val_metrics,

            "ffpp_replay":
                replay_metrics,

            "backbone_lr":
                backbone_lr,

            "classifier_lr":
                classifier_lr,

        }

        history.append(
            epoch_record
        )

        # --------------------------------------------------------
        # SAVE LAST
        # --------------------------------------------------------

        save_checkpoint(

            last_output,

            model,

            optimizer,

            epoch,

            synthetic_val_metrics,

            model_name,

            best_auc

        )

        # --------------------------------------------------------
        # BEST MODEL
        # --------------------------------------------------------

        current_auc = (
            synthetic_val_metrics["auc"]
        )

        if (

            np.isfinite(current_auc)

            and

            current_auc >
            best_auc + MIN_DELTA

        ):

            best_auc = current_auc

            best_epoch = epoch

            no_improvement = 0

            save_checkpoint(

                best_output,

                model,

                optimizer,

                epoch,

                synthetic_val_metrics,

                model_name,

                best_auc

            )

            print()
            print(
                "✓ BEST V2 MODEL SAVED"
            )

            print(
                f"  Synthetic Val AUC: "
                f"{best_auc:.4f}"
            )

        else:

            no_improvement += 1

            print()
            print(
                f"No improvement "
                f"({no_improvement}/"
                f"{PATIENCE})"
            )

        # --------------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------------

        if (
            no_improvement
            >= PATIENCE
        ):

            print()
            print("=" * 70)
            print("EARLY STOPPING")
            print("=" * 70)

            break

    # ============================================================
    # LOAD BEST MODEL
    # ============================================================

    print()
    print("=" * 70)

    print(
        f"LOADING BEST {model_name} V2"
    )

    print("=" * 70)

    best_checkpoint = torch.load(
        best_output,
        map_location=DEVICE
    )

    best_state_dict = (
        extract_state_dict(
            best_checkpoint
        )
    )

    best_state_dict = (
        clean_state_dict(
            best_state_dict
        )
    )

    model.load_state_dict(
        best_state_dict,
        strict=True
    )

    # ============================================================
    # FINAL SYNTHETIC TEST
    # ============================================================

    final_test_metrics = evaluate(

        model,

        test_loader,

        model_name,

        "SYNTHETIC TEST"

    )

    # ============================================================
    # FINAL FF++ REPLAY
    # ============================================================

    final_replay_metrics = evaluate(

        model,

        replay_val_loader,

        model_name,

        "FF++ REPLAY"

    )

    print()
    print("=" * 70)

    print(
        f"FINAL {model_name} V2"
    )

    print("=" * 70)

    print_metrics(
        "SYNTHETIC TEST",
        final_test_metrics
    )

    print_metrics(
        "FF++ REPLAY RETENTION",
        final_replay_metrics
    )

    print()

    print(
        f"Best epoch : "
        f"{best_epoch}"
    )

    print(
        f"Best Synthetic Val AUC : "
        f"{best_auc:.4f}"
    )

    return {

        "model":
            model,

        "best_epoch":
            best_epoch,

        "best_val_auc":
            float(best_auc),

        "final_synthetic_test":
            final_test_metrics,

        "final_ffpp_replay":
            final_replay_metrics,

        "history":
            history,

    }


# ================================================================
# TRAIN XCEPTION
# ================================================================

print()
print("=" * 70)
print("MODEL 1 / 2")
print("XCEPTION V2")
print("=" * 70)

xception_model = create_xception()

xception_results = train_model(

    xception_model,

    "Xception",

    XCEPTION_V1,

    XCEPTION_V2_BEST,

    XCEPTION_V2_LAST

)


# ================================================================
# FREE MEMORY
# ================================================================

del xception_model

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ================================================================
# TRAIN EFFICIENTNET
# ================================================================

print()
print("=" * 70)
print("MODEL 2 / 2")
print("EFFICIENTNET-B4 V2")
print("=" * 70)

efficientnet_model = (
    create_efficientnet()
)

efficientnet_results = train_model(

    efficientnet_model,

    "EfficientNet-B4",

    EFFICIENTNET_V1,

    EFFICIENTNET_V2_BEST,

    EFFICIENTNET_V2_LAST

)


# ================================================================
# FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("FORENSIQ V2 TRAINING COMPLETE")
print("=" * 70)

print()

print("XCEPTION V2")

print(
    f"Best Val AUC : "
    f"{xception_results['best_val_auc']:.4f}"
)

print(
    f"Synthetic Test AUC : "
    f"{xception_results['final_synthetic_test']['auc']:.4f}"
)

print(
    f"Synthetic Test F1 : "
    f"{xception_results['final_synthetic_test']['f1']:.4f}"
)

print(
    f"FF++ Replay AUC : "
    f"{xception_results['final_ffpp_replay']['auc']:.4f}"
)

print()

print("EFFICIENTNET-B4 V2")

print(
    f"Best Val AUC : "
    f"{efficientnet_results['best_val_auc']:.4f}"
)

print(
    f"Synthetic Test AUC : "
    f"{efficientnet_results['final_synthetic_test']['auc']:.4f}"
)

print(
    f"Synthetic Test F1 : "
    f"{efficientnet_results['final_synthetic_test']['f1']:.4f}"
)

print(
    f"FF++ Replay AUC : "
    f"{efficientnet_results['final_ffpp_replay']['auc']:.4f}"
)


# ================================================================
# SAVE RESULTS JSON
# ================================================================

results_json = {

    "version":
        "ForensIQ-V2",

    "description":
        "Synthetic AI-face fine-tuning "
        "with FF++ replay",

    "critical_label_mapping":
        {
            "FAKE": 0,
            "REAL": 1
        },

    "score_definition":
        "fake_probability = P(class 0)",

    "prediction_rule":
        "fake_probability >= threshold -> FAKE",

    "synthetic_dataset":
        {
            "total":
                len(synthetic_samples),

            "fake":
                synthetic_fake,

            "real":
                synthetic_real,

            "train":
                len(synthetic_train),

            "validation":
                len(synthetic_val),

            "test":
                len(synthetic_test),
        },

    "ffpp_replay":
        {
            "total":
                len(ffpp_replay),

            "fake":
                sum(
                    1
                    for _, label
                    in ffpp_replay
                    if label == FAKE
                ),

            "real":
                sum(
                    1
                    for _, label
                    in ffpp_replay
                    if label == REAL
                ),

            "source":
                str(
                    FFPP_TRAIN_CSV
                ),

            "path_column":
                "face_path",
        },

    "training":
        {
            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "backbone_lr":
                BACKBONE_LR,

            "classifier_lr":
                CLASSIFIER_LR,

            "weight_decay":
                WEIGHT_DECAY,

            "image_size":
                IMAGE_SIZE,

            "normalization":
                {
                    "mean":
                        [0.5, 0.5, 0.5],

                    "std":
                        [0.5, 0.5, 0.5],
                },

        },

    "xception_v2":
        {
            "best_checkpoint":
                str(
                    XCEPTION_V2_BEST
                ),

            "last_checkpoint":
                str(
                    XCEPTION_V2_LAST
                ),

            "best_epoch":
                xception_results[
                    "best_epoch"
                ],

            "best_val_auc":
                xception_results[
                    "best_val_auc"
                ],

            "synthetic_test":
                xception_results[
                    "final_synthetic_test"
                ],

            "ffpp_replay":
                xception_results[
                    "final_ffpp_replay"
                ],

            "history":
                xception_results[
                    "history"
                ],
        },

    "efficientnet_b4_v2":
        {
            "best_checkpoint":
                str(
                    EFFICIENTNET_V2_BEST
                ),

            "last_checkpoint":
                str(
                    EFFICIENTNET_V2_LAST
                ),

            "best_epoch":
                efficientnet_results[
                    "best_epoch"
                ],

            "best_val_auc":
                efficientnet_results[
                    "best_val_auc"
                ],

            "synthetic_test":
                efficientnet_results[
                    "final_synthetic_test"
                ],

            "ffpp_replay":
                efficientnet_results[
                    "final_ffpp_replay"
                ],

            "history":
                efficientnet_results[
                    "history"
                ],
        },

}


RESULT_JSON = (
    V2_RESULT_DIR /
    "finetune_v2_results.json"
)

with open(
    RESULT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results_json,
        file,
        indent=4
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 70)
print("V2 CHECKPOINTS")
print("=" * 70)

print()

print(
    "Xception V2 BEST:"
)

print(
    XCEPTION_V2_BEST
)

print()

print(
    "EfficientNet-B4 V2 BEST:"
)

print(
    EFFICIENTNET_V2_BEST
)

print()

print(
    "Results JSON:"
)

print(
    RESULT_JSON
)

print()
print("=" * 70)

print(
    "✓ FORENSIQ V2 FINISHED SUCCESSFULLY"
)

print("=" * 70)