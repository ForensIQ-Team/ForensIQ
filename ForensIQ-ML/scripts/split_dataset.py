import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# FORENSIQ VIDEO-LEVEL DATASET SPLIT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = ROOT / "dataset" / "faces_metadata.csv"
SPLIT_DIR = ROOT / "dataset" / "split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)


# ============================================================
# VALIDATE RATIOS
# ============================================================

if abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) > 1e-6:

    raise RuntimeError(
        "TRAIN + VAL + TEST ratios must equal 1.0"
    )


# ============================================================
# CHECK METADATA
# ============================================================

if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Missing face metadata:\n{METADATA_FILE}"
    )


print("\n========================================")
print("FORENSIQ VIDEO-LEVEL DATASET SPLIT")
print("========================================")


# ============================================================
# LOAD FACE METADATA
# ============================================================

print("\nLoading face metadata...")

metadata_rows = []

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    required_columns = {
        "face_path",
        "frame_path",
        "video_id",
        "label",
        "manipulation_type",
        "frame_number",
        "face_index",
        "detector_confidence",
    }

    missing = (
        required_columns
        - set(reader.fieldnames or [])
    )

    if missing:

        raise RuntimeError(
            f"Missing columns in faces_metadata.csv: {missing}"
        )

    for row in reader:
        metadata_rows.append(row)


print(
    f"Total face records: "
    f"{len(metadata_rows)}"
)


# ============================================================
# GROUP BY VIDEO
#
# IMPORTANT:
# Every video stays entirely inside ONE split.
# ============================================================

video_groups = defaultdict(list)

for row in metadata_rows:

    key = (
        row["label"],
        row["manipulation_type"],
        row["video_id"]
    )

    video_groups[key].append(row)


# ============================================================
# DISPLAY SOURCE DISTRIBUTION
# ============================================================

print("\n========================================")
print("SOURCE DISTRIBUTION")
print("========================================")


source_counts = defaultdict(set)

for label, manipulation, video_id in video_groups:

    source_counts[
        (label, manipulation)
    ].add(video_id)


for (
    label,
    manipulation
), video_ids in sorted(source_counts.items()):

    print(
        f"{label.upper():5} | "
        f"{manipulation:20} | "
        f"{len(video_ids):4} videos"
    )


print(
    f"\nTotal unique videos: "
    f"{len(video_groups)}"
)


# ============================================================
# CREATE SPLIT CONTAINERS
# ============================================================

splits = {
    "train": [],
    "val": [],
    "test": []
}


# ============================================================
# STRATIFIED VIDEO SPLIT
#
# We split each manipulation type independently.
# ============================================================

print("\n========================================")
print("CREATING STRATIFIED VIDEO SPLIT")
print("========================================")


for (
    label,
    manipulation
) in sorted(
    set(
        (
            key[0],
            key[1]
        )
        for key in video_groups
    )
):

    video_ids = [

        video_id

        for (
            video_label,
            video_manipulation,
            video_id
        ) in video_groups

        if (
            video_label == label
            and
            video_manipulation == manipulation
        )
    ]


    random.shuffle(video_ids)


    total = len(video_ids)


    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )


    # Ensure very small groups don't disappear
    # completely from validation/test where possible.

    if total >= 3:

        train_count = max(
            1,
            train_count
        )

        val_count = max(
            1,
            val_count
        )


    test_count = (
        total
        - train_count
        - val_count
    )


    # Safety adjustment

    if test_count < 1 and total >= 3:

        test_count = 1
        train_count -= 1


    train_videos = video_ids[
        :train_count
    ]

    val_videos = video_ids[
        train_count:
        train_count + val_count
    ]

    test_videos = video_ids[
        train_count + val_count:
    ]


    splits["train"].extend(
        (
            label,
            manipulation,
            video_id
        )
        for video_id in train_videos
    )

    splits["val"].extend(
        (
            label,
            manipulation,
            video_id
        )
        for video_id in val_videos
    )

    splits["test"].extend(
        (
            label,
            manipulation,
            video_id
        )
        for video_id in test_videos
    )


# ============================================================
# VIDEO SPLIT SUMMARY
# ============================================================

print("\n========================================")
print("VIDEO-LEVEL SPLIT")
print("========================================")


for split_name in [
    "train",
    "val",
    "test"
]:

    real_count = sum(
        1
        for label, _, _
        in splits[split_name]
        if label == "real"
    )

    fake_count = sum(
        1
        for label, _, _
        in splits[split_name]
        if label == "fake"
    )

    print(
        f"{split_name.upper():5} | "
        f"REAL: {real_count:4} | "
        f"FAKE: {fake_count:4} | "
        f"TOTAL: {real_count + fake_count:4}"
    )


# ============================================================
# MANIPULATION DISTRIBUTION
# ============================================================

print("\n========================================")
print("MANIPULATION DISTRIBUTION")
print("========================================")


for split_name in [
    "train",
    "val",
    "test"
]:

    print(f"\n{split_name.upper()}")


    counts = defaultdict(int)

    for (
        label,
        manipulation,
        video_id
    ) in splits[split_name]:

        counts[
            (label, manipulation)
        ] += 1


    for (
        label,
        manipulation
    ), count in sorted(
        counts.items()
    ):

        print(
            f"  {label.upper():5} | "
            f"{manipulation:20} | "
            f"{count:4} videos"
        )


# ============================================================
# REMOVE OLD SPLIT DATASET
# ============================================================

if SPLIT_DIR.exists():

    print("\n========================================")
    print("REMOVING OLD SPLIT DATASET")
    print("========================================")

    print(SPLIT_DIR)

    shutil.rmtree(SPLIT_DIR)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for split_name in [
    "train",
    "val",
    "test"
]:

    for label in [
        "real",
        "fake"
    ]:

        (
            SPLIT_DIR
            / split_name
            / label
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY FACE CROPS
# ============================================================

print("\n========================================")
print("COPYING FACE CROPS")
print("========================================")


split_rows = {
    "train": [],
    "val": [],
    "test": []
}


for split_name in [
    "train",
    "val",
    "test"
]:

    print(
        f"\nProcessing "
        f"{split_name.upper()}..."
    )


    copied = 0


    for (
        label,
        manipulation,
        video_id
    ) in splits[split_name]:


        rows = video_groups[
            (
                label,
                manipulation,
                video_id
            )
        ]


        for row in rows:

            source = (
                ROOT
                / row["face_path"]
            )


            if not source.exists():

                print(
                    f"[WARNING] Missing: "
                    f"{source}"
                )

                continue


            destination = (
                SPLIT_DIR
                / split_name
                / label
                / manipulation
                / source.name
            )


            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            shutil.copy2(
                source,
                destination
            )


            split_rows[
                split_name
            ].append([

                str(
                    destination.relative_to(
                        ROOT
                    )
                ),

                row["frame_path"],
                video_id,
                label,
                manipulation,
                row["frame_number"],
                row["face_index"],
                row["detector_confidence"],

            ])


            copied += 1


    print(
        f"Copied faces: {copied}"
    )


# ============================================================
# SAVE SPLIT METADATA
# ============================================================

print("\n========================================")
print("SAVING SPLIT METADATA")
print("========================================")


header = [
    "face_path",
    "frame_path",
    "video_id",
    "label",
    "manipulation_type",
    "frame_number",
    "face_index",
    "detector_confidence",
]


for split_name in [
    "train",
    "val",
    "test"
]:

    csv_path = (
        SPLIT_DIR
        / f"{split_name}.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            header
        )

        writer.writerows(
            split_rows[split_name]
        )


    print(
        f"{split_name}.csv saved"
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n========================================")
print("LEAKAGE CHECK")
print("========================================")


video_sets = {}

for split_name in [
    "train",
    "val",
    "test"
]:

    video_sets[split_name] = set(

        (
            label,
            manipulation,
            video_id
        )

        for (
            label,
            manipulation,
            video_id
        ) in splits[split_name]

    )


leakage_found = False


for a, b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:

    overlap = (
        video_sets[a]
        &
        video_sets[b]
    )


    if overlap:

        leakage_found = True

        print(
            f"[ERROR] "
            f"{a} <-> {b}: "
            f"{len(overlap)} overlapping videos"
        )

    else:

        print(
            f"✓ {a} <-> {b}: "
            f"No overlap"
        )


if leakage_found:

    raise RuntimeError(
        "\nDATASET LEAKAGE DETECTED.\n"
        "Training must NOT continue."
    )


# ============================================================
# FINAL FACE COUNTS
# ============================================================

print("\n========================================")
print("FINAL FACE DATASET")
print("========================================")


for split_name in [
    "train",
    "val",
    "test"
]:

    real_faces = len(
        list(
            (
                SPLIT_DIR
                / split_name
                / "real"
            ).rglob("*.jpg")
        )
    )


    fake_faces = len(
        list(
            (
                SPLIT_DIR
                / split_name
                / "fake"
            ).rglob("*.jpg")
        )
    )


    total_faces = (
        real_faces
        + fake_faces
    )


    print(
        f"{split_name.upper():5} | "
        f"REAL faces: {real_faces:5} | "
        f"FAKE faces: {fake_faces:5} | "
        f"TOTAL: {total_faces:5}"
    )


# ============================================================
# MANIPULATION FACE COUNTS
# ============================================================

print("\n========================================")
print("FACE COUNTS BY MANIPULATION")
print("========================================")


for split_name in [
    "train",
    "val",
    "test"
]:

    print(
        f"\n{split_name.upper()}"
    )


    manipulation_counts = defaultdict(int)


    for row in split_rows[split_name]:

        manipulation = row[4]

        manipulation_counts[
            manipulation
        ] += 1


    for manipulation, count in sorted(
        manipulation_counts.items()
    ):

        print(
            f"  {manipulation:20} : "
            f"{count:5}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("DATASET SPLIT COMPLETE")
print("========================================")

print("\nDataset:")
print(SPLIT_DIR)

print("\nMetadata:")
print(
    SPLIT_DIR / "train.csv"
)
print(
    SPLIT_DIR / "val.csv"
)
print(
    SPLIT_DIR / "test.csv"
)

print("\n========================================")
print("SAFE TO START MODEL TRAINING")
print("========================================")