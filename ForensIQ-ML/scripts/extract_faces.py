import csv
from pathlib import Path

import torch
from PIL import Image
from facenet_pytorch import MTCNN


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FRAMES_METADATA = ROOT / "dataset" / "frames_metadata.csv"
OUTPUT_DIR = ROOT / "dataset" / "faces"
METADATA_FILE = ROOT / "dataset" / "faces_metadata.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========================================")
print("FORENSIQ INCREMENTAL FACE EXTRACTION")
print("========================================")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# CHECK FRAME METADATA
# ============================================================

if not FRAMES_METADATA.exists():
    raise FileNotFoundError(
        f"Missing frame metadata:\n{FRAMES_METADATA}"
    )


# ============================================================
# LOAD ALL FRAME METADATA
# ============================================================

print("\nLoading frame metadata...")

frame_rows = []

with open(
    FRAMES_METADATA,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    required_columns = {
        "frame_path",
        "video_id",
        "label",
        "manipulation_type",
        "frame_number",
    }

    missing = required_columns - set(
        reader.fieldnames or []
    )

    if missing:
        raise RuntimeError(
            f"frames_metadata.csv is missing columns: {missing}"
        )

    for row in reader:
        frame_rows.append(row)


print(f"Total frames in dataset: {len(frame_rows)}")


# ============================================================
# LOAD EXISTING FACE METADATA
# ============================================================

existing_rows = []

processed_frames = set()

if METADATA_FILE.exists():

    print("\nExisting face metadata found.")
    print("Checking already processed frames...")

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            existing_rows.append(row)

            if row.get("frame_path"):
                processed_frames.add(
                    row["frame_path"]
                )

    print(
        f"Existing face records : "
        f"{len(existing_rows)}"
    )

    print(
        f"Frames already processed: "
        f"{len(processed_frames)}"
    )

else:

    print("\nNo existing face metadata found.")
    print("Starting fresh.")


# ============================================================
# FIND ONLY NEW FRAMES
# ============================================================

new_frame_rows = [
    row
    for row in frame_rows
    if row["frame_path"] not in processed_frames
]


print("\n========================================")
print("INCREMENTAL CHECK")
print("========================================")

print(
    f"Total frames      : {len(frame_rows)}"
)

print(
    f"Already processed : {len(processed_frames)}"
)

print(
    f"NEW frames        : {len(new_frame_rows)}"
)


if len(new_frame_rows) == 0:

    print("\n========================================")
    print("NOTHING NEW TO PROCESS")
    print("========================================")
    print("All frames already have face extraction.")
    print("No files were changed.")

    raise SystemExit(0)


# ============================================================
# FACE DETECTOR
# ============================================================

print("\nLoading MTCNN face detector...")

mtcnn = MTCNN(
    image_size=299,
    margin=0,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=False,
    keep_all=True,
    device=device,
)


# ============================================================
# PROCESS ONLY NEW FRAMES
# ============================================================

new_metadata = []

total_frames = 0
total_faces = 0
failed_images = 0
no_face = 0


print("\n========================================")
print("PROCESSING NEW FRAMES ONLY")
print("========================================")


for index, row in enumerate(
    new_frame_rows,
    start=1
):

    frame_path = ROOT / row["frame_path"]

    label = row["label"]
    manipulation_type = row["manipulation_type"]
    video_id = row["video_id"]

    if not frame_path.exists():

        print(
            f"[WARNING] Missing frame: "
            f"{frame_path}"
        )

        failed_images += 1
        continue

    total_frames += 1

    try:

        image = Image.open(
            frame_path
        ).convert("RGB")

        boxes, probabilities = mtcnn.detect(
            image
        )

        if boxes is None:

            no_face += 1
            continue

        for face_index, (
            box,
            probability
        ) in enumerate(
            zip(boxes, probabilities)
        ):

            if probability is None:
                continue

            if probability < 0.90:
                continue

            x1, y1, x2, y2 = [
                int(v)
                for v in box
            ]

            # ------------------------------------------------
            # 20% CONTEXT PADDING
            # ------------------------------------------------

            width = x2 - x1
            height = y2 - y1

            pad_x = int(
                width * 0.20
            )

            pad_y = int(
                height * 0.20
            )

            x1 = max(
                0,
                x1 - pad_x
            )

            y1 = max(
                0,
                y1 - pad_y
            )

            x2 = min(
                image.width,
                x2 + pad_x
            )

            y2 = min(
                image.height,
                y2 + pad_y
            )

            face_crop = image.crop(
                (x1, y1, x2, y2)
            )

            # ------------------------------------------------
            # XCEPTION INPUT SIZE
            # ------------------------------------------------

            face_crop = face_crop.resize(
                (299, 299),
                Image.Resampling.LANCZOS
            )

            # ------------------------------------------------
            # UNIQUE FILENAME
            # ------------------------------------------------

            output_name = (
                f"{label}_"
                f"{manipulation_type}_"
                f"{video_id}_"
                f"frame_{row['frame_number']}_"
                f"face_{face_index}.jpg"
            )

            output_path = (
                OUTPUT_DIR
                / label
                / manipulation_type
                / output_name
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # ------------------------------------------------
            # DO NOT OVERWRITE EXISTING FACE
            # ------------------------------------------------

            if output_path.exists():
                continue

            face_crop.save(
                output_path,
                "JPEG",
                quality=95
            )

            # ------------------------------------------------
            # ADD METADATA
            # ------------------------------------------------

            new_metadata.append([
                str(
                    output_path.relative_to(ROOT)
                ),

                str(
                    frame_path.relative_to(ROOT)
                ),

                video_id,
                label,
                manipulation_type,
                row["frame_number"],
                face_index,
                float(probability),
            ])

            total_faces += 1

    except Exception as e:

        failed_images += 1

        print(
            f"[WARNING] Failed: "
            f"{frame_path.name}"
        )

        print(
            f"         {e}"
        )

    if index % 100 == 0:

        print(
            f"Processed "
            f"{index}/{len(new_frame_rows)} "
            f"| New faces: {total_faces}"
        )


# ============================================================
# APPEND NEW METADATA
# ============================================================

if new_metadata:

    print("\nAppending new face metadata...")

    file_exists = METADATA_FILE.exists()

    with open(
        METADATA_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "face_path",
                "frame_path",
                "video_id",
                "label",
                "manipulation_type",
                "frame_number",
                "face_index",
                "detector_confidence",
            ])

        writer.writerows(
            new_metadata
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("INCREMENTAL FACE EXTRACTION COMPLETE")
print("========================================")

print(
    f"New frames processed : "
    f"{total_frames}"
)

print(
    f"New faces extracted  : "
    f"{total_faces}"
)

print(
    f"No face detected     : "
    f"{no_face}"
)

print(
    f"Failed images        : "
    f"{failed_images}"
)

print("\n========================================")
print("RESULT")
print("========================================")

print(
    f"Previous face records : "
    f"{len(existing_rows)}"
)

print(
    f"New face records      : "
    f"{len(new_metadata)}"
)

print(
    f"Total face records    : "
    f"{len(existing_rows) + len(new_metadata)}"
)

print("\nFaces directory:")
print(OUTPUT_DIR)

print("\nMetadata:")
print(METADATA_FILE)

print("\n========================================")
print("SAFE TO RUN DATASET SPLIT")
print("========================================")