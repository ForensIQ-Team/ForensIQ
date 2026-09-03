import cv2
import csv
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

OUTPUT = DATASET / "frames"
METADATA = DATASET / "frames_metadata.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FRAMES_PER_VIDEO = 30
JPEG_QUALITY = 95


# ============================================================
# DATASET SOURCES
# ============================================================

REAL_SOURCES = [
    (
        DATASET / "original_sequences" / "youtube" / "c23" / "videos",
        "original"
    ),
]

FAKE_SOURCES = [

    (
        DATASET / "manipulated_sequences" / "Deepfakes" / "c23" / "videos",
        "Deepfakes"
    ),

    (
        DATASET / "manipulated_sequences" / "Face2Face" / "c23" / "videos",
        "Face2Face"
    ),

    (
        DATASET / "manipulated_sequences" / "FaceSwap" / "c23" / "videos",
        "FaceSwap"
    ),

    (
        DATASET / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos",
        "NeuralTextures"
    ),
]


# ============================================================
# PREPARE OUTPUT
# ============================================================

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# GET FRAME INDICES
# ============================================================

def get_frame_indices(total_frames):

    if total_frames <= 0:
        return []

    if total_frames <= MAX_FRAMES_PER_VIDEO:
        return list(range(total_frames))

    return [
        int(
            i * (total_frames - 1)
            / (MAX_FRAMES_PER_VIDEO - 1)
        )
        for i in range(MAX_FRAMES_PER_VIDEO)
    ]


# ============================================================
# CHECK WHETHER FRAME EXISTS
# ============================================================

def frame_exists(path):

    if not path.exists():
        return False

    try:
        return path.stat().st_size > 1000
    except Exception:
        return False


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def process_video(
    video_path,
    output_dir,
    label,
    manipulation_type
):

    video_id = video_path.stem

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print(
            f"[WARNING] Could not open: "
            f"{video_path.name}"
        )

        return 0, 0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:

        print(
            f"[WARNING] No frames: "
            f"{video_path.name}"
        )

        cap.release()

        return 0, 0

    frame_indices = get_frame_indices(
        total_frames
    )

    # --------------------------------------------------------
    # FIND MISSING FRAMES
    # --------------------------------------------------------

    missing_indices = []

    for frame_number in frame_indices:

        output_name = (
            f"{video_id}"
            f"_frame_{frame_number:06d}.jpg"
        )

        output_path = output_dir / output_name

        if not frame_exists(output_path):

            missing_indices.append(
                frame_number
            )

    # --------------------------------------------------------
    # VIDEO ALREADY COMPLETE
    # --------------------------------------------------------

    if len(missing_indices) == 0:

        cap.release()

        return len(frame_indices), 0

    # --------------------------------------------------------
    # EXTRACT ONLY MISSING FRAMES
    # --------------------------------------------------------

    saved = 0

    for frame_number in missing_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        ret, frame = cap.read()

        if not ret:

            print(
                f"[WARNING] Could not read frame "
                f"{frame_number} from "
                f"{video_path.name}"
            )

            continue

        output_name = (
            f"{video_id}"
            f"_frame_{frame_number:06d}.jpg"
        )

        output_path = (
            output_dir / output_name
        )

        success = cv2.imwrite(
            str(output_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )

        if success:
            saved += 1

    cap.release()

    return len(frame_indices), saved


# ============================================================
# PROCESS DATASET SOURCE
# ============================================================

def process_source(
    video_dir,
    label,
    manipulation_type
):

    if not video_dir.exists():

        print(
            f"\n[WARNING] Dataset folder does not exist:"
            f"\n{video_dir}"
        )

        return []

    videos = sorted(
        video_dir.glob("*.mp4")
    )

    print("\n========================================")
    print(f"LABEL        : {label.upper()}")
    print(f"TYPE         : {manipulation_type}")
    print(f"VIDEOS FOUND : {len(videos)}")
    print("========================================")

    label_output = OUTPUT / label

    label_output.mkdir(
        parents=True,
        exist_ok=True
    )

    source_results = []

    for video_number, video_path in enumerate(
        videos,
        start=1
    ):

        expected, newly_saved = process_video(
            video_path,
            label_output,
            label,
            manipulation_type
        )

        if newly_saved == 0:

            print(
                f"[{video_number:3}/{len(videos)}] "
                f"{video_path.name}: "
                f"already complete "
                f"({expected} frames)"
            )

        else:

            print(
                f"[{video_number:3}/{len(videos)}] "
                f"{video_path.name}: "
                f"+{newly_saved} frames "
                f"(total {expected})"
            )

        source_results.append(
            (
                video_path.stem,
                label,
                manipulation_type,
                expected
            )
        )

    return source_results


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

all_videos = []

print("\n========================================")
print("FORENSIQ RESUMABLE FRAME EXTRACTION")
print("========================================")

print(
    "\nExisting frames will NOT be deleted."
)

# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

for video_dir, manipulation_type in REAL_SOURCES:

    all_videos.extend(
        process_source(
            video_dir,
            "real",
            manipulation_type
        )
    )


# ------------------------------------------------------------
# FAKE
# ------------------------------------------------------------

for video_dir, manipulation_type in FAKE_SOURCES:

    all_videos.extend(
        process_source(
            video_dir,
            "fake",
            manipulation_type
        )
    )


# ============================================================
# REBUILD METADATA
# ============================================================

print("\n========================================")
print("REBUILDING FRAME METADATA")
print("========================================")

rows = []


for video_id, label, manipulation_type, expected_count in all_videos:

    label_output = OUTPUT / label

    # Find all frames belonging to this video
    video_frames = sorted(
        label_output.glob(
            f"{video_id}_frame_*.jpg"
        )
    )

    for frame_path in video_frames:

        # Extract frame number from filename
        try:

            frame_number = int(
                frame_path.stem.split(
                    "_frame_"
                )[1]
            )

        except Exception:

            continue

        if not frame_exists(frame_path):
            continue

        rows.append([
            str(
                frame_path.relative_to(ROOT)
            ),
            video_id,
            label,
            manipulation_type,
            frame_number
        ])


# ============================================================
# SAVE METADATA CSV
# ============================================================

with open(
    METADATA,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame_path",
        "video_id",
        "label",
        "manipulation_type",
        "frame_number"
    ])

    writer.writerows(rows)


# ============================================================
# FINAL COUNTS
# ============================================================

print("\n========================================")
print("FRAME EXTRACTION COMPLETE")
print("========================================")

print(
    f"Total videos processed : "
    f"{len(all_videos)}"
)

print(
    f"Total valid frames     : "
    f"{len(rows)}"
)


# ------------------------------------------------------------
# COUNT BY TYPE
# ------------------------------------------------------------

counts = {}

for row in rows:

    manipulation_type = row[3]

    counts[manipulation_type] = (
        counts.get(
            manipulation_type,
            0
        ) + 1
    )


print("\nFrames by manipulation type:")

for manipulation_type in [
    "original",
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures"
]:

    print(
        f"{manipulation_type:20} : "
        f"{counts.get(manipulation_type, 0)}"
    )


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

print("\nFrames directory:")
print(OUTPUT)

print("\nMetadata:")
print(METADATA)

print("\n========================================")
print("SAFE TO CONTINUE TO FACE EXTRACTION")
print("========================================")