# ================================================================
# ForensIQ — Face Verifier
#
# Extracts 512-d L2-normalized face embeddings using facenet-pytorch
# (MTCNN for face detection + InceptionResnetV1 pretrained on VGGFace2).
# Compares facial identity via cosine similarity.
# ================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

logger = logging.getLogger("forensiq-propagation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mtcnn: Optional[MTCNN] = None
_resnet: Optional[InceptionResnetV1] = None


def _get_models() -> tuple[MTCNN, InceptionResnetV1]:
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(
            keep_all=True,
            device=DEVICE,
            post_process=True,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
        )
        logger.info("[FaceVerifier] MTCNN loaded on %s", DEVICE)
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        logger.info("[FaceVerifier] InceptionResnetV1 (vggface2) loaded on %s", DEVICE)
    return _mtcnn, _resnet


@dataclass
class FaceEmbeddingResult:
    success: bool
    embedding: Optional[np.ndarray]  # L2-normalized, shape (512,)
    num_faces: int
    error: Optional[str] = None


def extract_face_embedding(image: Image.Image) -> FaceEmbeddingResult:
    """
    Detects faces in the given image, crops the largest face,
    resizes to 160x160, passes through InceptionResnetV1, and returns
    a 512-d L2-normalized face embedding.

    Never raises an exception — returns FaceEmbeddingResult(success=False, ...)
    on error.
    """
    try:
        if image is None:
            return FaceEmbeddingResult(
                success=False, embedding=None, num_faces=0, error="image_is_none"
            )

        mtcnn, resnet = _get_models()

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Detect all faces with MTCNN
        boxes, _ = mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return FaceEmbeddingResult(
                success=False,
                embedding=None,
                num_faces=0,
                error="no_face_detected",
            )

        num_faces = len(boxes)

        # 3. Pick largest face by area
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = int(np.argmax(areas))
        box = boxes[largest_idx]

        # Clamp box coordinates to image dimensions
        w, h = image.size
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))

        if x2 <= x1 or y2 <= y1:
            return FaceEmbeddingResult(
                success=False,
                embedding=None,
                num_faces=num_faces,
                error="invalid_face_crop_bounds",
            )

        # 4. Crop, resize to 160x160, normalize to [-1, 1]
        face_crop = image.crop((x1, y1, x2, y2)).resize(
            (160, 160), Image.Resampling.BILINEAR
        )
        face_arr = np.array(face_crop, dtype=np.float32)
        # Normalize to [-1, 1]
        face_arr = (face_arr - 127.5) / 128.0

        # Tensor shape (1, 3, 160, 160)
        face_tensor = torch.from_numpy(face_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        # 5. Forward through InceptionResnetV1
        with torch.no_grad():
            raw_emb = resnet(face_tensor).cpu().numpy().squeeze(0)

        # 6. L2-normalize output
        norm = np.linalg.norm(raw_emb)
        if norm > 0:
            embedding = (raw_emb / norm).astype(np.float32)
        else:
            embedding = raw_emb.astype(np.float32)

        return FaceEmbeddingResult(
            success=True,
            embedding=embedding,
            num_faces=num_faces,
            error=None,
        )

    except Exception as exc:
        logger.warning(f"[FaceVerifier] Embedding extraction failed: {exc}")
        return FaceEmbeddingResult(
            success=False,
            embedding=None,
            num_faces=0,
            error=str(exc),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two unit vectors.
    Since both vectors are L2-normalized, this is equivalent to their dot product.
    """
    try:
        return float(np.dot(a, b))
    except Exception:
        return 0.0


def classify_face_match(sim: float) -> str:
    """
    Classifies face similarity:
      sim >= 0.65       -> 'same_person'
      0.40 <= sim < 0.65 -> 'similar_face'
      sim < 0.40        -> 'different_person'
    """
    if sim >= 0.65:
        return "same_person"
    if sim >= 0.40:
        return "similar_face"
    return "different_person"
