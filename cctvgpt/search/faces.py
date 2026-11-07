"""Face matching for investigations (opt-in, case-scoped, audited).

Detection: YuNet (MIT). Recognition: SFace (Apache-2.0). Both are OpenCV Zoo
ONNX models run with OpenCV on CPU.

What this is: investigator-led face MATCHING. An officer supplies a probe photo
of a person of interest; the system ranks faces found in that case's footage by
similarity, for a human to verify.

What this is not: identification against population databases, watch-lists,
live tracking or alerting. Those are not built.

Safeguards in the engine (cctvgpt/search/engine.py):
  * Off unless the deployment sets CCTVGPT_FACE_SEARCH_ENABLED=true. Faces are
    not even indexed until then.
  * Search is limited to sources ingested under the same case reference.
  * Every face search needs a written justification and is audited with a
    hash of the probe image.
  * Results below the model's verification threshold are labelled as weak, and
    tiny faces as unreliable. Every result is a lead for human review, never an
    identification.

Measured accuracy, including on CCTV-like degraded faces, is in docs/SEARCH.md.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

# SFace's published cosine threshold for "same person" on LFW-quality faces.
MATCH_THRESHOLD = 0.363
MIN_RELIABLE_FACE_PX = 40

_lock = threading.Lock()
_m: dict = {}


def _load():
    with _lock:
        if "rec" not in _m:
            from huggingface_hub import hf_hub_download

            det = hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx")
            rec = hf_hub_download("opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx")
            _m["det"] = cv2.FaceDetectorYN.create(det, "", (320, 320), 0.7, 0.3, 5000)
            _m["rec"] = cv2.FaceRecognizerSF.create(rec, "")
        return _m["det"], _m["rec"]


def detect_faces(img: np.ndarray, upscale_small: bool = True) -> list:
    """Return detection rows (x, y, w, h, 5 landmarks x/y, score) in input-image coordinates."""
    det, _ = _load()
    h, w = img.shape[:2]
    scale = 1.0
    if upscale_small and max(h, w) < 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    with _lock:
        det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = det.detect(img)
    if faces is None:
        return []
    faces = faces.copy()
    faces[:, :14] /= scale
    return [f for f in faces]


def embed_face(img: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    _, rec = _load()
    with _lock:
        aligned = rec.alignCrop(img, face_row)
        v = rec.feature(aligned).reshape(-1).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def largest_face(img: np.ndarray):
    faces = detect_faces(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])
