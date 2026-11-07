"""Number-plate reading (ANPR) for Indian plates.

Plate localisation: PP-OCRv3 DB text detector (Apache-2.0, OpenCV Zoo).
Plate reading: fast-plate-ocr "cct-xs-v2-global" (MIT), a recogniser trained on
licence plates from many countries; it read real Indian plate crops about twice
as accurately as a generic OCR model. If it is not installed, the generic CRNN
(Apache-2.0, OpenCV Zoo) is used instead. Recognised strings are
normalised to Indian plate formats with position-aware corrections (an O in a
digit position becomes 0, an 8 in a letter position becomes B, and so on).

Search accepts full or partial plates ("2780", "MH01 DE") and tolerates OCR
errors with edit distance, so a witness's partial recollection still finds
candidates.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
# Standard series (e.g. MH01DE2780, DL3CAB1234) and Bharat series (22BH1234AA).
PLATE_RES = [re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$"), re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")]
STATE_CODES = set("AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB".split())
# Look-alike corrections by slot, from errors observed on real Indian plates (benchmarks/plates_faces_eval.py).
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "J": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4", "Y": "4", "E": "3"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "7": "T", "4": "A", "3": "B"}
# Visually confusable letters, used to snap an invalid state code to a valid one ("WH" -> "MH").
CONFUSABLE = {frozenset(p) for p in ("WM", "NM", "VM", "HN", "KX", "PR", "DO", "OQ", "CG", "UV", "TI", "BR", "AR")}

_lock = threading.Lock()
_models: dict = {}


def _paths():
    from huggingface_hub import hf_hub_download

    return (hf_hub_download("opencv/text_detection_ppocr", "text_detection_en_ppocrv3_2023may.onnx"),
            hf_hub_download("opencv/text_recognition_crnn", "text_recognition_CRNN_EN_2021sep.onnx"))


def _load():
    with _lock:
        if "det" not in _models:
            det_path, rec_path = _paths()
            det = cv2.dnn_TextDetectionModel_DB(det_path)
            det.setBinaryThreshold(0.3)
            det.setPolygonThreshold(0.5)
            det.setMaxCandidates(200)
            det.setUnclipRatio(2.0)
            _models["det"] = det
            _models["rec"] = cv2.dnn.readNet(rec_path)
        return _models["det"], _models["rec"]


def _plate_model():
    with _lock:
        if "fpo" not in _models:
            try:
                from fast_plate_ocr import LicensePlateRecognizer

                _models["fpo"] = LicensePlateRecognizer("cct-xs-v2-global-model")
            except Exception:  # noqa: BLE001 - optional dependency or offline
                _models["fpo"] = None
        return _models["fpo"]


def recognise_plate_crop(img: np.ndarray) -> str:
    """Read one plate image (already cropped to the plate)."""
    m = _plate_model()
    if m is not None:
        out = m.run(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        t = out[0] if isinstance(out, (list, tuple)) else out
        return str(getattr(t, "plate", t)).upper().replace("_", "")
    _, rec = _load()
    return _recognise(rec, img, None)


def _recognise(rec, img: np.ndarray, quad: np.ndarray | None) -> str:
    if quad is not None:
        q = quad.astype(np.float32)
        # Order points: bottom-left, top-left, top-right, bottom-right (OpenCV DB output order).
        dst = np.array([[0, 31], [0, 0], [99, 0], [99, 31]], np.float32)
        crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(q, dst), (100, 32))
    else:
        crop = cv2.resize(img, (100, 32))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blob = cv2.dnn.blobFromImage(gray, scalefactor=1 / 127.5, size=(100, 32), mean=127.5)
    rec.setInput(blob)
    out = rec.forward()  # (T, 1, 37)
    chars, prev = [], 0
    for t in range(out.shape[0]):
        c = int(np.argmax(out[t][0]))
        if c != 0 and c != prev:
            chars.append(CHARSET[c - 1])
        prev = c
    return "".join(chars).upper()


def _fit(chars: str, pattern: str) -> str | None:
    """Coerce chars to a slot pattern of L (letter) / D (digit), correcting look-alikes; None if impossible."""
    if len(chars) != len(pattern):
        return None
    out = []
    for ch, slot in zip(chars, pattern):
        if slot == "D":
            ch = TO_DIGIT.get(ch, ch)
            if not ch.isdigit():
                return None
        else:
            ch = TO_LETTER.get(ch, ch)
            if not ch.isalpha():
                return None
        out.append(ch)
    return "".join(out)


def _snap_state(plate: str) -> str:
    st = plate[:2]
    if st in STATE_CODES:
        return plate
    cands = []
    for code in STATE_CODES:
        diff = [(a, b) for a, b in zip(st, code) if a != b]
        if len(diff) == 1:
            cands.append((frozenset(diff[0]) in CONFUSABLE, code))
    good = [c for ok, c in cands if ok]
    if len(good) == 1:
        return good[0] + plate[2:]
    return plate


def normalise_plate(raw: str) -> tuple[str, bool]:
    """Return (best plate string, matches_indian_format)."""
    s = re.sub(r"[^A-Z0-9]", "", (raw or "").upper())
    if s.startswith("IND"):
        s = s[3:]
    best = None
    for n_mid in (0, 1, 2, 3):
        for n_rto in (1, 2):
            pattern = "LL" + "D" * n_rto + "L" * n_mid + "DDDD"
            fixed = _fit(s, pattern)
            if fixed and any(r.match(fixed) for r in PLATE_RES):
                fixed = _snap_state(fixed)
                score = (fixed[:2] in STATE_CODES, n_rto == 2)
                if best is None or score > best[0]:
                    best = (score, fixed)
    bh = _fit(s, "DDLLDDDDL") or _fit(s, "DDLLDDDDLL")
    if bh and bh[2:4] == "BH":
        return bh, True
    if best:
        return best[1], True
    return s, False


@dataclass
class PlateRead:
    text: str
    raw: str
    valid_format: bool
    bbox: tuple  # in the input image


def read_plates(img: np.ndarray, max_side: int = 736) -> list:
    """Detect text regions in an image (e.g. a vehicle crop) and return plate candidates."""
    det, rec = _load()
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    W, H = max(32, int(round(w * scale / 32)) * 32), max(32, int(round(h * scale / 32)) * 32)
    det.setInputParams(1.0 / 255.0, (W, H), (122.67891434, 116.66876762, 104.00698793))
    quads, _conf = det.detect(img)
    pieces = []
    use_plate_model = _plate_model() is not None
    for q in quads:
        q = np.array(q, np.float32)
        if use_plate_model:
            x, y, bw, bh = cv2.boundingRect(q.astype(np.int32))
            px, py = int(bw * 0.08), int(bh * 0.25)
            crop = img[max(0, y - py): y + bh + py, max(0, x - px): x + bw + px]
            raw = recognise_plate_crop(crop) if crop.size else ""
        else:
            raw = _recognise(rec, img, q)
        if len(raw) >= 2:
            x, y, bw, bh = cv2.boundingRect(q.astype(np.int32))
            pieces.append((y, x, raw, (x, y, bw, bh)))
    reads = []
    # Single-line reads.
    for _, _, raw, box in pieces:
        text, ok = normalise_plate(raw)
        if len(text) >= 4:
            reads.append(PlateRead(text, raw, ok, box))
    # Two-line plates (common on two-wheelers): join vertically stacked neighbours.
    pieces.sort()
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            (y1, x1, r1, b1), (y2, x2, r2, b2) = pieces[i], pieces[j]
            if 0 < y2 - y1 < 2.5 * b1[3] and abs(x1 - x2) < max(b1[2], b2[2]):
                text, ok = normalise_plate(r1 + r2)
                if ok:
                    x0, y0 = min(b1[0], b2[0]), min(b1[1], b2[1])
                    box = (x0, y0, max(b1[0] + b1[2], b2[0] + b2[2]) - x0, max(b1[1] + b1[3], b2[1] + b2[3]) - y0)
                    reads.append(PlateRead(text, r1 + "|" + r2, True, box))
    reads.sort(key=lambda r: (not r.valid_format, -len(r.text)))
    return reads


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def plate_match_score(query: str, candidate: str) -> float:
    """1.0 = exact; partial queries match as substrings; OCR errors tolerated by edit distance."""
    q = re.sub(r"[^A-Z0-9]", "", query.upper())
    c = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if not q or not c:
        return 0.0
    if q == c:
        return 1.0
    if len(q) < len(c):
        best = min(edit_distance(q, c[i : i + len(q)]) for i in range(len(c) - len(q) + 1))
        return max(0.0, 0.95 - best / len(q))
    return max(0.0, 1.0 - edit_distance(q, c) / max(len(q), len(c)))
