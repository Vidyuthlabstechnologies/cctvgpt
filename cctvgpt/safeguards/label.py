"""SR-1: persistent visible "SYNTHETIC" marking on every output frame.

Three layers, so that cropping or covering one does not remove the marking:
  1. a solid badge in the bottom-right corner ("SYNTHETIC - AI GENERATED"),
  2. a smaller badge in the top-right corner ("SYNTHETIC"),
  3. a faint tiled "SYNTHETIC" pattern across the whole frame.

label_score() lets the pipeline check that a frame really carries the badge
before it is released (the "zero unlabelled outputs" metric in section 9).
"""

from __future__ import annotations

import functools

import cv2
import numpy as np

BADGE_TEXT = "SYNTHETIC - AI GENERATED"
TOP_TEXT = "SYNTHETIC"
BADGE_BG = (0, 0, 0)
BADGE_FG = (0, 215, 255)  # amber (BGR)
TILE_ALPHA = 0.07


def _font_scale(h: int) -> float:
    return max(0.38, h / 720 * 0.85)


@functools.lru_cache(maxsize=16)
def _layout(w: int, h: int):
    s = _font_scale(h)
    th = max(1, int(round(s * 2)))
    (tw, tht), base = cv2.getTextSize(BADGE_TEXT, cv2.FONT_HERSHEY_SIMPLEX, s, th)
    pad = max(3, int(h * 0.012))
    bx1, by1 = w - pad, h - pad
    bx0, by0 = bx1 - tw - 2 * pad, by1 - tht - base - 2 * pad
    badge = (bx0, by0, bx1, by1, (bx0 + pad, by1 - pad - base), s, th)
    s2 = s * 0.8
    th2 = max(1, int(round(s2 * 2)))
    (tw2, th2h), base2 = cv2.getTextSize(TOP_TEXT, cv2.FONT_HERSHEY_SIMPLEX, s2, th2)
    tx1, ty0 = w - pad, pad
    tx0, ty1 = tx1 - tw2 - 2 * pad, ty0 + th2h + base2 + 2 * pad
    top = (tx0, ty0, tx1, ty1, (tx0 + pad, ty1 - pad - base2), s2, th2)
    # Tiled pattern mask.
    tile = np.zeros((h, w), np.uint8)
    ts = s * 1.4
    tth = max(1, int(round(ts * 2)))
    (pw, ph), _ = cv2.getTextSize(TOP_TEXT, cv2.FONT_HERSHEY_SIMPLEX, ts, tth)
    step_x, step_y = pw + int(w * 0.12), ph * 5
    row = 0
    for y in range(ph * 3, h, step_y):
        off = (row % 2) * step_x // 2
        for x in range(-off, w, step_x):
            cv2.putText(tile, TOP_TEXT, (x, y), cv2.FONT_HERSHEY_SIMPLEX, ts, 255, tth, cv2.LINE_AA)
        row += 1
    # Reference rendering of the bottom badge, for verification.
    ref = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
    org = (badge[4][0] - bx0, badge[4][1] - by0)
    cv2.putText(ref, BADGE_TEXT, org, cv2.FONT_HERSHEY_SIMPLEX, s, 255, th, cv2.LINE_AA)
    return badge, top, (tile.astype(np.float32) / 255.0)[..., None], ref


def apply_label(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    badge, top, tile, _ = _layout(w, h)
    f = frame.astype(np.float32)
    f = f * (1 - TILE_ALPHA * tile) + 255.0 * TILE_ALPHA * tile
    out = np.clip(f, 0, 255).astype(np.uint8)
    for x0, y0, x1, y1, org, s, th in (badge, top):
        cv2.rectangle(out, (x0, y0), (x1, y1), BADGE_BG, -1)
        cv2.putText(out, BADGE_TEXT if (x0, y0) == badge[:2] else TOP_TEXT, org, cv2.FONT_HERSHEY_SIMPLEX, s, BADGE_FG, th, cv2.LINE_AA)
    return out


def label_score(frame: np.ndarray) -> float:
    """Correlation (0..1) between the expected badge and the frame's badge region."""
    h, w = frame.shape[:2]
    badge, _, _, ref = _layout(w, h)
    x0, y0, x1, y1 = badge[:4]
    roi = frame[y0:y1, x0:x1]
    if roi.shape[:2] != ref.shape:
        return 0.0
    # Amber text: high R and G, low B. Score text presence where expected and background darkness elsewhere.
    amber = (roi[..., 2].astype(np.float32) + roi[..., 1].astype(np.float32)) / 2 - roi[..., 0].astype(np.float32)
    a = amber - amber.mean()
    r = ref.astype(np.float32) - ref.mean()
    denom = float(np.sqrt((a * a).sum() * (r * r).sum()))
    return float((a * r).sum() / denom) if denom > 0 else 0.0


def has_label(frame: np.ndarray, threshold: float = 0.6) -> bool:
    return label_score(frame) >= threshold
