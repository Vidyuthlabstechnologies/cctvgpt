"""SR-2: invisible, keyed spread-spectrum watermark.

Scheme (vl-ss-dct-v1, implemented in the pixel domain):
  * A secret key seeds K = 1 + 48 pseudo-random +/-1 patterns on a 64x64 grid,
    upsampled to a 256x256 canonical grid. Pattern 0 is a "sync" pattern that
    signals presence; patterns 1..48 carry a 48-bit payload (32-bit clip token
    + 16-bit CRC).
  * Embedding adds s(t) * alpha * sum_k b_k P_k (resized to the frame,
    modulated by local texture) to the luma of every frame. The sign s(t)
    flips every FLIP_PERIOD_S seconds of clip time, so the mark is tied to
    time rather than frame index and survives frame-rate changes.
  * Detection resizes each frame's luma to 256x256, high-passes it, and
    correlates against the identically filtered patterns. Static scene
    content correlates identically in every frame, so subtracting the
    temporal median of the correlations removes it, leaving +/- the mark.
    Each frame is sign-aligned by its sync correlation and the payload is
    accumulated, so dropped, duplicated or trimmed frames cost little.

Because detection works on a canonical grid, the mark survives re-encoding,
resizing and moderate compression. It does NOT survive heavy cropping,
rotation or geometric warps; this is the "known limitation" in PRD 6.2 and is
stated in docs/SAFEGUARDS.md. Robustness numbers come from
scripts/eval_watermark.py and must be re-measured on every release [VERIFY].
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import zlib
from dataclasses import dataclass

import cv2
import numpy as np

GRID = 256
BASE = 64
PAYLOAD_BITS = 48
TOKEN_BITS = 32
N_PATTERNS = 1 + PAYLOAD_BITS
DEFAULT_ALPHA = 2.0  # luma std of the mark (0..255 scale)
FLIP_PERIOD_S = 0.25
MIN_FRAMES = 6


def new_token() -> int:
    return secrets.randbits(TOKEN_BITS)


def _crc16(token: int) -> int:
    return zlib.crc32(token.to_bytes(4, "big")) & 0xFFFF


def token_to_bits(token: int) -> np.ndarray:
    word = (token << 16) | _crc16(token)
    return np.array([1.0 if (word >> (PAYLOAD_BITS - 1 - i)) & 1 else -1.0 for i in range(PAYLOAD_BITS)], np.float32)


def bits_to_token(bits: np.ndarray) -> int | None:
    word = 0
    for b in bits:
        word = (word << 1) | (1 if b > 0 else 0)
    token, crc = word >> 16, word & 0xFFFF
    return token if _crc16(token) == crc else None


def _highpass(img: np.ndarray) -> np.ndarray:
    return img - cv2.GaussianBlur(img, (0, 0), 2.5)


@dataclass
class Detection:
    present: bool
    sync_z: float  # ratio of sync-pattern energy to payload-pattern energy (~1 when marked)
    token: int | None
    bit_confidence: float  # weakest payload bit t-statistic
    frames_used: int

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "sync_z": round(self.sync_z, 2),
            "token": self.token,
            "bit_confidence": round(self.bit_confidence, 2),
            "frames_used": self.frames_used,
        }


class Watermarker:
    def __init__(self, key: bytes, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        seed = int.from_bytes(hmac.new(key, b"vl-ss-dct-v1/patterns", hashlib.sha256).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        base = rng.choice(np.array([-1.0, 1.0], np.float32), size=(N_PATTERNS, BASE, BASE))
        pats = np.stack([cv2.resize(b, (GRID, GRID), interpolation=cv2.INTER_LINEAR) for b in base])
        pats -= pats.mean(axis=(1, 2), keepdims=True)
        pats /= pats.std(axis=(1, 2), keepdims=True)
        self.patterns = pats.astype(np.float32)  # (K, 256, 256), unit variance
        filt = np.stack([_highpass(p) for p in self.patterns])
        filt /= np.linalg.norm(filt.reshape(N_PATTERNS, -1), axis=1)[:, None, None]
        self.filtered = filt.reshape(N_PATTERNS, -1).astype(np.float32)  # unit L2 norm rows
        self._cache: dict = {}

    def _field(self, token: int, w: int, h: int) -> np.ndarray:
        key = (token, w, h)
        if key not in self._cache:
            bits = np.concatenate([[1.0], token_to_bits(token)]).astype(np.float32)
            wm = np.tensordot(bits, self.patterns, axes=1) / np.sqrt(N_PATTERNS)  # unit variance
            self._cache = {key: cv2.resize(wm, (w, h), interpolation=cv2.INTER_LINEAR)}
        return self._cache[key]

    def embed(self, frame_bgr: np.ndarray, token: int, t: float) -> np.ndarray:
        """Embed the mark into a frame shown at clip time t (seconds)."""
        h, w = frame_bgr.shape[:2]
        ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = ycc[..., 0]
        # Texture masking: stronger in busy regions, weaker in flat ones (less visible).
        act = cv2.GaussianBlur(np.abs(_highpass(y)), (0, 0), 3.0)
        mask = np.clip(0.6 + act / 6.0, 0.6, 2.0)
        # Keep away from clipping at black/white.
        room = np.clip(np.minimum(y, 255 - y) / 12.0, 0.2, 1.0)
        sign = 1.0 if int(np.floor(t / FLIP_PERIOD_S + 1e-6)) % 2 == 0 else -1.0
        ycc[..., 0] = y + sign * self.alpha * self._field(token, w, h) * mask * room
        return cv2.cvtColor(np.clip(ycc + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def correlate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return per-pattern z-scores for one frame."""
        if frame_bgr.ndim == 3:
            y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float32)
        else:
            y = frame_bgr.astype(np.float32)
        y = cv2.resize(y, (GRID, GRID), interpolation=cv2.INTER_AREA)
        res = _highpass(y).reshape(-1)
        sigma = float(res.std()) + 1e-6
        return (self.filtered @ res) / sigma

    def detect(self, frames, min_bit_t: float = 3.0) -> Detection:
        """Detect the mark in a sequence of frames from one clip.

        Presence requires (a) the 48-bit payload to pass its CRC-16 and (b) every
        payload bit to be decided with a t-statistic of at least min_bit_t.
        Under the null hypothesis the CRC alone passes with probability 2^-16.
        """
        zs = [self.correlate(f) for f in frames]
        if len(zs) < MIN_FRAMES:
            return Detection(False, 0.0, None, 0.0, len(zs))
        z = np.asarray(zs)
        # Frames carry content + mark or content - mark. Split them into two
        # clusters on the sync component (1-D 2-means), then take the midpoint of
        # the cluster means as the static-content estimate. This stays unbiased
        # when the two signs are unequally represented (e.g. after frame drops).
        s = z[:, 0]
        c = float(np.median(s))
        pos = s > c
        for _ in range(20):
            if pos.all() or not pos.any():
                break
            c = 0.5 * (s[pos].mean() + s[~pos].mean())
            new_pos = s > c
            if (new_pos == pos).all():
                break
            pos = new_pos
        if pos.sum() < 2 or (~pos).sum() < 2:
            return Detection(False, 0.0, None, 0.0, len(zs))
        content = 0.5 * (z[pos].mean(0) + z[~pos].mean(0))
        v = z - content
        sign = np.where(pos, 1.0, -1.0)
        aligned = v[:, 1:] * sign[:, None]
        n = aligned.shape[0]
        mean = aligned.mean(0)
        std = aligned.std(0, ddof=1) + 1e-6
        t = mean / (std / np.sqrt(n))
        sync = float(np.abs(v[:, 0]).mean() / (np.abs(v[:, 1:]).mean() + 1e-6))
        token = bits_to_token(t)
        conf = float(np.abs(t).min())
        present = token is not None and conf >= min_bit_t
        return Detection(present, sync, token if present else None, conf, len(zs))
