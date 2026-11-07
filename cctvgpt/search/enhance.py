"""Low-light and noise handling for indexing real CCTV footage.

Indian CCTV archives are dominated by dark, noisy, heavily compressed and IR
(monochrome) footage. Embedding models see better after a light, conservative
enhancement: denoise, then local contrast (CLAHE), then gamma towards a mid
brightness. The original frame is always indexed too, and search takes the
best match of the two, so enhancement can only add recall.

The enhanced frame is used for indexing and display only. It is never
exported as evidence: exports are cut from the untouched original file.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    brightness: float  # mean luma 0..255
    noise: float  # robust noise sigma estimate on luma
    monochrome: bool  # IR / grayscale footage
    dark: bool
    noisy: bool

    def to_dict(self) -> dict:
        return {"brightness": round(self.brightness, 1), "noise": round(self.noise, 2), "monochrome": self.monochrome, "dark": self.dark, "noisy": self.noisy}


def assess(frame: np.ndarray) -> FrameQuality:
    small = frame if frame.shape[1] <= 480 else cv2.resize(frame, (480, int(frame.shape[0] * 480 / frame.shape[1])), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Noise: median absolute Laplacian response (Immerkaer-style estimate).
    lap = cv2.Laplacian(y, cv2.CV_32F, ksize=3)
    noise = float(np.median(np.abs(lap)) / 0.6745 / 2.0)
    b, g, r = cv2.split(small.astype(np.float32))
    chroma = float(np.mean(np.abs(b - g)) + np.mean(np.abs(g - r)))
    brightness = float(y.mean())
    return FrameQuality(brightness, noise, chroma < 3.0, brightness < 75, noise > 6.0)


def enhance(frame: np.ndarray, q: FrameQuality | None = None) -> np.ndarray:
    q = q or assess(frame)
    out = frame
    if q.noisy:
        h = float(np.clip(q.noise * 1.2, 4, 20))
        out = cv2.fastNlMeansDenoisingColored(out, None, h, h if not q.monochrome else 0, 5, 15)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5 if q.dark else 1.8, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    mean = float(l_ch.mean())
    if mean < 100:
        gamma = float(np.clip(np.log(110 / 255) / np.log(max(mean, 5) / 255), 0.35, 1.0))
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        l_ch = cv2.LUT(l_ch, lut)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def needs_enhancement(q: FrameQuality) -> bool:
    return q.dark or q.noisy or q.monochrome


def degrade_to_night(frame: np.ndarray, rng: np.random.Generator, darkness: float = 0.25, noise: float = 14.0, jpeg_q: int = 30) -> np.ndarray:
    """Simulate poor night CCTV from a daytime frame (used by benchmarks)."""
    f = frame.astype(np.float32) / 255.0
    f = np.power(f, 1.6) * darkness
    f = f * 255.0 + rng.normal(0, noise, f.shape)
    out = np.clip(f, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
