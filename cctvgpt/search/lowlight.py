"""Learned low-light enhancement (Zero-DCE) for night footage.

Zero-DCE (Guo et al., CVPR 2020) estimates per-pixel light-enhancement curves
with a 79k-parameter CNN and applies them iteratively. It brightens dark
regions without blowing out lit ones, which suits CCTV frames with street
lights. Weights: the published "Epoch99.pth" checkpoint, mirrored on Hugging
Face as leonelhs/lowlight (MIT) [VERIFY licence of the original weights before
commercial distribution].

If torch or the weights are unavailable, callers fall back to the classical
CLAHE + gamma enhancer in enhance.py.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

_lock = threading.Lock()
_model: dict = {}


def _build():
    import torch
    import torch.nn as nn

    class ZeroDCE(nn.Module):
        def __init__(self):
            super().__init__()
            n = 32
            self.e_conv1 = nn.Conv2d(3, n, 3, 1, 1)
            self.e_conv2 = nn.Conv2d(n, n, 3, 1, 1)
            self.e_conv3 = nn.Conv2d(n, n, 3, 1, 1)
            self.e_conv4 = nn.Conv2d(n, n, 3, 1, 1)
            self.e_conv5 = nn.Conv2d(n * 2, n, 3, 1, 1)
            self.e_conv6 = nn.Conv2d(n * 2, n, 3, 1, 1)
            self.e_conv7 = nn.Conv2d(n * 2, 24, 3, 1, 1)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x1 = self.relu(self.e_conv1(x))
            x2 = self.relu(self.e_conv2(x1))
            x3 = self.relu(self.e_conv3(x2))
            x4 = self.relu(self.e_conv4(x3))
            x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
            return torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))  # 8 curve maps x 3 channels

    from huggingface_hub import hf_hub_download

    m = ZeroDCE()
    m.load_state_dict(torch.load(hf_hub_download("leonelhs/lowlight", "Epoch99.pth"), map_location="cpu", weights_only=True))
    m.eval()
    return m


def available() -> bool:
    with _lock:
        if "m" not in _model:
            try:
                _model["m"] = _build()
            except Exception:  # noqa: BLE001 - torch missing, offline, or weights unavailable
                _model["m"] = None
        return _model["m"] is not None


def zero_dce(img: np.ndarray, passes: int = 1, curve_side: int = 96) -> np.ndarray:
    """Enhance a BGR uint8 image.

    Curves are estimated on a copy downscaled to `curve_side` (the Zero-DCE++
    trick: the curve maps are smooth), then upsampled and applied at full
    resolution. `passes` > 1 re-applies the curves for very dark input.
    """
    if not available():
        raise RuntimeError("Zero-DCE unavailable")
    import torch

    h, w = img.shape[:2]
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    for _ in range(passes):
        scale = min(1.0, curve_side / max(h, w))
        small = cv2.resize(x, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else x
        with torch.no_grad():
            r = _model["m"](torch.from_numpy(small).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
        if scale < 1:
            r = cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)
        for i in range(8):
            rr = r[..., 3 * i : 3 * i + 3]
            x = x + rr * (x * x - x)
        x = np.clip(x, 0, 1)
    return cv2.cvtColor((x * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
