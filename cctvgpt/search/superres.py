"""Super-resolution (Real-ESRGAN general x4v3) for small faces, plates and display.

Super-resolution INVENTS plausible detail. An upscaled plate or face can show
characters or features that were never captured by the camera. Therefore:

- Measured, it HURTS recognition, so it is never used before a recogniser.
  On 300 real Indian plates shrunk to half size, exact reads fell from 25%
  (bicubic) to 19% (super-resolution). On LFW faces about 20 px wide, the
  true-match rate fell from 91% to 73%, and wrong-person matches rose from
  1.2% to 3.0% (benchmarks/events_sr_eval.py).
- It is offered only for an investigator's viewing, with a label burned into
  the image: "AI-enhanced, not evidence".
- It is never used to produce an evidence export. Exports are cut from the
  untouched original file.

Weights: realesr-general-x4v3 (Wang et al., Real-ESRGAN, BSD-3-Clause), a
1.2M-parameter compact network that runs on CPU.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
EVIDENCE_LABEL = "AI-enhanced (super-resolution): not evidence; detail may be invented"

_lock = threading.Lock()
_model: dict = {}


def _build():
    import os

    import torch
    from torch import nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + nn.functional.interpolate(x, scale_factor=self.upscale, mode="nearest")

    path = os.environ.get("CCTVGPT_SR_WEIGHTS") or os.path.expanduser("~/.cache/cctvgpt/realesr-general-x4v3.pth")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, path, progress=False)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("params", sd.get("params_ema", sd))
    m = SRVGGNetCompact()
    m.load_state_dict(sd)
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


def upscale(img: np.ndarray, max_input_side: int = 256) -> np.ndarray:
    """Return a 4x super-resolved BGR image. Large inputs are first shrunk to
    `max_input_side` (super-resolution is for small regions, not whole frames)."""
    if not available():
        raise RuntimeError("super-resolution unavailable")
    import torch

    h, w = img.shape[:2]
    if max(h, w) > max_input_side:
        s = max_input_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = _model["m"](x)[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((y * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def labelled_view(img: np.ndarray) -> np.ndarray:
    """Night-enhance and super-resolve a small region for viewing, with the label burned in."""
    from .enhance import assess, night_enhance

    q = assess(img)
    out = night_enhance(img, q) if (q.dark or q.noisy) else img
    if available() and max(out.shape[:2]) <= 256:
        out = upscale(out)
    bar = max(18, out.shape[0] // 14)
    canvas = cv2.copyMakeBorder(out, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 160))
    cv2.putText(canvas, "AI-ENHANCED - NOT EVIDENCE", (4, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, bar / 30, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
