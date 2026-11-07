"""CCTV realism stage (FR-3).

Applied to rendered frames: resolution loss, infrared night mode, lens
distortion, bloom, vignetting, sensor noise, compression artifacts and an
on-screen timestamp. The timestamp uses a fixed simulation epoch (year 2000)
and the "SYN" tag, so it is clearly marked synthetic and never asserts a real
date or time.
"""

from __future__ import annotations

import cv2
import numpy as np

from .params import RealismParams, SceneParams

JPEG_QUALITY = {"none": None, "low": 85, "medium": 45, "high": 18}
SIM_EPOCH_DATE = "2000-01-01"


class RealismProcessor:
    def __init__(self, realism: RealismParams, scene: SceneParams, depth: np.ndarray | None = None):
        self.r = realism
        self.s = scene
        self.W, self.H = scene.width, scene.height
        self.ir = realism.ir_night_mode == "on" or (realism.ir_night_mode == "auto" and scene.time_of_day == "night")
        self._maps = self._distortion_maps(realism.lens_distortion) if realism.lens_distortion > 0 else None
        yy, xx = np.mgrid[0 : self.H, 0 : self.W].astype(np.float32)
        rr = ((xx - self.W / 2) / (self.W / 2)) ** 2 + ((yy - self.H / 2) / (self.H / 2)) ** 2
        self._vignette = (1.0 - 0.28 * np.clip(rr / 2.0, 0, 1))[..., None].astype(np.float32)
        self._ir_falloff = None
        if self.ir and depth is not None:
            # IR illuminator mounted at the camera: brightness falls with distance.
            # Normalised so that mid-range distances keep their brightness.
            d = np.clip(depth, 0.5, 200).astype(np.float32)
            reach = 14.0 if scene.outdoor else 10.0
            self._ir_falloff = np.clip(1.25 / (1.0 + (d / reach) ** 2), 0.08, 1.25)[..., None]
        # Simulated clock start: deterministic from seed, within the chosen time of day.
        base_h = {"day": 10, "dusk": 18, "night": 23}[scene.time_of_day]
        rng = np.random.default_rng([scene.seed & 0xFFFFFFFF, 7177])
        self._clock0 = base_h * 3600 + int(rng.integers(0, 3000))

    def _distortion_maps(self, k: float):
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx = (xx - W / 2) / (W / 2)
        ny = (yy - H / 2) / (W / 2)
        r2 = nx * nx + ny * ny
        # Barrel distortion with a scale so the corners stay filled.
        scale = 1.0 / (1 + k * 0.6)
        f = (1 + k * r2) * scale
        mapx = (nx * f) * (W / 2) + W / 2
        mapy = (ny * f) * (W / 2) + H / 2
        return mapx.astype(np.float32), mapy.astype(np.float32)

    # ------------------------------------------------------------------
    def distort_points(self, pts) -> np.ndarray:
        """Map undistorted pixel coordinates to where they appear in the output frame.

        Inverse of the remap used in process(): the output pixel o samples the
        source at s = o * (1 + k r_o^2) * scale (normalised), so for each source
        point we solve for r_o with Newton's method.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        k = self.r.lens_distortion
        if k <= 0:
            return pts
        W, H = self.W, self.H
        sx = (pts[:, 0] - W / 2) / (W / 2)
        sy = (pts[:, 1] - H / 2) / (W / 2)
        rs = np.hypot(sx, sy)
        scale = 1.0 / (1 + k * 0.6)
        ro = rs.copy()
        for _ in range(20):
            g = ro * (1 + k * ro**2) * scale - rs
            dg = (1 + 3 * k * ro**2) * scale
            ro = ro - g / dg
        ratio = np.where(rs > 1e-9, ro / np.maximum(rs, 1e-9), 1.0 / scale)
        ox = sx * ratio * (W / 2) + W / 2
        oy = sy * ratio * (W / 2) + H / 2
        return np.stack([ox, oy], axis=-1)

    def distort_bbox(self, box) -> list:
        x, y, w, h = box
        if self.r.lens_distortion <= 0:
            return list(box)
        xs = np.linspace(x, x + w, 5)
        ys = np.linspace(y, y + h, 5)
        edge = np.concatenate([
            np.stack([xs, np.full(5, y)], 1), np.stack([xs, np.full(5, y + h)], 1),
            np.stack([np.full(5, x), ys], 1), np.stack([np.full(5, x + w), ys], 1),
        ])
        o = self.distort_points(edge)
        x0, y0 = np.clip(o.min(0), 0, [self.W, self.H])
        x1, y1 = np.clip(o.max(0), 0, [self.W, self.H])
        return [int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))]

    def process(self, frame: np.ndarray, index: int) -> np.ndarray:
        r = self.r
        img = frame
        if r.low_res_factor < 1.0:
            sw = max(16, int(self.W * r.low_res_factor))
            sh = max(16, int(self.H * r.low_res_factor))
            img = cv2.resize(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        f = img.astype(np.float32) / 255.0
        if self.ir:
            # Near-IR: clothing and skin reflect IR more uniformly; output is monochrome.
            gray = 0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]
            gray = np.power(np.clip(gray, 0, 1), 0.9)[..., None]
            if self._ir_falloff is not None:
                gray = gray * self._ir_falloff
            f = np.repeat(gray, 3, axis=2)
        if self._maps is not None:
            f = cv2.remap(f, self._maps[0], self._maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Bloom around highlights (stronger when glare is requested).
        thr = 0.85
        hi = np.clip(f - thr, 0, None)
        if hi.max() > 0:
            strength = 1.4 if self.s.glare else 0.6
            f = f + cv2.GaussianBlur(hi, (0, 0), max(2.0, self.W / 160)) * strength
        f = f * self._vignette
        if r.noise > 0:
            rng = np.random.default_rng([self.s.seed & 0xFFFFFFFF, 99, index])
            # Low light means more gain, which means more noise.
            gain = {"day": 1.0, "dusk": 1.6, "night": 2.4}[self.s.time_of_day]
            sigma = 0.018 * r.noise * gain
            noise = rng.normal(0, sigma, size=f.shape[:2]).astype(np.float32)[..., None]
            chroma = 0 if self.ir else rng.normal(0, sigma * 0.4, size=f.shape).astype(np.float32)
            f = f + noise + chroma
        out = (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)
        q = JPEG_QUALITY[r.compression]
        if q is not None:
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if r.timestamp_overlay:
            self._timestamp(out, index)
        return out

    def _timestamp(self, img: np.ndarray, index: int) -> None:
        secs = self._clock0 + int(index / self.s.fps)
        hh, mm, ss = (secs // 3600) % 24, (secs // 60) % 60, secs % 60
        text = f"SYN {SIM_EPOCH_DATE} {hh:02d}:{mm:02d}:{ss:02d}  {self.r.camera_label}"
        scale = max(0.35, self.H / 720 * 0.9)
        th = max(1, int(round(scale * 2)))
        org = (int(self.W * 0.015), int(self.H * 0.06) + int(10 * scale))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
