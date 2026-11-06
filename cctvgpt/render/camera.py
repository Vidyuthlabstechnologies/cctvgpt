"""Fixed pinhole camera (FR-2, FR-8).

World frame: x to the right, y forward (away from the camera mount), z up,
metres. The camera never moves during a clip, which is what guarantees a stable
view (FR-8) by construction rather than by hoping a model holds it.
"""

from __future__ import annotations

import math

import numpy as np


class Camera:
    def __init__(self, width: int, height: int, position, tilt_deg: float, pan_deg: float, fov_deg: float):
        self.width = int(width)
        self.height = int(height)
        self.position = np.asarray(position, dtype=np.float64)
        t = math.radians(tilt_deg)
        p = math.radians(pan_deg)
        self.forward = np.array([math.sin(p) * math.cos(t), math.cos(p) * math.cos(t), -math.sin(t)])
        self.right = np.array([math.cos(p), -math.sin(p), 0.0])
        self.up = np.cross(self.right, self.forward)
        self.focal = (self.width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def to_camera(self, pts) -> np.ndarray:
        """World points (..., 3) -> camera coordinates (..., 3) as (x_right, y_up, z_forward)."""
        d = np.asarray(pts, dtype=np.float64) - self.position
        return np.stack([d @ self.right, d @ self.up, d @ self.forward], axis=-1)

    def project(self, pts) -> tuple[np.ndarray, np.ndarray]:
        """World points (..., 3) -> (pixel coords (..., 2), depth (...)). Depth <= 0 is behind the camera."""
        c = self.to_camera(pts)
        z = c[..., 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = self.cx + self.focal * c[..., 0] / zs
        v = self.cy - self.focal * c[..., 1] / zs
        return np.stack([u, v], axis=-1), z

    def pixel_rays(self) -> np.ndarray:
        """Unit ray direction for every pixel centre, shape (H, W, 3)."""
        us = (np.arange(self.width) + 0.5 - self.cx) / self.focal
        vs = (np.arange(self.height) + 0.5 - self.cy) / self.focal
        uu, vv = np.meshgrid(us, vs)
        d = self.forward[None, None, :] + uu[..., None] * self.right + (-vv)[..., None] * self.up
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def pixels_per_metre(self, depth: float) -> float:
        return self.focal / max(depth, 1e-3)
