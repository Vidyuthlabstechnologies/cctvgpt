"""Dynamic actors: people (as faceless articulated figures) and objects.

People are drawn as simple 3D skeletons with clothing colours only. They have
no faces or identifying features, so outputs are privacy-safe by construction
and cannot depict a real, identifiable person (non-goal in PRD section 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SHIRTS = [(0.2, 0.2, 0.7), (0.6, 0.3, 0.1), (0.15, 0.15, 0.15), (0.8, 0.8, 0.8), (0.2, 0.5, 0.2), (0.1, 0.4, 0.8), (0.5, 0.1, 0.5), (0.7, 0.6, 0.2)]
TROUSERS = [(0.15, 0.12, 0.1), (0.35, 0.25, 0.15), (0.2, 0.2, 0.2), (0.45, 0.4, 0.35), (0.6, 0.35, 0.1)]
SKIN = [(0.45, 0.55, 0.7), (0.35, 0.45, 0.6), (0.25, 0.35, 0.5), (0.55, 0.65, 0.8)]


@dataclass
class PersonState:
    x: float
    y: float
    heading: float  # radians, 0 = +y (away from camera)
    phase: float  # gait phase
    speed: float
    fall_angle: float = 0.0  # 0 upright, pi/2 lying
    fall_dir: float = 0.0  # direction of fall in the ground plane
    carrying: bool = False
    visible: bool = True


@dataclass
class Person:
    track_id: int
    height: float
    shirt: tuple
    trousers: tuple
    skin: tuple
    # Function of time -> PersonState; set by the event/background scripts.
    script: object = None
    role: str = "background"  # background | victim | intruder | owner | crowd
    states: list = field(default_factory=list)

    @classmethod
    def random(cls, track_id: int, rng: np.random.Generator, role: str = "background") -> "Person":
        return cls(
            track_id=track_id,
            height=float(rng.uniform(1.55, 1.9)),
            shirt=SHIRTS[int(rng.integers(len(SHIRTS)))],
            trousers=TROUSERS[int(rng.integers(len(TROUSERS)))],
            skin=SKIN[int(rng.integers(len(SKIN)))],
            role=role,
        )


def skeleton(person: Person, st: PersonState) -> dict:
    """World-space joint positions for a person in a given state."""
    h = person.height / 1.75
    swing = math.sin(st.phase) * min(1.0, st.speed / 1.2) * 0.45
    arm = -swing * 0.8
    # Local frame: x lateral, y forward, z up.
    j = {
        "hip": (0.0, 0.0, 0.95 * h),
        "neck": (0.0, 0.0, 1.45 * h),
        "head": (0.0, 0.02, 1.62 * h),
        "l_sh": (-0.2 * h, 0.0, 1.42 * h),
        "r_sh": (0.2 * h, 0.0, 1.42 * h),
        "l_hip": (-0.1 * h, 0.0, 0.95 * h),
        "r_hip": (0.1 * h, 0.0, 0.95 * h),
    }

    def limb(origin, angle, length):
        ox, oy, oz = origin
        return (ox, oy + math.sin(angle) * length, oz - math.cos(angle) * length)

    j["l_knee"] = limb(j["l_hip"], swing, 0.48 * h)
    j["r_knee"] = limb(j["r_hip"], -swing, 0.48 * h)
    j["l_foot"] = limb(j["l_knee"], swing * 0.6 - 0.05, 0.47 * h)
    j["r_foot"] = limb(j["r_knee"], -swing * 0.6 - 0.05, 0.47 * h)
    if st.carrying:
        j["l_hand"] = limb(j["l_sh"], 0.05, 0.62 * h)
        j["r_hand"] = limb(j["r_sh"], -arm, 0.62 * h)
    else:
        j["l_hand"] = limb(j["l_sh"], arm, 0.62 * h)
        j["r_hand"] = limb(j["r_sh"], -arm, 0.62 * h)

    names = list(j)
    pts = np.array([j[n] for n in names], dtype=np.float64)
    if st.fall_angle > 0:
        # Rotate the whole body about the feet towards fall_dir (local frame axis).
        rel = st.fall_dir - st.heading
        fx, fy = math.sin(rel), math.cos(rel)
        a = st.fall_angle
        along = pts[:, 0] * fx + pts[:, 1] * fy
        perp_x, perp_y = pts[:, 0] - along * fx, pts[:, 1] - along * fy
        z = pts[:, 2]
        new_along = along * math.cos(a) + z * math.sin(a)
        new_z = -along * math.sin(a) + z * math.cos(a)
        pts[:, 0] = perp_x + new_along * fx
        pts[:, 1] = perp_y + new_along * fy
        pts[:, 2] = np.maximum(new_z, 0.0) + 0.1 * math.sin(a)
    # Rotate by heading and translate.
    c, s = math.cos(st.heading), math.sin(st.heading)
    wx = pts[:, 0] * c + pts[:, 1] * s + st.x
    wy = -pts[:, 0] * s + pts[:, 1] * c + st.y
    world = np.stack([wx, wy, pts[:, 2]], axis=-1)
    return dict(zip(names, world))


BONES = [
    ("l_hip", "l_knee", "trousers", 0.15),
    ("l_knee", "l_foot", "trousers", 0.13),
    ("r_hip", "r_knee", "trousers", 0.15),
    ("r_knee", "r_foot", "trousers", 0.13),
    ("hip", "neck", "shirt", 0.36),
    ("l_sh", "l_hand", "shirt", 0.1),
    ("r_sh", "r_hand", "shirt", 0.1),
]


@dataclass
class Prop:
    """A static or scripted object, e.g. an abandoned bag."""

    track_id: int
    kind: str  # bag
    size: tuple  # (w, d, h) metres
    color: tuple
    script: object = None  # t -> (x, y, z) or None when not present
