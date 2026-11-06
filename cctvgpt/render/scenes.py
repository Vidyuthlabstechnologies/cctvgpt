"""Static scene catalogue: geometry, materials and lights.

Each scene is described as a set of infinite planes (floor, walls, ceiling) and
axis-aligned boxes (pillars, racks, parked cars). The static background is
ray-cast once per clip, producing colour, depth and world-position buffers. The
dynamic layer (people, objects, fire) is depth-tested against that buffer, so
static geometry correctly occludes actors (analytics testing under occlusion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera


@dataclass
class Plane:
    axis: int  # 0=x, 1=y, 2=z
    value: float
    material: str
    # Only rays hitting the plane from this side are accepted (+1 / -1); 0 = both.
    facing: int = 0
    # Optional bounds on the other two axes: {axis: (lo, hi)}
    bounds: dict = field(default_factory=dict)


@dataclass
class Box:
    lo: tuple
    hi: tuple
    material: str
    color: tuple | None = None  # BGR 0..1 override


@dataclass
class Light:
    position: tuple
    intensity: float
    color: tuple = (1.0, 1.0, 1.0)  # BGR
    kind: str = "point"  # point | lamp (visible bulb)
    on_at: tuple = ("day", "dusk", "night")


@dataclass
class Zone:
    """A restricted ground region (used by the intrusion event)."""

    polygon: list  # [(x, y), ...] on the ground plane


@dataclass
class SceneLayout:
    name: str
    planes: list
    boxes: list
    lights: list
    walk_lanes: list  # list of ((x0, y0), (x1, y1)) ground segments people walk along
    walk_area: tuple  # (xmin, xmax, ymin, ymax) free area for milling crowds
    camera_xy: tuple  # camera mount ground position
    restricted_zone: Zone | None
    hazard_point: tuple  # where fire / falls are staged by default
    drop_point: tuple  # where an object is abandoned
    exit_point: tuple  # where a surging crowd runs to
    sky: bool = False
    ceiling_h: float | None = None


def build_layout(scene: str, occlusion: str, rng: np.random.Generator) -> SceneLayout:
    if scene == "corridor":
        lay = _corridor()
    elif scene == "parking_lot":
        lay = _parking_lot(rng)
    elif scene == "warehouse":
        lay = _warehouse()
    else:  # pragma: no cover - validated earlier
        raise ValueError(scene)
    _add_occluders(lay, occlusion, rng)
    return lay


def _corridor() -> SceneLayout:
    L = 32.0
    planes = [
        Plane(2, 0.0, "tile", bounds={0: (-2.0, 2.0)}),
        Plane(2, 3.0, "ceiling", bounds={0: (-2.0, 2.0)}),
        Plane(0, -2.0, "wall_left"),
        Plane(0, 2.0, "wall_right"),
        Plane(1, L, "end_wall"),
        Plane(1, -1.0, "end_wall"),
    ]
    lights = [Light((0.0, y, 2.95), 2.2, (0.95, 1.0, 1.0), "lamp") for y in np.arange(2.0, L, 5.0)]
    # Doors as shallow boxes recessed into the walls.
    boxes = []
    for i, y in enumerate(np.arange(5.0, L - 2, 7.0)):
        x = -2.0 if i % 2 == 0 else 1.95
        boxes.append(Box((x, y, 0.0), (x + 0.05, y + 1.0, 2.1), "door"))
    return SceneLayout(
        name="corridor",
        planes=planes,
        boxes=boxes,
        lights=lights,
        walk_lanes=[((-0.8, 30.0), (-0.8, 1.0)), ((0.8, 1.0), (0.8, 30.0)), ((0.0, 28.0), (0.0, 2.0))],
        walk_area=(-1.6, 1.6, 4.0, 20.0),
        camera_xy=(0.0, 0.0),
        restricted_zone=Zone([(-2.0, 22.0), (2.0, 22.0), (2.0, 27.0), (-2.0, 27.0)]),
        hazard_point=(0.2, 9.0),
        drop_point=(1.4, 8.0),
        exit_point=(0.0, 31.0),
        ceiling_h=3.0,
    )


def _parking_lot(rng: np.random.Generator) -> SceneLayout:
    planes = [
        Plane(2, 0.0, "asphalt"),
        Plane(1, 55.0, "building", bounds={2: (0.0, 14.0)}),
    ]
    boxes = []
    # Two rows of parked cars either side of a central aisle.
    palette = [(0.75, 0.75, 0.78), (0.2, 0.2, 0.22), (0.15, 0.15, 0.6), (0.55, 0.35, 0.15), (0.85, 0.85, 0.85), (0.2, 0.35, 0.2)]
    for row_x in (-7.5, 4.5):
        for y in np.arange(6.0, 30.0, 3.0):
            if rng.random() < 0.25:
                continue  # empty bay
            c = palette[int(rng.integers(len(palette)))]
            boxes.append(Box((row_x, y + 0.3, 0.0), (row_x + 4.4, y + 2.2, 1.45), "car", c))
    # Perimeter fence posts along the far edge, restricted compound behind.
    for x in np.arange(-14.0, 14.5, 3.0):
        boxes.append(Box((x, 34.0, 0.0), (x + 0.12, 34.12, 2.2), "metal"))
    boxes.append(Box((-14.0, 34.0, 1.9), (14.0, 34.08, 2.0), "metal"))
    lights = [Light((x, y, 7.0), 60.0, (0.55, 0.8, 1.0), "lamp", ("dusk", "night")) for x in (-3.0, 3.5) for y in (10.0, 24.0)]
    for x, y in ((-3.0, 10.0), (-3.0, 24.0), (3.5, 10.0), (3.5, 24.0)):
        boxes.append(Box((x - 0.08, y - 0.08, 0.0), (x + 0.08, y + 0.08, 7.0), "metal"))
    return SceneLayout(
        name="parking_lot",
        planes=planes,
        boxes=boxes,
        lights=lights,
        walk_lanes=[((-1.0, 33.0), (-1.0, 3.0)), ((1.5, 3.0), (1.5, 33.0)), ((-12.0, 18.0), (12.0, 19.0)), ((12.0, 31.0), (-12.0, 30.5))],
        walk_area=(-2.5, 3.5, 6.0, 28.0),
        camera_xy=(0.0, 0.0),
        restricted_zone=Zone([(-14.0, 34.2), (14.0, 34.2), (14.0, 44.0), (-14.0, 44.0)]),
        hazard_point=(0.5, 16.0),
        drop_point=(2.8, 14.0),
        exit_point=(0.0, 33.0),
        sky=True,
    )


def _warehouse() -> SceneLayout:
    planes = [
        Plane(2, 0.0, "concrete", bounds={0: (-15.0, 15.0)}),
        Plane(2, 10.0, "ceiling_metal", bounds={0: (-15.0, 15.0)}),
        Plane(0, -15.0, "wall_metal"),
        Plane(0, 15.0, "wall_metal"),
        Plane(1, 45.0, "wall_metal"),
        Plane(1, -1.0, "wall_metal"),
    ]
    boxes = []
    for x in (-12.5, -6.5, 4.5, 10.5):
        for y0 in (8.0, 24.0):
            boxes.append(Box((x, y0, 0.0), (x + 1.8, y0 + 13.0, 5.5), "rack"))
    # Pallets with goods in the aisles.
    for x, y in ((-2.8, 14.0), (2.0, 28.0), (-3.2, 33.0)):
        boxes.append(Box((x, y, 0.0), (x + 1.2, y + 1.0, 1.3), "goods"))
    lights = [Light((x, y, 9.5), 20.0, (0.85, 0.95, 1.0), "lamp") for x in (-8.0, 0.0, 8.0) for y in (8.0, 20.0, 32.0)]
    return SceneLayout(
        name="warehouse",
        planes=planes,
        boxes=boxes,
        lights=lights,
        walk_lanes=[((-1.2, 42.0), (-1.2, 2.0)), ((1.2, 2.0), (1.2, 42.0)), ((-14.0, 22.5), (14.0, 22.5)), ((14.0, 6.5), (-14.0, 6.5))],
        walk_area=(-4.0, 4.0, 5.0, 38.0),
        camera_xy=(0.0, 0.0),
        restricted_zone=Zone([(-4.0, 28.0), (4.0, 28.0), (4.0, 34.0), (-4.0, 34.0)]),
        hazard_point=(0.3, 12.0),
        drop_point=(-0.5, 10.0),
        exit_point=(14.0, 22.5),
        ceiling_h=10.0,
    )


def _add_occluders(lay: SceneLayout, occlusion: str, rng: np.random.Generator) -> None:
    if occlusion == "none":
        return
    n = 2 if occlusion == "light" else 5
    xmin, xmax, ymin, ymax = lay.walk_area
    h = lay.ceiling_h or 4.0
    for i in range(n):
        x = float(rng.uniform(xmin, xmax))
        y = float(ymin + (i + 0.5) * (ymax - ymin) / n * 0.7)
        w = 0.4 if lay.name == "corridor" else 0.6
        lay.boxes.append(Box((x - w / 2, y - w / 2, 0.0), (x + w / 2, y + w / 2, h), "pillar"))
    if occlusion == "heavy":
        # A hanging sign close to the camera partially blocks the view.
        cy = 3.0
        lay.boxes.append(Box((-0.9, cy, 2.0), (0.9, cy + 0.05, 2.45), "sign"))


# ---------------------------------------------------------------------------
# Ray casting


def raycast(layout: SceneLayout, cam: Camera):
    """Return (material_id map, depth map, world position map, normal map, materials list)."""
    rays = cam.pixel_rays()
    H, W = rays.shape[:2]
    o = cam.position
    best_t = np.full((H, W), np.inf)
    mat = np.full((H, W), -1, dtype=np.int32)
    normal = np.zeros((H, W, 3))
    materials: list = []
    colors: list = []

    def _mat_id(name, color=None):
        key = (name, color)
        if key not in materials:
            materials.append(key)
        return materials.index(key)

    for pl in layout.planes:
        a = pl.axis
        d = rays[..., a]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (pl.value - o[a]) / d
        ok = (t > 1e-4) & np.isfinite(t)
        if pl.bounds:
            p = o + rays * np.where(ok, t, 0)[..., None]
            for ax, (lo, hi) in pl.bounds.items():
                ok &= (p[..., ax] >= lo - 1e-6) & (p[..., ax] <= hi + 1e-6)
        closer = ok & (t < best_t)
        best_t = np.where(closer, t, best_t)
        mid = _mat_id(pl.material)
        mat = np.where(closer, mid, mat)
        n = np.zeros(3)
        n[a] = 1.0 if o[a] > pl.value else -1.0  # normal faces the camera side
        normal = np.where(closer[..., None], n, normal)

    inv = np.where(np.abs(rays) < 1e-9, 1e9, 1.0 / np.where(np.abs(rays) < 1e-9, 1.0, rays))
    for bx in layout.boxes:
        lo = np.asarray(bx.lo, dtype=np.float64)
        hi = np.asarray(bx.hi, dtype=np.float64)
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
        tmin3 = np.minimum(t1, t2)
        tmax3 = np.maximum(t1, t2)
        tnear = tmin3.max(axis=-1)
        tfar = tmax3.min(axis=-1)
        hit = (tnear <= tfar) & (tfar > 1e-4) & (tnear > 1e-4)
        closer = hit & (tnear < best_t)
        if not closer.any():
            continue
        best_t = np.where(closer, tnear, best_t)
        mid = _mat_id(bx.material, bx.color)
        mat = np.where(closer, mid, mat)
        axis = tmin3.argmax(axis=-1)
        n = np.zeros((H, W, 3))
        sgn = -np.sign(np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0])
        np.put_along_axis(n, axis[..., None], sgn[..., None], axis=-1)
        normal = np.where(closer[..., None], n, normal)

    sky = ~np.isfinite(best_t)
    depth_t = np.where(sky, 1e4, best_t)
    world = o + rays * depth_t[..., None]
    # Depth along the optical axis, which is what sprites are compared against.
    zdepth = depth_t * (rays @ cam.forward)
    mat = np.where(sky, -1, mat)
    return mat, zdepth, world, normal, materials


# ---------------------------------------------------------------------------
# Materials (procedural textures). Colours are BGR in 0..1.


def _hash_noise(x, y, scale, seed=0):
    xi = np.floor(x * scale).astype(np.int64)
    yi = np.floor(y * scale).astype(np.int64)
    h = (xi * 73856093) ^ (yi * 19349663) ^ (seed * 83492791)
    return ((h % 1000) / 1000.0).astype(np.float64)


def shade_materials(mat, world, normal, materials, layout: SceneLayout) -> np.ndarray:
    H, W = mat.shape
    albedo = np.zeros((H, W, 3))
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    for idx, (name, color) in enumerate(materials):
        m = mat == idx
        if not m.any():
            continue
        xs, ys, zs = x[m], y[m], z[m]
        n = _hash_noise(xs + zs, ys + zs, 7.0)
        if name == "tile":
            grout = (np.abs(((xs + 10) % 0.6) - 0.3) > 0.285) | (np.abs((ys % 0.6) - 0.3) > 0.285)
            base = np.array([0.70, 0.72, 0.72])
            c = base * (0.92 + 0.08 * n)[:, None]
            c[grout] *= 0.6
        elif name in ("ceiling", "ceiling_metal"):
            c = np.array([0.8, 0.8, 0.8])[None, :] * (0.9 + 0.1 * n)[:, None]
            if name == "ceiling_metal":
                rib = (np.abs((xs % 1.5) - 0.75) > 0.68)
                c[rib] *= 0.7
        elif name in ("wall_left", "wall_right", "end_wall"):
            c = np.array([0.72, 0.80, 0.84])[None, :] * (0.95 + 0.05 * n)[:, None]
            skirting = zs < 0.12
            c[skirting] = np.array([0.30, 0.32, 0.35])
        elif name == "door":
            c = np.array([0.25, 0.40, 0.55])[None, :] * (0.9 + 0.1 * n)[:, None]
        elif name == "asphalt":
            c = np.array([0.30, 0.31, 0.32])[None, :] * (0.8 + 0.25 * _hash_noise(xs, ys, 23.0))[:, None]
            # Bay markings
            line = (
                ((np.abs(xs + 7.6) < 0.06) | (np.abs(xs - 8.95) < 0.06) | (np.abs(xs + 3.05) < 0.06) | (np.abs(xs - 4.45) < 0.06))
                & (ys > 6) & (ys < 30)
            ) | ((((ys % 3.0) < 0.1)) & (ys > 6) & (ys < 30) & (((xs > -7.6) & (xs < -3.05)) | ((xs > 4.45) & (xs < 8.95))))
            c[line] = np.array([0.85, 0.88, 0.9])
            if layout.restricted_zone is not None:
                hatch = (ys > 34.2) & (ys < 44.0) & (((xs + ys) % 1.2) < 0.25)
                c[hatch] = np.array([0.15, 0.75, 0.85])
        elif name == "building":
            win = ((xs % 4.0) > 1.0) & ((xs % 4.0) < 3.0) & ((zs % 3.5) > 1.0) & ((zs % 3.5) < 2.6)
            c = np.array([0.55, 0.58, 0.62])[None, :] * (0.9 + 0.1 * n)[:, None]
            c[win] = np.array([0.35, 0.30, 0.25])
        elif name == "concrete":
            c = np.array([0.55, 0.56, 0.56])[None, :] * (0.85 + 0.2 * _hash_noise(xs, ys, 3.0))[:, None]
            lane = (np.abs(np.abs(xs) - 4.2) < 0.08)
            c[lane] = np.array([0.1, 0.75, 0.9])
            if layout.restricted_zone is not None:
                (x0, y0), _, (x1, y1), _ = layout.restricted_zone.polygon
                inside = (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
                hatch = inside & (((xs + ys) % 1.0) < 0.3)
                c[hatch] = np.array([0.1, 0.75, 0.9])
        elif name == "wall_metal":
            c = np.array([0.55, 0.52, 0.48])[None, :] * (0.85 + 0.15 * (np.abs((xs + ys) % 0.8 - 0.4) > 0.37))[:, None]
        elif name == "rack":
            frame = (np.abs((ys % 2.6) - 1.3) > 1.2) | (np.abs((zs % 1.8) - 0.9) > 0.84)
            c = np.array([0.25, 0.45, 0.75])[None, :] * np.ones((xs.size, 1))
            goods = ~frame
            gn = _hash_noise(ys, zs, 1.5)
            c[goods] = np.stack([0.35 + 0.2 * gn[goods], 0.5 + 0.15 * gn[goods], 0.6 + 0.1 * gn[goods]], axis=-1)
            c[frame] = np.array([0.1, 0.35, 0.85])
        elif name == "goods":
            c = np.array([0.35, 0.55, 0.65])[None, :] * (0.85 + 0.15 * n)[:, None]
        elif name == "pillar":
            c = np.array([0.62, 0.62, 0.6])[None, :] * (0.92 + 0.08 * n)[:, None]
            stripe = (zs > 0.9) & (zs < 1.2)
            c[stripe] = np.array([0.1, 0.7, 0.9])
        elif name == "sign":
            c = np.tile(np.array([0.2, 0.55, 0.15]), (xs.size, 1))
        elif name == "metal":
            c = np.tile(np.array([0.45, 0.45, 0.47]), (xs.size, 1))
        elif name == "car":
            base = np.array(color if color is not None else (0.5, 0.5, 0.5))
            glass = zs > 0.95
            c = np.tile(base, (xs.size, 1)) * (0.9 + 0.1 * n)[:, None]
            c[glass] = np.array([0.18, 0.16, 0.14])
            c[zs < 0.35] *= 0.5
        else:
            c = np.tile(np.array([0.5, 0.5, 0.5]), (xs.size, 1))
        albedo[m] = c
    return albedo


def sky_color(time_of_day: str, rays_v: np.ndarray) -> np.ndarray:
    top = {"day": (0.85, 0.65, 0.45), "dusk": (0.45, 0.35, 0.40), "night": (0.08, 0.05, 0.04)}[time_of_day]
    hor = {"day": (0.95, 0.88, 0.80), "dusk": (0.45, 0.55, 0.85), "night": (0.15, 0.12, 0.10)}[time_of_day]
    t = np.clip(rays_v, 0, 1)[..., None]
    return np.array(hor) * (1 - t) + np.array(top) * t


def ambient_level(scene: SceneLayout, time_of_day: str) -> float:
    if scene.sky:
        return {"day": 1.0, "dusk": 0.35, "night": 0.04}[time_of_day]
    # Indoor: some daylight leak by day, emergency/ambient light at night.
    return {"day": 0.45, "dusk": 0.3, "night": 0.12}[time_of_day]


def lighting(points: np.ndarray, normals: np.ndarray | None, layout: SceneLayout, time_of_day: str) -> np.ndarray:
    """Irradiance (BGR multiplier) at world points (..., 3)."""
    shape = points.shape[:-1]
    amb = ambient_level(layout, time_of_day)
    total = np.full(shape + (3,), amb)
    if layout.sky and time_of_day != "night":
        sun = np.array([-0.4, 0.5, 0.77])
        sun /= np.linalg.norm(sun)
        if normals is not None:
            lam = np.clip(normals @ sun, 0, 1)
            total = total * (0.55 + 0.45 * lam[..., None])
    for li in layout.lights:
        if time_of_day not in li.on_at:
            continue
        d = np.asarray(li.position) - points
        # The +1.0 term models the finite size of the luminaire and avoids hot spots.
        dist2 = (d**2).sum(-1) + 1.0
        if normals is not None:
            nd = (normals * d).sum(-1) / np.sqrt(dist2)
            lam = np.clip(nd, 0, 1) * 0.95 + 0.05
        else:
            lam = 1.0
        contrib = li.intensity * lam / dist2
        total = total + contrib[..., None] * np.asarray(li.color)
    return np.minimum(total, 2.0)
