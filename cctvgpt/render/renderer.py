"""Procedural CCTV scene renderer (PRD section 8, Option B).

Produces clean frames plus exact ground truth. CCTV realism (compression,
IR, distortion, noise, timestamp) is applied afterwards by
:mod:`cctvgpt.realism`, and safeguards after that.

Determinism (FR-5): every random choice comes from generators seeded by
(seed, purpose[, frame]); rendering the same parameters with the same
RENDERER_VERSION yields bit-identical frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..params import SceneParams
from .actors import BONES, SHIRT_NAMES, TROUSER_NAMES, Person, skeleton
from .camera import Camera
from .events import Cast, build_cast
from .scenes import build_layout, lighting, raycast, shade_materials, sky_color

FOG_DISTANCE = {"fog": 14.0}


def _rng(seed: int, *purpose) -> np.random.Generator:
    key = [int(seed) & 0xFFFFFFFF] + [abs(hash_str(p)) for p in purpose]
    return np.random.default_rng(key)


def hash_str(x) -> int:
    # Stable across processes (unlike built-in hash for str).
    if isinstance(x, int):
        return x
    h = 2166136261
    for ch in str(x).encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


@dataclass
class FrameTruth:
    index: int
    t: float
    objects: list


class SimRenderer:
    def __init__(self, params: SceneParams):
        self.p = params
        self.notes: list[str] = []
        rng = _rng(params.seed, "layout")
        self.layout = build_layout(params.scene, params.occlusion, rng)
        cx, cy = self.layout.camera_xy
        self.cam = Camera(params.width, params.height, (cx, cy, params.camera_height_m), params.camera_tilt_deg, params.camera_pan_deg, params.fov_deg)
        self.weather = params.weather
        if not params.outdoor and params.weather != "clear":
            self.notes.append(f"weather '{params.weather}' has no effect in indoor scene '{params.scene}'")
            self.weather = "clear"
        if params.event_distance_m:
            self._stage_event_at(params.event_distance_m)
        self.cast: Cast = build_cast(self.layout, params.event, params.crowd_density, _rng(params.seed, "cast"), params.duration_s)
        self._build_background()

    def _stage_event_at(self, dist: float) -> None:
        """Move the event staging point to a ground distance along the camera's view direction."""
        cx, cy = self.layout.camera_xy
        pan = np.radians(self.p.camera_pan_deg)
        xmin, xmax, _, _ = self.layout.walk_area
        x = float(np.clip(cx + dist * np.sin(pan), xmin, xmax))
        y = float(cy + dist * np.cos(pan))
        self.layout.hazard_point = (x, y)
        self.layout.drop_point = (float(np.clip(x + 1.0, xmin, xmax)), y - 0.5)

    # ------------------------------------------------------------------
    def _build_background(self) -> None:
        p = self.p
        mat, depth, world, normal, materials = raycast(self.layout, self.cam)
        self.depth = depth.astype(np.float32)
        self.world = world.astype(np.float32)
        albedo = shade_materials(mat, world, normal, materials, self.layout)
        light = lighting(world, normal, self.layout, p.time_of_day)
        bg = albedo * light
        sky = mat < 0
        if sky.any():
            rays = self.cam.pixel_rays()
            bg[sky] = sky_color(p.time_of_day, rays[..., 2])[sky]
        if self.weather in ("rain", "snow"):
            # Overcast and wet: lower contrast, darker ground.
            ground = (world[..., 2] < 0.01) & ~sky
            bg[ground] *= 0.8
            bg = bg * 0.85 + 0.04
        self.albedo = albedo.astype(np.float32)
        self.sky_mask = sky
        self._draw_lamps(bg)
        if self.weather == "fog":
            bg = self._fog(bg, depth)
        # Auto-exposure, as a real CCTV camera would apply. Computed once from the
        # static background so brightness never pumps during the clip (FR-8).
        lum = bg[..., 0] * 0.114 + bg[..., 1] * 0.587 + bg[..., 2] * 0.299
        self.exposure = float(np.clip(0.42 / max(float(np.median(lum)), 1e-3), 0.3, 2.5))
        self.background = bg.astype(np.float32)

    def _draw_lamps(self, img: np.ndarray) -> None:
        for li in self.layout.lights:
            if self.p.time_of_day not in li.on_at or li.kind != "lamp":
                continue
            (u, v), z = self.cam.project(np.asarray(li.position))
            if z <= 0.2 or not (0 <= u < self.p.width and 0 <= v < self.p.height):
                continue
            if self.depth[int(v), int(u)] < z - 0.5:
                continue
            r = max(1, int(self.cam.pixels_per_metre(z) * 0.25))
            cv2.ellipse(img, (int(u), int(v)), (r * 2, max(1, r // 2)), 0, 0, 360, (1.6, 1.7, 1.7), -1, cv2.LINE_AA)

    def _fog(self, img: np.ndarray, depth) -> np.ndarray:
        k = 1.0 - np.exp(-np.asarray(depth) / FOG_DISTANCE["fog"])
        fog_col = {"day": 0.78, "dusk": 0.45, "night": 0.12}[self.p.time_of_day]
        if np.ndim(k) == 2:
            k = k[..., None]
        return img * (1 - k) + fog_col * k

    # ------------------------------------------------------------------
    def frames(self):
        """Yield (frame_bgr_uint8, FrameTruth) for every frame of the clip."""
        p = self.p
        for i in range(p.n_frames):
            t = i / p.fps
            frame = self.background.copy()
            objects: list = []
            self._apply_fire_light(frame, t)
            sprites = []
            for person in self.cast.people:
                st = person.script(t)
                if st is not None:
                    _, z = self.cam.project(np.array([st.x, st.y, 0.9]))
                    sprites.append((float(z), "person", person, st))
            for prop in self.cast.props:
                pos = prop.script(t)
                if pos is not None:
                    _, z = self.cam.project(np.asarray(pos))
                    sprites.append((float(z), "prop", prop, pos))
            for fire in self.cast.fires:
                if t >= fire.ignite_t:
                    _, z = self.cam.project(np.array([fire.x, fire.y, 0.3]))
                    sprites.append((float(z), "fire", fire, None))
            sprites.sort(key=lambda s: -s[0])
            for z, kind, obj, st in sprites:
                if z < 0.5:
                    continue
                if kind == "person":
                    ann = self._draw_person(frame, obj, st, z, t)
                elif kind == "prop":
                    ann = self._draw_prop(frame, obj, st, z)
                else:
                    ann = self._draw_fire(frame, obj, z, t, i)
                if ann is not None:
                    objects.append(ann)
            self._apply_weather(frame, i)
            if p.glare:
                self._apply_glare(frame, t)
            out = (np.clip(_tonemap(frame * self.exposure), 0, 1) * 255.0 + 0.5).astype(np.uint8)
            yield out, FrameTruth(i, round(t, 4), objects)

    # ------------------------------------------------------------------
    def _composite(self, frame, layer, alpha, x0, y0, z):
        """Depth-tested alpha composite of an ROI layer. Returns (amodal_mask, visible_mask)."""
        h, w = alpha.shape
        H, W = frame.shape[:2]
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(W, x0 + w), min(H, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return None, None
        lx0, ly0 = fx0 - x0, fy0 - y0
        a = alpha[ly0 : ly0 + (fy1 - fy0), lx0 : lx0 + (fx1 - fx0)]
        col = layer[ly0 : ly0 + (fy1 - fy0), lx0 : lx0 + (fx1 - fx0)]
        static_d = self.depth[fy0:fy1, fx0:fx1]
        vis = a * (static_d > z - 0.35)
        if self.weather == "fog":
            col = self._fog(col, z)
        region = frame[fy0:fy1, fx0:fx1]
        region[:] = region * (1 - vis[..., None]) + col * vis[..., None]
        return (fx0, fy0, a), (fx0, fy0, vis)

    @staticmethod
    def _bbox(mask_info, thresh=0.3):
        if mask_info is None:
            return None
        x0, y0, m = mask_info
        ys, xs = np.nonzero(m > thresh)
        if xs.size == 0:
            return None
        return [int(x0 + xs.min()), int(y0 + ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]

    def _shadow(self, frame, x, y, radius, strength):
        pts = np.array([[x + radius * math.cos(a), y + radius * 0.6 * math.sin(a), 0.0] for a in np.linspace(0, 2 * math.pi, 16)])
        uv, z = self.cam.project(pts)
        if (z <= 0.3).any():
            return
        poly = uv.astype(np.int32)
        mask = np.zeros(frame.shape[:2], np.float32)
        cv2.fillPoly(mask, [poly], 1.0, cv2.LINE_AA)
        x0, y0, w, h = cv2.boundingRect(poly)
        if w <= 0 or h <= 0:
            return
        x0, y0 = max(0, x0 - 4), max(0, y0 - 4)
        x1, y1 = min(frame.shape[1], x0 + w + 8), min(frame.shape[0], y0 + h + 8)
        if x1 <= x0 or y1 <= y0:
            return
        roi = cv2.GaussianBlur(mask[y0:y1, x0:x1], (0, 0), 2.0)
        frame[y0:y1, x0:x1] *= (1 - strength * roi)[..., None]

    def _draw_person(self, frame, person: Person, st, z, t):
        joints = skeleton(person, st)
        names = list(joints)
        pts = np.array([joints[n] for n in names])
        uv, depth = self.cam.project(pts)
        if (depth <= 0.3).any():
            return None
        ppm = self.cam.pixels_per_metre(z)
        margin = int(ppm * 0.35) + 3
        x0, y0 = int(uv[:, 0].min()) - margin, int(uv[:, 1].min()) - margin
        x1, y1 = int(uv[:, 0].max()) + margin, int(uv[:, 1].max()) + margin
        H, W = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H or (x1 - x0) * (y1 - y0) > W * H * 2:
            return None
        self._shadow(frame, st.x, st.y, 0.35 + 0.5 * math.sin(st.fall_angle), 0.35 if self.p.time_of_day == "day" else 0.15)
        w, h = x1 - x0, y1 - y0
        layer = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros((h, w), np.float32)
        light = lighting(np.array([st.x, st.y, 1.0]), None, self.layout, self.p.time_of_day)
        light = light + self._fire_light_at(np.array([st.x, st.y, 1.0]), t)
        light = np.clip(light, 0, 1.4)
        loc = {n: (float(uv[k, 0] - x0), float(uv[k, 1] - y0)) for k, n in enumerate(names)}
        # Draw far-side limbs first for a little depth cue.
        cols = {"shirt": person.shirt, "trousers": person.trousers}
        for a, b, part, width in BONES:
            th = max(1, int(round(width * ppm)))
            c = tuple(float(v) for v in (np.asarray(cols[part]) * light))
            pa = (int(round(loc[a][0])), int(round(loc[a][1])))
            pb = (int(round(loc[b][0])), int(round(loc[b][1])))
            cv2.line(layer, pa, pb, c, th, cv2.LINE_AA)
            cv2.line(alpha, pa, pb, 1.0, th, cv2.LINE_AA)
        hr = max(1, int(round(0.11 * ppm * person.height / 1.75)))
        hc = (int(round(loc["head"][0])), int(round(loc["head"][1])))
        skin = tuple(float(v) for v in (np.asarray(person.skin) * light))
        cv2.circle(layer, hc, hr, skin, -1, cv2.LINE_AA)
        cv2.circle(alpha, hc, hr, 1.0, -1, cv2.LINE_AA)
        hair = tuple(float(v) for v in (np.array([0.08, 0.08, 0.1]) * light))
        cv2.ellipse(layer, hc, (hr, hr), 0, 180, 360, hair, -1, cv2.LINE_AA)
        amodal, visible = self._composite(frame, layer, alpha, x0, y0, z)
        box = self._bbox(amodal)
        if box is None:
            return None
        vis_frac = float(visible[2].sum() / max(amodal[2].sum(), 1e-6))
        if vis_frac < 0.02:
            return None
        involuntary = person.role == "victim"
        if st.fall_angle > 1.2:
            state = "fallen" if involuntary else "lying"
        elif st.fall_angle > 0.05 or (involuntary and st.crouch > 0.05):
            state = "falling" if involuntary else "lying_down"
        elif st.crouch > 0.3:
            state = "crouching"
        elif st.speed > 2.0:
            state = "running"
        elif st.speed > 0.1:
            state = "walking"
        else:
            state = "standing"
        return {
            "track_id": person.track_id,
            "class": "person",
            "role": person.role,
            "bbox": _clip_box(box, W, H),
            "visibility": round(vis_frac, 3),
            "state": state,
            "upper_color": SHIRT_NAMES.get(tuple(person.shirt)),
            "lower_color": TROUSER_NAMES.get(tuple(person.trousers)),
            "ground_xy": [round(st.x, 3), round(st.y, 3)],
        }

    def _draw_prop(self, frame, prop, pos, z):
        w_, d_, h_ = prop.size
        x, y, zb = pos
        corners = np.array([[x + dx, y + dy, zb + dz] for dx in (-w_ / 2, w_ / 2) for dy in (-d_ / 2, d_ / 2) for dz in (0, h_)])
        uv, depth = self.cam.project(corners)
        if (depth <= 0.3).any():
            return None
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(hull)
        H, W = frame.shape[:2]
        if bx > W or by > H or bx + bw < 0 or by + bh < 0:
            return None
        layer = np.zeros((bh + 2, bw + 2, 3), np.float32)
        alpha = np.zeros((bh + 2, bw + 2), np.float32)
        hull_l = hull - np.array([bx, by])
        light = np.clip(lighting(np.array([x, y, 0.3]), None, self.layout, self.p.time_of_day), 0, 3.0)
        col = tuple(float(v) for v in np.asarray(prop.color) * light)
        cv2.fillPoly(layer, [hull_l], col, cv2.LINE_AA)
        cv2.fillPoly(alpha, [hull_l], 1.0, cv2.LINE_AA)
        # Top face slightly lighter for a 3D cue.
        top = uv[[1, 3, 7, 5]].astype(np.int32) - np.array([bx, by])
        cv2.fillPoly(layer, [top], tuple(c * 1.3 for c in col), cv2.LINE_AA)
        if zb < 0.05:
            self._shadow(frame, x, y, 0.3, 0.3)
        amodal, visible = self._composite(frame, layer, alpha, bx, by, z)
        box = self._bbox(amodal)
        if box is None:
            return None
        vis_frac = float(visible[2].sum() / max(amodal[2].sum(), 1e-6))
        return {"track_id": prop.track_id, "class": prop.kind, "bbox": _clip_box(box, W, H), "visibility": round(vis_frac, 3), "state": "carried" if zb > 0.05 else "on_ground", "ground_xy": [round(x, 3), round(y, 3)]}

    # ------------------------------------------------------------------ fire
    def _fire_radius(self, fire, t):
        return min(fire.max_radius, 0.15 + 0.35 * (t - fire.ignite_t))

    def _fire_light_at(self, pts, t):
        total = 0.0
        for fire in self.cast.fires:
            if t < fire.ignite_t:
                continue
            r = self._fire_radius(fire, t)
            flick = 0.8 + 0.2 * math.sin(t * 23.0) * math.sin(t * 7.3 + 1.0)
            d2 = ((np.asarray(pts) - np.array([fire.x, fire.y, 0.6])) ** 2).sum(-1) + 0.4
            total = total + (6.0 * r * flick / d2)[..., None] * np.array([0.25, 0.6, 1.0])
        return total

    def _apply_fire_light(self, frame, t):
        if not self.cast.fires or all(t < f.ignite_t for f in self.cast.fires):
            return
        add = self._fire_light_at(self.world, t)
        frame += (self.albedo * add).astype(np.float32)

    def _draw_fire(self, frame, fire, z, t, frame_idx):
        r = self._fire_radius(fire, t)
        rng = _rng(self.p.seed, "fire", frame_idx)
        H, W = frame.shape[:2]
        ppm = self.cam.pixels_per_metre(z)
        layer = np.zeros((H, W, 3), np.float32)
        smoke = np.zeros((H, W), np.float32)
        n = 70
        flame_h = 1.0 + 1.6 * r
        for k in range(n):
            u = ((t * 1.6 + k / n + rng.uniform(0, 0.05)) % 1.0)
            ang = k * 2.399
            rad = r * math.sqrt((k % 17) / 17.0) * (1 - u * 0.7)
            px = fire.x + rad * math.cos(ang) + 0.1 * math.sin(t * 9 + k)
            py = fire.y + rad * math.sin(ang)
            pz = u * flame_h
            (uu, vv), d = self.cam.project(np.array([px, py, pz]))
            if d <= 0.3:
                continue
            size = max(1, int(ppm * r * 0.35 * (1 - u) + 1))
            col = (0.25 * (1 - u), 0.9 - 0.6 * u, 1.3) if u < 0.6 else (0.05, 0.3 * (1 - u), 0.9 * (1 - u) + 0.2)
            cv2.circle(layer, (int(uu), int(vv)), size, col, -1, cv2.LINE_AA)
        for k in range(12):
            u = ((t * 0.35 + k / 12.0) % 1.0)
            (uu, vv), d = self.cam.project(np.array([fire.x + 0.3 * math.sin(k + t), fire.y, flame_h + u * 2.5]))
            if d <= 0.3:
                continue
            cv2.circle(smoke, (int(uu), int(vv)), max(2, int(ppm * (0.3 + u * 0.9) * r)), 0.5 * (1 - u), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), max(1.0, ppm * 0.05))
        smoke = cv2.GaussianBlur(smoke, (0, 0), max(2.0, ppm * 0.15))
        visible = (self.depth > z - 1.0).astype(np.float32)
        layer *= visible[..., None]
        smoke *= visible
        smoke_col = 0.25 if self.p.time_of_day == "night" else 0.4
        frame[:] = frame * (1 - smoke[..., None]) + smoke_col * smoke[..., None]
        frame += layer
        # Ground-truth box: projected flame volume.
        corners = np.array([[fire.x + dx, fire.y + dy, dz] for dx in (-r, r) for dy in (-r, r) for dz in (0, flame_h)])
        uv, d = self.cam.project(corners)
        if (d <= 0.3).any():
            return None
        x0, y0 = uv.min(0)
        x1, y1 = uv.max(0)
        box = _clip_box([int(x0), int(y0), int(x1 - x0), int(y1 - y0)], W, H)
        if box[2] <= 0 or box[3] <= 0:
            return None
        return {"track_id": fire.track_id, "class": "fire", "bbox": box, "visibility": 1.0, "state": "burning", "radius_m": round(r, 3), "ground_xy": [round(fire.x, 3), round(fire.y, 3)]}

    # ------------------------------------------------------------------ weather
    def _apply_weather(self, frame, i):
        if self.weather not in ("rain", "snow"):
            return
        rng = _rng(self.p.seed, "weather", i)
        H, W = frame.shape[:2]
        layer = np.zeros((H, W), np.float32)
        if self.weather == "rain":
            n = int(W * H / 900)
            xs = rng.uniform(0, W, n)
            ys = rng.uniform(0, H, n)
            lens = rng.uniform(H * 0.02, H * 0.07, n)
            for x, y, L in zip(xs, ys, lens):
                cv2.line(layer, (int(x), int(y)), (int(x + L * 0.12), int(y + L)), 0.35, 1, cv2.LINE_AA)
            bright = 0.85 if self.p.time_of_day == "day" else 0.5
        else:
            n = int(W * H / 1500)
            # Flakes drift slowly: positions depend on time, not just frame.
            base = _rng(self.p.seed, "snowbase")
            bx = base.uniform(0, W, n)
            by = base.uniform(0, H, n)
            sp = base.uniform(H * 0.01, H * 0.03, n)
            t = i / self.p.fps
            xs = (bx + np.sin(t * 1.3 + bx) * 4) % W
            ys = (by + sp * t * self.p.fps / 3) % H
            for x, y, s in zip(xs, ys, sp):
                cv2.circle(layer, (int(x), int(y)), max(1, int(s / 6)), 0.8, -1, cv2.LINE_AA)
            bright = 0.95 if self.p.time_of_day != "night" else 0.6
        frame[:] = frame * (1 - layer[..., None]) + bright * layer[..., None]

    def _apply_glare(self, frame, t):
        H, W = frame.shape[:2]
        # A bright source (sun reflection or headlights) near the upper right with a flare streak.
        cx, cy = int(W * 0.78), int(H * 0.22)
        g = np.zeros((H, W), np.float32)
        cv2.circle(g, (cx, cy), int(H * 0.05), 1.0, -1, cv2.LINE_AA)
        cv2.line(g, (0, cy), (W, cy), 0.35, max(1, H // 120), cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0, 0), H * 0.03) * (1.6 + 0.1 * math.sin(t * 3))
        frame += g[..., None]

    # ------------------------------------------------------------------
    def event_annotations(self) -> list:
        fps = self.p.fps
        out = []
        for ev in self.cast.events:
            start_f = min(self.p.n_frames - 1, int(math.ceil(ev.start_t * fps)))
            end_f = self.p.n_frames - 1
            occurred = ev.start_t < self.p.duration_s
            d = {
                "type": ev.type,
                "occurred": occurred,
                "start_time_s": round(ev.start_t, 3),
                "end_time_s": round(ev.end_t, 3),
                "start_frame": start_f if occurred else None,
                "end_frame": end_f if occurred else None,
                "track_ids": ev.track_ids,
                **ev.details,
            }
            if "zone_polygon_world" in ev.details:
                pts = np.array([[x, y, 0.0] for x, y in ev.details["zone_polygon_world"]])
                uv, zz = self.cam.project(pts)
                d["zone_polygon_image"] = [[round(float(u), 1), round(float(v), 1)] for u, v in uv] if (zz > 0).all() else None
            out.append(d)
        return out

    def scene_metadata(self) -> dict:
        return {
            "camera": {
                "position_m": [round(float(v), 3) for v in self.cam.position],
                "tilt_deg": self.p.camera_tilt_deg,
                "pan_deg": self.p.camera_pan_deg,
                "fov_deg": self.p.fov_deg,
                "focal_px": round(self.cam.focal, 3),
                "intrinsics": [[self.cam.focal, 0, self.cam.cx], [0, self.cam.focal, self.cam.cy], [0, 0, 1]],
            },
            "effective_weather": self.weather,
            "notes": self.notes,
        }


def _tonemap(x: np.ndarray) -> np.ndarray:
    """Linear up to 0.8, then a soft shoulder so highlights (fire, lamps, glare) roll off."""
    knee = 0.8
    over = np.maximum(x - knee, 0)
    return np.where(x < knee, x, knee + 0.2 * (1 - np.exp(-over / 0.2)))


def _clip_box(box, W, H):
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return [int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0))]
