"""Event library (FR-4) and background activity.

Every event is a deterministic script over time, so ground truth (who, where,
when) is exact rather than estimated (FR-7). Adding an event means adding a
builder function to EVENT_BUILDERS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .actors import Person, PersonState, Prop
from .scenes import SceneLayout

STRIDE_M = 0.7


@dataclass
class EventSpec:
    type: str
    start_t: float
    end_t: float
    track_ids: list
    details: dict = field(default_factory=dict)


@dataclass
class FireSpec:
    x: float
    y: float
    ignite_t: float
    max_radius: float = 1.0
    track_id: int = 900


@dataclass
class Cast:
    people: list = field(default_factory=list)
    props: list = field(default_factory=list)
    fires: list = field(default_factory=list)
    events: list = field(default_factory=list)


def _heading(dx: float, dy: float) -> float:
    return math.atan2(dx, dy)


def path_script(points, speed: float, start_t: float, wobble: float = 0.0, rng=None, pauses=None):
    """Walk along a polyline at constant speed starting at start_t. Visible only while on the path.

    pauses: optional list of (vertex_index, seconds) to stand still at a vertex.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    segs = []
    t = start_t
    pause_map = dict(pauses or [])
    dist_acc = 0.0
    for i in range(len(pts) - 1):
        if i in pause_map:
            segs.append(("pause", t, t + pause_map[i], pts[i], pts[i], dist_acc))
            t += pause_map[i]
        L = float(np.linalg.norm(pts[i + 1] - pts[i]))
        dur = L / speed
        segs.append(("walk", t, t + dur, pts[i], pts[i + 1], dist_acc))
        t += dur
        dist_acc += L
    end_t = t
    wob_phase = float(rng.uniform(0, 6.28)) if rng is not None else 0.0

    def script(time: float):
        if time < start_t or time > end_t:
            return None
        for kind, t0, t1, a, b, d0 in segs:
            if t0 <= time <= t1:
                frac = 0.0 if t1 == t0 else (time - t0) / (t1 - t0)
                p = a + (b - a) * frac
                d = b - a
                if kind == "pause" or np.allclose(d, 0):
                    # Keep heading of the next walking segment.
                    nxt = [s for s in segs if s[1] >= t1 and s[0] == "walk"]
                    d = (nxt[0][4] - nxt[0][3]) if nxt else np.array([0.0, 1.0])
                    dist = d0
                    spd = 0.0
                else:
                    dist = d0 + float(np.linalg.norm(b - a)) * frac
                    spd = speed
                h = _heading(float(d[0]), float(d[1]))
                lat = wobble * math.sin(time * 0.9 + wob_phase)
                px = float(p[0]) + lat * math.cos(h)
                py = float(p[1]) - lat * math.sin(h)
                return PersonState(px, py, h, dist / STRIDE_M * math.pi, spd)
        return None

    script.end_t = end_t
    return script


def _lane_walker(layout: SceneLayout, rng, duration: float):
    lane = layout.walk_lanes[int(rng.integers(len(layout.walk_lanes)))]
    a, b = np.asarray(lane[0], float), np.asarray(lane[1], float)
    if rng.random() < 0.5:
        a, b = b, a
    off = rng.uniform(-0.4, 0.4)
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    a, b = a + n * off, b + n * off
    speed = float(rng.uniform(0.9, 1.5))
    length = float(np.linalg.norm(b - a))
    # Start anywhere such that the person is on screen for part of the clip.
    start = float(rng.uniform(-length / speed * 0.9, duration * 0.85))
    return path_script([a, b], speed, start, wobble=0.08, rng=rng)


def background(layout: SceneLayout, density: str, rng, duration: float, first_id: int = 1) -> list:
    n = {"empty": 0, "low": 3, "medium": 8, "high": 18}[density]
    people = []
    for i in range(n):
        p = Person.random(first_id + i, rng)
        p.script = _lane_walker(layout, rng, duration)
        people.append(p)
    return people


# ---------------------------------------------------------------------------
# Events


def ev_none(layout, rng, duration, cast: Cast, next_id):
    return next_id


def ev_fall(layout, rng, duration, cast: Cast, next_id):
    victim = Person.random(next_id, rng, role="victim")
    hx, hy = layout.hazard_point
    t_fall = float(rng.uniform(0.3, 0.45)) * duration
    speed = float(rng.uniform(0.9, 1.3))
    # Approach from further down the scene, walking towards the camera at an angle.
    approach = np.array([float(rng.uniform(-0.6, 0.6)), 1.0])
    approach /= np.linalg.norm(approach)
    start = np.array([hx, hy]) + approach * speed * t_fall
    walk = path_script([start, (hx, hy)], speed, 0.0)
    fall_dur = float(rng.uniform(0.6, 0.9))
    fall_dir = float(rng.uniform(-1.2, 1.2)) + math.pi  # fall roughly forward (towards camera)
    heading = _heading(hx - start[0], hy - start[1])

    def script(t):
        if t < t_fall:
            return walk(t)
        k = min(1.0, (t - t_fall) / fall_dur)
        ang = (math.pi / 2) * (k * k)  # accelerate under gravity
        # Small residual movement while on the ground.
        wig = 0.03 * math.sin(t * 2.0) if k >= 1.0 else 0.0
        return PersonState(hx, hy, heading, 0.0, 0.0, fall_angle=ang + wig, fall_dir=heading + (fall_dir - math.pi))

    victim.script = script
    cast.people.append(victim)
    cast.events.append(EventSpec("fall", t_fall, duration, [victim.track_id], {"fall_duration_s": round(fall_dur, 3), "ground_point": [hx, hy]}))
    return next_id + 1


def ev_intrusion(layout, rng, duration, cast: Cast, next_id):
    zone = layout.restricted_zone
    poly = np.asarray(zone.polygon)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    intruder = Person.random(next_id, rng, role="intruder")
    target = np.array([float(rng.uniform(x0 + 0.5, x1 - 0.5)) * 0.6, float(rng.uniform(y0 + 1.0, y1 - 0.5))])
    t_start = float(rng.uniform(0.0, 0.1)) * duration
    speed = float(rng.uniform(1.0, 1.4))
    pause = float(rng.uniform(0.5, 1.2))
    # Scale the approach so the boundary is crossed around the middle of the clip.
    approach = float(np.clip(speed * (0.45 * duration - t_start - pause), 1.5, 9.0))
    start = np.array([float(rng.uniform(-1.0, 1.0)), y0 - 0.6 - approach])
    # Cautious approach: pause before crossing the boundary.
    edge = np.array([target[0] * 0.7, y0 - 0.6])
    script = path_script([start, edge, target], speed, t_start, pauses=[(1, pause)])
    intruder.script = script
    # Find when the feet first cross into the zone.
    t_enter = None
    for t in np.arange(t_start, duration, 1.0 / 50):
        st = script(t)
        if st is not None and x0 <= st.x <= x1 and y0 <= st.y <= y1:
            t_enter = float(t)
            break
    cast.people.append(intruder)
    if t_enter is not None:
        cast.events.append(EventSpec("intrusion", t_enter, duration, [intruder.track_id], {"zone_polygon_world": poly.tolist()}))
    else:
        cast.events.append(EventSpec("intrusion", duration, duration, [intruder.track_id], {"zone_polygon_world": poly.tolist(), "note": "clip too short for entry"}))
    # After reaching the target the intruder lingers in the zone.
    end_state = {"t": script.end_t}

    def linger(t, _s=script):
        st = _s(t)
        if st is None and t > end_state["t"]:
            st = _s(end_state["t"])
            if st is not None:
                st.speed = 0.0
                st.heading += 0.4 * math.sin(t)
        return st

    intruder.script = linger
    return next_id + 1


def ev_fire(layout, rng, duration, cast: Cast, next_id):
    hx, hy = layout.hazard_point
    t0 = float(rng.uniform(0.2, 0.35)) * duration
    fire = FireSpec(hx + float(rng.uniform(-0.5, 0.5)), hy + float(rng.uniform(0, 2.0)), t0, float(rng.uniform(0.7, 1.3)), track_id=next_id)
    cast.fires.append(fire)
    cast.events.append(EventSpec("fire", t0, duration, [fire.track_id], {"ground_point": [fire.x, fire.y]}))
    return next_id + 1


def ev_crowd_surge(layout, rng, duration, cast: Cast, next_id):
    xmin, xmax, ymin, ymax = layout.walk_area
    n = int(rng.integers(26, 36))
    t_surge = float(rng.uniform(0.35, 0.5)) * duration
    ex, ey = layout.exit_point
    ids = []
    for i in range(n):
        p = Person.random(next_id + i, rng, role="crowd")
        x0 = float(rng.uniform(xmin, xmax))
        y0 = float(rng.uniform(ymin, ymax * 0.8))
        ph = rng.uniform(0, 6.28, size=4)
        fr = rng.uniform(0.15, 0.4, size=2)
        run_speed = float(rng.uniform(2.6, 4.0))
        react = float(rng.uniform(0.0, 0.6))  # reaction delay

        def script(t, x0=x0, y0=y0, ph=ph, fr=fr, run_speed=run_speed, react=react):
            # Milling: slow wandering around the start point.
            def mill(tt):
                return (x0 + 0.6 * math.sin(fr[0] * tt + ph[0]), y0 + 0.6 * math.sin(fr[1] * tt + ph[1]))

            ts = t_surge + react
            if t < ts:
                x, y = mill(t)
                x2, y2 = mill(t + 0.1)
                spd = math.hypot(x2 - x, y2 - y) / 0.1
                return PersonState(x, y, _heading(x2 - x, y2 - y), t * spd / STRIDE_M * math.pi, spd)
            sx, sy = mill(ts)
            dx, dy = ex - sx, ey - sy
            L = math.hypot(dx, dy)
            # Accelerate over 0.8 s, then run.
            dt = t - ts
            dist = run_speed * (dt - 0.4) if dt > 0.8 else run_speed * dt * dt / 1.6
            if dist >= L:
                return None
            k = dist / L
            return PersonState(sx + dx * k, sy + dy * k, _heading(dx, dy), dist / STRIDE_M * math.pi * 0.8, run_speed)

        p.script = script
        cast.people.append(p)
        ids.append(p.track_id)
    cast.events.append(EventSpec("crowd_surge", t_surge, duration, ids, {"exit_point": [ex, ey], "crowd_size": n}))
    return next_id + n


def ev_abandoned_object(layout, rng, duration, cast: Cast, next_id):
    owner = Person.random(next_id, rng, role="owner")
    dx_, dy_ = layout.drop_point
    t_drop = 0.25 * duration
    threshold = min(3.0, 0.35 * duration)
    speed = float(rng.uniform(1.0, 1.3))
    start = np.array([dx_ + float(rng.uniform(-0.5, 0.5)), dy_ + speed * t_drop])
    leave = np.array([dx_ - 3.0 if dx_ > 0 else dx_ + 3.0, -2.0])
    pause = 1.2
    walk = path_script([start, (dx_, dy_), leave], speed, 0.0, pauses=[(1, pause)])

    def owner_script(t):
        st = walk(t)
        if st is not None:
            st.carrying = t < t_drop + pause * 0.5
        return st

    owner.script = owner_script
    cast.people.append(owner)
    bag = Prop(next_id + 1, "bag", (0.45, 0.25, 0.4), (0.12, 0.2, 0.25))
    t_place = t_drop + pause * 0.5

    def bag_script(t):
        if t >= t_place:
            return (dx_ + 0.35, dy_ - 0.1, 0.0)
        st = walk(t)
        if st is None:
            return None
        # Carried in the hand, beside the owner.
        c, s = math.cos(st.heading), math.sin(st.heading)
        return (st.x - 0.3 * c, st.y + 0.3 * s, 0.25)

    bag.script = bag_script
    cast.props.append(bag)
    cast.events.append(
        EventSpec(
            "abandoned_object",
            min(duration, t_place + threshold),
            duration,
            [bag.track_id, owner.track_id],
            {"drop_time_s": round(t_place, 3), "unattended_threshold_s": threshold, "ground_point": [dx_ + 0.35, dy_ - 0.1]},
        )
    )
    return next_id + 2


EVENT_BUILDERS = {
    "none": ev_none,
    "fall": ev_fall,
    "intrusion": ev_intrusion,
    "fire": ev_fire,
    "crowd_surge": ev_crowd_surge,
    "abandoned_object": ev_abandoned_object,
}


def build_cast(layout: SceneLayout, event: str, density: str, rng, duration: float) -> Cast:
    cast = Cast()
    next_id = 100
    next_id = EVENT_BUILDERS[event](layout, rng, duration, cast, next_id)
    cast.people.extend(background(layout, density if event != "crowd_surge" else "low", rng, duration, first_id=1))
    return cast
