"""Cross-camera fusion: a map-and-clock model of the investigation area.

Investigators rarely have one camera. They have footage from shops, homes,
traffic junctions, toll plazas and phones, each with its own clock (DVR clocks
in India are often minutes or hours wrong). This module turns sightings from
any of those sources into one movement trail:

1. Cameras are registered with a location (lat/lon) and a clock offset
   (seconds to ADD to the recorder's time to get true time, measured once
   against a known event or a reference clock).
2. Every sighting (a plate read, a face match, a description hit) is placed
   on true time: source start_time + offset within the file + camera offset.
3. A trail is the highest-scoring chain of sightings that is physically
   possible: the speed needed to travel between consecutive cameras must not
   exceed a limit (default 80 km/h for city traffic). Sightings that would
   need an impossible speed are reported as conflicts, which usually means an
   OCR misread, a lookalike vehicle or a wrong camera clock.
4. "Where to look next": cameras the target could have reached since its
   last sighting, nearest first, with the time window to pull footage for.

The trail is an investigative lead, not evidence of movement: each sighting
must be confirmed visually, and camera clocks must be verified.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

CAMERA_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    name TEXT PRIMARY KEY, lat REAL, lon REAL, clock_offset_s REAL NOT NULL DEFAULT 0,
    kind TEXT, note TEXT, updated_at TEXT, updated_by TEXT
);
"""

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class Camera:
    name: str
    lat: float | None = None
    lon: float | None = None
    clock_offset_s: float = 0.0
    kind: str | None = None  # e.g. "traffic junction", "shop DVR", "toll plaza", "phone video"
    note: str | None = None


@dataclass
class Sighting:
    camera: str | None
    time: datetime | None  # true (corrected) time
    score: float  # match confidence 0..1
    source_id: str
    t_in_file: float
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time"] = self.time.isoformat() if self.time else None
        return d


def haversine_km(a: Camera, b: Camera) -> float | None:
    if None in (a.lat, a.lon, b.lat, b.lon):
        return None
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_time(s: str | None) -> datetime | None:
    """ISO 8601 recorder start time; naive times are taken as IST (Indian DVRs)."""
    if not s:
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=IST)


def true_time(start_time: str | None, t_in_file: float, cam: Camera | None) -> datetime | None:
    t0 = parse_time(start_time)
    if t0 is None:
        return None
    return t0 + timedelta(seconds=t_in_file + (cam.clock_offset_s if cam else 0.0))


def _feasible(a: Sighting, b: Sighting, cams: dict, max_kmh: float, slack_s: float) -> tuple[bool, dict]:
    """Can the target get from sighting a to sighting b? Returns (ok, leg info)."""
    dt = (b.time - a.time).total_seconds()
    leg = {"from": a.camera, "to": b.camera, "minutes": round(dt / 60, 1)}
    if a.camera == b.camera:
        return dt >= -slack_s, leg
    ca, cb = cams.get(a.camera), cams.get(b.camera)
    km = haversine_km(ca, cb) if ca and cb else None
    if km is None:
        leg["km"] = None
        return dt >= -slack_s, leg  # unknown geometry: only order in time is checked
    leg["km"] = round(km, 2)
    needed = km / max(dt + slack_s, 1e-6) * 3600 if dt + slack_s > 0 else math.inf
    leg["needed_kmh"] = round(needed, 1) if math.isfinite(needed) else None
    return needed <= max_kmh, leg


def build_trail(sightings: list, cams: dict, max_kmh: float = 80.0, slack_s: float = 60.0) -> dict:
    """Best physically possible chain of sightings (dynamic programming over time order).

    `slack_s` absorbs small clock errors between cameras. Untimed sightings are
    returned separately: they cannot be placed on a trail until the camera's
    start time is known.
    """
    timed = sorted([s for s in sightings if s.time is not None], key=lambda s: s.time)
    untimed = [s for s in sightings if s.time is None]
    n = len(timed)
    best = [s.score for s in timed]
    prev = [-1] * n
    for j in range(n):
        for i in range(j):
            ok, _ = _feasible(timed[i], timed[j], cams, max_kmh, slack_s)
            if ok and best[i] + timed[j].score > best[j]:
                best[j], prev[j] = best[i] + timed[j].score, i
    chain = []
    if n:
        j = max(range(n), key=lambda k: best[k])
        while j != -1:
            chain.append(j)
            j = prev[j]
        chain.reverse()
    on_trail = set(chain)
    legs = []
    for a, b in itertools.pairwise(chain):
        _, leg = _feasible(timed[a], timed[b], cams, max_kmh, slack_s)
        legs.append(leg)
    conflicts = []
    for k in range(n):
        if k in on_trail:
            continue
        # Explain why it is off the trail: the first trail point it cannot be reconciled with.
        why = None
        for c in chain:
            a, b = (timed[k], timed[c]) if timed[k].time <= timed[c].time else (timed[c], timed[k])
            ok, leg = _feasible(a, b, cams, max_kmh, slack_s)
            if not ok:
                why = leg
                break
        conflicts.append({"sighting": timed[k].to_dict(), "conflicts_with": why,
                          "likely_cause": "misread or lookalike, or a wrong camera clock" if why else "lower score than an overlapping sighting"})
    return {"trail": [timed[k].to_dict() for k in chain], "legs": legs, "conflicts": conflicts,
            "untimed": [s.to_dict() for s in untimed], "max_kmh": max_kmh}


def where_next(last: Sighting, cams: dict, until: datetime, max_kmh: float = 80.0, already: set | None = None, k: int = 10) -> list:
    """Cameras the target could have reached after `last`, before `until`, nearest first.

    Each gets the window to pull: from the earliest possible arrival to `until`.
    """
    src = cams.get(last.camera)
    out = []
    for name, cam in cams.items():
        if name == last.camera or (already and name in already):
            continue
        km = haversine_km(src, cam) if src else None
        if km is None:
            continue
        earliest = last.time + timedelta(hours=km / max_kmh)
        if earliest > until:
            continue
        out.append({"camera": name, "kind": cam.kind, "km": round(km, 2), "pull_from": earliest.isoformat(), "pull_to": until.isoformat()})
    return sorted(out, key=lambda r: r["km"])[:k]
