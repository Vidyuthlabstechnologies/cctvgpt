"""Forensic search over photo galleries, video files and CCTV recordings.

Ingest
  * Walks a folder (or takes single files): images and videos, including
    common NVR/DVR export containers that FFmpeg/OpenCV can read.
  * Hashes every original (SHA-256) and never modifies it.
  * Flags platform-generated synthetic footage (watermark/provenance check) so
    synthetic clips can never be mistaken for real records in an investigation.
  * Samples video at a fixed interval plus extra frames on scene changes.
  * Indexes each sampled frame twice when it is dark/noisy/IR (original and
    enhanced), and every detected person / vehicle / bag as its own crop.

Search
  * Text ("man in a red shirt on a motorcycle", Hindi with the multilingual
    model) or a query photo. Cosine similarity over CLIP embeddings; the best
    of the original/enhanced variants counts; hits in the same video within a
    few seconds merge into one segment.
  * Every search requires a case reference and a purpose and is written to the
    hash-chained audit log.

Export
  * Cuts the segment from the ORIGINAL file by stream copy (no re-encoding),
    hashes it, and writes a chain-of-custody record with the details an
    officer needs for a Bharatiya Sakshya Adhiniyam s.63 certificate
    [VERIFY form and content with counsel].

Not included by design: face recognition / biometric identification (a policy
and legal decision; see docs/SEARCH.md).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..store import new_id, now_iso
from .enhance import assess, enhance, needs_enhancement

COLOUR_WORDS = {"red", "blue", "black", "white", "green", "yellow", "orange", "purple", "pink", "brown", "grey", "gray", "maroon", "saffron", "navy"}

VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle"}
VEHICLE_COLOURS = ["white", "black", "silver", "grey", "red", "blue", "yellow", "green", "brown"]
# Scene vocabulary for "place" tags. Religious sites are deliberately not tagged, to avoid
# footage being sorted by community.
SCENES = ["a road or street", "a highway", "a traffic junction", "a busy market or bazaar", "a railway station platform",
          "a bus stand", "a parking lot", "a petrol pump", "a shop interior", "an office or corridor", "a residential lane",
          "a home interior", "an ATM or bank", "a school or college campus", "a hospital", "a warehouse or factory",
          "an open ground or park", "a bridge or flyover"]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".ts", ".mts", ".3gp", ".flv", ".wmv", ".mpg", ".mpeg", ".dav", ".h264", ".264", ".asf"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL, kind TEXT NOT NULL,
    camera TEXT, start_time TEXT, duration_s REAL, fps REAL, width INTEGER, height INTEGER,
    case_ref TEXT, synthetic INTEGER NOT NULL DEFAULT 0, quality_json TEXT,
    ingested_at TEXT NOT NULL, ingested_by TEXT, frames_indexed INTEGER, objects_indexed INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS sources_sha ON sources(sha256);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL, t REAL NOT NULL,
    kind TEXT NOT NULL, cls TEXT, score REAL, x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    variant TEXT NOT NULL, thumb TEXT, vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS items_source ON items(source_id);
CREATE TABLE IF NOT EXISTS plates (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL, t REAL NOT NULL, text TEXT NOT NULL, raw TEXT,
    valid INTEGER NOT NULL, vehicle TEXT, x INTEGER, y INTEGER, w INTEGER, h INTEGER, thumb TEXT
);
CREATE INDEX IF NOT EXISTS plates_source ON plates(source_id);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL, t REAL NOT NULL, x INTEGER, y INTEGER, w INTEGER, h INTEGER,
    score REAL, size_px INTEGER, thumb TEXT, vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_source ON faces(source_id);
CREATE TABLE IF NOT EXISTS exports (
    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, t0 REAL, t1 REAL, path TEXT, sha256 TEXT,
    case_ref TEXT, actor TEXT, created_at TEXT, custody_json TEXT
);
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(path: Path) -> list:
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXT | VIDEO_EXT else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT | VIDEO_EXT)


def sample_video(path: Path, every_s: float = 1.0, scene_threshold: float = 18.0, max_side: int = 960):
    """Yield (t_seconds, frame) at a fixed interval plus on scene changes (at most 4 per second)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    i = 0
    last_t = -1e9
    last_small = None
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if f.shape[1] > max_side:
                f = cv2.resize(f, (max_side, int(f.shape[0] * max_side / f.shape[1])), interpolation=cv2.INTER_AREA)
            small = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32)
            due = t - last_t >= every_s
            changed = last_small is not None and float(np.abs(small - last_small).mean()) > scene_threshold and t - last_t >= 0.25
            if due or changed:
                last_t, last_small = t, small
                yield t, f
    finally:
        cap.release()
    sample_video.fps = fps


@dataclass
class SearchHit:
    source_id: str
    path: str
    kind: str
    t_start: float
    t_end: float
    t_best: float
    score: float
    match: str  # frame | object:<class>
    bbox: list | None
    item_id: int
    synthetic: bool
    camera: str | None

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class SearchEngine:
    def __init__(self, base_dir: Path, store=None, watermarker=None, keys=None, variant: str = "en", detect_objects: bool = True,
                 embedder=None, detector=None, index_enhanced: bool = False, enable_plates: bool = True,
                 enable_faces: bool = False, enable_tags: bool = True, plate_reader=None, face_backend=None, translator=None):
        self.dir = Path(base_dir)
        (self.dir / "thumbs").mkdir(parents=True, exist_ok=True)
        (self.dir / "exports").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.dir / "search.sqlite3"), check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        for table, col in (("items", "tags"), ("sources", "scene_tags")):
            cols = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
        self.lock = threading.RLock()
        self.enable_plates = enable_plates
        # Face indexing is biometric processing: off unless the deployment enables it.
        self.enable_faces = enable_faces
        self.enable_tags = enable_tags
        self._plate_reader = plate_reader
        self._face_backend = face_backend
        self._translator = translator
        self._text_cache: dict = {}
        self.store = store  # platform store, for the shared audit log
        self.watermarker = watermarker
        self.keys = keys
        self.variant = variant
        self.detect_objects = detect_objects
        # Measured on real night-degraded footage (benchmarks/search_eval.py): adding an
        # enhanced-frame embedding did not improve retrieval (AP 0.42 -> 0.40), so it is
        # off by default. Enhancement is still used for the detector and for display.
        self.index_enhanced = index_enhanced
        self._embedder_override = embedder
        self._detector_override = detector
        self._matrix = None

    # ------------------------------------------------------------------ helpers
    def _audit(self, actor: str, action: str, details: dict) -> None:
        if self.store is not None:
            self.store.audit(actor, action, details)

    def _embedder(self):
        if self._embedder_override is not None:
            return self._embedder_override
        from .models import get_embedder

        return get_embedder(self.variant)

    def _detector(self):
        if self._detector_override is not None:
            return self._detector_override
        from .models import get_detector

        return get_detector()

    def _text_vecs(self, texts: list) -> np.ndarray:
        key = tuple(texts)
        if key not in self._text_cache:
            self._text_cache[key] = np.stack([self._embedder().encode_text(t) for t in texts])
        return self._text_cache[key]

    def _plates_fn(self):
        if self._plate_reader is not None:
            return self._plate_reader
        from .plates import read_plates

        return read_plates

    def _faces_fns(self):
        if self._face_backend is not None:
            return self._face_backend
        from . import faces

        return faces

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def sources(self) -> list:
        rows = self._q("SELECT id, path, sha256, kind, camera, start_time, duration_s, case_ref, synthetic, quality_json, scene_tags, ingested_at, frames_indexed, objects_indexed FROM sources ORDER BY ingested_at DESC")
        for r in rows:
            r["scene_tags"] = json.loads(r["scene_tags"] or "[]")
            r["plates_read"] = self._q("SELECT COUNT(*) AS n FROM plates WHERE source_id=?", (r["id"],))[0]["n"]
            r["faces_indexed"] = self._q("SELECT COUNT(*) AS n FROM faces WHERE source_id=?", (r["id"],))[0]["n"]
        return rows

    def _is_synthetic(self, path: Path, frames: list) -> bool:
        """True if the file carries our provenance manifest or our watermark."""
        try:
            from ..safeguards import provenance
            from ..video_io import read_tags

            if provenance.decode_from_container(read_tags(path).get("comment")):
                return True
        except Exception:
            pass
        if self.watermarker is not None:
            # Watermark detection needs consecutive frames, not the sparse index samples.
            from ..video_io import read_frames

            dense = read_frames(path, max_frames=120)
            if len(dense) >= 6:
                return bool(self.watermarker.detect(dense).present)
        return False

    # ------------------------------------------------------------------ ingest
    def ingest(self, path, actor: str, case_ref: str, camera: str | None = None, start_time: str | None = None, every_s: float = 1.0) -> dict:
        files = discover(Path(path))
        summary = {"files": len(files), "indexed": 0, "skipped_duplicates": 0, "synthetic_flagged": 0, "frames": 0, "objects": 0, "errors": []}
        for f in files:
            try:
                r = self._ingest_file(f, actor, case_ref, camera, start_time, every_s)
            except Exception as e:  # noqa: BLE001 - one bad file must not stop a bulk ingest
                summary["errors"].append({"file": str(f), "error": str(e)})
                continue
            if r is None:
                summary["skipped_duplicates"] += 1
                continue
            summary["indexed"] += 1
            summary["frames"] += r["frames"]
            summary["objects"] += r["objects"]
            summary["synthetic_flagged"] += int(r["synthetic"])
        self._matrix = None
        self._audit(actor, "search_ingest", {"path": str(path), "case_ref": case_ref, **{k: v for k, v in summary.items() if k != "errors"}, "errors": len(summary["errors"])})
        return summary

    def _ingest_file(self, path: Path, actor, case_ref, camera, start_time, every_s):
        digest = sha256_file(path)
        if self._q("SELECT 1 FROM sources WHERE sha256=?", (digest,)):
            return None
        kind = "image" if path.suffix.lower() in IMAGE_EXT else "video"
        if kind == "image":
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError("unreadable image")
            samples = [(0.0, img)]
            fps = None
        else:
            samples = list(sample_video(path, every_s))
            if not samples:
                raise ValueError("no decodable video frames")
            fps = getattr(sample_video, "fps", None)
        sid = new_id("src")
        frames = [f for _, f in samples]
        synthetic = self._is_synthetic(path, frames) if kind == "video" else False
        quals = [assess(f) for f in frames]
        emb = self._embedder()
        rows = []  # (t, kind, cls, score, bbox, variant, thumb, image)
        for (t, f), q in zip(samples, quals):
            thumb = self._thumb(sid, f"{t:.2f}", f)
            rows.append((t, "frame", None, None, None, "original", thumb, f))
            if self.index_enhanced and needs_enhancement(q):
                rows.append((t, "frame", None, None, None, "enhanced", thumb, enhance(f, q)))
        n_obj = 0
        plate_rows: list = []
        face_rows: list = []
        if self.enable_faces:
            face_rows = self._index_faces(sid, samples)
        if self.detect_objects:
            det_frames = [enhance(f, q) if needs_enhancement(q) else f for f, q in zip(frames, quals)]
            for (t, f), df, dets in zip(samples, det_frames, self._detector().detect(det_frames)):
                H, W = f.shape[:2]
                for cls, sc, (x, y, w, h) in dets:
                    if w < 12 or h < 12:
                        continue
                    px, py = int(w * 0.1), int(h * 0.1)
                    x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py)
                    crop = df[y0:y1, x0:x1]
                    thumb = self._thumb(sid, f"{t:.2f}_{n_obj}", crop)
                    rows.append((t, "object", cls, sc, [x, y, w, h], "enhanced" if df is not f else "original", thumb, crop))
                    n_obj += 1
                    if self.enable_plates and cls in VEHICLES - {"bicycle"} and w >= 40:
                        plate_rows.extend(self._read_vehicle_plates(sid, t, cls, f[y0:y1, x0:x1], x0, y0))
        self._save_background(sid, frames)
        if self.enable_plates and kind == "video" and len(plate_rows) > 1:
            plate_rows.extend(self._fuse_plate_reads(plate_rows))
        if self.enable_plates and kind == "image" and not plate_rows:
            # Photos (e.g. a close-up of a plate) get a whole-image pass when no vehicle plate was read.
            plate_rows.extend(self._read_vehicle_plates(sid, 0.0, None, frames[0], 0, 0))
        vecs = emb.encode_images([r[7] for r in rows])
        tags = [None] * len(rows)
        scene_tags: list = []
        if self.enable_tags and len(rows):
            frame_idx = [i for i, r in enumerate(rows) if r[1] == "frame" and r[5] == "original"]
            if frame_idx:
                sims = vecs[frame_idx] @ self._text_vecs(SCENES).T
                mean = sims.mean(0)
                top = np.argsort(-mean)[:3]
                scene_tags = [{"tag": SCENES[i].split(" ", 1)[1], "score": round(float(mean[i]), 3)} for i in top]
            for i, r in enumerate(rows):
                if r[1] == "object" and r[2] in VEHICLES:
                    cs = vecs[i] @ self._text_vecs([f"a {c} {r[2]}" for c in VEHICLE_COLOURS]).T
                    tags[i] = json.dumps({"colour": VEHICLE_COLOURS[int(np.argmax(cs))]})
        h0, w0 = frames[0].shape[:2]
        with self.lock:
            self.db.execute("BEGIN")
            self.db.execute(
                "INSERT INTO sources (id, path, sha256, kind, camera, start_time, duration_s, fps, width, height, case_ref, synthetic, quality_json, ingested_at, ingested_by, frames_indexed, objects_indexed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, str(path.resolve()), digest, kind, camera, start_time, samples[-1][0] if kind == "video" else None, fps, w0, h0, case_ref, int(synthetic),
                 json.dumps({"dark_frames": sum(q.dark for q in quals), "noisy_frames": sum(q.noisy for q in quals), "ir_frames": sum(q.monochrome for q in quals), "sampled": len(quals)}),
                 now_iso(), actor, len(samples), n_obj),
            )
            self.db.execute("UPDATE sources SET scene_tags=? WHERE id=?", (json.dumps(scene_tags), sid))
            for pr in plate_rows:
                self.db.execute("INSERT INTO plates (source_id, t, text, raw, valid, vehicle, x, y, w, h, thumb) VALUES (?,?,?,?,?,?,?,?,?,?,?)", pr)
            for fr in face_rows:
                self.db.execute("INSERT INTO faces (source_id, t, x, y, w, h, score, size_px, thumb, vec) VALUES (?,?,?,?,?,?,?,?,?,?)", fr)
            for (t, k, cls, sc, bbox, var, thumb, _), v, tg in zip(rows, vecs, tags):
                x, y, w, h = bbox if bbox else (None, None, None, None)
                self.db.execute("INSERT INTO items (source_id, t, kind, cls, score, x, y, w, h, variant, thumb, vec, tags) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (sid, t, k, cls, sc, x, y, w, h, var, thumb, v.astype(np.float16).tobytes(), tg))
            self.db.execute("COMMIT")
        return {"frames": len(samples), "objects": n_obj, "synthetic": synthetic}

    @staticmethod
    def _fuse_plate_reads(rows: list, window_s: float = 4.0, min_reads: int = 3) -> list:
        """Multi-frame consensus: reads of one vehicle across nearby frames are fused by per-character vote.

        Night frames carry independent sensor noise, so different frames make different OCR
        errors (benchmarks/plates_faces_eval.py measures the gain). Returns extra rows with
        raw text "CONSENSUS n=<reads> agree=<share>".
        """
        from .plates import consensus, plate_match_score

        rows = sorted(rows, key=lambda r: r[1])
        used = [False] * len(rows)
        out = []
        for i, r in enumerate(rows):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(rows)):
                if not used[j] and rows[j][1] - r[1] <= window_s and plate_match_score(r[2], rows[j][2]) >= 0.6:
                    group.append(j)
            if len(group) < min_reads:
                continue
            for j in group:
                used[j] = True
            text, agree = consensus([rows[j][2] for j in group])
            if not text or any(rows[j][2] == text for j in group) and agree == 1.0:
                continue
            best = rows[group[len(group) // 2]]
            from .plates import normalise_plate

            out.append((best[0], best[1], text, f"CONSENSUS n={len(group)} agree={agree:.2f}", int(normalise_plate(text)[1]), best[5], *best[6:]))
        return out

    def _read_vehicle_plates(self, sid, t, cls, crop, ox, oy) -> list:
        """Plate rows (source_id, t, text, raw, valid, vehicle, x, y, w, h, thumb) for one vehicle crop."""
        if crop.size == 0:
            return []
        q = assess(crop)
        if q.dark or q.noisy:
            crop = enhance(crop, q)
        scale = 2.0 if max(crop.shape[:2]) < 600 else 1.0
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else crop
        out, seen = [], set()
        for r in self._plates_fn()(big)[:2]:
            if r.text in seen or len(r.text) < 5:
                continue
            seen.add(r.text)
            x, y, w, h = (int(v / scale) for v in r.bbox)
            thumb = self._thumb(sid, f"{t:.2f}_plate_{len(seen)}_{ox}", crop[max(0, y - 4): y + h + 4, max(0, x - 4): x + w + 4] if h > 0 and w > 0 else crop)
            out.append((sid, t, r.text, r.raw, int(r.valid_format), cls, ox + x, oy + y, w, h, thumb))
        return out

    def _index_faces(self, sid, samples) -> list:
        fb = self._faces_fns()
        rows = []
        for t, f in samples:
            # Dark/noisy frames are enhanced first: on night-degraded faces this lifted
            # matches from 1% to 30% (benchmarks/plates_faces_eval.py).
            q = assess(f)
            f = enhance(f, q) if (q.dark or q.noisy) else f
            for k, row in enumerate(fb.detect_faces(f)):
                x, y, w, h = (int(v) for v in row[:4])
                if w < 12 or h < 12:
                    continue
                v = fb.embed_face(f, row)
                thumb = self._thumb(sid, f"{t:.2f}_face_{k}", f[max(0, y): y + h, max(0, x): x + w])
                rows.append((sid, t, x, y, w, h, float(row[-1]), int(min(w, h)), thumb, v.astype(np.float16).tobytes()))
        return rows

    def _save_background(self, sid: str, frames: list) -> None:
        """Median of the sampled frames (moving people vanish): the camera's view of the place."""
        sel = frames[:: max(1, len(frames) // 15)][:15]
        small = [cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1]))) for f in sel]
        bg = np.median(np.stack(small), axis=0).astype(np.uint8) if len(small) > 1 else small[0]
        (self.dir / "backgrounds").mkdir(exist_ok=True)
        cv2.imwrite(str(self.dir / "backgrounds" / f"{sid}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _thumb(self, sid: str, tag: str, img: np.ndarray) -> str:
        p = self.dir / "thumbs" / sid
        p.mkdir(exist_ok=True)
        scale = 240 / max(img.shape[:2])
        t = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))) if scale < 1 else img
        out = p / f"{tag}.jpg"
        cv2.imwrite(str(out), t, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return str(out.relative_to(self.dir))

    # ------------------------------------------------------------------ search
    def _load(self):
        if self._matrix is None:
            rows = self._q("SELECT i.id, i.source_id, i.t, i.kind, i.cls, i.x, i.y, i.w, i.h, i.vec, s.path, s.kind AS skind, s.synthetic, s.camera, s.case_ref FROM items i JOIN sources s ON s.id = i.source_id")
            if not rows:
                self._matrix = (np.zeros((0, 1), np.float32), [])
            else:
                M = np.stack([np.frombuffer(r.pop("vec"), dtype=np.float16) for r in rows]).astype(np.float32)
                self._matrix = (M, rows)
        return self._matrix

    def query_vector(self, text: str | None = None, image: np.ndarray | None = None) -> np.ndarray:
        emb = self._embedder()
        parts = []
        if text:
            parts.append(emb.encode_text(text))
        if image is not None:
            parts.append(emb.encode_images([image])[0])
        if not parts:
            raise ValueError("give a text query, an image, or both")
        v = np.mean(parts, axis=0)
        return v / np.linalg.norm(v)

    def search(self, actor: str, case_ref: str, purpose: str, text: str | None = None, image: np.ndarray | None = None,
               k: int = 30, kinds=("frame", "object"), classes=None, source_ids=None, include_synthetic: bool = False, merge_s: float = 3.0,
               scene: str | None = None, vehicle_colour: str | None = None) -> dict:
        if not case_ref or not purpose:
            raise ValueError("every search needs a case reference and a purpose (accountability)")
        translation = None
        original_text = text
        if text:
            from .translate import translate_query

            translation = translate_query(text, self._translator)
            text = translation["text"]
        q = self.query_vector(text, image)
        if scene:
            allowed = {r["id"] for r in self._q("SELECT id, scene_tags FROM sources") if any(scene.lower() in tg["tag"].lower() for tg in json.loads(r["scene_tags"] or "[]"))}
            source_ids = list(allowed & set(source_ids)) if source_ids else list(allowed)
            if not source_ids:
                source_ids = ["__none__"]
        colour_items = None
        if vehicle_colour:
            colour_items = {r["id"] for r in self._q("SELECT id, tags FROM items WHERE tags IS NOT NULL") if json.loads(r["tags"]).get("colour") == vehicle_colour.lower()}
        M, meta = self._load()
        scores = M @ q if meta else np.zeros(0, np.float32)
        keep = np.ones(len(meta), bool)
        for i, m in enumerate(meta):
            if m["kind"] not in kinds or (classes and m["kind"] == "object" and m["cls"] not in classes) or (classes and m["kind"] == "frame" and "frame" not in classes):
                keep[i] = False
            elif source_ids and m["source_id"] not in source_ids:
                keep[i] = False
            elif m["synthetic"] and not include_synthetic:
                keep[i] = False
            elif colour_items is not None and m["id"] not in colour_items:
                keep[i] = False
        order = [i for i in np.argsort(-scores) if keep[i]]
        hits: list = []
        for i in order:
            m = meta[i]
            s = float(scores[i])
            # Merge with an existing hit from the same video within merge_s seconds.
            merged = False
            for h in hits:
                if h.source_id == m["source_id"] and m["skind"] == "video" and h.t_start - merge_s <= m["t"] <= h.t_end + merge_s:
                    h.t_start, h.t_end = min(h.t_start, m["t"]), max(h.t_end, m["t"])
                    merged = True
                    break
                if h.source_id == m["source_id"] and m["skind"] == "image":
                    merged = True
                    break
            if merged:
                continue
            hits.append(SearchHit(m["source_id"], m["path"], m["skind"], m["t"], m["t"], m["t"], s,
                                  "frame" if m["kind"] == "frame" else f"object:{m['cls']}",
                                  [m["x"], m["y"], m["w"], m["h"]] if m["kind"] == "object" else None, m["id"], bool(m["synthetic"]), m["camera"]))
            if len(hits) >= k:
                break
        warnings = []
        words = set((text or "").lower().replace(",", " ").split())
        if words & COLOUR_WORDS:
            ir_sources = {r["id"] for r in self._q("SELECT id, quality_json FROM sources") if json.loads(r["quality_json"] or "{}").get("ir_frames", 0) > 0}
            n_ir = sum(1 for h in hits if h.source_id in ir_sources)
            if n_ir:
                warnings.append(f"{n_ir} result(s) come from infrared/monochrome footage, where colours cannot be judged. Measured colour-search accuracy on IR footage is near chance.")
        if translation and translation["source_lang"] and not translation["translated"]:
            warnings.append(f"Query in {translation['source_lang']} could not be translated ({translation['error']}); Indian-language queries without translation score near chance. Use English.")
        elif translation and translation["translated"]:
            warnings.append(f"Searched for the translation \"{text}\" ({translation['backend']}). Check it matches what you meant.")
        out = {"results": [h.to_dict() for h in hits], "searched_items": int(keep.sum()),
               "query": {"text": original_text, "searched_text": text, "translation": translation, "image": image is not None, "scene": scene, "vehicle_colour": vehicle_colour},
               "warnings": warnings,
               "note": "Scores are relative similarities, not probabilities. Review every result; a search hit is a lead, not an identification."}
        self._audit(actor, "search_query", {"case_ref": case_ref, "purpose": purpose, "text": original_text, "searched_text": text, "image_query": image is not None, "results": len(hits),
                                            "top": [h.source_id for h in hits[:5]]})
        return out

    # ------------------------------------------------------------------ plates
    def search_plate(self, actor: str, case_ref: str, purpose: str, plate: str, k: int = 30, min_score: float = 0.6, include_synthetic: bool = False) -> dict:
        """Full or partial plate search, tolerant of OCR errors (edit distance)."""
        from .plates import plate_match_score

        if not case_ref or not purpose:
            raise ValueError("every search needs a case reference and a purpose (accountability)")
        rows = self._q("SELECT p.*, s.path, s.kind AS skind, s.synthetic, s.camera FROM plates p JOIN sources s ON s.id = p.source_id")
        groups: dict = {}
        for r in rows:
            if r["synthetic"] and not include_synthetic:
                continue
            sc = plate_match_score(plate, r["text"])
            if sc < min_score:
                continue
            key = (r["source_id"], r["text"])
            g = groups.get(key)
            if g is None or sc > g["score"]:
                prev = g or {}
                groups[key] = {"source_id": r["source_id"], "path": r["path"], "kind": r["skind"], "plate_read": r["text"], "raw_ocr": r["raw"],
                               "valid_format": bool(r["valid"]), "vehicle": r["vehicle"], "score": round(sc, 3), "plate_id": r["id"],
                               "bbox": [r["x"], r["y"], r["w"], r["h"]], "camera": r["camera"],
                               "t_start": min(r["t"], prev.get("t_start", r["t"])), "t_end": max(r["t"], prev.get("t_end", r["t"]))}
            else:
                g["t_start"], g["t_end"] = min(g["t_start"], r["t"]), max(g["t_end"], r["t"])
        res = sorted(groups.values(), key=lambda g: -g["score"])[:k]
        self._audit(actor, "search_plate", {"case_ref": case_ref, "purpose": purpose, "plate_query": plate, "results": len(res)})
        return {"results": res, "plates_indexed": len(rows),
                "note": "Plate reads are OCR output and can be wrong (measured: about half of real Indian plate crops read within one character). Confirm visually."}

    def plate_thumb(self, plate_id: int) -> Path | None:
        r = self._q("SELECT thumb FROM plates WHERE id=?", (plate_id,))
        return (self.dir / r[0]["thumb"]) if r and r[0]["thumb"] else None

    # ------------------------------------------------------------------ faces
    def search_face(self, actor: str, case_ref: str, purpose: str, justification: str, image: np.ndarray, k: int = 20, all_cases: bool = False) -> dict:
        """Rank faces in this case's footage by similarity to a probe photo. Opt-in, case-scoped, audited."""
        if not self.enable_faces:
            raise PermissionError("face search is disabled on this deployment (CCTVGPT_FACE_SEARCH_ENABLED)")
        if not case_ref or not purpose or len((justification or "").strip()) < 20:
            raise ValueError("face search needs a case reference, a purpose and a written justification (20+ characters)")
        fb = self._faces_fns()
        pq = assess(image)
        if pq.dark or pq.noisy:
            image = enhance(image, pq)
        probe = fb.largest_face(image)
        if probe is None:
            raise ValueError("no face found in the probe image")
        pv = fb.embed_face(image, probe)
        if all_cases:
            rows = self._q("SELECT f.*, s.path, s.kind AS skind, s.camera, s.synthetic, s.case_ref FROM faces f JOIN sources s ON s.id = f.source_id")
        else:
            rows = self._q("SELECT f.*, s.path, s.kind AS skind, s.camera, s.synthetic, s.case_ref FROM faces f JOIN sources s ON s.id = f.source_id WHERE s.case_ref=?", (case_ref,))
        rows = [r for r in rows if not r["synthetic"]]
        out = []
        if rows:
            M = np.stack([np.frombuffer(r["vec"], np.float16).astype(np.float32) for r in rows])
            sims = M @ pv
            for i in np.argsort(-sims)[:k]:
                r = rows[i]
                out.append({"face_id": r["id"], "source_id": r["source_id"], "path": r["path"], "kind": r["skind"], "t": r["t"], "camera": r["camera"],
                            "bbox": [r["x"], r["y"], r["w"], r["h"]], "similarity": round(float(sims[i]), 3),
                            "above_threshold": bool(sims[i] >= fb.MATCH_THRESHOLD),
                            "reliability": "low (face under 40 px)" if r["size_px"] < fb.MIN_RELIABLE_FACE_PX else "normal"})
        probe_hash = hashlib.sha256(cv2.imencode(".png", image)[1].tobytes()).hexdigest()
        self._audit(actor, "search_face", {"case_ref": case_ref, "purpose": purpose, "justification": justification, "probe_sha256": probe_hash,
                                           "scope": "all cases" if all_cases else "this case", "faces_compared": len(rows), "above_threshold": sum(o["above_threshold"] for o in out)})
        return {"results": out, "faces_compared": len(rows), "threshold": fb.MATCH_THRESHOLD,
                "note": "Possible matches for human verification only. A similarity score is not an identification; accuracy drops sharply on small, blurred or dark faces."}

    def face_thumb(self, face_id: int) -> Path | None:
        r = self._q("SELECT thumb FROM faces WHERE id=?", (face_id,))
        return (self.dir / r[0]["thumb"]) if r and r[0]["thumb"] else None

    # ------------------------------------------------------------------ places
    def places(self, min_inliers: int = 30) -> list:
        """Group sources filmed at the same place.

        Uses geometric matching of each source's background (median frame):
        ORB keypoints, ratio test, then a RANSAC homography. A fixed camera seen
        by day and by night, or two cameras with overlapping views, share many
        consistent keypoints; different rooms of the same type do not. (Embedding
        similarity was measured to confuse two rooms of the same lab, so it is
        only used for the scene-type tags.)
        """
        meta = {s["id"]: s for s in self.sources()}
        orb = cv2.ORB_create(2000)
        feats = {}
        for sid in meta:
            p = self.dir / "backgrounds" / f"{sid}.jpg"
            img = cv2.imread(str(p)) if p.exists() else None
            if img is None:
                continue
            q = assess(img)
            g = cv2.cvtColor(enhance(img, q) if (q.dark or q.noisy) else img, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(g, None)
            if des is not None and len(kp) >= 20:
                feats[sid] = (kp, des)
        parent = {sid: sid for sid in meta}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        ids = list(feats)
        links = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                (ka, da), (kb, db) = feats[ids[i]], feats[ids[j]]
                good = [m for m, n in (pair for pair in bf.knnMatch(da, db, k=2) if len(pair) == 2) if m.distance < 0.75 * n.distance]
                if len(good) < min_inliers:
                    continue
                src = np.float32([ka[m.queryIdx].pt for m in good])
                dst = np.float32([kb[m.trainIdx].pt for m in good])
                _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                inl = int(mask.sum()) if mask is not None else 0
                if inl >= min_inliers:
                    parent[find(ids[i])] = find(ids[j])
                    links[(ids[i], ids[j])] = inl
        groups: dict = {}
        for sid in meta:
            groups.setdefault(find(sid), []).append(sid)
        out = []
        for root, members in groups.items():
            tags: dict = {}
            for sid in members:
                for tg in meta[sid].get("scene_tags", []):
                    tags[tg["tag"]] = tags.get(tg["tag"], 0) + 1
            out.append({"place_id": f"place_{root[4:12]}", "sources": [{"id": sid, "path": meta[sid]["path"], "camera": meta[sid]["camera"]} for sid in members],
                        "scene_tags": sorted(tags, key=lambda t: -tags[t])[:3],
                        "evidence": [{"a": a, "b": b, "matched_keypoints": n} for (a, b), n in links.items() if a in members]})
        return sorted(out, key=lambda g: -len(g["sources"]))

    def thumb_path(self, item_id: int) -> Path | None:
        r = self._q("SELECT thumb FROM items WHERE id=?", (item_id,))
        return (self.dir / r[0]["thumb"]) if r and r[0]["thumb"] else None

    def source(self, source_id: str) -> dict | None:
        r = self._q("SELECT * FROM sources WHERE id=?", (source_id,))
        return r[0] if r else None

    # ------------------------------------------------------------------ export
    def export_segment(self, source_id: str, t0: float, t1: float, actor: str, case_ref: str, pad_s: float = 2.0) -> dict:
        src = self.source(source_id)
        if src is None:
            raise ValueError("unknown source")
        path = Path(src["path"])
        if sha256_file(path) != src["sha256"]:
            raise ValueError("original file has changed since ingest; integrity cannot be confirmed")
        eid = new_id("exp")
        start = max(0.0, t0 - pad_s)
        end = t1 + pad_s
        if src["kind"] == "image":
            out = self.dir / "exports" / f"{eid}{path.suffix.lower()}"
            out.write_bytes(path.read_bytes())
            method = "byte-for-byte copy of the original image"
        else:
            out = self.dir / "exports" / f"{eid}.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(path), "-map", "0", "-c", "copy", str(out)], check=True, capture_output=True)
            method = "stream copy from the original (no re-encoding); cut points snap to the nearest preceding keyframe"
        custody = {
            "export_id": eid, "case_ref": case_ref, "exported_by": actor, "exported_at": now_iso(),
            "original": {"path": str(path), "sha256": src["sha256"], "ingested_at": src["ingested_at"], "ingested_by": src["ingested_by"], "camera": src["camera"]},
            "segment": {"requested_start_s": round(t0, 3), "requested_end_s": round(t1, 3), "exported_start_s": round(start, 3), "exported_end_s": round(end, 3)},
            "export": {"file": out.name, "sha256": sha256_file(out), "method": method},
            "synthetic_flag": bool(src["synthetic"]),
            "notes": [
                "Enhanced frames shown in search are for viewing only; this export is cut from the unmodified original.",
                "For use in court, the responsible officer must issue the certificate required under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 covering the source device and this copy [VERIFY form with counsel].",
            ],
        }
        if src["synthetic"]:
            custody["notes"].insert(0, "WARNING: this file carries CCTVGPT synthetic-content marks. It is NOT a real recording and must not be used as evidence.")
        (self.dir / "exports" / f"{eid}.custody.json").write_text(json.dumps(custody, indent=2))
        with self.lock:
            self.db.execute("INSERT INTO exports (id, source_id, t0, t1, path, sha256, case_ref, actor, created_at, custody_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (eid, source_id, start, end, str(out), custody["export"]["sha256"], case_ref, actor, custody["exported_at"], json.dumps(custody)))
        self._audit(actor, "search_export", {"case_ref": case_ref, "export_id": eid, "source_id": source_id, "sha256": custody["export"]["sha256"]})
        return {**custody, "path": str(out)}
