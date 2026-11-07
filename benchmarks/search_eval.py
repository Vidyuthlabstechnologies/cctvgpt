#!/usr/bin/env python3
"""How well does CCTVGPT search find what an investigator asks for, including in poor night footage?

Part A, real footage (URFD, per-frame labels):
  Query "a person lying on the floor" over every sampled frame (2 fps) of all
  70 clips. A frame is relevant when URFD labels it "lying" (1). Measured
  three ways:
    * day     - the original footage
    * night   - the same frames degraded to poor night CCTV (dark, noisy, JPEG)
    * night+E - the night frames indexed as the product does: original and
                enhanced variants, detector crops, best match counts
  plus the same query in Hindi with the multilingual model.

Part B, synthetic footage with exact clothing ground truth:
  "a person wearing a <colour> shirt" over person crops (ground-truth boxes,
  to isolate retrieval from detection) under day / dusk / night IR / night
  colour / rain / fog. Shows where colour search works and where it cannot
  (IR night footage has no colour).

Metric: average precision (AP) and precision@10/@50 per query; mean over queries.

    python benchmarks/search_eval.py --urfd external_data/urfd --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cctvgpt.evaluation import average_precision  # noqa: E402
from cctvgpt.search.enhance import assess, degrade_to_night, enhance, needs_enhancement  # noqa: E402
from cctvgpt.search.models import get_detector, get_embedder  # noqa: E402

LYING_QUERIES = {
    "en": "a person lying on the floor",
    "hi": "फर्श पर लेटा हुआ व्यक्ति",
}


def retrieval_metrics(scores: np.ndarray, relevant: np.ndarray) -> dict:
    order = np.argsort(-scores)
    rel = relevant[order]
    ap = average_precision(list(scores[order]), list(rel.astype(int)), int(relevant.sum()))
    return {"ap": round(ap, 3), "p@10": round(float(rel[:10].mean()), 3), "p@50": round(float(rel[:50].mean()), 3),
            "relevant": int(relevant.sum()), "pool": int(len(relevant)), "random_ap": round(float(relevant.mean()), 3)}


# ---------------------------------------------------------------------------
# Part A: real footage


def load_urfd_frames(root: Path, step: int = 15):
    labels = {}
    for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for line in (root / fn).read_text().splitlines():
            parts = line.split(",")
            if len(parts) > 2:
                labels[(parts[0], int(parts[1]))] = int(parts[2])
    frames, rel = [], []
    names = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]
    for n in names:
        cap = cv2.VideoCapture(str(root / f"{n}-cam0.mp4"))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            i += 1  # URFD frame numbers are 1-based
            if i % step == 0 and (n, i) in labels:
                frames.append(f[:, f.shape[1] // 2 :].copy())
                rel.append(labels[(n, i)] == 1)
        cap.release()
    return frames, np.array(rel)


_PREP: dict = {}
_EMB: dict = {}


def _prepare(frames, det):
    """Enhanced variants and person crops for a frame set; computed once and reused across queries."""
    key = id(frames)
    if key not in _PREP:
        quals = [assess(f) for f in frames]
        enhanced = {i: enhance(frames[i], q) for i, q in enumerate(quals) if needs_enhancement(q)}
        det_frames = [enhanced.get(i, f) for i, f in enumerate(frames)]
        crops, owner = [], []
        for i, dets in enumerate(det.detect(det_frames)):
            H, W = det_frames[i].shape[:2]
            for cls, sc, (x, y, w, h) in dets:
                if cls != "person" or w < 12 or h < 12:
                    continue
                px, py = int(w * 0.1), int(h * 0.1)
                crops.append(det_frames[i][max(0, y - py): min(H, y + h + py), max(0, x - px): min(W, x + w + px)])
                owner.append(i)
        _PREP[key] = (enhanced, crops, owner)
    return _PREP[key]


def _vecs(tag, emb, images):
    key = (tag, emb.variant)
    if key not in _EMB:
        _EMB[key] = emb.encode_images(images)
    return _EMB[key]


def index_scores(frames, query_vec, emb, product_mode: bool, det=None):
    """Per-frame score: the frame alone, or (product mode) the best of original/enhanced frame and person crops."""
    base = _vecs(("frames", id(frames)), emb, frames) @ query_vec
    if not product_mode:
        return base
    best = base.copy()
    enhanced, crops, owner = _prepare(frames, det)
    if enhanced:
        idx = sorted(enhanced)
        s = _vecs(("enh", id(frames)), emb, [enhanced[i] for i in idx]) @ query_vec
        for j, i in enumerate(idx):
            best[i] = max(best[i], s[j])
    if crops:
        s = _vecs(("crops", id(frames)), emb, crops) @ query_vec
        for j, i in enumerate(owner):
            best[i] = max(best[i], s[j])
    return best


def part_a(root: Path) -> dict:
    frames, rel = load_urfd_frames(root)
    rng = np.random.default_rng(0)
    night = [degrade_to_night(f, rng) for f in frames]
    out = {"frames": len(frames), "lying_frames": int(rel.sum())}
    det = get_detector()
    for variant in ARGS.variants.split(","):
        emb = get_embedder(variant)
        langs = ["en"] if variant == "en" else ["en", "hi"]
        for lang in langs:
            q = emb.encode_text(LYING_QUERIES[lang])
            key = f"{variant}_model_{lang}_query"
            out[key] = {
                "day_frame_only": retrieval_metrics(index_scores(frames, q, emb, False), rel),
                "day_product": retrieval_metrics(index_scores(frames, q, emb, True, det), rel),
                "night_frame_only": retrieval_metrics(index_scores(night, q, emb, False), rel),
                "night_product_enhanced": retrieval_metrics(index_scores(night, q, emb, True, det), rel),
            }
            print(key, json.dumps(out[key]), file=sys.stderr)
    cv2.imwrite(str(Path(ARGS.out) / "search_night_example.jpg"), np.hstack([frames[len(frames) // 3], night[len(frames) // 3], enhance(night[len(frames) // 3])]))
    return out


# ---------------------------------------------------------------------------
# Part B: synthetic attribute search


CONDITIONS = {
    "day": dict(time_of_day="day", weather="clear", ir="off"),
    "dusk": dict(time_of_day="dusk", weather="clear", ir="off"),
    "night_colour": dict(time_of_day="night", weather="clear", ir="off"),
    "night_ir": dict(time_of_day="night", weather="clear", ir="on"),
    "rain": dict(time_of_day="day", weather="rain", ir="off"),
    "fog": dict(time_of_day="day", weather="fog", ir="off"),
}


def part_b(clips_per_condition: int = 8) -> dict:
    from cctvgpt.params import RealismParams, SceneParams
    from cctvgpt.realism import RealismProcessor
    from cctvgpt.render.renderer import SimRenderer

    emb = get_embedder("en")
    colours = ["red", "blue", "black", "white", "green", "orange", "purple"]
    qv = {c: emb.encode_text(f"a person wearing a {c} shirt") for c in colours}
    out = {}
    for cname, cond in CONDITIONS.items():
        crops, truth = [], []
        for i in range(clips_per_condition):
            scene = ["corridor", "parking_lot", "warehouse"][i % 3]
            cam = {"corridor": (2.8, 22.0), "parking_lot": (4.0, 18.0), "warehouse": (5.0, 22.0)}[scene]
            weather = cond["weather"] if scene == "parking_lot" else "clear"
            if cond["weather"] != "clear" and scene != "parking_lot":
                scene, weather, cam = "parking_lot", cond["weather"], (4.0, 18.0)
            p = SceneParams(scene=scene, time_of_day=cond["time_of_day"], weather=weather, crowd_density="medium", camera_height_m=cam[0], camera_tilt_deg=cam[1],
                            width=640, height=360, fps=2, duration_s=4.0, seed=500 + i)
            r = SimRenderer(p)
            rp = RealismProcessor(RealismParams(ir_night_mode=cond["ir"], noise=0.5, compression="medium"), p, r.depth)
            for f, t in r.frames():
                img = rp.process(f, t.index)
                for o in t.objects:
                    if o["class"] != "person" or o["visibility"] < 0.6:
                        continue
                    x, y, w, h = rp.distort_bbox(o["bbox"])
                    if h < 40:
                        continue
                    px, py = int(w * 0.1), int(h * 0.1)
                    crop = img[max(0, y - py): y + h + py, max(0, x - px): x + w + px]
                    if crop.size == 0:
                        continue
                    q = assess(crop)
                    crops.append(enhance(crop, q) if needs_enhancement(q) else crop)
                    truth.append(o["upper_color"])
        V = emb.encode_images(crops)
        truth = np.array(truth)
        per = {}
        for c in colours:
            relevant = truth == c
            if relevant.sum() == 0:
                continue
            per[c] = retrieval_metrics(V @ qv[c], relevant)
        out[cname] = {"crops": len(crops), "mean_ap": round(float(np.mean([m["ap"] for m in per.values()])), 3),
                      "mean_random_ap": round(float(np.mean([m["random_ap"] for m in per.values()])), 3),
                      "mean_p@10": round(float(np.mean([m["p@10"] for m in per.values()])), 3), "per_colour_ap": {c: m["ap"] for c, m in per.items()}}
        print(cname, json.dumps(out[cname]), file=sys.stderr)
    return out


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--urfd", default="external_data/urfd")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--variants", default="en,multilingual")
    ARGS = ap.parse_args()
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    res = {}
    if not ARGS.skip_real:
        res["real_urfd_lying_search"] = part_a(Path(ARGS.urfd))
    if not ARGS.skip_synthetic:
        res["synthetic_colour_search"] = part_b()
    out_path = Path(ARGS.out) / "search_eval.json"
    if out_path.exists():  # merge with earlier partial runs
        prev = json.loads(out_path.read_text())
        for k, v in res.items():
            if isinstance(v, dict) and isinstance(prev.get(k), dict):
                prev[k].update(v)
            else:
                prev[k] = v
        res = prev
    out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


ARGS = None

if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    _ = defaultdict
    main()
