#!/usr/bin/env python3
"""Sim-to-real benchmark: does CCTVGPT synthetic data help detect real falls?

Real data: UR Fall Detection Dataset (URFD), camera 0 RGB: 30 falls and 40
activities of daily living (ADL: walking, sitting, crouching, picking things
up, lying down on a bed). See benchmarks/download_urfd.py for its licence.

Pipeline (identical for both domains, so the only difference is the data):
  frames -> MOG2 background subtraction -> largest foreground blob -> person
  box per frame (10 fps) -> scale-invariant motion features (height drop,
  aspect-ratio change, descent speed and duration, end state) -> classifier.

Experiments:
  A  real-only, repeated stratified 5-fold cross-validation (in-domain upper bound)
  B  synthetic-only training, tested on all 70 real clips (zero real labels)
  C  as B without the hard-negative events (ablation: are confusers needed?)
  D  few real clips per class, with vs without synthetic data (the rare-event case)
  E  hand-tuned rule baseline

    python benchmarks/fall_sim2real.py --urfd external_data/urfd --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FPS = 10
FEATURES = ["max_drop_1s", "max_ar_rise_1s", "end_ar", "end_h", "min_h", "max_descent_05s", "fastest_drop_s", "frac_low", "valid_frac"]

# ---------------------------------------------------------------------------
# Shared perception front-end


def boxes_from_frames(frames, ignore_mask=None) -> list:
    """MOG2 foreground -> largest blob bounding box per frame (or None)."""
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=True)
    H, W = frames[0].shape[:2]
    min_area = 0.003 * H * W
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = []
    for i, f in enumerate(frames):
        g = cv2.GaussianBlur(f, (5, 5), 0)
        m = mog.apply(g, learningRate=0.05 if i < 5 else 0.002)
        m = (m > 200).astype(np.uint8) * 255  # drop shadows (127)
        if ignore_mask is not None:
            m[ignore_mask] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        if i < 3 or not cnts:
            out.append(None)
            continue
        # Merge blobs near the largest one (a person often splits into parts).
        cnts.sort(key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(cnts[0])
        for c in cnts[1:]:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx < x + w + 10 and bx + bw > x - 10 and by < y + h + 10 and by + bh > y - 10:
                x0, y0 = min(x, bx), min(y, by)
                x, y, w, h = x0, y0, max(x + w, bx + bw) - x0, max(y + h, by + bh) - y0
        out.append((x, y, w, h))
    return out


_DET = None


def _detector():
    global _DET
    if _DET is None:
        import torch
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn

        torch.set_num_threads(1)
        _DET = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT, min_size=480, max_size=640).eval()
    return _DET


def detect_people(frames, score=0.5) -> list:
    """Off-the-shelf COCO person detector (torchvision Faster R-CNN MobileNetV3). Returns all person boxes per frame."""
    import torch

    m = _detector()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), 8):
            batch = [torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255 for f in frames[i : i + 8]]
            for r in m(batch):
                keep = (r["labels"] == 1) & (r["scores"] >= score)
                out.append([(tuple(b), float(s_)) for b, s_ in zip(r["boxes"][keep].tolist(), r["scores"][keep].tolist())])
    return out


def track_main_person(dets) -> list:
    """Follow one person: highest score at first, then the box that best overlaps the previous one."""

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    prev, out = None, []
    for d in dets:
        if not d:
            out.append(None)
            continue
        if prev is None:
            box = max(d, key=lambda x: x[1])[0]
        else:
            box = max(d, key=lambda x: iou(prev, x[0]) + 0.1 * x[1])[0]
        prev = box
        x0, y0, x1, y1 = box
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def features_from_boxes(boxes) -> dict:
    """Scale- and perspective-robust motion features from a per-frame box track.

    Heights are expressed relative to the running maximum height so far (the
    person's most recent full standing height), which stays meaningful when
    people walk towards or away from the camera.
    """
    n = len(boxes)
    valid = [b for b in boxes if b is not None]
    feats = {k: 0.0 for k in FEATURES}
    feats["valid_frac"] = len(valid) / max(n, 1)
    feats["fastest_drop_s"] = 5.0
    feats["end_h"] = 1.0
    feats["min_h"] = 1.0
    if len(valid) < 8:
        return feats
    first = next(i for i, b in enumerate(boxes) if b is not None)
    filled, last = [], None
    for b in boxes[first:]:
        last = b if b is not None else last
        filled.append(last)
    arr = np.array(filled, dtype=float)
    w, h = arr[:, 2], arr[:, 3]
    yc = arr[:, 1] + h / 2
    ref = np.maximum.accumulate(h)
    rel = h / ref
    ar = w / h
    ycn = yc / ref
    L = FPS

    def max_delta(series, lag, sign=1):
        if len(series) <= lag:
            return 0.0
        return float((sign * (series[lag:] - series[:-lag])).max())

    feats["max_drop_1s"] = max_delta(rel, L, -1)
    feats["max_ar_rise_1s"] = max_delta(ar, L, 1)
    tail = slice(max(0, len(rel) - L), len(rel))
    feats["end_ar"] = float(np.median(ar[tail]))
    feats["end_h"] = float(np.median(rel[tail]))
    feats["min_h"] = float(rel.min())
    feats["max_descent_05s"] = max_delta(ycn, L // 2, 1)
    hi_idx = np.where(rel > 0.8)[0]
    lo_idx = np.where(rel < 0.5)[0]
    best = 5.0
    for t0 in hi_idx:
        later = lo_idx[lo_idx > t0]
        if later.size:
            best = min(best, (later[0] - t0) / FPS)
    feats["fastest_drop_s"] = best
    feats["frac_low"] = float(np.mean(rel < 0.5))
    return feats


# ---------------------------------------------------------------------------
# Real data


def _urfd_one(args):
    path, frontend = args if len(args) == 2 else (args[0], "mog2")
    cap = cv2.VideoCapture(str(path))
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % 3 == 0:  # 30 fps -> 10 fps
            frames.append(f[:, f.shape[1] // 2 :])  # right half = RGB, left half = depth
        i += 1
    cap.release()
    if frontend == "detector":
        return features_from_boxes(track_main_person(detect_people(frames)))
    return features_from_boxes(boxes_from_frames(frames))


def load_urfd(root: Path, frontend: str = "detector") -> tuple[np.ndarray, np.ndarray, list]:
    names = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]
    paths = [(root / f"{n}-cam0.mp4", frontend) for n in names]
    with ProcessPoolExecutor(4) as ex:
        feats = list(ex.map(_urfd_one, paths))
    X = np.array([[f[k] for k in FEATURES] for f in feats])
    y = np.array([1 if n.startswith("fall") else 0 for n in names])
    return X, y, names


# ---------------------------------------------------------------------------
# Synthetic data (generated with the product's renderer + realism + label)


def _synth_one(args):
    event, seed, frontend = args if len(args) == 3 else (*args, "gt")
    from cctvgpt.params import RealismParams, SceneParams
    from cctvgpt.realism import RealismProcessor
    from cctvgpt.render.renderer import SimRenderer
    from cctvgpt.safeguards.label import apply_label

    rng = np.random.default_rng([seed, 4242])
    scene = str(rng.choice(["corridor", "warehouse", "parking_lot"]))
    hmax = 2.9 if scene == "corridor" else 3.5
    # Indoor-camera viewpoints: events close to the camera, as in typical room/corridor CCTV.
    p = SceneParams(event_distance_m=float(rng.uniform(2.5, 8.0)),
        scene=scene, event=event, crowd_density="empty", occlusion=str(rng.choice(["none", "none", "light"])),
        time_of_day=str(rng.choice(["day", "day", "dusk", "night"])),
        camera_height_m=float(rng.uniform(2.0, hmax)), camera_tilt_deg=float(rng.uniform(10, 35)),
        camera_pan_deg=float(rng.uniform(-20, 20)), fov_deg=float(rng.uniform(55, 80)),
        width=320, height=240, fps=FPS, duration_s=float(rng.uniform(6.0, 9.0)), seed=int(seed),
    )
    rl = RealismParams(compression=str(rng.choice(["low", "medium", "high"])), noise=float(rng.uniform(0.1, 0.6)),
                       lens_distortion=float(rng.uniform(0.0, 0.25)), timestamp_overlay=True)
    r = SimRenderer(p)
    rp = RealismProcessor(rl, p, r.depth)
    actor_ids = set(r.cast.events[0].track_ids)
    frames, gt = [], []
    for f, t in r.frames():
        frames.append(apply_label(rp.process(f, t.index)))
        box = next((rp.distort_bbox(o["bbox"]) for o in t.objects if o["track_id"] in actor_ids and o["visibility"] > 0.3), None)
        gt.append(box)
    if frontend == "detector":
        dets = detect_people(frames)
        boxes = track_main_person(dets)
        hits = sum(1 for g, d in zip(gt, dets) if g is not None and d)
        feats = features_from_boxes(boxes)
        feats["_detector_recall"] = hits / max(1, sum(g is not None for g in gt))
        return feats
    if frontend == "mog2":
        return features_from_boxes(boxes_from_frames(frames))
    # Exact ground truth, jittered like a detector (+-4% of box size per frame).
    jit = np.random.default_rng([seed, 7])
    noisy = []
    for g in gt:
        if g is None or g[2] < 2 or g[3] < 2:
            noisy.append(None)
            continue
        x, y, w, h = g
        e = jit.normal(0, 0.04, 4)
        noisy.append((x + e[0] * w, y + e[1] * h, max(2.0, w * (1 + e[2])), max(2.0, h * (1 + e[3]))))
    return features_from_boxes(noisy)


def make_synthetic(n_fall: int, n_neg_each: int, seed0: int = 10_000, frontend: str = "gt"):
    jobs = [("fall", seed0 + i, frontend) for i in range(n_fall)]
    for j, ev in enumerate(["walk", "crouch", "lie_down"]):
        jobs += [(ev, seed0 + 100_000 * (j + 1) + i, frontend) for i in range(n_neg_each)]
    with ProcessPoolExecutor(4) as ex:
        feats = list(ex.map(_synth_one, jobs, chunksize=2))
    if frontend == "detector":
        make_synthetic.detector_recall = {ev: round(float(np.mean([f["_detector_recall"] for f, (e, *_) in zip(feats, jobs) if e == ev])), 3) for ev in ("fall", "walk", "crouch", "lie_down")}
    X = np.array([[f[k] for k in FEATURES] for f in feats])
    y = np.array([1 if ev == "fall" else 0 for ev, *_ in jobs])
    events = np.array([ev for ev, *_ in jobs])
    return X, y, events


# ---------------------------------------------------------------------------
# Evaluation


def model(kind: str = "lr"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


def metrics(y, p, thr=0.5) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    pred = (p >= thr).astype(int)
    out = {
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    out["auroc"] = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
    return {k: (round(v, 3) if v is not None else None) for k, v in out.items()}


def fit_predict(kind, Xtr, ytr, Xte, w=None):
    m = model(kind)
    if w is not None:
        step = "logisticregression__sample_weight" if kind == "lr" else "sample_weight"
        m.fit(Xtr, ytr, **{step: w})
    else:
        m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run(a) -> dict:
    from sklearn.model_selection import RepeatedStratifiedKFold

    t0 = time.time()
    cache = Path(a.out) / f"urfd_features_{a.frontend}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        Xr, yr, names = z["X"], z["y"], list(z["names"])
    else:
        Xr, yr, names = load_urfd(Path(a.urfd), a.frontend)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=Xr, y=yr, names=np.array(names))
    print(f"real features: {Xr.shape} in {time.time() - t0:.0f}s", file=sys.stderr)
    t0 = time.time()
    Xs, ys, evs = make_synthetic(a.n_fall, a.n_neg, frontend="gt")
    print(f"synthetic features: {Xs.shape} in {time.time() - t0:.0f}s", file=sys.stderr)
    realism = None
    if a.realism_check:
        t0 = time.time()
        Xd, yd, evd = make_synthetic(a.realism_check, a.realism_check // 3, seed0=777_000, frontend="detector")
        realism = {"clips": len(yd), "detector_recall_on_synthetic_people": make_synthetic.detector_recall}
        print(f"realism check in {time.time() - t0:.0f}s", file=sys.stderr)
    res: dict = {"real": {"clips": len(yr), "falls": int(yr.sum()), "adl": int((1 - yr).sum())},
                 "synthetic": {"clips": len(ys), "falls": int(ys.sum()), "by_event": {e: int((evs == e).sum()) for e in sorted(set(evs))}},
                 "features": FEATURES, "real_frontend": a.frontend, "synthetic_labels": "renderer ground truth + 4% box jitter",
                 "realism_check": realism}

    for kind in ("lr", "rf"):
        # A. real-only CV
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        accs = []
        for tr, te in rskf.split(Xr, yr):
            accs.append(metrics(yr[te], fit_predict(kind, Xr[tr], yr[tr], Xr[te])))
        res[f"A_real_only_cv_{kind}"] = {k: round(float(np.mean([m[k] for m in accs])), 3) for k in accs[0]}
        # B. synthetic only -> real
        res[f"B_synthetic_only_{kind}"] = metrics(yr, fit_predict(kind, Xs, ys, Xr))
        # C. ablation without hard negatives
        keep = (evs == "fall") | (evs == "walk")
        res[f"C_synthetic_no_hard_negatives_{kind}"] = metrics(yr, fit_predict(kind, Xs[keep], ys[keep], Xr))

    # D. few-shot: k real clips per class, with and without synthetic
    rng = np.random.default_rng(0)
    fall_idx, adl_idx = np.where(yr == 1)[0], np.where(yr == 0)[0]
    for k in (1, 2, 3, 5, 10):
        rows = {"real_only": [], "real_plus_synthetic": []}
        for _ in range(a.draws):
            tr = np.concatenate([rng.choice(fall_idx, k, replace=False), rng.choice(adl_idx, k, replace=False)])
            te = np.setdiff1d(np.arange(len(yr)), tr)
            rows["real_only"].append(metrics(yr[te], fit_predict("lr", Xr[tr], yr[tr], Xr[te])))
            Xtr = np.vstack([Xs, Xr[tr]])
            ytr = np.concatenate([ys, yr[tr]])
            w = np.concatenate([np.ones(len(ys)), np.full(len(tr), len(ys) / (2.0 * len(tr)))])  # real and synthetic weigh equally
            rows["real_plus_synthetic"].append(metrics(yr[te], fit_predict("lr", Xtr, ytr, Xr[te], w)))
        res[f"D_{k}_real_per_class"] = {name: {m: round(float(np.mean([r[m] for r in v if r[m] is not None])), 3) for m in v[0]} for name, v in rows.items()}

    # E. rule baseline (classic aspect-ratio + height-drop heuristic)
    i = {f: FEATURES.index(f) for f in FEATURES}
    rule = ((Xr[:, i["end_ar"]] > 1.0) & (Xr[:, i["max_drop_1s"]] > 0.35)).astype(float)
    res["E_rule_baseline"] = metrics(yr, rule)

    # Diagnostic: per-feature means by domain and class.
    res["feature_means"] = {
        "real_fall": dict(zip(FEATURES, np.round(Xr[yr == 1].mean(0), 3).tolist())),
        "real_adl": dict(zip(FEATURES, np.round(Xr[yr == 0].mean(0), 3).tolist())),
        "synthetic_fall": dict(zip(FEATURES, np.round(Xs[ys == 1].mean(0), 3).tolist())),
        "synthetic_negative": dict(zip(FEATURES, np.round(Xs[ys == 0].mean(0), 3).tolist())),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urfd", default="external_data/urfd")
    ap.add_argument("--n-fall", type=int, default=150)
    ap.add_argument("--n-neg", type=int, default=50, help="clips per hard-negative event (walk, crouch, lie_down)")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--frontend", choices=["detector", "mog2"], default="detector", help="person boxes on real video")
    ap.add_argument("--realism-check", type=int, default=30, help="synthetic fall clips to run the real detector on (0 = skip)")
    a = ap.parse_args()
    res = run(a)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "fall_sim2real.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
