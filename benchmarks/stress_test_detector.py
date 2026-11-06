#!/usr/bin/env python3
"""Analytics stress test: score an off-the-shelf person detector across conditions.

Generates a controlled scenario suite (scene x time of day x weather x
occlusion, with fall / lie-down / crouch / crowd events) using the product's
renderer and realism stage, runs torchvision's COCO-pretrained Faster R-CNN
(MobileNetV3) on every frame, and scores it with cctvgpt.evaluation.

It serves two purposes:
  1. A demonstration of the acceptance-testing product: a per-condition
     scorecard for any detector.
  2. A realism probe: a detector trained only on real photos (COCO) either
     recognises our synthetic people or it does not.

    python benchmarks/stress_test_detector.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctvgpt.evaluation import scorecard, scorecard_html  # noqa: E402


def suite() -> list:
    cams = {"corridor": (2.8, 22.0), "parking_lot": (4.0, 18.0), "warehouse": (5.0, 22.0)}
    jobs = []
    seed = 0
    for scene, tod, occ, ev in itertools.product(["corridor", "parking_lot", "warehouse"], ["day", "night"], ["none", "heavy"], ["fall", "crowd_surge", "lie_down"]):
        weathers = ["clear", "rain", "fog"] if scene == "parking_lot" else ["clear"]
        for w in weathers:
            seed += 1
            h, t = cams[scene]
            jobs.append(dict(scene=scene, time_of_day=tod, weather=w, occlusion=occ, event=ev, camera_height_m=h, camera_tilt_deg=t,
                             crowd_density="low", event_distance_m=7.0, width=640, height=360, fps=4, duration_s=6.0, seed=seed))
    return jobs


def run_clip(cfg: dict):
    from fall_sim2real import detect_people

    from cctvgpt.params import RealismParams, SceneParams
    from cctvgpt.realism import RealismProcessor
    from cctvgpt.render.renderer import SimRenderer
    from cctvgpt.safeguards.label import apply_label

    p = SceneParams(**cfg)
    rl = RealismParams()
    r = SimRenderer(p)
    rp = RealismProcessor(rl, p, r.depth)
    frames, gt_frames = [], []
    for f, t in r.frames():
        frames.append(apply_label(rp.process(f, t.index)))
        objs = []
        for o in t.objects:
            o = dict(o)
            o["bbox"] = rp.distort_bbox(o["bbox"])
            if o["bbox"][2] > 0 and o["bbox"][3] > 0:
                objs.append(o)
        gt_frames.append({"frame": t.index, "t": t.t, "objects": objs})
    dets = detect_people(frames, score=0.3)
    pred = {"frames": [{"frame": i, "detections": [{"bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s, "class": "person"} for b, s in d]}
                       for i, d in enumerate(dets)]}
    clip_id = f"{cfg['scene']}-{cfg['time_of_day']}-{cfg['weather']}-{cfg['occlusion']}-{cfg['event']}-{cfg['seed']}"
    ann = {"clip_id": clip_id, "parameters": {"scene": cfg, "realism": rl.__dict__}, "video": {"fps": p.fps},
           "frames": gt_frames, "events": r.event_annotations()}
    return clip_id, ann, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = suite()
    if a.limit:
        jobs = jobs[: a.limit]
    with ProcessPoolExecutor(4) as ex:
        results = list(ex.map(run_clip, jobs))
    anns = {c: ann for c, ann, _ in results}
    preds = {c: pr for c, _, pr in results}
    # Events are not predicted by a plain detector; score detection only.
    for p in preds.values():
        p["events"] = []
    card = scorecard(anns, preds)
    card["detector"] = "torchvision fasterrcnn_mobilenet_v3_large_fpn (COCO), score>=0.5 for P/R, all scores for AP"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stress_test_scorecard.json").write_text(json.dumps(card, indent=1))
    (out / "stress_test_scorecard.html").write_text(scorecard_html(card, "Stress test: COCO person detector on CCTVGPT scenario suite"))
    o = card["overall"]["detection"].get("person", {})
    print(json.dumps({"clips": len(jobs), "person": o, "by_condition": {k: {v: g["detection"].get("person", {}).get("recall") for v, g in grp.items()}
                                                                        for k, grp in card["by_condition"].items()}}, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
