"""Analytics acceptance testing: score a customer's detector against ground truth.

A customer runs their video-analytics system on a generated scenario suite
(e.g. a batch sweep over night/rain/fog/occlusion/distance) and submits its
outputs. Because every clip has exact ground truth, the platform can return a
condition-by-condition scorecard. That is the "pre-deployment testing of
analytics" use the PRD permits for public-safety buyers, and the kind of
evidence a Safe City tender's acceptance test needs.

Prediction format (per clip, JSON):

    {"frames": [{"frame": 0, "detections": [{"bbox": [x, y, w, h], "score": 0.9, "class": "person"}]}],
     "events": [{"type": "fall", "start_frame": 41}]}

Detection metrics: greedy IoU matching per frame and class, precision/recall,
and AP (all-point interpolation). Event metrics: an alarm event is detected if
a prediction of the same type starts between (gt_start - tolerance) and the
event end. Latency is measured from the true start. A predicted event that
matches nothing is a false alarm; hard-negative clips (walk, crouch, lie_down)
exist to provoke these.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

IOU_THRESHOLD = 0.5
EVENT_TOLERANCE_S = 1.0


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores: list, matched: list, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, dtype=float)[order]
    fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class ClipScore:
    clip_id: str
    conditions: dict
    det: dict = field(default_factory=dict)  # per class: tp, fp, fn, scores, matched, by_bucket
    events: list = field(default_factory=list)
    false_alarms: list = field(default_factory=list)


def _height_bucket(h: float) -> str:
    return "<32px" if h < 32 else "32-64px" if h < 64 else "64-128px" if h < 128 else ">=128px"


def score_clip(ann: dict, pred: dict, min_visibility: float = 0.3, score_threshold: float = 0.5) -> ClipScore:
    """Score one clip. GT objects below min_visibility are 'ignore' (neither TP nor FN, matching them is not an FP)."""
    params = ann.get("parameters", {}).get("scene", {})
    conditions = {k: params.get(k) for k in ("scene", "time_of_day", "weather", "occlusion", "event", "crowd_density")}
    conditions["ir"] = (ann.get("parameters", {}).get("realism", {}).get("ir_night_mode") == "on") or (
        ann.get("parameters", {}).get("realism", {}).get("ir_night_mode") == "auto" and params.get("time_of_day") == "night")
    cs = ClipScore(ann.get("clip_id", "?"), conditions)
    fps = ann["video"]["fps"]
    pred_frames = {f["frame"]: f.get("detections", []) for f in pred.get("frames", [])}
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "matched": [], "n_gt": 0,
                                 "by_bucket": defaultdict(lambda: [0, 0]), "by_state": defaultdict(lambda: [0, 0])})
    for fr in ann["frames"]:
        dets = pred_frames.get(fr["frame"], [])
        for cls in {o["class"] for o in fr["objects"]} | {d.get("class", "person") for d in dets}:
            gts = [o for o in fr["objects"] if o["class"] == cls]
            care = [g for g in gts if g["visibility"] >= min_visibility]
            ignore = [g for g in gts if g["visibility"] < min_visibility]
            ds = sorted([d for d in dets if d.get("class", "person") == cls], key=lambda d: -d.get("score", 1.0))
            s = stats[cls]
            s["n_gt"] += len(care)
            used = set()
            for d in ds:
                best, bj = 0.0, -1
                for j, g in enumerate(care):
                    if j in used:
                        continue
                    v = iou(d["bbox"], g["bbox"])
                    if v > best:
                        best, bj = v, j
                ok = best >= IOU_THRESHOLD
                if ok:
                    used.add(bj)
                elif any(iou(d["bbox"], g["bbox"]) >= IOU_THRESHOLD for g in ignore):
                    continue  # matched an ignore-region object: not counted
                s["scores"].append(d.get("score", 1.0))
                s["matched"].append(1 if ok else 0)
                if d.get("score", 1.0) >= score_threshold:
                    s["tp" if ok else "fp"] += 1
            for j, g in enumerate(care):
                hit = j in used and any(
                    iou(d["bbox"], g["bbox"]) >= IOU_THRESHOLD and d.get("score", 1.0) >= score_threshold for d in ds)
                if not hit:
                    s["fn"] += 1
                b = s["by_bucket"][_height_bucket(g["bbox"][3])]
                b[0] += int(hit)
                b[1] += 1
                st = s["by_state"][g.get("state", "?")]
                st[0] += int(hit)
                st[1] += 1
    cs.det = {k: v for k, v in stats.items()}

    preds = list(pred.get("events", []))
    used = set()
    for ev in ann.get("events", []):
        if not ev.get("alarm", True) or not ev.get("occurred"):
            continue
        tol = int(round(EVENT_TOLERANCE_S * fps))
        match = None
        for i, p in enumerate(preds):
            if i in used or p.get("type") != ev["type"]:
                continue
            if ev["start_frame"] - tol <= p["start_frame"] <= ev["end_frame"]:
                match = i
                break
        if match is not None:
            used.add(match)
        cs.events.append({"type": ev["type"], "detected": match is not None,
                          "latency_s": round((preds[match]["start_frame"] - ev["start_frame"]) / fps, 3) if match is not None else None})
    cs.false_alarms = [p for i, p in enumerate(preds) if i not in used]
    return cs


def _summarise(clips: list) -> dict:
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "scores": [], "matched": [], "n_gt": 0,
                               "by_bucket": defaultdict(lambda: [0, 0]), "by_state": defaultdict(lambda: [0, 0])})
    for c in clips:
        for cls, s in c.det.items():
            a = agg[cls]
            for k in ("tp", "fp", "fn", "n_gt"):
                a[k] += s[k]
            a["scores"] += s["scores"]
            a["matched"] += s["matched"]
            for k, (h, n) in s["by_bucket"].items():
                a["by_bucket"][k][0] += h
                a["by_bucket"][k][1] += n
            for k, (h, n) in s["by_state"].items():
                a["by_state"][k][0] += h
                a["by_state"][k][1] += n
    det = {}
    for cls, a in agg.items():
        p = a["tp"] / max(a["tp"] + a["fp"], 1)
        r = a["tp"] / max(a["tp"] + a["fn"], 1)
        ap = average_precision(a["scores"], a["matched"], a["n_gt"])
        det[cls] = {
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3),
            "ap50": round(ap, 3) if ap is not None else None, "gt_objects": a["n_gt"],
            "recall_by_size": {k: round(h / n, 3) for k, (h, n) in sorted(a["by_bucket"].items()) if n},
            "recall_by_state": {k: round(h / n, 3) for k, (h, n) in sorted(a["by_state"].items()) if n},
        }
    evs = [e for c in clips for e in c.events]
    by_type = defaultdict(list)
    for e in evs:
        by_type[e["type"]].append(e)
    events = {t: {"detected": sum(e["detected"] for e in v), "total": len(v),
                  "rate": round(sum(e["detected"] for e in v) / len(v), 3),
                  "median_latency_s": (round(float(np.median([e["latency_s"] for e in v if e["detected"]])), 3) if any(e["detected"] for e in v) else None)}
              for t, v in by_type.items()}
    fa = sum(len(c.false_alarms) for c in clips)
    return {"clips": len(clips), "detection": det, "events": events, "false_alarms": fa,
            "false_alarms_per_clip": round(fa / max(len(clips), 1), 3)}


def scorecard(annotations: dict, predictions: dict, group_by=("time_of_day", "weather", "occlusion", "scene", "event")) -> dict:
    """annotations / predictions: {clip_id: dict}. Returns overall and per-condition results."""
    missing = sorted(set(annotations) - set(predictions))
    clips = [score_clip(annotations[c], predictions.get(c, {})) for c in annotations]
    out = {"overall": _summarise(clips), "missing_predictions": missing, "by_condition": {}}
    for key in group_by:
        groups = defaultdict(list)
        for c in clips:
            groups[str(c.conditions.get(key))].append(c)
        if len(groups) > 1:
            out["by_condition"][key] = {k: _summarise(v) for k, v in sorted(groups.items())}
    out["method"] = {"iou_threshold": IOU_THRESHOLD, "event_tolerance_s": EVENT_TOLERANCE_S,
                     "note": "Ground truth is exact (rendered). Objects under 30% visible are ignore regions."}
    return out


def scorecard_html(card: dict, title: str = "Analytics acceptance test") -> str:
    """Self-contained HTML rendering of a scorecard."""
    def table(rows, head):
        th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
        trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"

    parts = [f"<h1>{html.escape(title)}</h1>", "<p class=note>All test footage is SYNTHETIC with exact ground truth. Results describe performance on this scenario suite only.</p>"]
    o = card["overall"]
    parts.append("<h2>Overall</h2>")
    parts.append(table([[cls, d["precision"], d["recall"], d["f1"], d["ap50"], d["gt_objects"]] for cls, d in o["detection"].items()],
                       ["class", "precision", "recall", "F1", "AP50", "GT objects"]))
    if o["events"]:
        parts.append(table([[t, f"{e['detected']}/{e['total']}", e["rate"], e["median_latency_s"]] for t, e in o["events"].items()],
                           ["event", "detected", "rate", "median latency (s)"]))
    parts.append(f"<p>False alarms: {o['false_alarms']} ({o['false_alarms_per_clip']} per clip)</p>")
    for key, groups in card["by_condition"].items():
        parts.append(f"<h2>By {html.escape(key)}</h2>")
        rows = []
        for val, g in groups.items():
            p = g["detection"].get("person", {})
            ev = "; ".join(f"{t} {e['detected']}/{e['total']}" for t, e in g["events"].items())
            rows.append([val, g["clips"], p.get("recall"), p.get("precision"), p.get("ap50"), ev or "-", g["false_alarms"]])
        parts.append(table(rows, [key, "clips", "person recall", "person precision", "person AP50", "events detected", "false alarms"]))
    style = ("body{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#1d2330;background:#fff}table{border-collapse:collapse;margin:8px 0 16px}"
             "td,th{border-bottom:1px solid #dde1e8;padding:5px 10px;text-align:left}th{color:#5d6678}.note{color:#a15c00}"
             "@media (prefers-color-scheme:dark){body{background:#14171c;color:#e7eaf0}td,th{border-color:#2e3440}}")
    return f"<!doctype html><html><head><meta charset=utf-8><title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(parts)}</body></html>"


def load_json(path) -> dict:
    with open(path) as f:
        return json.load(f)
