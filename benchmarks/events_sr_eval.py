#!/usr/bin/env python3
"""Violence and accident detection, and super-resolution, on real data.

Datasets (downloaded by hand into external_data/; internal evaluation only
unless the licence says otherwise):
  - Real Life Violence Situations (RLVS): 500 violent + 500 non-violent clips
    (34data/real-life-violence zips 0001 and 0003 on Hugging Face; licence not
    stated) -> external_data/violence/clips/{Violence,NonViolence}
  - Surveillance Camera Fight Dataset (Akti et al. 2019, MIT licence): 150
    fight + 150 no-fight CCTV clips -> external_data/survfight/{fight,noFight}
  - CCTV traffic accident detection (justjuu/traffic-accident-cctv-object-
    detection, CC0): accident vs non-accident vehicle boxes -> external_data/accident
  - Indian number plates (zenitsu09/indian-number-plate) and LFW pairs, as in
    plates_faces_eval.py.

    python benchmarks/events_sr_eval.py --out benchmarks/results [--only violence,accident,superres]

Findings this script reproduces (results/events_sr_eval.json):
  - A classifier trained on RLVS reaches AUROC 0.999 on RLVS but only 0.80 on
    real CCTV fights; zero-shot CLIP prompts reach 0.87 there -> the product
    uses the zero-shot score.
  - The accident crop classifier reaches test AUROC 0.93 after removing test
    images that are near-copies (flips) of training images (67 of 325).
  - Super-resolution hurts plate reading and face matching -> display only.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1] / "external_data"


def clip_frames(path: str, k: int = 8) -> list:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "fps=4,scale=320:180", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=False).stdout
    allf = np.frombuffer(raw, np.uint8).reshape(-1, 180, 320, 3) if raw else np.zeros((1, 180, 320, 3), np.uint8)
    return [allf[j] for j in np.linspace(0, max(len(allf) - 1, 0), k).astype(int)]


def clip_set(pos_glob: str, neg_glob: str, emb) -> tuple:
    files = sorted(glob.glob(pos_glob)) + sorted(glob.glob(neg_glob))
    y = np.array([1] * len(glob.glob(pos_glob)) + [0] * len(glob.glob(neg_glob)))
    return np.array([emb.encode_images(clip_frames(f)) for f in files]), y


def violence(emb) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from cctvgpt.search.events import VIOLENCE_PROMPTS, VIOLENCE_THRESHOLD

    Ea, ya = clip_set(str(ROOT / "violence/clips/Violence/*"), str(ROOT / "violence/clips/NonViolence/*"), emb)
    Eb, yb = clip_set(str(ROOT / "survfight/fight/*"), str(ROOT / "survfight/noFight/*"), emb)
    pos, neg = emb.encode_text(VIOLENCE_PROMPTS[0]), emb.encode_text(VIOLENCE_PROMPTS[1])
    zs = lambda E: (E @ pos - E @ neg).mean(1)
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = zs(Eb)
    p = s > VIOLENCE_THRESHOLD
    return {
        "rlvs_clips": len(ya), "cctv_clips": len(yb),
        "trained_on_rlvs_auroc_rlvs_5fold": round(float(roc_auc_score(ya, cross_val_predict(clf(), Ea.mean(1), ya, cv=cv, method="decision_function"))), 3),
        "trained_on_rlvs_auroc_cctv": round(float(roc_auc_score(yb, clf().fit(Ea.mean(1), ya).decision_function(Eb.mean(1)))), 3),
        "zero_shot_auroc_rlvs": round(float(roc_auc_score(ya, zs(Ea))), 3),
        "zero_shot_auroc_cctv": round(float(roc_auc_score(yb, s)), 3),
        "zero_shot_cctv_at_threshold": {"threshold": VIOLENCE_THRESHOLD, "precision": round(float(yb[p].mean()), 3), "recall": round(float(p[yb == 1].mean()), 3)},
    }


def accident(emb) -> dict:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    def load(name):
        df = pd.read_parquet(ROOT / "accident/data" / name)
        crops, ys, gid, thumbs = [], [], [], []
        for i, r in df.iterrows():
            img = cv2.imdecode(np.frombuffer(r.image["bytes"], np.uint8), 1)
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
            thumbs.append((g - g.mean()) / (g.std() + 1e-6))
            H, W = img.shape[:2]
            for (x, y, w, h), c in zip(r.objects["bbox"], r.objects["category"]):
                cr = img[int(max(0, y - .15 * h)):int(min(H, y + 1.15 * h)), int(max(0, x - .15 * w)):int(min(W, x + 1.15 * w))]
                if cr.size and min(cr.shape[:2]) >= 8:
                    crops.append(cr), ys.append(int(c == 0)), gid.append(i)
        return emb.encode_images(crops), np.array(ys), np.array(gid), np.array(thumbs)

    Xtr, ytr, _, Ttr = load("train-00000-of-00002.parquet")
    Xte, yte, gte, Tte = load("test-00000-of-00001.parquet")
    flip = Ttr.reshape(-1, 16, 16)[:, :, ::-1].reshape(len(Ttr), -1)
    near = (np.maximum(Tte @ Ttr.T, Tte @ flip.T) / 256).max(1) > 0.95  # near-copies of training images
    keep = ~near[gte]
    clf = LogisticRegression(C=10, max_iter=3000).fit(Xtr, ytr)
    p = clf.decision_function(Xte[keep])
    pr, rc, _, _ = precision_recall_fscore_support(yte[keep], p > 0, average="binary")
    return {"test_images_removed_as_near_copies": int(near.sum()), "test_crops": int(keep.sum()),
            "auroc": round(float(roc_auc_score(yte[keep], p)), 3), "accuracy": round(float(accuracy_score(yte[keep], p > 0)), 3),
            "precision": round(float(pr), 3), "recall": round(float(rc), 3)}


def superres() -> dict:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from sklearn.datasets import fetch_lfw_pairs

    from cctvgpt.search import faces as F
    from cctvgpt.search import superres as S
    from cctvgpt.search.plates import normalise_plate, read_plates, recognise_plate_crop

    out = {"plates": {}, "faces": {}}
    df = pd.read_parquet(hf_hub_download("zenitsu09/indian-number-plate", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    df = df.drop_duplicates("plate_text").reset_index(drop=True).sample(300, random_state=0)

    def read(img):
        canvas = cv2.copyMakeBorder(img, 40, 40, 60, 60, cv2.BORDER_CONSTANT, value=(90, 90, 90))
        rp = read_plates(cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        return rp[0].text if rp else normalise_plate(recognise_plate_crop(img))[0]

    for f in (0.5, 0.33):
        hits = {"bicubic": 0, "superres": 0}
        for _, r in df.iterrows():
            img = cv2.imdecode(np.frombuffer(r.image["bytes"], np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            gt = normalise_plate(r.plate_text)[0]
            hits["bicubic"] += read(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)) == gt
            hits["superres"] += read(cv2.resize(S.upscale(small), (w, h), interpolation=cv2.INTER_AREA)) == gt
        out["plates"][f"scale_{f}_exact"] = {k: round(v / len(df), 3) for k, v in hits.items()}
    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    idx = np.random.default_rng(1).permutation(len(data.target))[:300]
    for name in ("bicubic", "superres"):
        sims = []
        for a, b in data.pairs[idx]:
            vs = []
            for im in (a, b):
                bgr = cv2.cvtColor(np.clip(im if im.max() > 1.5 else im * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                small = cv2.resize(bgr, (40, 40), interpolation=cv2.INTER_AREA)
                up = cv2.resize(small, (250, 250), interpolation=cv2.INTER_CUBIC) if name == "bicubic" else cv2.resize(S.upscale(small), (250, 250), interpolation=cv2.INTER_AREA)
                fc = F.largest_face(up)
                vs.append(None if fc is None else F.embed_face(up, fc))
            sims.append(-1.0 if vs[0] is None or vs[1] is None else float(vs[0] @ vs[1]))
        p = np.array(sims) >= F.MATCH_THRESHOLD
        y = data.target[idx]
        out["faces"][f"face_20px_{name}"] = {"true_match": round(float(p[y == 1].mean()), 3), "false_match": round(float(p[y == 0].mean()), 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--only", default="violence,accident,superres")
    a = ap.parse_args()
    from cctvgpt.search.models import get_embedder

    emb = get_embedder("en")
    res = {}
    for part in a.only.split(","):
        res[part] = violence(emb) if part == "violence" else accident(emb) if part == "accident" else superres()
        print(part, json.dumps(res[part]), file=sys.stderr, flush=True)
    path = Path(a.out) / "events_sr_eval.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(res)
    path.write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
