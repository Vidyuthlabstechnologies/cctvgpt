#!/usr/bin/env python3
"""Accuracy of plate reading, face matching and Hindi query translation on real data.

Plates: 965 unique real Indian plates (zenitsu09/indian-number-plate on Hugging
Face; no licence stated, so used for internal evaluation only). Metrics: exact
read, read within one character, and whether a fuzzy plate search (score >=
0.8) returns the right image at rank 1.

Faces: LFW verification pairs (sklearn fetch_lfw_pairs, test split, 1,000 pairs;
research dataset, internal evaluation only). YuNet detection + SFace matching
at the published threshold. Conditions model CCTV: low resolution (face about
24 px), night (dark, noisy, compressed), heavy JPEG.

Translation: Hindi versions of the URFD "person lying on the floor" query,
translated to English, then searched with the English model.

    python benchmarks/plates_faces_eval.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cctvgpt.search.enhance import assess, degrade_to_night, enhance, night_enhance, night_variants  # noqa: E402

RNG = np.random.default_rng(0)


def low_res(img, target_px=24):
    h, w = img.shape[:2]
    s = target_px / min(h, w) * 4  # whole image scaled so the face (about 1/4 of LFW frame) is ~target px
    small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def jpeg(img, q=15):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def tiny(img):
    """Face about 20 px across: the whole 250 px LFW frame shrunk to 40 px."""
    h, w = img.shape[:2]
    return cv2.resize(cv2.resize(img, (40, 40), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)


def as_product(img):
    """What the engine does: enhance dark/noisy images before detection."""
    q = assess(img)
    return enhance(img, q) if (q.dark or q.noisy) else img


CONDITIONS = {"clean": lambda x: x, "low_res_face_45px": low_res, "tiny_face_20px": tiny, "jpeg_q15": jpeg,
              "night": lambda x: degrade_to_night(x, RNG), "night_enhanced": lambda x: as_product(degrade_to_night(x, RNG)),
              # What the engine does now: denoise -> Zero-DCE, then classical, then Zero-DCE x2, until a face is found.
              "night_cascade": lambda x: degrade_to_night(x, RNG)}
CASCADE = {"night_cascade"}


def plates(limit: int | None) -> dict:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from cctvgpt.search.plates import normalise_plate, plate_match_score, read_plates, recognise_plate_crop

    df = pd.read_parquet(hf_hub_download("zenitsu09/indian-number-plate", "data/train-00000-of-00001.parquet", repo_type="dataset"))
    df = df.drop_duplicates("plate_text").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    out = {"plates": len(df)}
    plate_conditions = {"clean": lambda x: x, "half_res": lambda x: cv2.resize(cv2.resize(x, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), (x.shape[1], x.shape[0])),
                        "night": lambda x: degrade_to_night(x, RNG, darkness=0.35, noise=10),
                        "night_enhanced_classical": lambda x: as_product(degrade_to_night(x, RNG, darkness=0.35, noise=10)),
                        # What the engine does now: denoise -> Zero-DCE on dark plate crops.
                        "night_enhanced": lambda x: night_enhance(degrade_to_night(x, RNG, darkness=0.35, noise=10))}
    for cname, fn in plate_conditions.items():
        gts, reads = [], []
        for _, r in df.iterrows():
            img = fn(cv2.imdecode(np.frombuffer(r.image["bytes"], np.uint8), cv2.IMREAD_COLOR))
            gt = normalise_plate(r.plate_text)[0]
            canvas = cv2.copyMakeBorder(img, 40, 40, 60, 60, cv2.BORDER_CONSTANT, value=(90, 90, 90))
            rp = read_plates(cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            txt = rp[0].text if rp else normalise_plate(recognise_plate_crop(img))[0]
            gts.append(gt)
            reads.append(txt)
        exact = np.mean([a == b for a, b in zip(gts, reads)])
        within1 = np.mean([plate_match_score(a, b) >= 1 - 1 / len(a) for a, b in zip(gts, reads)])
        # Search: query each true plate against all reads; is its own image ranked first with score >= 0.8?
        hit = 0
        for i, gt in enumerate(gts):
            scores = np.array([plate_match_score(gt, rd) for rd in reads])
            best = int(np.argmax(scores))
            hit += int(scores[i] >= 0.8 and scores[i] == scores[best])
        out[cname] = {"exact": round(float(exact), 3), "within_1_char": round(float(within1), 3), "search_top1": round(hit / len(gts), 3)}
        print("plates", cname, out[cname], file=sys.stderr)
    return out


def faces(limit: int | None) -> dict:
    from sklearn.datasets import fetch_lfw_pairs

    from cctvgpt.search import faces as F

    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=(slice(0, 250), slice(0, 250)))
    pairs, y = data.pairs, data.target
    if limit:  # keep both classes: the split lists same-person pairs first
        idx = np.random.default_rng(1).permutation(len(y))[:limit]
        pairs, y = pairs[idx], y[idx]
    out = {"pairs": int(len(y)), "threshold": F.MATCH_THRESHOLD}
    for cname, fn in CONDITIONS.items():
        sims, missed = [], 0
        for a, b in pairs:
            vecs = []
            for im in (a, b):
                bgr = cv2.cvtColor(np.clip(im if im.max() > 1.5 else im * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                bgr = fn(bgr)
                face = None
                for v in (night_variants(bgr) if cname in CASCADE else [bgr]):
                    face = F.largest_face(v)
                    if face is not None:
                        bgr = v
                        break
                vecs.append(None if face is None else F.embed_face(bgr, face))
            if vecs[0] is None or vecs[1] is None:
                missed += 1
                sims.append(-1.0)
            else:
                sims.append(float(vecs[0] @ vecs[1]))
        sims = np.array(sims)
        pred = sims >= F.MATCH_THRESHOLD
        same, diff = y == 1, y == 0
        from sklearn.metrics import roc_auc_score

        out[cname] = {
            "accuracy": round(float((pred == (y == 1)).mean()), 3),
            "true_match_rate": round(float(pred[same].mean()), 3),
            "false_match_rate": round(float(pred[diff].mean()), 3),
            "auroc": round(float(roc_auc_score(y, sims)), 3),
            "face_not_detected": round(missed / len(y), 3),
        }
        print("faces", cname, out[cname], file=sys.stderr)
    return out


def translation() -> dict:
    from search_eval import index_scores, load_urfd_frames, retrieval_metrics

    from cctvgpt.search.models import get_embedder
    from cctvgpt.search.translate import translate_query

    frames, rel = load_urfd_frames(Path("external_data/urfd"))
    rng = np.random.default_rng(0)
    night = [degrade_to_night(f, rng) for f in frames]
    emb = get_embedder("en")
    out = {}
    for q in ["फर्श पर लेटा हुआ व्यक्ति", "जमीन पर गिरा हुआ आदमी", "फर्श पर पड़ा व्यक्ति"]:
        tr = translate_query(q)
        v = emb.encode_text(tr["text"])
        out[q] = {"translation": tr["text"], "backend": tr["backend"],
                  "day": retrieval_metrics(index_scores(frames, v, emb, False), rel),
                  "night": retrieval_metrics(index_scores(night, v, emb, False), rel)}
        print("translation", q, json.dumps(out[q], ensure_ascii=False), file=sys.stderr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default="plates,faces,translation")
    a = ap.parse_args()
    res = {}
    for part in a.only.split(","):
        res[part] = {"plates": plates, "faces": faces, "translation": translation}[part](a.limit if part == "faces" else None) if part != "translation" else translation()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    path = Path(a.out) / "plates_faces_eval.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(res)
    path.write_text(json.dumps(prev, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
