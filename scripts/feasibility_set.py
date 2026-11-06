#!/usr/bin/env python3
"""Phase 1 feasibility set (PRD section 8 and roadmap Phase 1).

Generates the single-scene / single-event test set through the full pipeline
(SR-1 to SR-3 applied from day one), varying lighting, weather, camera and
seed, and writes a summary CSV with the automatic metrics:

  * fixed-view stability (FR-8) - 1 = background pixels identical across frames
  * event correctness - whether the scripted event occurred inside the clip
  * safeguard QA - label and watermark re-detected from the encoded file

The practitioner-acceptance part of the gate is a human review; the CSV has
blank columns for the reviewer's ratings.

    python scripts/feasibility_set.py --scene corridor --event fall --out out/feasibility
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cctvgpt import release_gate  # noqa: E402
from cctvgpt.context import TERMS_VERSION, Platform  # noqa: E402
from cctvgpt.pipeline import generate  # noqa: E402
from cctvgpt.prompt import resolve_request  # noqa: E402

INTERNAL_CUSTOMER = "VidyuthLabs internal - feasibility"


def internal_customer(p: Platform) -> str:
    row = p.store.q1("SELECT id FROM customers WHERE name=?", (INTERNAL_CUSTOMER,))
    if row:
        return row["id"]
    c = p.store.create_customer(INTERNAL_CUSTOMER, "internal", "safeguards@vidyuthlabs.invalid", actor="feasibility-script")
    p.store.set_kyc(c["id"], "verified", "internal-evaluation", "feasibility-script")
    p.store.accept_terms(c["id"], TERMS_VERSION, "feasibility-script")
    return c["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="corridor")
    ap.add_argument("--event", default="fall")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--out", default="out/feasibility")
    a = ap.parse_args()

    p = Platform.open()
    if not release_gate.is_approved(p.store, *release_gate.renderer_component()):
        sys.exit("The renderer is not approved. Ask the safeguards owner to run: cctvgpt release approve --owner NAME")
    cid = internal_customer(p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    times = ["day", "dusk", "night"]
    weathers = ["clear", "rain", "fog"] if a.scene == "parking_lot" else ["clear"]
    tilts = [15.0, 30.0]
    rows = []
    for tod, weather, tilt, seed in itertools.product(times, weathers, tilts, range(a.seeds)):
        payload = {
            "prompt": f"{a.event} in a {a.scene}",
            "use_case": "analytics_testing",
            "scene": {"scene": a.scene, "event": a.event, "time_of_day": tod, "weather": weather, "camera_tilt_deg": tilt, "seed": seed, "duration_s": a.seconds},
        }
        req, interp = resolve_request(payload)
        res = generate(p, cid, "feasibility-script", req, interp)
        ann = json.loads(res.annotations_path.read_text())
        ev = ann["events"][0] if ann["events"] else {}
        name = f"{a.scene}_{a.event}_{tod}_{weather}_tilt{int(tilt)}_s{seed}"
        for f in res.video_path.parent.iterdir():
            shutil.copy2(f, out / f"{name}{''.join(f.suffixes)}")
        rows.append({
            "file": f"{name}.mp4", "clip_id": res.clip_id, "time_of_day": tod, "weather": weather, "tilt_deg": tilt, "seed": seed,
            "fixed_view_stability": res.qa["fixed_view_stability"], "event_occurred": ev.get("occurred"),
            "event_start_s": ev.get("start_time_s"), "label_ok": res.qa["label_min_score_decoded"],
            "watermark_confidence": res.qa["watermark"]["bit_confidence"],
            "reviewer_realism_1to5": "", "reviewer_event_correct_yn": "", "reviewer_would_use_yn": "", "reviewer_notes": "",
        })
        print(f"{name}: stability={res.qa['fixed_view_stability']} event={ev.get('occurred')}", file=sys.stderr)
    with open(out / "feasibility_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} clips and feasibility_summary.csv to {out}")


if __name__ == "__main__":
    main()
