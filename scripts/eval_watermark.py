#!/usr/bin/env python3
"""Watermark robustness evaluation (SR-2 [VERIFY]; section 9 safeguards metric).

Renders a set of clips across scenes and conditions, embeds the watermark,
applies a suite of attacks with ffmpeg and reports the detection rate per
attack, plus the false-positive rate on unmarked clips. The safeguards owner
should re-run this for every release and attach the output to the release
decision.

    python scripts/eval_watermark.py --clips 6 --out out/watermark_eval.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cctvgpt.params import RealismParams, SceneParams  # noqa: E402
from cctvgpt.realism import RealismProcessor  # noqa: E402
from cctvgpt.render.renderer import SimRenderer  # noqa: E402
from cctvgpt.safeguards.label import apply_label  # noqa: E402
from cctvgpt.safeguards.watermark import Watermarker, new_token  # noqa: E402
from cctvgpt.video_io import VideoWriter, read_frames  # noqa: E402

ATTACKS = {
    "none (platform encode)": None,
    "re-encode crf 28": ["-c:v", "libx264", "-crf", "28"],
    "re-encode crf 35": ["-c:v", "libx264", "-crf", "35"],
    "resize 75%": ["-vf", "scale=iw*3/4:-2", "-c:v", "libx264", "-crf", "23"],
    "resize 50% + crf 28": ["-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "28"],
    "resize 50% + crf 32": ["-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "32"],
    "half frame rate": ["-vf", "fps=6", "-c:v", "libx264", "-crf", "23"],
    "mpeg4 transcode": ["-c:v", "mpeg4", "-q:v", "6"],
    "brightness/contrast": ["-vf", "eq=brightness=0.06:contrast=1.15", "-c:v", "libx264", "-crf", "23"],
    "grayscale": ["-vf", "format=gray,format=yuv420p", "-c:v", "libx264", "-crf", "23"],
    "trim first 1s": ["-ss", "1", "-c:v", "libx264", "-crf", "23"],
    "crop 10% (expected to fail)": ["-vf", "crop=iw*0.9:ih*0.9", "-c:v", "libx264", "-crf", "23"],
    "rotate 3deg (expected to fail)": ["-vf", "rotate=3*PI/180", "-c:v", "libx264", "-crf", "23"],
}

CONFIGS = [
    dict(scene="corridor", event="fall", time_of_day="day"),
    dict(scene="corridor", event="crowd_surge", time_of_day="night", crowd_density="high"),
    dict(scene="parking_lot", event="intrusion", time_of_day="night", weather="rain", camera_height_m=6.0, camera_tilt_deg=20.0),
    dict(scene="parking_lot", event="abandoned_object", time_of_day="day", weather="fog", camera_height_m=6.0, camera_tilt_deg=20.0),
    dict(scene="warehouse", event="fire", time_of_day="day", camera_height_m=7.0, camera_tilt_deg=22.0),
    dict(scene="warehouse", event="none", time_of_day="dusk", occlusion="heavy", camera_height_m=7.0, camera_tilt_deg=22.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=len(CONFIGS))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    wm = Watermarker(b"evaluation-key-not-for-production")
    results = {k: [] for k in ATTACKS}
    false_pos = 0
    psnrs = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i in range(a.clips):
            cfg = CONFIGS[i % len(CONFIGS)]
            p = SceneParams(**cfg, fps=12, duration_s=a.seconds, seed=1000 + i)
            r = SimRenderer(p)
            rp = RealismProcessor(RealismParams(), p, r.depth)
            token = new_token()
            src = td / f"c{i}.mp4"
            plain = []
            w = VideoWriter(src, p.width, p.height, p.fps)
            for f, t in r.frames():
                base = apply_label(rp.process(f, t.index))
                marked = wm.embed(base, token, t.t)
                mse = np.mean((base.astype(float) - marked.astype(float)) ** 2)
                psnrs.append(10 * np.log10(255**2 / max(mse, 1e-9)))
                plain.append(base)
                w.write(marked)
            w.close()
            if wm.detect(plain).present:
                false_pos += 1
            for name, args in ATTACKS.items():
                path = src
                if args is not None:
                    path = td / f"c{i}_{abs(hash(name))}.mp4"
                    pre, post = (args[:2], args[2:]) if args[0] == "-ss" else ([], args)
                    subprocess.run(["ffmpeg", "-y", "-v", "error", *pre, "-i", str(src), *post, "-pix_fmt", "yuv420p", str(path)], check=True)
                det = wm.detect(read_frames(path))
                results[name].append(det.token == token)
            print(f"clip {i + 1}/{a.clips} {cfg['scene']}/{cfg['event']} done", file=sys.stderr)
    summary = {
        "clips": a.clips,
        "seconds_per_clip": a.seconds,
        "mean_psnr_db": round(float(np.mean(psnrs)), 2),
        "false_positive_rate_unmarked": false_pos / a.clips,
        "detection_rate": {k: round(sum(v) / len(v), 3) for k, v in results.items()},
    }
    print(json.dumps(summary, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
