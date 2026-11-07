"""Event detection over indexed footage: violence and road accidents.

Both detectors reuse the CLIP embeddings that indexing already computes, so
they add almost no processing cost.

Violence: a window of consecutive sampled frames is scored by zero-shot
similarity to "people fighting" minus "people walking calmly". This was chosen
by measurement (benchmarks/events_eval.py). A classifier trained on the Real
Life Violence Situations dataset scored AUROC 0.999 on held-out clips from that
dataset, but only 0.80 on real CCTV fights (Akti et al. 2019). The zero-shot
score reached 0.87 there. Dataset shortcuts do not transfer; the zero-shot
score does.

Accident: each detected vehicle crop is scored by a logistic probe trained on
CCTV accident crops (CC0 dataset, near-duplicate test images removed): test
AUROC 0.93, precision 0.88, recall 0.92 at the default threshold.

Both are for RANKING footage for review: "show me the moments most likely to
be a fight / a crash". Real footage has few such moments, so even a good score
produces false alarms. They must not be used as automatic alerts without
validation on the deployment's own cameras.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROBE_DIR = Path(__file__).parent / "probes"
VIOLENCE_PROMPTS = ("people fighting, punching and kicking each other", "people walking, talking or playing calmly")
VIOLENCE_THRESHOLD = 0.02  # CCTV fights: recall 0.91, precision 0.67 on a balanced set
ACCIDENT_THRESHOLD = 0.5  # probability; test precision 0.88, recall 0.92 on vehicle crops
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source_id TEXT NOT NULL, kind TEXT NOT NULL,
    t0 REAL NOT NULL, t1 REAL NOT NULL, score REAL NOT NULL, item_id INTEGER, thumb TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS events_source ON events(source_id);
"""


def violence_windows(times: list, vecs: np.ndarray, pos: np.ndarray, neg: np.ndarray, window: int = 8, stride: int = 4,
                     threshold: float = VIOLENCE_THRESHOLD) -> list:
    """Score sliding windows of frame embeddings; return merged (t0, t1, score, peak_index) above threshold."""
    if len(times) == 0:
        return []
    per = vecs @ pos - vecs @ neg
    n = len(times)
    w = min(window, n)
    starts = list(range(0, max(n - w, 0) + 1, stride))
    if starts[-1] != n - w:
        starts.append(n - w)
    hits = []
    for s in starts:
        sc = float(per[s : s + w].mean())
        if sc > threshold:
            hits.append([times[s], times[s + w - 1], sc, s + int(np.argmax(per[s : s + w]))])
    merged: list = []
    for h in hits:  # overlapping windows become one event, keeping the best score
        if merged and h[0] <= merged[-1][1]:
            m = merged[-1]
            m[1] = max(m[1], h[1])
            if h[2] > m[2]:
                m[2], m[3] = h[2], h[3]
        else:
            merged.append(h)
    return [tuple(m) for m in merged]


class AccidentProbe:
    def __init__(self, path: Path = PROBE_DIR / "accident_vitb32_laion2b.npz"):
        d = np.load(path)
        self.coef = d["coef"].astype(np.float32)
        self.intercept = float(d["intercept"])

    def prob(self, vecs: np.ndarray) -> np.ndarray:
        if len(vecs) == 0 or vecs.shape[1] != self.coef.shape[0]:
            return np.zeros(len(vecs), np.float32)
        return 1.0 / (1.0 + np.exp(-(vecs.astype(np.float32) @ self.coef + self.intercept)))
