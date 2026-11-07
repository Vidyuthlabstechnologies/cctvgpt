"""Event detection (violence windows, accident probe) and cross-camera trails through the engine."""

import numpy as np

from cctvgpt.search.engine import SearchEngine
from cctvgpt.search.events import AccidentProbe, violence_windows

from .test_search import COLOURS, StubDetector, StubEmbedder, _video, stub_plates


class FightEmbedder(StubEmbedder):
    """'fighting' is red, 'walking calmly' is blue: a red clip is a fight."""

    def encode_text(self, q):
        if "fighting" in q:
            return super().encode_text("red")
        if "walking" in q:
            return super().encode_text("blue")
        return super().encode_text(q)


def test_violence_windows_merge_overlaps_and_respect_threshold():
    pos, neg = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    calm, fight = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    vecs = np.array([calm] * 10 + [fight] * 12 + [calm] * 10)
    ev = violence_windows(list(range(32)), vecs, pos, neg, window=8, stride=4, threshold=0.2)
    assert len(ev) == 1
    t0, t1, score, peak = ev[0]
    assert 6 <= t0 <= 12 and 20 <= t1 <= 26 and score > 0.5 and 10 <= peak < 22
    assert violence_windows(list(range(32)), np.array([calm] * 32), pos, neg) == []


def test_accident_probe_loads_and_ignores_wrong_dimension():
    p = AccidentProbe()
    assert p.coef.shape == (512,)
    assert p.prob(np.zeros((2, 3), np.float32)).tolist() == [0.0, 0.0]
    v = np.random.default_rng(0).normal(size=(4, 512)).astype(np.float32)
    assert ((p.prob(v) >= 0) & (p.prob(v) <= 1)).all()


def test_engine_finds_violent_clip_and_audits_event_search(platform, tmp_path):
    eng = SearchEngine(tmp_path / "s", platform.store, embedder=FightEmbedder(), detector=StubDetector(), plate_reader=stub_plates)
    d = tmp_path / "in"
    d.mkdir()
    _video(d / "fight.mp4", COLOURS["red"], n=80)
    _video(d / "calm.mp4", COLOURS["blue"], n=80)
    s = eng.ingest(d, "inv", "FIR-9/2026")
    assert s["events"] >= 1
    r = eng.events("inv", "FIR-9/2026", "assault complaint", kind="violence")
    assert r["results"] and all("fight.mp4" in e["path"] for e in r["results"])
    assert "not alerts" in r["note"]
    assert "search_events" in [r["action"] for r in platform.store.q("SELECT action FROM audit")]


def test_trail_plate_orders_sightings_by_corrected_camera_time(platform, tmp_path):
    eng = SearchEngine(tmp_path / "s", platform.store, embedder=StubEmbedder(), detector=StubDetector(), plate_reader=stub_plates)
    for name, sub, n in (("junction_a", "a", 24), ("junction_b", "b", 32)):  # different lengths: not duplicates
        d = tmp_path / sub
        d.mkdir()
        _video(d / f"{name}.mp4", COLOURS["red"], n=n)
    eng.set_camera("inv", "junction_a", 12.9172, 77.6229, 0, "traffic junction")
    eng.set_camera("inv", "junction_b", 12.9116, 77.6412, -600, "shop DVR")  # DVR 10 min fast
    eng.ingest(tmp_path / "a", "inv", "FIR-10/2026", camera="junction_a", start_time="2026-11-01T21:00:00")
    # Recorder says 21:15, true time 21:05: 2 km in 5 min is possible.
    eng.ingest(tmp_path / "b", "inv", "FIR-10/2026", camera="junction_b", start_time="2026-11-01T21:15:00")
    out = eng.trail_plate("inv", "FIR-10/2026", "vehicle theft", "MH01DE2780")
    assert [p["camera"] for p in out["trail"]] == ["junction_a", "junction_b"]
    assert out["trail"][1]["time"].startswith("2026-11-01T21:05")
    assert out["legs"][0]["km"] > 1.5
