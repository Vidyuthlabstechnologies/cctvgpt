"""Forensic search tests. A colour-based stub embedder keeps these fast and offline (no model downloads)."""

import json
import subprocess

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cctvgpt.api.app import create_app
from cctvgpt.pipeline import generate
from cctvgpt.prompt import resolve_request
from cctvgpt.search.engine import SearchEngine
from cctvgpt.search.enhance import assess, degrade_to_night, enhance, needs_enhancement

from .conftest import FAST

COLOURS = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}


class StubEmbedder:
    """Embeds an image as its mean colour; text by colour words. Enough to test ranking plumbing."""

    dim = 3

    def encode_images(self, images):
        v = np.array([im.reshape(-1, 3).mean(0) for im in images], np.float32) + 1e-3
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_text(self, q):
        for name, bgr in COLOURS.items():
            if name in q:
                v = np.array(bgr, np.float32) + 1e-3
                return v / np.linalg.norm(v)
        return np.ones(3, np.float32) / np.sqrt(3)


class StubDetector:
    def detect(self, frames):
        return [[("person", 0.9, (2, 2, 40, 60))] if f.shape[0] > 70 else [] for f in frames]


def _video(path, bgr, n=24, fps=8, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), bgr, np.uint8)
        cv2.putText(f, str(i), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        w.write(f)
    w.release()


@pytest.fixture
def engine(platform, tmp_path):
    return SearchEngine(tmp_path / "search", platform.store, platform.watermarker, platform.keys, embedder=StubEmbedder(), detector=StubDetector())


@pytest.fixture
def footage(platform):
    d = platform.settings.evidence_inbox / "case1"
    d.mkdir(parents=True)
    _video(d / "cam_red.mp4", COLOURS["red"])
    _video(d / "cam_blue.mp4", COLOURS["blue"])
    cv2.imwrite(str(d / "photo_green.jpg"), np.full((90, 90, 3), COLOURS["green"], np.uint8))
    return d


def test_enhancement_brightens_dark_noisy_frames():
    rng = np.random.default_rng(0)
    day = np.full((120, 160, 3), 150, np.uint8)
    cv2.rectangle(day, (40, 30), (120, 100), (30, 60, 200), -1)
    night = degrade_to_night(day, rng)
    q = assess(night)
    assert q.dark and needs_enhancement(q)
    assert assess(enhance(night, q)).brightness > q.brightness + 20


def test_ingest_search_and_dedup(engine, footage):
    s = engine.ingest(footage, "officer", "FIR-1")
    assert s["indexed"] == 3 and s["errors"] == [] and s["objects"] > 0
    assert engine.ingest(footage, "officer", "FIR-1")["skipped_duplicates"] == 3  # same files by hash
    r = engine.search("officer", "FIR-1", "test", text="red")
    assert r["results"][0]["path"].endswith("cam_red.mp4")
    assert r["results"][0]["kind"] == "video" and r["results"][0]["t_end"] >= r["results"][0]["t_start"]
    g = engine.search("officer", "FIR-1", "test", text="green")
    assert g["results"][0]["path"].endswith("photo_green.jpg")
    # Query by example image.
    img = engine.search("officer", "FIR-1", "test", image=np.full((50, 50, 3), COLOURS["blue"], np.uint8))
    assert img["results"][0]["path"].endswith("cam_blue.mp4")
    with pytest.raises(ValueError):
        engine.search("officer", "", "", text="red")


def test_ir_and_language_warnings(engine, platform):
    d = platform.settings.evidence_inbox / "case_ir"
    d.mkdir(parents=True)
    grey = np.full((90, 90, 3), 60, np.uint8)
    cv2.circle(grey, (45, 45), 20, (200, 200, 200), -1)
    cv2.imwrite(str(d / "ir_frame.png"), grey)
    engine.ingest(d, "officer", "FIR-7")
    r = engine.search("officer", "FIR-7", "test", text="man in red shirt")
    assert any("infrared" in w for w in r["warnings"])
    assert any("Indian-language" in w for w in engine.search("officer", "FIR-7", "test", text="लाल शर्ट वाला आदमी")["warnings"])


def test_searches_are_audited(engine, footage, platform):
    engine.ingest(footage, "officer", "FIR-2")
    engine.search("officer", "FIR-2", "trace red vehicle", text="red")
    rows = platform.store.q("SELECT action, details_json FROM audit WHERE action LIKE 'search_%' ORDER BY seq")
    actions = [r["action"] for r in rows]
    assert "search_ingest" in actions and "search_query" in actions
    q = json.loads([r for r in rows if r["action"] == "search_query"][-1]["details_json"])
    assert q["case_ref"] == "FIR-2" and q["purpose"] == "trace red vehicle" and q["text"] == "red"
    assert platform.store.verify_audit_chain()["ok"]


def test_export_preserves_original_and_records_custody(engine, footage):
    engine.ingest(footage, "officer", "FIR-3")
    hit = engine.search("officer", "FIR-3", "test", text="red")["results"][0]
    ex = engine.export_segment(hit["source_id"], 0.5, 1.5, "officer", "FIR-3")
    assert ex["original"]["sha256"] and ex["export"]["sha256"]
    assert "no re-encoding" in ex["export"]["method"]
    assert (engine.dir / "exports" / f"{ex['export_id']}.custody.json").exists()
    # Tampering with the original after ingest blocks export.
    with open(footage / "cam_red.mp4", "ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="changed since ingest"):
        engine.export_segment(hit["source_id"], 0.5, 1.5, "officer", "FIR-3")


def test_platform_synthetic_clips_are_flagged_and_excluded(engine, platform, customer):
    req, interp = resolve_request({"scene": {**FAST, "duration_s": 3.0, "fps": 6}})
    res = generate(platform, customer["id"], "u", req, interp)
    d = platform.settings.evidence_inbox / "case_syn"
    d.mkdir(parents=True)
    # Strip our metadata so only the invisible watermark can reveal it.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(res.video_path), "-map_metadata", "-1", "-c", "copy", str(d / "suspicious.mp4")], check=True)
    s = engine.ingest(d, "officer", "FIR-4")
    assert s["synthetic_flagged"] == 1
    assert engine.search("officer", "FIR-4", "test", text="anything")["results"] == []
    assert len(engine.search("officer", "FIR-4", "test", text="anything", include_synthetic=True)["results"]) == 1


def test_search_api_roles_paths_and_flow(platform, engine, footage, tmp_path):
    st = platform.store
    inv = st.create_api_key("investigator", "si.sharma", None, "admin")
    cust_c = st.create_customer("Acme", "video_analytics", "a@x.example")
    cust = st.create_api_key("customer", "alice", cust_c["id"], "admin")
    with TestClient(create_app(platform, workers=1, search_engine=engine)) as c:
        H = {"X-API-Key": inv}
        assert c.get("/search").status_code == 200
        assert c.post("/v1/search/query", json={"text": "red", "case_ref": "FIR-5", "purpose": "x y z"}, headers={"X-API-Key": cust}).status_code == 403
        assert c.post("/v1/search/ingest", json={"path": "/etc", "case_ref": "FIR-5"}, headers=H).status_code == 403
        outside = tmp_path / "outside"
        outside.mkdir()
        assert c.post("/v1/search/ingest", json={"path": str(outside), "case_ref": "FIR-5"}, headers=H).status_code == 403
        r = c.post("/v1/search/ingest", json={"path": str(footage), "case_ref": "FIR-5", "camera": "Gate 2"}, headers=H)
        assert r.status_code == 200 and r.json()["indexed"] == 3
        assert c.post("/v1/search/query", json={"text": "red", "case_ref": "bad ref!", "purpose": "trace"}, headers=H).status_code == 422
        q = c.post("/v1/search/query", json={"text": "red", "case_ref": "FIR-5", "purpose": "trace red car"}, headers=H).json()
        top = q["results"][0]
        assert top["path"].endswith("cam_red.mp4") and top["camera"] == "Gate 2"
        assert c.get(f"/v1/search/thumb/{top['item_id']}", headers=H).headers["content-type"] == "image/jpeg"
        assert c.get(f"/v1/search/media/{top['source_id']}?case_ref=FIR-5", headers=H).status_code == 200
        _, buf = cv2.imencode(".png", np.full((40, 40, 3), COLOURS["blue"], np.uint8))
        qi = c.post("/v1/search/query-image", data={"case_ref": "FIR-5", "purpose": "match photo"}, files={"image": ("q.png", buf.tobytes(), "image/png")}, headers=H).json()
        assert qi["results"][0]["path"].endswith("cam_blue.mp4")
        ex = c.post("/v1/search/export", json={"source_id": top["source_id"], "t0": 0.5, "t1": 1.0, "case_ref": "FIR-5"}, headers=H).json()
        dl = c.get(f"/v1/search/exports/{ex['export_id']}/download", headers=H)
        assert dl.status_code == 200 and len(dl.content) > 0
        up = c.post("/v1/search/upload", data={"case_ref": "FIR-6"}, files=[("files", ("green2.png", cv2.imencode(".png", np.full((80, 80, 3), (0, 200, 0), np.uint8))[1].tobytes(), "image/png"))], headers=H)
        assert up.status_code == 200 and up.json()["indexed"] == 1
        assert len(c.get("/v1/search/sources", headers=H).json()) == 4
