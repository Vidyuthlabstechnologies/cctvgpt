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
        return [[("person", 0.9, (2, 2, 40, 60)), ("car", 0.9, (60, 10, 90, 70))] if f.shape[0] > 70 else [] for f in frames]


class StubTranslator:
    name = "stub"

    def translate(self, text, src_lang):
        return {"लाल": "red", "नीली": "blue"}.get(text.split()[0], "something")


class _Read:
    def __init__(self, text):
        self.text, self.raw, self.valid_format, self.bbox = text, text, True, (2, 2, 30, 10)


def stub_plates(img):
    """Pretend every vehicle crop in a red video carries plate MH01DE2780, blue ones KA05MN1234."""
    b, g, r = img.reshape(-1, 3).mean(0)
    return [_Read("MH01DE2780" if r > b else "KA05MN1234")]


class StubFaces:
    MATCH_THRESHOLD = 0.363
    MIN_RELIABLE_FACE_PX = 40

    @staticmethod
    def detect_faces(img):
        return [np.array([10, 10, 50, 50] + [0] * 10 + [0.95], np.float32)] if img.shape[0] > 70 else []

    @staticmethod
    def embed_face(img, row):
        v = img.reshape(-1, 3).mean(0).astype(np.float32) + 1e-3
        return v / np.linalg.norm(v)

    @classmethod
    def largest_face(cls, img):
        f = cls.detect_faces(cv2.resize(img, (100, 100)))
        return f[0] if f else None


def _video(path, bgr, n=24, fps=8, size=(160, 120)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        f = np.full((size[1], size[0], 3), bgr, np.uint8)
        cv2.putText(f, str(i), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        w.write(f)
    w.release()


@pytest.fixture
def engine(platform, tmp_path):
    return SearchEngine(tmp_path / "search", platform.store, platform.watermarker, platform.keys, embedder=StubEmbedder(), detector=StubDetector(),
                        plate_reader=stub_plates, face_backend=StubFaces(), translator=StubTranslator(), enable_faces=True)


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
    r = engine.search("officer", "FIR-7", "test", text="लाल शर्ट वाला आदमी")
    assert r["query"]["searched_text"] == "red" and r["query"]["translation"]["source_lang"] == "hin_Deva"
    assert any("translation" in w for w in r["warnings"])


def test_hindi_query_is_translated_before_search(engine, footage):
    engine.ingest(footage, "officer", "FIR-8")
    r = engine.search("officer", "FIR-8", "test", text="लाल गाड़ी")
    assert r["results"][0]["path"].endswith("cam_red.mp4")


def test_untranslatable_query_warns(platform, tmp_path):
    class Broken:
        name = "broken"

        def translate(self, text, lang):
            raise RuntimeError("model missing")

    eng = SearchEngine(tmp_path / "s2", platform.store, embedder=StubEmbedder(), detector=StubDetector(), translator=Broken(), plate_reader=stub_plates)
    r = eng.search("o", "FIR-9", "test", text="லால் கார்")
    assert r["query"]["translation"]["source_lang"] == "tam_Taml"
    assert any("could not be translated" in w for w in r["warnings"])


def test_plate_search_exact_partial_and_fuzzy(engine, footage, platform):
    engine.ingest(footage, "officer", "FIR-10")
    exact = engine.search_plate("officer", "FIR-10", "trace vehicle", "MH01DE2780")
    assert exact["results"][0]["path"].endswith("cam_red.mp4") and exact["results"][0]["score"] == 1.0
    partial = engine.search_plate("officer", "FIR-10", "witness saw last digits", "2780")
    assert partial["results"] and partial["results"][0]["plate_read"] == "MH01DE2780"
    fuzzy = engine.search_plate("officer", "FIR-10", "ocr tolerant", "MH01DE278O")
    assert fuzzy["results"][0]["plate_read"] == "MH01DE2780"
    assert not engine.search_plate("officer", "FIR-10", "none", "DL3CAB9999")["results"]
    assert "search_plate" in [r["action"] for r in platform.store.q("SELECT action FROM audit")]


def test_face_search_is_gated_scoped_and_audited(engine, footage, platform, tmp_path):
    engine.ingest(footage, "officer", "FIR-11")
    probe = np.full((120, 120, 3), COLOURS["blue"], np.uint8)
    with pytest.raises(ValueError, match="justification"):
        engine.search_face("officer", "FIR-11", "find suspect", "short", probe)
    r = engine.search_face("officer", "FIR-11", "find suspect", "CCTV of the chain-snatching suspect from the victim's phone", probe)
    assert r["results"][0]["path"].endswith("cam_blue.mp4") and r["results"][0]["above_threshold"]
    # Another case's footage is not searched.
    assert engine.search_face("officer", "OTHER-CASE", "find suspect", "Different case, must not see FIR-11 footage", probe)["results"] == []
    rows = [json.loads(x["details_json"]) for x in platform.store.q("SELECT action, details_json FROM audit") if x["action"] == "search_face"]
    d = [r for r in rows if r["case_ref"] == "FIR-11"][-1]
    assert d["justification"].startswith("CCTV of") and len(d["probe_sha256"]) == 64
    off = SearchEngine(tmp_path / "s3", platform.store, embedder=StubEmbedder(), detector=StubDetector(), plate_reader=stub_plates)
    with pytest.raises(PermissionError):
        off.search_face("officer", "FIR-11", "x", "a long enough justification text", probe)


def test_scene_tags_vehicle_colour_and_places(engine, footage):
    engine.ingest(footage, "officer", "FIR-12")
    srcs = engine.sources()
    assert all(len(s["scene_tags"]) == 3 for s in srcs)
    assert any(s["plates_read"] > 0 for s in srcs) and any(s["faces_indexed"] > 0 for s in srcs)
    places = engine.places()
    assert sum(len(p["sources"]) for p in places) == 3
    tag = srcs[0]["scene_tags"][0]["tag"]
    assert engine.search("officer", "FIR-12", "t", text="red", scene=tag)["results"]


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


def test_plate_face_places_api(platform, engine, footage):
    inv = platform.store.create_api_key("investigator", "si.rao", None, "admin")
    with TestClient(create_app(platform, workers=1, search_engine=engine)) as c:
        H = {"X-API-Key": inv}
        c.post("/v1/search/ingest", json={"path": str(footage), "case_ref": "FIR-20"}, headers=H)
        cap = c.get("/v1/search/capabilities", headers=H).json()
        assert cap["face_search"] is True and cap["gender_classification"] is False
        pr = c.post("/v1/search/plate", json={"plate": "MH01DE2780", "case_ref": "FIR-20", "purpose": "trace car"}, headers=H).json()
        assert pr["results"][0]["plate_read"] == "MH01DE2780"
        assert c.get(f"/v1/search/plate-thumb/{pr['results'][0]['plate_id']}", headers=H).status_code == 200
        _, buf = cv2.imencode(".png", np.full((120, 120, 3), COLOURS["red"], np.uint8))
        form = {"case_ref": "FIR-20", "purpose": "find suspect", "justification": "Suspect photo provided by complainant in FIR-20"}
        fr = c.post("/v1/search/face", data=form, files={"image": ("p.png", buf.tobytes(), "image/png")}, headers=H).json()
        assert fr["results"][0]["path"].endswith("cam_red.mp4")
        assert c.get(f"/v1/search/face-thumb/{fr['results'][0]['face_id']}", headers=H).status_code == 200
        allc = c.post("/v1/search/face", data={**form, "all_cases": "true"}, files={"image": ("p.png", buf.tobytes(), "image/png")}, headers=H)
        assert allc.status_code == 403  # only admins may search across cases
        assert c.get("/v1/search/places", headers=H).status_code == 200


def test_multiframe_plate_consensus_fuses_noisy_reads():
    from cctvgpt.search.plates import consensus

    assert consensus(["MH01DE2780", "MH01DF2780", "MHO1DE27B0", "MH01DE2788", "H01DE2780"])[0] == "MH01DE2780"
    rows = [("src", t, txt, txt, 1, "car", 0, 0, 10, 10, "th") for t, txt in
            [(0.0, "MH01DE2788"), (1.0, "MH01DF2780"), (2.0, "MH01DE2780"), (3.0, "MHO1DE27B0"), (20.0, "KA05MN1234")]]
    fused = SearchEngine._fuse_plate_reads(rows)
    assert len(fused) == 1 and fused[0][2] == "MH01DE2780" and fused[0][3].startswith("CONSENSUS n=4")
