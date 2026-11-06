import time

import pytest
from fastapi.testclient import TestClient

from cctvgpt.api.app import create_app
from cctvgpt.context import TERMS_VERSION

from .conftest import FAST


@pytest.fixture
def client(platform):
    app = create_app(platform, workers=1)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def keys(platform):
    st = platform.store
    return {
        "admin": st.create_api_key("admin", "root", None, "test"),
        "owner": st.create_api_key("safeguards_owner", "sg", None, "test"),
    }


def H(k):
    return {"X-API-Key": k}


def _onboard(client, keys):
    c = client.post("/v1/admin/customers", json={"name": "Acme", "org_type": "video_analytics", "contact_email": "a@acme.example"}, headers=H(keys["admin"])).json()
    k = client.post(f"/v1/admin/customers/{c['id']}/api-keys", json={"principal": "alice"}, headers=H(keys["admin"])).json()["api_key"]
    return c["id"], k


def _wait(client, key, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/v1/jobs/{job_id}", headers=H(key)).json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_public_endpoints(client):
    assert client.get("/").status_code == 200
    assert "Verify a clip" in client.get("/verify").text
    assert client.get("/v1/health").json()["renderer_approved"] is True
    assert client.get("/v1/public-key").json()["algorithm"] == "Ed25519"
    assert client.get("/v1/terms").json()["version"] == TERMS_VERSION
    assert "fall" in client.get("/v1/options").json()["event_library"]
    r = client.post("/v1/misuse-reports", json={"description": "Clip shown on social media as a real incident", "clip_reference": "clip_x"})
    assert r.status_code == 201


def test_no_anonymous_generation_and_roles(client, keys):
    body = {"prompt": "a fall", "scene": FAST}
    assert client.post("/v1/generate", json=body).status_code == 401
    assert client.post("/v1/generate", json=body, headers=H("bogus")).status_code == 401
    assert client.post("/v1/generate", json=body, headers=H(keys["admin"])).status_code == 403
    assert client.get("/v1/admin/audit", headers=H(keys["owner"])).status_code == 403
    assert client.post("/v1/releases/decide", json={"decision": "blocked"}, headers=H(keys["admin"])).status_code == 403


def test_customer_flow_end_to_end(client, keys):
    cid, key = _onboard(client, keys)
    body = {"prompt": "someone leaves a bag unattended in the hallway", "scene": {**FAST, "duration_s": 4.0}}
    # KYC pending -> refused; job is recorded as blocked.
    r = client.post("/v1/generate?wait=true", json=body, headers=H(key))
    assert r.status_code == 403 and r.json()["error"]["reason"] == "kyc_not_verified"
    client.post(f"/v1/admin/customers/{cid}/kyc", json={"status": "verified", "reference": "GSTIN"}, headers=H(keys["admin"]))
    r = client.post("/v1/generate?wait=true", json=body, headers=H(key))
    assert r.json()["error"]["reason"] == "terms_not_accepted"
    assert client.post("/v1/me/accept-terms", json={"version": TERMS_VERSION}, headers=H(key)).status_code == 200
    me = client.get("/v1/me", headers=H(key)).json()
    assert me["customer"]["kyc_status"] == "verified"

    # Interpretation preview and filter.
    pre = client.post("/v1/interpret", json={"prompt": "fire in Mumbai warehouse"}, headers=H(key)).json()
    assert pre["filter"]["allowed"] is False

    job = client.post("/v1/generate", json=body, headers=H(key)).json()
    job = _wait(client, key, job["id"])
    assert job["status"] == "done", job
    assert job["qa"]["passed"]
    video = client.get(job["links"]["video"], headers=H(key))
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    ann = client.get(job["links"]["annotations"], headers=H(key)).json()
    assert ann["events"][0]["type"] == "abandoned_object"
    assert client.get(job["links"]["tracks"], headers=H(key)).text.startswith("frame,track_id")
    assert client.get(job["links"]["provenance"], headers=H(key)).json()["alg"] == "Ed25519"

    # Another customer cannot see the clip.
    _, other = _onboard(client, keys)
    assert client.get(job["links"]["video"], headers=H(other)).status_code == 404

    # Public verification of the downloaded file.
    rep = client.post("/v1/verify", files={"file": ("clip.mp4", video.content, "video/mp4")}).json()
    assert rep["verdict"] == "verified_platform_original"
    full = client.post("/v1/admin/verify", files={"file": ("clip.mp4", video.content, "video/mp4")}, headers=H(keys["admin"])).json()
    assert full["clip"]["customer_id"] == cid

    # Audit trail is complete and intact.
    actions = [a["action"] for a in client.get("/v1/admin/audit?limit=200", headers=H(keys["admin"])).json()]
    for a in ("customer_created", "kyc_updated", "terms_accepted", "generation_refused", "clip_generated", "clip_downloaded", "verification_performed"):
        assert a in actions, a
    assert client.get("/v1/admin/audit/verify", headers=H(keys["admin"])).json()["ok"]


def test_batch_sweep_and_export(client, keys, platform):
    cid, key = _onboard(client, keys)
    platform.store.set_kyc(cid, "verified", "ref", "admin")
    platform.store.accept_terms(cid, TERMS_VERSION, "alice")
    body = {"base": {"prompt": "a person falls", "scene": FAST}, "sweep": {"weather": ["clear", "fog"], "scene": ["parking_lot"], "seed": [1, 2]}}
    r = client.post("/v1/batches", json=body, headers=H(key)).json()
    assert len(r["jobs"]) == 4
    for jid in r["jobs"]:
        assert _wait(client, key, jid)["status"] == "done"
    st = client.get(f"/v1/batches/{r['batch_id']}", headers=H(key)).json()
    assert st["counts"] == {"done": 4}
    z = client.get(f"/v1/batches/{r['batch_id']}/export.zip", headers=H(key))
    assert z.status_code == 200 and z.content[:2] == b"PK"
    too_big = {"base": {"scene": FAST}, "sweep": {"seed": list(range(100))}}
    assert client.post("/v1/batches", json=too_big, headers=H(key)).status_code == 422


def test_release_decision_by_safeguards_owner(client, keys):
    r = client.post("/v1/releases/decide", json={"decision": "blocked", "notes": "pause"}, headers=H(keys["owner"]))
    assert r.status_code == 200 and r.json()["status"] == "blocked"
    assert client.get("/v1/health").json()["renderer_approved"] is False
    r = client.post("/v1/releases/decide", json={"decision": "approved", "notes": "resume"}, headers=H(keys["owner"]))
    assert r.status_code == 200 and r.json()["status"] == "approved"


def test_batch_acceptance_test_endpoint(client, keys, platform):
    import json as _json

    cid, key = _onboard(client, keys)
    platform.store.set_kyc(cid, "verified", "ref", "admin")
    platform.store.accept_terms(cid, TERMS_VERSION, "alice")
    r = client.post("/v1/batches", json={"base": {"scene": {**FAST, "event": "fall", "crowd_density": "empty"}}, "sweep": {"time_of_day": ["day", "night"]}}, headers=H(key)).json()
    preds = {}
    for jid in r["jobs"]:
        j = _wait(client, key, jid)
        ann = client.get(j["links"]["annotations"], headers=H(key)).json()
        preds[j["clip_id"]] = {"frames": [{"frame": f["frame"], "detections": [{"bbox": o["bbox"], "score": 0.9, "class": o["class"]} for o in f["objects"]]} for f in ann["frames"]], "events": []}
    card = client.post(f"/v1/batches/{r['batch_id']}/evaluate", json={"predictions": preds}, headers=H(key)).json()
    assert card["overall"]["detection"]["person"]["recall"] == 1.0
    assert set(card["by_condition"]["time_of_day"]) == {"day", "night"}
    html_ = client.post(f"/v1/batches/{r['batch_id']}/evaluate?format=html", json={"predictions": preds}, headers=H(key))
    assert "Acceptance test" in html_.text
    bad = client.post(f"/v1/batches/{r['batch_id']}/evaluate", json={"predictions": {"clip_nope": {}}}, headers=H(key))
    assert bad.status_code == 422
    _ = _json
