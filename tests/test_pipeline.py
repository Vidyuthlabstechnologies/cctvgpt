import json
import subprocess

import pytest

from cctvgpt import release_gate
from cctvgpt.context import TERMS_VERSION
from cctvgpt.pipeline import GenerationRefused, generate
from cctvgpt.prompt import resolve_request
from cctvgpt.verify import verify_file
from cctvgpt.video_io import read_tags

from .conftest import FAST, make_platform


def _req(prompt="a person falls in the corridor", **kw):
    scene = {**FAST, **kw.pop("scene", {})}
    req, interp = resolve_request({"prompt": prompt, "scene": scene, **kw})
    return req, interp


def _actions(platform):
    return [r["action"] for r in platform.store.q("SELECT action FROM audit ORDER BY seq")]


def test_full_generation_outputs_and_verification(platform, customer):
    req, interp = _req()
    res = generate(platform, customer["id"], "alice", req, interp)
    assert res.qa["passed"] and res.qa["watermark"]["present"]
    assert res.video_path.exists() and res.manifest_path.exists() and res.annotations_path.exists()
    ann = json.loads(res.annotations_path.read_text())
    assert ann["synthetic"] is True and ann["events"][0]["type"] == "fall"
    assert len(ann["frames"]) == req.scene.n_frames
    m = res.manifest
    assert m["customer_id"] == customer["id"] and m["synthetic"] and m["visible_label"]
    assert m["generator"]["renderer_version"]
    assert read_tags(res.video_path)["comment"].startswith("CCTVGPT-PROVENANCE-1:")

    rep = verify_file(platform, res.video_path)
    assert rep["verdict"] == "verified_platform_original"
    assert rep["checks"]["visible_label_detected"]
    assert rep["clip"]["customer_id"].endswith("...")  # public view hides customer id
    assert "clip_generated" in _actions(platform)
    assert platform.store.verify_audit_chain()["ok"]


def test_verification_of_modified_copies(platform, customer, tmp_path):
    req, interp = _req(scene={"duration_s": 3.0, "fps": 8})
    res = generate(platform, customer["id"], "alice", req, interp)
    # Metadata stripped, stream untouched.
    stripped = tmp_path / "stripped.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(res.video_path), "-map", "0", "-c", "copy", "-map_metadata", "-1", str(stripped)], check=True)
    assert verify_file(platform, stripped)["verdict"] == "platform_original_metadata_altered"
    # Re-encoded and resized: only the watermark survives.
    reenc = tmp_path / "reenc.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(res.video_path), "-vf", "scale=240:-2", "-map_metadata", "-1", "-c:v", "libx264", "-crf", "26", str(reenc)], check=True)
    rep = verify_file(platform, reenc)
    assert rep["verdict"] == "platform_origin_modified", rep
    assert rep["clip"]["clip_id"] == res.clip_id
    # Unrelated video: no marks, with the caveat that this does not prove it is real.
    other = tmp_path / "other.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=6:duration=2", "-pix_fmt", "yuv420p", str(other)], check=True)
    rep = verify_file(platform, other)
    assert rep["verdict"] == "no_platform_marks_found" and "does NOT mean" in rep["message"]


def test_forged_manifest_is_flagged(platform, customer, tmp_path):
    from cctvgpt.safeguards import provenance
    from cctvgpt.video_io import embed_comment

    other = tmp_path / "other.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=6:duration=2", "-pix_fmt", "yuv420p", str(other)], check=True)
    env = {"manifest": {"clip_id": "clip_fake"}, "alg": "Ed25519", "key_id": platform.keys.key_id, "signature": "AAAA"}
    forged = tmp_path / "forged.mp4"
    embed_comment(other, forged, provenance.encode_for_container(env))
    assert verify_file(platform, forged)["verdict"] == "invalid_provenance_claim"


@pytest.mark.parametrize("setup,reason", [
    (lambda st, c: st.set_kyc(c, "pending", "-", "admin"), "kyc_not_verified"),
    (lambda st, c: st.update_customer(c, "admin", status="suspended"), "customer_not_active"),
    (lambda st, c: st.accept_terms(c, "old-version", "u"), "terms_not_accepted"),
])
def test_customer_gates(platform, customer, setup, reason):
    setup(platform.store, customer["id"])
    with pytest.raises(GenerationRefused) as e:
        generate(platform, customer["id"], "alice", *_req())
    assert e.value.reason == reason
    assert "generation_refused" in _actions(platform)


def test_prompt_filter_blocks_and_audits(platform, customer):
    with pytest.raises(GenerationRefused) as e:
        generate(platform, customer["id"], "alice", *_req("footage of Ravi Kumar falling, to use as evidence"))
    assert e.value.reason == "prompt_blocked"
    row = platform.store.q("SELECT details_json FROM audit WHERE action='generation_refused'")[-1]
    details = json.loads(row["details_json"])
    assert details["prompt"].startswith("footage of Ravi Kumar")
    assert set(details["details"]["categories"]) >= {"real_person", "evidence_misuse"}


def test_release_gate_blocks_unapproved_renderer(tmp_path, customer, platform):
    release_gate.decide(platform.store, *release_gate.renderer_component(), "blocked", "owner", "regression found")
    with pytest.raises(GenerationRefused) as e:
        generate(platform, customer["id"], "alice", *_req())
    assert e.value.reason == "release_gate"
    with pytest.raises(release_gate.ReleaseBlocked):
        release_gate.decide(platform.store, *release_gate.renderer_component(), "approved", "owner", "", {"passed": False})


def test_restricted_use_requires_enterprise_contract(platform, customer):
    with pytest.raises(GenerationRefused) as e:
        generate(platform, customer["id"], "alice", *_req(use_case="public_safety_training"))
    assert e.value.reason == "restricted_use_requires_contract"
    platform.store.update_customer(customer["id"], "admin", tier="enterprise", restricted_use_contract="MSA-42")
    res = generate(platform, customer["id"], "alice", *_req(use_case="public_safety_training"))
    assert res.manifest["declared_use"] == "public_safety_training"


def test_clean_export_exception(tmp_path, selftest_result):
    # Disabled platform-wide by default (SR-9 pending legal review).
    p = make_platform(tmp_path / "a")
    release_gate.decide(p.store, *release_gate.renderer_component(), "approved", "owner", "", selftest_result)
    c = p.store.create_customer("Ent", "integrator", "e@x.example", tier="enterprise")
    p.store.set_kyc(c["id"], "verified", "ref", "admin")
    p.store.accept_terms(c["id"], TERMS_VERSION, "u")
    p.store.update_customer(c["id"], "admin", clean_export_contract="CLEAN-7")
    with pytest.raises(GenerationRefused) as e:
        generate(p, c["id"], "u", *_req(clean_export=True))
    assert e.value.reason == "clean_export_disabled"

    # Enabled, with contract, and the feature approved by the safeguards owner.
    p2 = make_platform(tmp_path / "b", clean_export_enabled=True)
    release_gate.decide(p2.store, *release_gate.renderer_component(), "approved", "owner", "", selftest_result)
    c2 = p2.store.create_customer("Ent", "integrator", "e@x.example", tier="enterprise")
    p2.store.set_kyc(c2["id"], "verified", "ref", "admin")
    p2.store.accept_terms(c2["id"], TERMS_VERSION, "u")
    with pytest.raises(GenerationRefused) as e:
        generate(p2, c2["id"], "u", *_req(clean_export=True))
    assert e.value.reason == "clean_export_requires_enterprise_contract"
    p2.store.update_customer(c2["id"], "admin", clean_export_contract="CLEAN-7")
    with pytest.raises(GenerationRefused) as e:
        generate(p2, c2["id"], "u", *_req(clean_export=True))
    assert e.value.reason == "release_gate"
    release_gate.decide(p2.store, "feature", "clean_export", "approved", "owner", "", selftest_result)
    res = generate(p2, c2["id"], "u", *_req(clean_export=True))
    assert res.manifest["visible_label"] is False and res.manifest["clean_export_contract"] == "CLEAN-7"
    rep = verify_file(p2, res.video_path)
    # Unlabelled, but watermark, metadata and logging are retained.
    assert rep["verdict"] == "verified_platform_original" and not rep["checks"]["visible_label_detected"]
    assert rep["checks"]["watermark"]["present"]
