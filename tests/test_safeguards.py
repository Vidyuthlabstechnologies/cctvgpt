import json
import subprocess

import numpy as np
import pytest

from cctvgpt.params import RealismParams, SceneParams
from cctvgpt.realism import RealismProcessor
from cctvgpt.render.renderer import SimRenderer
from cctvgpt.safeguards import provenance
from cctvgpt.safeguards.keys import KeyRing
from cctvgpt.safeguards.label import apply_label, has_label, label_score
from cctvgpt.safeguards.watermark import Watermarker, bits_to_token, token_to_bits
from cctvgpt.store import Store
from cctvgpt.video_io import VideoWriter, read_frames


@pytest.fixture(scope="module")
def clip_frames():
    p = SceneParams(event="fall", width=640, height=360, fps=12, duration_s=3.0, seed=4)
    r = SimRenderer(p)
    rp = RealismProcessor(RealismParams(), p, r.depth)
    return p, [(rp.process(f, t.index), t.t) for f, t in r.frames()]


def test_label_applied_and_detected(clip_frames):
    _, frames = clip_frames
    f = frames[0][0]
    assert not has_label(f)
    labelled = apply_label(f)
    assert has_label(labelled) and label_score(labelled) > 0.95


def test_payload_roundtrip_and_crc():
    bits = token_to_bits(0x12345678)
    assert bits_to_token(bits) == 0x12345678
    bits[3] *= -1
    assert bits_to_token(bits) is None


def test_watermark_invisible_detectable_and_no_false_positive(clip_frames):
    _, frames = clip_frames
    wm = Watermarker(b"k" * 32)
    marked = [wm.embed(f, 0xCAFEBABE, t) for f, t in frames]
    psnr = np.mean([10 * np.log10(255**2 / np.mean((a.astype(float) - b.astype(float)) ** 2)) for (a, _), b in zip(frames, marked)])
    assert psnr > 36, psnr
    assert wm.detect(marked).token == 0xCAFEBABE
    assert not wm.detect([f for f, _ in frames]).present
    # A different key does not read the mark.
    assert Watermarker(b"x" * 32).detect(marked).token is None


def test_watermark_survives_reencode_resize_and_frame_drops(clip_frames, tmp_path):
    p, frames = clip_frames
    wm = Watermarker(b"k" * 32)
    a = tmp_path / "a.mp4"
    w = VideoWriter(a, p.width, p.height, p.fps)
    for f, t in frames:
        w.write(wm.embed(f, 0x0BADF00D, t))
    w.close()
    b = tmp_path / "b.mp4"
    # Re-encode at lower quality, smaller size and half the frame rate.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(a), "-vf", "scale=426:-2,fps=6", "-c:v", "libx264", "-crf", "28", str(b)], check=True)
    assert wm.detect(read_frames(a)).token == 0x0BADF00D
    assert wm.detect(read_frames(b)).token == 0x0BADF00D


def test_provenance_sign_verify_and_tamper(tmp_path):
    keys = KeyRing(tmp_path / "keys")
    env = provenance.sign_manifest({"clip_id": "c1", "content": {"video_stream_sha256": "ab"}}, keys)
    assert provenance.verify_envelope(env, keys).valid
    enc = provenance.encode_for_container(env)
    assert provenance.decode_from_container(enc) == json.loads(json.dumps(env))
    forged = json.loads(json.dumps(env))
    forged["manifest"]["customer_id"] = "someone-else"
    assert not provenance.verify_envelope(forged, keys).valid
    other = KeyRing(tmp_path / "other")
    assert not provenance.verify_envelope(env, other).valid  # unknown key
    assert KeyRing(tmp_path / "keys").key_id == keys.key_id  # keys persist


def test_audit_chain_detects_tampering(tmp_path):
    st = Store(tmp_path / "db.sqlite3")
    for i in range(5):
        st.audit("u", "act", {"i": i})
    assert st.verify_audit_chain() == {"ok": True, "entries": 5, "head": st.q("SELECT entry_hash FROM audit ORDER BY seq DESC LIMIT 1")[0]["entry_hash"]}
    st.x("UPDATE audit SET details_json=? WHERE seq=3", (json.dumps({"i": 99}),))
    res = st.verify_audit_chain()
    assert not res["ok"] and res["broken_at_seq"] == 3
