"""End-to-end generation pipeline.

    customer checks (SR-5) -> release gate (SR-10) -> prompt filter (SR-6)
    -> render (FR-1/2/4/8) -> CCTV realism (FR-3) -> visible label (SR-1)
    -> invisible watermark (SR-2) -> encode -> QA gate (label + watermark
    re-detected on the encoded file) -> signed provenance (SR-3)
    -> annotations (FR-7) -> audit log (SR-7)

Nothing leaves the pipeline unless every safeguard check passes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config, release_gate
from .backends import get_backend
from .context import TERMS_VERSION, Platform
from .params import USE_CASES_RESTRICTED, GenerationRequest
from .prompt import PromptInterpretation
from .realism import RealismProcessor
from .safeguards import provenance
from .safeguards.content_filter import check_text
from .safeguards.label import apply_label, label_score
from .safeguards.watermark import new_token
from .store import new_id, now_iso
from .video_io import VideoWriter, embed_comment, read_frames, stream_sha256


class GenerationRefused(Exception):
    """The request is not allowed (policy). Carries machine-readable reasons."""

    def __init__(self, reason: str, details: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class QAFailed(RuntimeError):
    pass


@dataclass
class ClipResult:
    clip_id: str
    video_path: Path
    manifest_path: Path
    annotations_path: Path | None
    manifest: dict
    qa: dict


def check_customer(customer: dict | None, request: GenerationRequest, settings) -> str | None:
    """Return the clean-export contract reference if a clean export is permitted, else None.

    Raises GenerationRefused when the customer may not generate this request.
    """
    if customer is None:
        raise GenerationRefused("unknown_customer")
    if customer["status"] != "active":
        raise GenerationRefused("customer_not_active", {"status": customer["status"]})
    if customer["kyc_status"] != "verified":
        raise GenerationRefused("kyc_not_verified", {"message": "Business KYC must be verified before generating (SR-5)."})
    if customer["terms_accepted_version"] != TERMS_VERSION:
        raise GenerationRefused("terms_not_accepted", {"required_version": TERMS_VERSION})
    if request.use_case in USE_CASES_RESTRICTED:
        if customer["tier"] != "enterprise" or not customer["restricted_use_contract"]:
            raise GenerationRefused("restricted_use_requires_contract", {"use_case": request.use_case})
    clean_contract = None
    if request.clean_export:
        if not settings.clean_export_enabled:
            raise GenerationRefused("clean_export_disabled", {"message": "Clean exports are disabled platform-wide pending legal review (SR-9)."})
        if customer["tier"] != "enterprise" or not customer["clean_export_contract"]:
            raise GenerationRefused("clean_export_requires_enterprise_contract")
        clean_contract = customer["clean_export_contract"]
    return clean_contract


def screen_prompt(request: GenerationRequest) -> dict:
    texts = [request.prompt, request.realism.camera_label]
    for t in texts:
        res = check_text(t)
        if not res.allowed:
            return res.to_dict()
    return {"allowed": True}


def generate(
    platform: Platform,
    customer_id: str,
    principal: str,
    request: GenerationRequest,
    interpretation: PromptInterpretation | None = None,
    job_id: str | None = None,
    backend: str = "sim",
) -> ClipResult:
    st = platform.store
    settings = platform.settings
    customer = st.customer(customer_id)
    audit_base = {"job_id": job_id, "prompt": request.prompt, "params": request.to_dict(), "principal": principal}
    try:
        request.validate(settings.max_duration_s)
        clean_contract = check_customer(customer, request, settings)
        release_gate.require_approved(st, *release_gate.renderer_component(), enforce=settings.enforce_release_gate)
        if request.clean_export:
            release_gate.require_approved(st, "feature", "clean_export", enforce=True)
        screen = screen_prompt(request)
        if not screen["allowed"]:
            raise GenerationRefused("prompt_blocked", screen)
    except GenerationRefused as e:
        st.audit(principal, "generation_refused", {**audit_base, "reason": e.reason, "details": e.details}, customer_id)
        raise
    except release_gate.ReleaseBlocked as e:
        st.audit(principal, "generation_refused", {**audit_base, "reason": "release_gate", "details": str(e)}, customer_id)
        raise GenerationRefused("release_gate", {"message": str(e)}) from e

    clip_id = new_id("clip")
    token = new_token()
    while st.token_in_use(token):
        token = new_token()
    labelled = clean_contract is None
    created_at = now_iso()
    clip_dir = settings.clips_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    st.x(
        "INSERT INTO clips (id, customer_id, job_id, token, created_at, request_json, status, renderer_version, labelled) VALUES (?,?,?,?,?,?,?,?,?)",
        (clip_id, customer_id, job_id, token, created_at, json.dumps(request.to_dict()), "rendering", config.RENDERER_VERSION, int(labelled)),
    )
    try:
        result = _render_and_seal(platform, clip_id, clip_dir, token, customer_id, request, interpretation, created_at, labelled, clean_contract, backend)
    except Exception as e:
        st.x("UPDATE clips SET status=?, qa_json=? WHERE id=?", ("failed", json.dumps({"error": str(e)}), clip_id))
        shutil.rmtree(clip_dir, ignore_errors=True)
        st.audit(principal, "generation_failed", {**audit_base, "clip_id": clip_id, "error": str(e)}, customer_id)
        raise
    st.audit(
        principal,
        "clip_generated",
        {**audit_base, "clip_id": clip_id, "output_stream_sha256": result.manifest["content"]["video_stream_sha256"],
         "output_file_sha256": _file_sha256(result.video_path), "labelled": labelled, "clean_export_contract": clean_contract},
        customer_id,
    )
    return result


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_and_seal(platform, clip_id, clip_dir, token, customer_id, request, interpretation, created_at, labelled, clean_contract, backend_name) -> ClipResult:
    st = platform.store
    sp = request.scene
    renderer = get_backend(backend_name).create(sp)
    realism = RealismProcessor(request.realism, sp, getattr(renderer, "depth", None))
    wm = platform.watermarker

    raw_path = clip_dir / "raw.mp4"
    writer = VideoWriter(raw_path, sp.width, sp.height, sp.fps)
    frame_truth = []
    label_scores = []
    stability_ref = None
    stability = []
    try:
        for frame, truth in renderer.frames():
            out = realism.process(frame, truth.index)
            if labelled:
                out = apply_label(out)
                label_scores.append(label_score(out))
            out = wm.embed(out, token, truth.t)
            writer.write(out)
            objs = []
            for o in truth.objects:
                o = dict(o)
                o["bbox"] = realism.distort_bbox(o["bbox"])
                if o["bbox"][2] > 0 and o["bbox"][3] > 0:
                    objs.append(o)
            frame_truth.append({"frame": truth.index, "t": truth.t, "objects": objs})
            stability_ref, s = _stability(out, objs, stability_ref)
            stability.append(s)
    finally:
        writer.close()

    # ---- QA gate: the encoded file itself must carry the safeguards.
    decoded = read_frames(raw_path, max_frames=120)
    det = wm.detect(decoded)
    qa = {
        "frames": len(frame_truth),
        "label_required": labelled,
        "label_min_score_pre_encode": round(min(label_scores), 3) if label_scores else None,
        "label_min_score_decoded": round(min(label_score(f) for f in decoded[:: max(1, len(decoded) // 10)]), 3) if labelled and decoded else None,
        "watermark": det.to_dict(),
        "fixed_view_stability": round(float(np.median(stability)), 4) if stability else None,
    }
    failures = []
    if labelled and (qa["label_min_score_pre_encode"] is None or qa["label_min_score_pre_encode"] < 0.6):
        failures.append("visible label missing before encode")
    if labelled and (qa["label_min_score_decoded"] is None or qa["label_min_score_decoded"] < 0.5):
        failures.append("visible label not detectable in encoded file")
    if det.token != token:
        failures.append("watermark not recovered from encoded file")
    qa["passed"] = not failures
    qa["failures"] = failures
    if failures:
        raise QAFailed("; ".join(failures))

    # ---- Provenance (SR-3): hash the encoded stream, sign, embed.
    s_hash = stream_sha256(raw_path)
    manifest = provenance.build_manifest(
        clip_id=clip_id, created_at=created_at, customer_id=customer_id, declared_use=request.use_case,
        params_hash=request.params_hash(), prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
        stream_sha256=s_hash, width=sp.width, height=sp.height, fps=sp.fps, n_frames=len(frame_truth),
        visible_label=labelled, clean_export_contract=clean_contract, backend=backend_name, key_id=platform.keys.key_id,
    )
    envelope = provenance.sign_manifest(manifest, platform.keys)
    video_path = clip_dir / f"{clip_id}.mp4"
    embed_comment(raw_path, video_path, provenance.encode_for_container(envelope))
    raw_path.unlink(missing_ok=True)
    if stream_sha256(video_path) != s_hash:
        raise QAFailed("stream hash changed while embedding provenance")
    manifest_path = clip_dir / f"{clip_id}.provenance.json"
    manifest_path.write_text(json.dumps(envelope, indent=2))

    ann_path = None
    if request.annotations:
        ann_path = clip_dir / f"{clip_id}.annotations.json"
        ann = build_annotations(clip_id, request, interpretation, renderer, realism, frame_truth, manifest)
        ann_path.write_text(json.dumps(ann, indent=1))
        write_mot(clip_dir / f"{clip_id}.tracks.csv", frame_truth)

    st.x(
        "UPDATE clips SET status=?, manifest_json=?, stream_sha256=?, file_sha256=?, path=?, qa_json=? WHERE id=?",
        ("ready", json.dumps(envelope), s_hash, _file_sha256(video_path), str(video_path), json.dumps(qa), clip_id),
    )
    return ClipResult(clip_id, video_path, manifest_path, ann_path, manifest, qa)


def _stability(frame: np.ndarray, objects: list, ref):
    """Fixed-view stability (section 9 quality metric): 1 - mean abs difference to the
    first frame over pixels not covered by annotated objects or overlays."""
    g = frame.mean(axis=2).astype(np.float32)
    if ref is None:
        return g, 1.0
    H, W = g.shape
    mask = np.ones((H, W), bool)
    mask[: int(H * 0.14), :] = False  # timestamp and top badge
    mask[int(H * 0.9) :, :] = False  # bottom badge
    for o in objects:
        x, y, w, h = o["bbox"]
        pad = 6
        mask[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = False
    if mask.sum() < 100:
        return ref, 1.0
    return ref, float(1.0 - np.abs(g - ref)[mask].mean() / 255.0)


def build_annotations(clip_id, request, interpretation, renderer, realism, frame_truth, manifest) -> dict:
    events = renderer.event_annotations()
    for ev in events:
        if ev.get("zone_polygon_image"):
            ev["zone_polygon_image"] = [[round(float(x), 1), round(float(y), 1)] for x, y in realism.distort_points(ev["zone_polygon_image"])]
    meta = renderer.scene_metadata()
    if request.realism.lens_distortion > 0:
        meta["camera"]["note"] = "intrinsics describe the undistorted pinhole model; boxes are in distorted output pixels"
    return {
        "schema": "cctvgpt.annotations/1",
        "clip_id": clip_id,
        "synthetic": True,
        "disclaimer": provenance.DISCLAIMER,
        "video": {"width": request.scene.width, "height": request.scene.height, "fps": request.scene.fps, "frames": len(frame_truth)},
        "generator": manifest["generator"],
        "parameters": {"scene": asdict(request.scene), "realism": asdict(request.realism)},
        "prompt_interpretation": interpretation.to_dict() if interpretation else None,
        "scene": meta,
        "classes": ["person", "bag", "fire"],
        "bbox_format": "xywh_pixels",
        "events": events,
        "frames": frame_truth,
    }


def write_mot(path: Path, frame_truth: list) -> None:
    """MOTChallenge-style tracks: frame,id,x,y,w,h,conf,class,visibility (1-based frames)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "x", "y", "w", "h", "conf", "class", "visibility"])
        for fr in frame_truth:
            for o in fr["objects"]:
                x, y, bw, bh = o["bbox"]
                w.writerow([fr["frame"] + 1, o["track_id"], x, y, bw, bh, 1, o["class"], o["visibility"]])
