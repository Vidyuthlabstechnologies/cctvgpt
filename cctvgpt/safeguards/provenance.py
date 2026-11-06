"""SR-3: signed provenance metadata embedded in each file.

The manifest records generator, model/renderer version, timestamp, customer ID,
declared use and a hash of the encoded video stream. It is signed with
Ed25519, embedded in the MP4 container's comment tag, and written as a sidecar
JSON file.

The assertion vocabulary follows C2PA concepts (claim generator, actions,
"trainedAlgorithmicMedia" digital source type) so that it can be migrated to a
real C2PA/JUMBF manifest via the c2pa SDK. This is NOT itself a conformant C2PA
manifest [VERIFY against the current C2PA specification before claiming
conformance].
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature

from .. import config
from .keys import KeyRing

EMBED_PREFIX = "CCTVGPT-PROVENANCE-1:"
DISCLAIMER = "SYNTHETIC - AI-generated simulation. Not real footage. Must not be used or presented as evidence or as a record of real events."


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def build_manifest(
    *,
    clip_id: str,
    created_at: str,
    customer_id: str,
    declared_use: str,
    params_hash: str,
    prompt_sha256: str,
    stream_sha256: str,
    width: int,
    height: int,
    fps: int,
    n_frames: int,
    visible_label: bool,
    clean_export_contract: str | None,
    backend: str,
    key_id: str,
) -> dict:
    return {
        "schema": config.MANIFEST_SCHEMA,
        "claim_generator": f"{config.GENERATOR_NAME}/{config.PACKAGE_VERSION}",
        "clip_id": clip_id,
        "created_at": created_at,
        "synthetic": True,
        "digital_source_type": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "disclaimer": DISCLAIMER,
        "generator": {
            "backend": backend,
            "renderer_version": config.RENDERER_VERSION,
            "package_version": config.PACKAGE_VERSION,
            "watermark_scheme": config.WATERMARK_SCHEME,
        },
        "customer_id": customer_id,
        "declared_use": declared_use,
        "params_sha256": params_hash,
        "prompt_sha256": prompt_sha256,
        "visible_label": visible_label,
        "clean_export_contract": clean_export_contract,
        "content": {
            "video_stream_sha256": stream_sha256,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": n_frames,
        },
        "actions": [{"action": "c2pa.created", "when": created_at, "softwareAgent": f"{config.GENERATOR_NAME}/{config.RENDERER_VERSION}"}],
        "signing_key_id": key_id,
    }


def sign_manifest(manifest: dict, keys: KeyRing) -> dict:
    sig = keys.signing_key.sign(canonical(manifest))
    return {"manifest": manifest, "alg": "Ed25519", "key_id": keys.key_id, "signature": base64.b64encode(sig).decode()}


def encode_for_container(envelope: dict) -> str:
    return EMBED_PREFIX + base64.urlsafe_b64encode(canonical(envelope)).decode()


def decode_from_container(comment: str | None) -> dict | None:
    if not comment or not comment.startswith(EMBED_PREFIX):
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(comment[len(EMBED_PREFIX) :].encode()))
    except Exception:
        return None


@dataclass
class SignatureCheck:
    valid: bool
    reason: str
    key_id: str | None = None


def verify_envelope(envelope: dict, keys: KeyRing) -> SignatureCheck:
    try:
        manifest = envelope["manifest"]
        kid = envelope["key_id"]
        sig = base64.b64decode(envelope["signature"])
    except Exception:
        return SignatureCheck(False, "malformed provenance envelope")
    pub = keys.trusted.get(kid)
    if pub is None:
        return SignatureCheck(False, f"signed with unknown key {kid}", kid)
    try:
        pub.verify(sig, canonical(manifest))
    except InvalidSignature:
        return SignatureCheck(False, "signature does not match manifest (manifest altered or forged)", kid)
    return SignatureCheck(True, "signature valid", kid)
