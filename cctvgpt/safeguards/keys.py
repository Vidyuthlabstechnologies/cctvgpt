"""Key management for signing (SR-3) and watermarking (SR-2).

Keys are created on first use under <data_dir>/keys with 0600 permissions. In
production these belong in a KMS/HSM; the file layout is the Phase 1 stand-in
[VERIFY: key custody arrangement with the safeguards owner].
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _write_secret(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def key_id(pub: Ed25519PublicKey) -> str:
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def pub_b64(pub: Ed25519PublicKey) -> str:
    return base64.b64encode(pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


def pub_from_b64(s: str) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(base64.b64decode(s))


class KeyRing:
    def __init__(self, keys_dir: Path):
        self.dir = Path(keys_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        sk_path = self.dir / "signing_ed25519.pem"
        if not sk_path.exists():
            sk = Ed25519PrivateKey.generate()
            _write_secret(sk_path, sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        self.signing_key: Ed25519PrivateKey = serialization.load_pem_private_key(sk_path.read_bytes(), password=None)
        self.public_key = self.signing_key.public_key()
        self.key_id = key_id(self.public_key)
        wm_path = self.dir / "watermark.key"
        if not wm_path.exists():
            _write_secret(wm_path, secrets.token_bytes(32))
        self.watermark_key = wm_path.read_bytes()
        # Trusted public keys: the current key plus any retired keys kept for verifying older clips.
        trust_path = self.dir / "trusted_public_keys.json"
        trusted = json.loads(trust_path.read_text()) if trust_path.exists() else {}
        if self.key_id not in trusted:
            trusted[self.key_id] = pub_b64(self.public_key)
            trust_path.write_text(json.dumps(trusted, indent=2))
        self.trusted = {kid: pub_from_b64(b) for kid, b in trusted.items()}

    def public_info(self) -> dict:
        return {
            "algorithm": "Ed25519",
            "current_key_id": self.key_id,
            "keys": {kid: pub_b64(k) for kid, k in self.trusted.items()},
        }
