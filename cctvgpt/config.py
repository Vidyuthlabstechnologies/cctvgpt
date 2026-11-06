"""Runtime configuration.

Everything that the PRD marks as [OPEN] or [VERIFY] and that changes runtime
behaviour is surfaced here so it can be set by the owner named in the PRD
without code changes. Values are read from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Version identifiers recorded in every provenance manifest (FR-5, SR-3).
GENERATOR_NAME = "VidyuthLabs CCTVGPT"
PACKAGE_VERSION = "0.1.0"
# The procedural simulation renderer (PRD section 8, Option B). Bump this whenever
# rendering output for a given (prompt, params, seed) can change, so that
# reproducibility claims stay truthful.
RENDERER_VERSION = "sim-1.1.0"
WATERMARK_SCHEME = "vl-ss-dct-v1"
MANIFEST_SCHEMA = "cctvgpt.provenance/1"
# Unique identifier of this computer resource, embedded in every output's
# permanent metadata (IT Rules 2026 amendment: SGI must carry "permanent
# metadata ... including a unique identifier" of the intermediary's computer
# resource [VERIFY exact wording against the Gazette notification]).
PLATFORM_ID = "in.vidyuthlabs.cctvgpt"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CCTVGPT_DATA_DIR", "data")).resolve())
    # SR-9: clean (unlabelled) exports. Disabled platform-wide. The IT Rules 2026
    # amendment requires intermediaries enabling SGI creation to "prevent any
    # modification, suppression or removal of the label", which this exception
    # likely conflicts with. Do not enable for India without written sign-off
    # from counsel (docs/COMPLIANCE.md). Even when enabled, a customer needs an
    # enterprise contract reference on file.
    clean_export_enabled: bool = field(default_factory=lambda: _env_bool("CCTVGPT_CLEAN_EXPORT_ENABLED", False))
    # SR-7: audit-log retention in days. [OPEN] to be set by counsel; None means
    # "retain indefinitely" until a period is decided.
    audit_retention_days: int | None = field(default_factory=lambda: _env_int("CCTVGPT_AUDIT_RETENTION_DAYS", None))
    # SR-10: when True, generation is refused for any renderer version the
    # safeguards owner has not approved.
    enforce_release_gate: bool = field(default_factory=lambda: _env_bool("CCTVGPT_ENFORCE_RELEASE_GATE", True))
    # Hard limits that keep cost per clip bounded until real compute figures
    # exist (section 6.3 [VERIFY with cloud provider quotes]).
    max_duration_s: int = field(default_factory=lambda: _env_int("CCTVGPT_MAX_DURATION_S", 30) or 30)
    max_batch_size: int = field(default_factory=lambda: _env_int("CCTVGPT_MAX_BATCH", 50) or 50)
    worker_threads: int = field(default_factory=lambda: _env_int("CCTVGPT_WORKERS", 2) or 2)
    # Forensic search: folders the server may ingest from (colon-separated). Ingest
    # requests for any other path are refused, so the API cannot read arbitrary files.
    search_roots: list = field(default_factory=lambda: [Path(p).resolve() for p in os.environ.get("CCTVGPT_SEARCH_ROOTS", "").split(":") if p])
    # Face matching is biometric processing: off unless the deployment opts in after legal
    # review (docs/SEARCH.md). When off, faces are not indexed at all.
    face_search_enabled: bool = field(default_factory=lambda: _env_bool("CCTVGPT_FACE_SEARCH_ENABLED", False))
    # "en" (default) or "multilingual" (Hindi and other Indian-language queries).
    search_model: str = field(default_factory=lambda: os.environ.get("CCTVGPT_SEARCH_MODEL", "en"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "cctvgpt.sqlite3"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def search_dir(self) -> Path:
        return self.data_dir / "search"

    @property
    def evidence_inbox(self) -> Path:
        return self.data_dir / "evidence_inbox"

    def allowed_search_roots(self) -> list:
        return [self.evidence_inbox.resolve(), *self.search_roots]

    @property
    def keys_dir(self) -> Path:
        return self.data_dir / "keys"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.clips_dir, self.keys_dir):
            p.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
