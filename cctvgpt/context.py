"""Shared platform context: settings, store, keys and watermarker."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, load_settings
from .safeguards.keys import KeyRing
from .safeguards.watermark import Watermarker
from .store import Store

TERMS_VERSION = "2026-11-draft-1"


@dataclass
class Platform:
    settings: Settings
    store: Store
    keys: KeyRing
    watermarker: Watermarker

    @classmethod
    def open(cls, settings: Settings | None = None) -> "Platform":
        settings = settings or load_settings()
        settings.ensure_dirs()
        store = Store(settings.db_path)
        keys = KeyRing(settings.keys_dir)
        return cls(settings, store, keys, Watermarker(keys.watermark_key))
