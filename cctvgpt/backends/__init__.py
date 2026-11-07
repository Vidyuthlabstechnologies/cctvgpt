"""Generation backends (PRD section 8).

* ``sim`` - Option B: procedural simulation renderer with exact ground truth.
  This is the Phase 1 feasibility backend and the default.
* ``foundation`` - Option A: a text-to-video foundation model fine-tuned or
  prompted for CCTV style. Not wired up: the model choice, licence and compute
  are [OPEN] pending the technical lead and the feasibility test. The adapter
  interface below is what such a backend must implement; every safeguard
  (label, watermark, provenance, audit, filter) is applied by the pipeline
  regardless of backend.
"""

from __future__ import annotations

from typing import Protocol

from ..params import SceneParams


class Renderer(Protocol):
    depth: object  # per-pixel depth map (H, W) or None

    def frames(self):  # yields (frame_bgr_uint8, FrameTruth)
        ...

    def event_annotations(self) -> list: ...

    def scene_metadata(self) -> dict: ...


class Backend(Protocol):
    name: str

    def create(self, params: SceneParams) -> Renderer: ...


class SimBackend:
    name = "sim"

    def create(self, params: SceneParams):
        from ..render.renderer import SimRenderer

        return SimRenderer(params)


class FoundationModelBackend:
    """Placeholder for Option A. Raises until a licensed model is selected [OPEN]."""

    name = "foundation"

    def create(self, params: SceneParams):
        raise NotImplementedError(
            "The foundation-model backend is not configured. Model selection, licence review "
            "(PRD section 7) and compute quotes are open decisions for the technical lead."
        )


BACKENDS = {"sim": SimBackend(), "foundation": FoundationModelBackend()}


def get_backend(name: str = "sim"):
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}") from None
