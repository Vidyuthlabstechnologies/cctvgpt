"""Generation parameters (FR-1, FR-2, FR-3, FR-4, FR-5).

Generation is driven only by this closed vocabulary. Free text in the prompt is
mapped onto these fields (see :mod:`cctvgpt.prompt`) and is never passed to the
renderer directly, which is itself a safeguard: the renderer cannot depict
anything outside the event library and scene catalogue.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCENES = ("corridor", "parking_lot", "warehouse")
TIMES_OF_DAY = ("day", "dusk", "night")
WEATHER = ("clear", "rain", "fog", "snow")
EVENTS = ("none", "fall", "intrusion", "fire", "crowd_surge", "abandoned_object")
CROWD_DENSITY = ("empty", "low", "medium", "high")
OCCLUSION = ("none", "light", "heavy")
COMPRESSION = ("none", "low", "medium", "high")
IR_MODES = ("auto", "on", "off")
RESOLUTIONS = ((320, 240), (640, 360), (640, 480), (960, 540), (1280, 720))

# Declared use is recorded in the provenance manifest and audit log. Anything
# not listed here (e.g. evidence, investigation, media) is refused.
USE_CASES_IN_SCOPE = (
    "rare_event_training",
    "analytics_testing",
    "privacy_safe_dataset",
    "operator_training",
    "sales_demo",
)
USE_CASES_RESTRICTED = ("public_safety_training",)  # enterprise contract + verified customer only
USE_CASES = USE_CASES_IN_SCOPE + USE_CASES_RESTRICTED


class ParamError(ValueError):
    pass


def _check_choice(name: str, value: Any, choices: tuple) -> None:
    if value not in choices:
        raise ParamError(f"{name}={value!r} is not one of {list(choices)}")


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ParamError(f"{name}={value} is outside [{lo}, {hi}]")


@dataclass
class SceneParams:
    scene: str = "corridor"
    time_of_day: str = "day"
    weather: str = "clear"
    event: str = "none"
    crowd_density: str = "low"
    occlusion: str = "none"
    glare: bool = False
    # Camera (FR-2): mounting height in metres, downward tilt, horizontal pan, field of view.
    camera_height_m: float = 2.8
    camera_tilt_deg: float = 25.0
    camera_pan_deg: float = 0.0
    fov_deg: float = 70.0
    width: int = 640
    height: int = 360
    fps: int = 12
    duration_s: float = 8.0
    seed: int = 0

    def validate(self, max_duration_s: float = 30.0) -> None:
        _check_choice("scene", self.scene, SCENES)
        _check_choice("time_of_day", self.time_of_day, TIMES_OF_DAY)
        _check_choice("weather", self.weather, WEATHER)
        _check_choice("event", self.event, EVENTS)
        _check_choice("crowd_density", self.crowd_density, CROWD_DENSITY)
        _check_choice("occlusion", self.occlusion, OCCLUSION)
        _check_range("camera_height_m", self.camera_height_m, 2.0, 12.0)
        _check_range("camera_tilt_deg", self.camera_tilt_deg, 5.0, 75.0)
        _check_range("camera_pan_deg", self.camera_pan_deg, -45.0, 45.0)
        _check_range("fov_deg", self.fov_deg, 40.0, 110.0)
        if (self.width, self.height) not in RESOLUTIONS:
            raise ParamError(f"resolution {self.width}x{self.height} is not one of {RESOLUTIONS}")
        _check_range("fps", self.fps, 2, 30)
        _check_range("duration_s", self.duration_s, 1.0, max_duration_s)
        if not (0 <= int(self.seed) < 2**32):
            raise ParamError("seed must be in [0, 2^32)")
        if self.scene == "corridor" and self.camera_height_m > 2.9:
            # The corridor ceiling is 3 m; a camera cannot be mounted above it.
            raise ParamError("corridor cameras must be mounted at or below 2.9 m")

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))

    @property
    def outdoor(self) -> bool:
        return self.scene == "parking_lot"


@dataclass
class RealismParams:
    """CCTV realism options (FR-3)."""

    low_res_factor: float = 1.0  # 1.0 = native; 0.5 renders detail at half resolution then upscales
    compression: str = "medium"
    ir_night_mode: str = "auto"  # auto switches to IR monochrome at night
    lens_distortion: float = 0.15  # barrel distortion strength
    noise: float = 0.35  # sensor noise strength 0..1
    timestamp_overlay: bool = True
    camera_label: str = "CAM-01"

    def validate(self) -> None:
        _check_range("low_res_factor", self.low_res_factor, 0.25, 1.0)
        _check_choice("compression", self.compression, COMPRESSION)
        _check_choice("ir_night_mode", self.ir_night_mode, IR_MODES)
        _check_range("lens_distortion", self.lens_distortion, 0.0, 0.5)
        _check_range("noise", self.noise, 0.0, 1.0)
        if len(self.camera_label) > 16 or not all(c.isalnum() or c in "-_ " for c in self.camera_label):
            raise ParamError("camera_label must be <=16 alphanumeric/-/_ characters")


@dataclass
class GenerationRequest:
    prompt: str = ""
    use_case: str = "analytics_testing"
    scene: SceneParams = field(default_factory=SceneParams)
    realism: RealismParams = field(default_factory=RealismParams)
    annotations: bool = True  # FR-7
    clean_export: bool = False  # SR-9; honoured only under the exception rules

    def validate(self, max_duration_s: float = 30.0) -> None:
        if len(self.prompt) > 1000:
            raise ParamError("prompt is longer than 1000 characters")
        _check_choice("use_case", self.use_case, USE_CASES)
        self.scene.validate(max_duration_s)
        self.realism.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def params_hash(self) -> str:
        """Hash of everything that determines the rendered content (FR-5)."""
        payload = {"scene": asdict(self.scene), "realism": asdict(self.realism)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "GenerationRequest":
        d = dict(d or {})
        scene = _dataclass_from(SceneParams, d.pop("scene", {}) or {})
        realism = _dataclass_from(RealismParams, d.pop("realism", {}) or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ParamError(f"unknown request fields: {sorted(unknown)}")
        return cls(scene=scene, realism=realism, **d)


def _dataclass_from(cls, d: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ParamError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    kwargs = {}
    defaults = cls()
    for k, v in d.items():
        default = getattr(defaults, k)
        try:
            if isinstance(default, bool):
                if isinstance(v, str):
                    v = v.strip().lower() in {"1", "true", "yes", "on"}
                kwargs[k] = bool(v)
            elif isinstance(default, int):
                kwargs[k] = int(v)
            elif isinstance(default, float):
                kwargs[k] = float(v)
            else:
                kwargs[k] = str(v)
        except (TypeError, ValueError) as e:
            raise ParamError(f"{cls.__name__}.{k}: {e}") from e
    return cls(**kwargs)


def option_schema() -> dict:
    """Machine-readable description of the controllable parameters, for the dashboard and API."""
    return {
        "scene": list(SCENES),
        "time_of_day": list(TIMES_OF_DAY),
        "weather": list(WEATHER),
        "event": list(EVENTS),
        "crowd_density": list(CROWD_DENSITY),
        "occlusion": list(OCCLUSION),
        "compression": list(COMPRESSION),
        "ir_night_mode": list(IR_MODES),
        "resolutions": [f"{w}x{h}" for w, h in RESOLUTIONS],
        "use_cases": {"in_scope": list(USE_CASES_IN_SCOPE), "restricted": list(USE_CASES_RESTRICTED)},
        "ranges": {
            "camera_height_m": [2.0, 12.0],
            "camera_tilt_deg": [5.0, 75.0],
            "camera_pan_deg": [-45.0, 45.0],
            "fov_deg": [40.0, 110.0],
            "fps": [2, 30],
            "duration_s": [1.0, 30.0],
            "low_res_factor": [0.25, 1.0],
            "lens_distortion": [0.0, 0.5],
            "noise": [0.0, 1.0],
        },
    }
