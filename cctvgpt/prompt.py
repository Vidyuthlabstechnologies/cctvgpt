"""Text prompt -> structured parameters (FR-1).

The prompt is interpreted against a closed vocabulary. Words that do not map to
a known control are ignored and reported back, so the user can see exactly what
the generator understood. Explicit parameters always override the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .params import GenerationRequest

# Each entry: field -> list of (value, regex). First match wins, in order.
_RULES: dict[str, list[tuple[object, str]]] = {
    "scene": [
        ("parking_lot", r"\b(parking|car ?park|parking lot|garage|driveway|vehicles?)\b"),
        ("warehouse", r"\b(warehouse|factory|depot|storage|loading dock|racks?|shelv\w*|logistics)\b"),
        ("corridor", r"\b(corridor|hallway|hall|passage|lobby|office|hospital|indoor)\b"),
    ],
    "event": [
        ("fall", r"\b(falls?|fell|falling|slips?|slipped|collaps\w*|trips?|tripped)\b"),
        ("intrusion", r"\b(intru\w*|trespass\w*|break[- ]?in|restricted area|unauthori[sz]ed|climb\w* (?:over|the) fence)\b"),
        ("fire", r"\b(fire|flames?|burning|smoke)\b"),
        ("crowd_surge", r"\b(crowd surge|surge|stampede|crush|crowd runs?|people run\w*|panic)\b"),
        ("lie_down", r"\b(lies? down|lying down to rest|reclin\w*|lies on the (?:floor|ground) deliberately)\b"),
        ("crouch", r"\b(crouch\w*|squat\w*|bends? down|picks? (?:something )?up|ties? (?:a |his |her )?shoe\w*)\b"),
        ("abandoned_object", r"\b(abandon\w*|unattended|left (?:behind|a|an|the)|suspicious (?:bag|package|object)|drops? (?:a|the) bag)\b"),
    ],
    "time_of_day": [
        ("night", r"\b(night|midnight|dark|nighttime|dimly lit|dim|2 ?am|3 ?am)\b"),
        ("dusk", r"\b(dusk|evening|sunset|twilight|dawn)\b"),
        ("day", r"\b(day|daytime|morning|afternoon|noon|sunny|daylight)\b"),
    ],
    "weather": [
        ("rain", r"\b(rain\w*|drizzl\w*|monsoon|storm\w*|wet)\b"),
        ("fog", r"\b(fog\w*|mist\w*|haz[ey]|smog)\b"),
        ("snow", r"\b(snow\w*|blizzard|sleet)\b"),
        ("clear", r"\b(clear|dry)\b"),
    ],
    "crowd_density": [
        ("high", r"\b(crowded|busy|packed|dense|rush hour|many people|crowd)\b"),
        ("medium", r"\b(several people|some people|moderate)\b"),
        ("empty", r"\b(empty|deserted|nobody else|alone)\b"),
        ("low", r"\b(quiet|few people|sparse)\b"),
    ],
    "occlusion": [
        ("heavy", r"\b(heavily occluded|heavy occlusion|obstructed|cluttered)\b"),
        ("light", r"\b(occlu\w*|partially hidden|pillars?|columns?)\b"),
    ],
    "glare": [
        (True, r"\b(glare|headlights?|lens flare|backlit|bright lights?)\b"),
    ],
}

_REALISM_RULES: dict[str, list[tuple[object, str]]] = {
    "ir_night_mode": [("on", r"\b(infra-?red|ir mode|night vision|ir camera)\b")],
    "compression": [
        ("high", r"\b(heavily compressed|blocky|low bitrate|grainy|very low quality)\b"),
        ("low", r"\b(high quality|crisp|sharp)\b"),
    ],
    "low_res_factor": [(0.5, r"\b(low[- ]res\w*|old camera|blurry|analog)\b")],
}

_CAMERA_RULES = [
    # (regex, field, value)
    (r"\b(high[- ]mounted|overhead|top[- ]down|bird'?s[- ]eye|high angle)\b", "camera_tilt_deg", 50.0),
    (r"\b(low[- ]mounted|low angle|eye level)\b", "camera_tilt_deg", 12.0),
    (r"\b(wide[- ]angle|fisheye|wide view)\b", "fov_deg", 95.0),
    (r"\b(narrow|zoomed|telephoto)\b", "fov_deg", 50.0),
]


@dataclass
class PromptInterpretation:
    prompt: str
    inferred_scene: dict = field(default_factory=dict)
    inferred_realism: dict = field(default_factory=dict)
    matched_terms: dict = field(default_factory=dict)
    conflicts: dict = field(default_factory=dict)
    ignored_terms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "inferred_scene": self.inferred_scene,
            "inferred_realism": self.inferred_realism,
            "matched_terms": self.matched_terms,
            "conflicts": self.conflicts,
            "ignored_terms": self.ignored_terms,
        }


_STOPWORDS = set(
    "a an the of in on at to with and or is are was were be by for from into near as it its this that "
    "some very camera cctv footage video clip scene shot view showing show shows person someone man woman "
    "people who while then there their his her they one two three during under over through".split()
)


def interpret(prompt: str) -> PromptInterpretation:
    text = (prompt or "").lower()
    out = PromptInterpretation(prompt=prompt or "")
    consumed: list[tuple[int, int]] = []

    def _apply(rules, target):
        for fld, options in rules.items():
            seen = []
            for value, pattern in options:
                for m in re.finditer(pattern, text):
                    consumed.append(m.span())
                    if fld not in target:
                        target[fld] = value
                        out.matched_terms[fld] = m.group(0)
                    if value not in seen:
                        seen.append(value)
            if len(seen) > 1:
                out.conflicts[fld] = {"chosen": target[fld], "also_mentioned": [v for v in seen if v != target[fld]]}

    _apply(_RULES, out.inferred_scene)
    _apply(_REALISM_RULES, out.inferred_realism)
    for pattern, fld, value in _CAMERA_RULES:
        m = re.search(pattern, text)
        if m and fld not in out.inferred_scene:
            out.inferred_scene[fld] = value
            out.matched_terms[fld] = m.group(0)
            consumed.append(m.span())

    # A crowd surge needs a crowd; outdoor-only weather is dropped indoors by the renderer.
    if out.inferred_scene.get("event") == "crowd_surge":
        out.inferred_scene.setdefault("crowd_density", "high")

    # Report words that were not understood, so the user knows they had no effect.
    for m in re.finditer(r"[a-z][a-z'-]+", text):
        if any(s <= m.start() < e for s, e in consumed):
            continue
        w = m.group(0)
        if w not in _STOPWORDS and len(w) > 2:
            out.ignored_terms.append(w)
    return out


# Typical mounting geometry per scene, used unless the caller sets camera fields.
SCENE_CAMERA_DEFAULTS = {
    "corridor": {"camera_height_m": 2.8, "camera_tilt_deg": 25.0},
    "parking_lot": {"camera_height_m": 6.0, "camera_tilt_deg": 20.0},
    "warehouse": {"camera_height_m": 7.0, "camera_tilt_deg": 22.0},
}


def resolve_request(payload: dict) -> tuple[GenerationRequest, PromptInterpretation]:
    """Build a GenerationRequest: defaults <- scene camera defaults <- prompt-inferred values <- explicit values."""
    payload = dict(payload or {})
    interp = interpret(payload.get("prompt", ""))
    explicit = dict(payload.get("scene") or {})
    scene_name = explicit.get("scene") or interp.inferred_scene.get("scene") or "corridor"
    scene = {**SCENE_CAMERA_DEFAULTS.get(scene_name, {}), **interp.inferred_scene, **explicit}
    realism = {**interp.inferred_realism, **(payload.get("realism") or {})}
    payload["scene"] = scene
    payload["realism"] = realism
    return GenerationRequest.from_dict(payload), interp
