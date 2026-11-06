import json

import pytest

from cctvgpt.evaluation import average_precision, iou, scorecard, scorecard_html
from cctvgpt.params import HARD_NEGATIVE_EVENTS, ParamError, SceneParams
from cctvgpt.pipeline import generate
from cctvgpt.prompt import interpret, resolve_request
from cctvgpt.render.renderer import SimRenderer

from .conftest import FAST


def test_iou_and_ap():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert average_precision([0.9, 0.8], [1, 1], 2) == pytest.approx(1.0)
    assert average_precision([0.9, 0.8], [0, 1], 2) == pytest.approx(0.25)


@pytest.mark.parametrize("event,state", [("crouch", "crouching"), ("lie_down", "lying"), ("walk", "walking")])
def test_hard_negative_events_are_non_alarm(event, state):
    assert event in HARD_NEGATIVE_EVENTS
    r = SimRenderer(SceneParams(event=event, crowd_density="empty", duration_s=8.0, fps=6, width=320, height=240, seed=1))
    states = {o["state"] for _, t in r.frames() for o in t.objects}
    ev = r.event_annotations()[0]
    assert ev["type"] == event and ev["alarm"] is False
    assert state in states
    assert "fallen" not in states  # a deliberate lie-down is never labelled as a fall


def test_alarm_events_flagged():
    r = SimRenderer(SceneParams(event="fall", **FAST))
    assert r.event_annotations()[0]["alarm"] is True


def test_prompt_maps_hard_negatives():
    assert interpret("a man crouches to tie his shoe in the corridor").inferred_scene["event"] == "crouch"
    assert interpret("someone lies down to rest in the warehouse").inferred_scene["event"] == "lie_down"


def test_event_distance_moves_the_event():
    def victim_height(d):
        r = SimRenderer(SceneParams(event="fall", crowd_density="empty", event_distance_m=d, duration_s=4.0, fps=4, width=320, height=240, seed=2))
        hs = [o["bbox"][3] for _, t in r.frames() for o in t.objects if o["role"] == "victim"]
        return max(hs)
    assert victim_height(4.0) > 2 * victim_height(12.0)
    with pytest.raises(ParamError):
        SceneParams(event_distance_m=100).validate()


def _gt_as_predictions(ann, drop_state=None, false_event=False):
    frames = [{"frame": f["frame"], "detections": [{"bbox": o["bbox"], "score": 0.9, "class": o["class"]}
                                                    for o in f["objects"] if o["visibility"] >= 0.3 and o.get("state") != drop_state]}
              for f in ann["frames"]]
    events = [{"type": e["type"], "start_frame": e["start_frame"]} for e in ann["events"] if e.get("alarm") and e["occurred"]]
    if false_event:
        events.append({"type": "fall", "start_frame": 0})
    return {"frames": frames, "events": events}


def test_scorecard_end_to_end(platform, customer):
    anns = {}
    for ev, tod in (("fall", "day"), ("lie_down", "night")):
        req, interp = resolve_request({"prompt": "", "scene": {**FAST, "duration_s": 6.0, "event": ev, "time_of_day": tod, "crowd_density": "empty"}})
        res = generate(platform, customer["id"], "u", req, interp)
        anns[res.clip_id] = json.loads(res.annotations_path.read_text())
    perfect = scorecard(anns, {c: _gt_as_predictions(a) for c, a in anns.items()})
    person = perfect["overall"]["detection"]["person"]
    assert person["recall"] == 1.0 and person["precision"] == 1.0 and person["ap50"] == pytest.approx(1.0)
    assert perfect["overall"]["events"]["fall"]["rate"] == 1.0
    assert perfect["overall"]["false_alarms"] == 0
    assert set(perfect["by_condition"]["time_of_day"]) == {"day", "night"}
    # A detector that misses fallen people and raises an alarm on a lie-down.
    flawed = scorecard(anns, {c: _gt_as_predictions(a, drop_state="fallen", false_event=(a["events"][0]["type"] == "lie_down")) for c, a in anns.items()})
    assert flawed["overall"]["detection"]["person"]["recall_by_state"]["fallen"] == 0.0
    assert flawed["overall"]["false_alarms"] == 1
    assert "<table>" in scorecard_html(flawed)


def test_manifest_has_platform_uid(platform, customer):
    req, interp = resolve_request({"scene": FAST})
    res = generate(platform, customer["id"], "u", req, interp)
    assert res.manifest["platform_uid"].startswith("in.vidyuthlabs.cctvgpt:")
