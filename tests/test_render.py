import hashlib

import numpy as np
import pytest

from cctvgpt.params import EVENTS, RealismParams, SceneParams
from cctvgpt.realism import RealismProcessor
from cctvgpt.render.renderer import SimRenderer

from .conftest import FAST


def _digest(params: SceneParams) -> str:
    h = hashlib.sha256()
    for f, _ in SimRenderer(params).frames():
        h.update(f.tobytes())
    return h.hexdigest()


def test_same_seed_is_bit_identical_and_seed_matters():
    a = SceneParams(event="fall", seed=7, **FAST)
    assert _digest(a) == _digest(SceneParams(event="fall", seed=7, **FAST))
    assert _digest(a) != _digest(SceneParams(event="fall", seed=8, **FAST))


@pytest.mark.parametrize("event", [e for e in EVENTS if e != "none"])
@pytest.mark.parametrize("scene", ["corridor", "parking_lot", "warehouse"])
def test_every_event_occurs_with_ground_truth(scene, event):
    cam = {"corridor": (2.8, 25.0), "parking_lot": (6.0, 20.0), "warehouse": (7.0, 22.0)}[scene]
    p = SceneParams(scene=scene, event=event, camera_height_m=cam[0], camera_tilt_deg=cam[1], width=320, height=240, fps=6, duration_s=8.0, seed=3)
    r = SimRenderer(p)
    truth = [t for _, t in r.frames()]
    evs = r.event_annotations()
    assert len(evs) == 1 and evs[0]["type"] == event
    assert evs[0]["occurred"], evs[0]
    # The actors involved in the event are annotated in frames after it starts.
    ids = set(evs[0]["track_ids"])
    after = [o for t in truth[evs[0]["start_frame"]:] for o in t.objects if o["track_id"] in ids]
    assert after, "event actors never visible after event start"
    for t in truth:
        for o in t.objects:
            x, y, w, h = o["bbox"]
            assert 0 <= x and 0 <= y and x + w <= p.width and y + h <= p.height
            assert 0 <= o["visibility"] <= 1


def test_fall_state_progresses():
    p = SceneParams(event="fall", duration_s=8.0, fps=6, width=320, height=240, seed=5)
    r = SimRenderer(p)
    states = [o["state"] for _, t in r.frames() for o in t.objects if o["role"] == "victim"]
    assert "walking" in states and "fallen" in states
    assert states.index("fallen") > states.index("walking")


def test_fixed_camera_background_is_stable():
    """FR-8: with no actors and no weather, consecutive frames are identical."""
    p = SceneParams(event="none", crowd_density="empty", **FAST)
    frames = [f for f, _ in SimRenderer(p).frames()]
    assert all(np.array_equal(frames[0], f) for f in frames[1:])


def test_indoor_weather_is_ignored_with_note():
    r = SimRenderer(SceneParams(scene="corridor", weather="rain", **FAST))
    assert r.weather == "clear" and r.notes


def test_realism_ir_produces_monochrome_and_distortion_maps_boxes():
    p = SceneParams(time_of_day="night", **FAST)
    rp = RealismProcessor(RealismParams(lens_distortion=0.3, timestamp_overlay=False), p)
    f = np.full((p.height, p.width, 3), 120, np.uint8)
    out = rp.process(f, 0)
    assert np.abs(out[..., 0].astype(int) - out[..., 2].astype(int)).max() <= 2  # IR: grey
    # Barrel distortion pushes off-centre points towards the centre; centre is fixed.
    c = rp.distort_points([[p.width / 2, p.height / 2], [10, 10]])
    assert np.allclose(c[0], [p.width / 2, p.height / 2])
    assert c[1][0] > 10 and c[1][1] > 10
