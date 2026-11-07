"""Cross-camera trails: true-time placement, physical plausibility, where to look next."""

from datetime import timedelta

from cctvgpt.search.fusion import (
    Camera,
    Sighting,
    build_trail,
    haversine_km,
    parse_time,
    true_time,
    where_next,
)

# Three Bengaluru junctions, roughly 2 km and 9 km apart.
CAMS = {
    "silk_board": Camera("silk_board", 12.9172, 77.6229, 0.0, "traffic junction"),
    "hsr_bda": Camera("hsr_bda", 12.9116, 77.6412, 0.0, "traffic junction"),
    "mg_road": Camera("mg_road", 12.9756, 77.6050, -300.0, "shop DVR"),  # DVR runs 5 min fast
}
T0 = parse_time("2026-11-01T21:00:00")


def s(cam, minutes, score=1.0, sid="src"):
    return Sighting(cam, T0 + timedelta(minutes=minutes), score, sid, 0.0)


def test_naive_times_are_ist_and_camera_offset_applies():
    t = true_time("2026-11-01T21:00:00", 30.0, CAMS["mg_road"])
    assert t.utcoffset() == timedelta(hours=5, minutes=30)
    assert t == T0 + timedelta(seconds=30 - 300)


def test_distance_is_plausible():
    assert 1.5 < haversine_km(CAMS["silk_board"], CAMS["hsr_bda"]) < 2.5


def test_trail_keeps_possible_chain_and_flags_impossible_sighting():
    # Silk Board 21:00 -> HSR 21:05 (2 km in 5 min: fine). MG Road at 21:06 would need ~400 km/h: a misread.
    out = build_trail([s("silk_board", 0), s("hsr_bda", 5), s("mg_road", 6, score=0.85)], CAMS, max_kmh=80)
    assert [p["camera"] for p in out["trail"]] == ["silk_board", "hsr_bda"]
    assert len(out["conflicts"]) == 1 and out["conflicts"][0]["sighting"]["camera"] == "mg_road"
    assert out["conflicts"][0]["conflicts_with"]["needed_kmh"] > 80
    assert out["legs"][0]["km"] is not None


def test_untimed_sightings_are_not_placed():
    out = build_trail([s("silk_board", 0), Sighting("hsr_bda", None, 1.0, "x", 0.0)], CAMS)
    assert len(out["trail"]) == 1 and len(out["untimed"]) == 1


def test_where_next_lists_reachable_cameras_nearest_first():
    # HSR (~2 km) is reachable within 5 min at 60 km/h; MG Road (~6.8 km) needs ~7 min.
    assert [r["camera"] for r in where_next(s("silk_board", 0), CAMS, T0 + timedelta(minutes=5), max_kmh=60)] == ["hsr_bda"]
    assert [r["camera"] for r in where_next(s("silk_board", 0), CAMS, T0 + timedelta(minutes=10), max_kmh=60)] == ["hsr_bda", "mg_road"]
