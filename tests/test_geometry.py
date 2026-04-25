from src.geometry import (
    circle_intersects_bbox,
    circle_intersects_polygon,
    find_gaze_hits,
    point_in_bbox,
    point_in_polygon,
)


def test_point_in_bbox_returns_true_for_inside_point():
    assert point_in_bbox(50, 60, [10, 20, 100, 120])


def test_point_in_bbox_includes_edges():
    assert point_in_bbox(10, 20, [10, 20, 100, 120])
    assert point_in_bbox(100, 120, [10, 20, 100, 120])


def test_point_in_bbox_returns_false_for_outside_point():
    assert not point_in_bbox(9, 60, [10, 20, 100, 120])
    assert not point_in_bbox(50, 121, [10, 20, 100, 120])


def test_point_in_bbox_ignores_malformed_bbox():
    assert not point_in_bbox(50, 60, [10, 20, 100])
    assert not point_in_bbox(50, 60, [100, 120, 10, 20])
    assert not point_in_bbox("x", 60, [10, 20, 100, 120])


def test_point_in_polygon_returns_true_for_inside_point_and_edges():
    polygon = [[10, 10], [100, 10], [100, 100], [10, 100]]

    assert point_in_polygon(50, 50, polygon)
    assert point_in_polygon(10, 50, polygon)


def test_point_in_polygon_returns_false_for_outside_point_or_bad_polygon():
    polygon = [[10, 10], [100, 10], [100, 100], [10, 100]]

    assert not point_in_polygon(9, 50, polygon)
    assert not point_in_polygon(50, 50, [[10, 10], [100, 10]])


def test_circle_intersects_bbox_for_nearby_points():
    assert circle_intersects_bbox(5, 50, 5, [10, 20, 100, 120])
    assert not circle_intersects_bbox(4, 50, 5, [10, 20, 100, 120])


def test_circle_intersects_polygon_for_nearby_points():
    polygon = [[10, 10], [100, 10], [100, 100], [10, 100]]

    assert circle_intersects_polygon(5, 50, 5, polygon)
    assert not circle_intersects_polygon(4, 50, 5, polygon)


def test_find_gaze_hits_returns_target_matches():
    detections = [
        {"label": "person", "bbox": [10, 20, 100, 120], "confidence": 0.8},
        {"label": "laptop", "bbox": [10, 20, 100, 120], "confidence": 0.9},
    ]

    hits = find_gaze_hits((50, 60), detections, {"person"}, 0.5)

    assert len(hits) == 1
    assert hits[0]["label"] == "person"
    assert hits[0]["hit_region"] == "bbox"


def test_find_gaze_hits_prefers_mask_polygons_when_available():
    detections = [
        {
            "label": "upper_torso",
            "bbox": [0, 0, 100, 100],
            "confidence": 0.9,
            "mask_polygons": [[[10, 10], [40, 10], [40, 40], [10, 40]]],
        },
    ]

    assert find_gaze_hits((80, 80), detections, {"upper_torso"}, 0.5) == []
    hits = find_gaze_hits((20, 20), detections, {"upper_torso"}, 0.5)
    assert len(hits) == 1
    assert hits[0]["hit_region"] == "mask"


def test_find_gaze_hits_can_use_gaze_radius_against_mask():
    detections = [
        {
            "label": "upper_torso",
            "bbox": [0, 0, 100, 100],
            "confidence": 0.9,
            "mask_polygons": [[[10, 10], [40, 10], [40, 40], [10, 40]]],
        },
    ]

    hits = find_gaze_hits((45, 20), detections, {"upper_torso"}, 0.5, gaze_radius_px=5)

    assert len(hits) == 1
    assert hits[0]["hit_region"] == "mask_radius"
    assert hits[0]["gaze_radius_px"] == 5.0


def test_find_gaze_hits_filters_low_confidence():
    detections = [
        {"label": "person", "bbox": [10, 20, 100, 120], "confidence": 0.49},
    ]

    assert find_gaze_hits((50, 60), detections, {"person"}, 0.5) == []


def test_find_gaze_hits_accepts_mapping_gaze_point():
    detections = [
        {"label": "person", "bbox": [10, 20, 100, 120], "confidence": 0.5},
    ]

    hits = find_gaze_hits({"x": 50, "y": 60}, detections, {"person"}, 0.5)

    assert len(hits) == 1


def test_find_gaze_hits_handles_missing_inputs():
    detections = [
        {"label": "person", "bbox": [10, 20, 100, 120], "confidence": 0.9},
    ]

    assert find_gaze_hits(None, detections, {"person"}, 0.5) == []
    assert find_gaze_hits((50, 60), None, {"person"}, 0.5) == []


def test_find_gaze_hits_ignores_bad_objects():
    detections = [
        None,
        "bad",
        {"label": "person", "bbox": [10, 20, 100], "confidence": 0.9},
        {"label": "person", "bbox": [10, 20, 100, 120], "confidence": "bad"},
        {"label": "person", "bbox": [200, 220, 300, 320], "confidence": 0.9},
    ]

    assert find_gaze_hits((50, 60), detections, {"person"}, 0.5) == []
