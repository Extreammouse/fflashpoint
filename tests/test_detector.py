from src.detector import clip_bbox_to_frame, label_from_names, make_detection, scale_bbox


def test_clip_bbox_to_frame_clips_to_screen_pixels():
    assert clip_bbox_to_frame([-10, 5, 110, 220], (100, 200)) == [
        0.0,
        5.0,
        99.0,
        199.0,
    ]


def test_clip_bbox_to_frame_rejects_invalid_boxes():
    assert clip_bbox_to_frame([10, 10, 5, 20], (100, 100)) is None
    assert clip_bbox_to_frame([10, 10, 20], (100, 100)) is None
    assert clip_bbox_to_frame([10, 10, 20, 20], (0, 100)) is None


def test_scale_bbox_expands_around_center_and_clips_to_frame():
    assert scale_bbox([10, 20, 30, 40], (100, 100), 1.1) == [9.0, 19.0, 31.0, 41.0]
    assert scale_bbox([0, 0, 10, 10], (100, 100), 1.2) == [0.0, 0.0, 11.0, 11.0]


def test_make_detection_uses_expected_schema():
    detection = make_detection("person", [1.234, 2.345, 30.456, 40.567], 0.91234, (100, 100))

    assert detection == {
        "label": "person",
        "bbox": [1.2, 2.3, 30.5, 40.6],
        "confidence": 0.9123,
    }


def test_make_detection_can_expand_bbox_and_mask_geometry():
    detection = make_detection(
        "upper_torso",
        [10, 20, 30, 40],
        0.9,
        (100, 100),
        mask_polygons=[[[10, 10], [20, 10], [20, 20], [10, 20]]],
        geometry_scale=1.1,
    )

    assert detection["bbox"] == [9.0, 19.0, 31.0, 41.0]
    assert detection["geometry_scale"] == 1.1
    assert detection["mask_polygons"][0][0] == [9.5, 9.5]


def test_label_from_names_accepts_dict_and_sequence():
    assert label_from_names({0: "person"}, 0) == "person"
    assert label_from_names(["person", "laptop"], 1) == "laptop"
    assert label_from_names({}, 99) == "99"
