from pathlib import Path

from src.gaze_model import L2CSAngleGazeModel, L2CSLocalKnnGazeModel, load_gaze_model


def test_l2cs_angle_gaze_model_fit_predict_save_load(tmp_path: Path):
    screen_size = (1200, 900)
    features = []
    targets = []

    for target_x, yaw in ((180.0, -0.24), (600.0, 0.0), (1020.0, 0.24)):
        for target_y, pitch in ((135.0, -0.18), (450.0, 0.0), (765.0, 0.18)):
            features.append([yaw, pitch])
            targets.append((target_x, target_y))

    model = L2CSAngleGazeModel.fit(
        features,
        targets,
        screen_size=screen_size,
        degree=2,
        ridge_lambda=0.0001,
        range_margin=0.1,
        validation_error_px=25.0,
    )
    point, debug = model.predict_with_diagnostics([0.0, 0.0])

    assert abs(point[0] - 600.0) < 2.0
    assert abs(point[1] - 450.0) < 2.0
    assert debug["out_of_calibration_range"] is False
    assert model.signal_stats["yaw_range_rad"] > 0
    assert model.validation_error_px == 25.0

    _point, debug = model.predict_with_diagnostics([0.8, 0.0])
    assert debug["out_of_calibration_range"] is True
    assert 0 in debug["out_of_range_feature_indices"]

    path = tmp_path / "l2cs_calibration.json"
    model.save(path)
    loaded = load_gaze_model(path)
    loaded_point = loaded.predict([-0.24, -0.18])

    assert abs(loaded_point[0] - 180.0) < 2.0
    assert abs(loaded_point[1] - 135.0) < 2.0


def test_l2cs_local_knn_gaze_model_fit_predict_save_load(tmp_path: Path):
    screen_size = (1200, 900)
    features = []
    targets = []

    for target_x, yaw in (
        (120.0, -0.30),
        (360.0, -0.15),
        (600.0, 0.0),
        (840.0, 0.15),
        (1080.0, 0.30),
    ):
        for target_y, pitch in (
            (90.0, -0.24),
            (270.0, -0.12),
            (450.0, 0.0),
            (630.0, 0.12),
            (810.0, 0.24),
        ):
            features.append([yaw, pitch])
            targets.append((target_x, target_y))

    model = L2CSLocalKnnGazeModel.fit(
        features,
        targets,
        screen_size=screen_size,
        neighbor_count=4,
        range_margin=0.08,
        validation_error_px=18.0,
    )
    point, debug = model.predict_with_diagnostics([0.0, 0.0])

    assert abs(point[0] - 600.0) < 2.0
    assert abs(point[1] - 450.0) < 2.0
    assert debug["out_of_calibration_range"] is False

    _point, debug = model.predict_with_diagnostics([0.65, 0.0])
    assert debug["out_of_calibration_range"] is True
    assert 0 in debug["out_of_range_feature_indices"]

    path = tmp_path / "l2cs_local_knn_calibration.json"
    model.save(path)
    loaded = load_gaze_model(path)
    loaded_point = loaded.predict([-0.30, -0.24])

    assert abs(loaded_point[0] - 120.0) < 2.0
    assert abs(loaded_point[1] - 90.0) < 2.0
