import calibrate_gaze


def test_should_finish_collection_waits_for_min_samples_until_hard_cap():
    assert (
        calibrate_gaze._should_finish_collection(
            elapsed=2.1,
            total_seconds=2.0,
            max_total_seconds=5.0,
            sample_count=5,
            min_samples=6,
        )
        is False
    )

    assert (
        calibrate_gaze._should_finish_collection(
            elapsed=2.1,
            total_seconds=2.0,
            max_total_seconds=5.0,
            sample_count=6,
            min_samples=6,
        )
        is True
    )

    assert (
        calibrate_gaze._should_finish_collection(
            elapsed=5.0,
            total_seconds=2.0,
            max_total_seconds=5.0,
            sample_count=5,
            min_samples=6,
        )
        is True
    )


def test_max_sample_seconds_for_calibration_uses_l2cs_window():
    assert calibrate_gaze.max_sample_seconds_for_calibration() >= 5.0


def test_dense_profile_has_5x5_grid_plus_corner_refinement():
    profile = calibrate_gaze.calibration_profile((2000, 1000), "dense5x5")

    assert profile.name == "dense5x5"
    assert profile.selection_mode == "corner_aware"
    assert len(profile.targets) == 29
    assert sum(1 for target in profile.targets if target.stage == "refinement") == 4
    assert sum(1 for target in profile.targets if target.emphasis == "corner") == 8


def test_corner_aware_selection_prefers_lower_corner_error():
    profile = calibrate_gaze.calibration_profile((2000, 1000), "dense5x5")
    result_a = {
        "validation_average_error_px": 120.0,
        "training_error_px": 80.0,
        "point_errors_px": [220.0 if target.emphasis == "corner" else 90.0 for target in profile.targets],
    }
    result_b = {
        "validation_average_error_px": 126.0,
        "training_error_px": 82.0,
        "point_errors_px": [140.0 if target.emphasis == "corner" else 98.0 for target in profile.targets],
    }

    assert calibrate_gaze._candidate_selection_score(
        result_b,
        profile.selection_mode,
        profile.targets,
    ) < calibrate_gaze._candidate_selection_score(
        result_a,
        profile.selection_mode,
        profile.targets,
    )
