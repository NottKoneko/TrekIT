"""
test_utils.py
-------------
Unit tests for frame quality scoring, homography rectification, and regional syntax validation.
"""

import numpy as np
from traffic_tracker.syntax_validator import RegionalSyntaxValidator, default_syntax_validator
from traffic_tracker.utils import (
    assess_image_quality,
    rectify_plate_quad,
    score_plate_keyframe,
)


def test_assess_image_quality():
    # Test blank / flat image
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    var, q_sharp, lum_bal = assess_image_quality(flat)
    assert var == 0.0
    assert q_sharp == 0.0
    assert lum_bal == 1.0

    # Test high-contrast textured image
    checker = np.zeros((100, 100, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    var_c, q_sharp_c, _ = assess_image_quality(checker)
    assert var_c > 10.0
    assert q_sharp_c > 0.0


def test_rectify_plate_quad():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    corners = np.array([[20, 30], [180, 25], [175, 85], [15, 80]], dtype=np.float32)
    warped = rectify_plate_quad(img, corners, target_size=(94, 24))
    assert warped is not None
    assert warped.shape == (24, 94, 3)


def test_score_plate_keyframe():
    crop = np.random.randint(0, 255, (30, 90, 3), dtype=np.uint8)
    score = score_plate_keyframe(
        conf_det=0.90,
        crop_bgr=crop,
        frame_shape=(720, 1280),
        plate_bbox=(100, 100, 190, 130),
    )
    assert 0.0 <= score <= 2.0


def test_regional_syntax_validator_us_ca():
    val = RegionalSyntaxValidator(default_jurisdiction="us-ca")

    # Standard CA 7-char: 1 digit - 3 letters - 3 digits
    res, is_valid, jur, w = val.validate_and_correct("9GAD429")
    assert res == "9GAD429"
    assert is_valid is True
    assert jur == "us-ca"

    # Disambiguation: Letter in digit position 0 (J -> 9)
    res_j, is_valid_j, _, _ = val.validate_and_correct("JGAD429")
    assert res_j == "9GAD429"
    assert is_valid_j is True

    # Disambiguation: Digit in letter position 1 (4 -> A)
    res_a, is_valid_a, _, _ = val.validate_and_correct("94AD429")
    assert res_a == "9AAD429"
    assert is_valid_a is True

    # Disambiguation: Letter in digit position 5 (Z -> 2)
    res_z, is_valid_z, _, _ = val.validate_and_correct("9GAD4Z9")
    assert res_z == "9GAD429"
    assert is_valid_z is True


def test_rescore_candidates():
    val = RegionalSyntaxValidator(default_jurisdiction="us-ca")
    candidates = [
        ("9G4D429", 0.70),  # Valid CA after 4->A correction
        ("BBBBBBB", 0.95),  # Repetitive hallucination
    ]
    rescored = val.rescore_candidates(candidates)
    assert len(rescored) == 1
    assert rescored[0][0] == "9GAD429"
