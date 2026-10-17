"""Unit tests for the landmark normalization and feature pipeline."""

import numpy as np
import pytest

from signtranslator.tracking.normalization import FrameLandmarks, LandmarkNormalizer


def test_empty_frame_normalization():
    """Verify that an empty frame (no hands/pose/face) produces zero-padded vector with false presence flags."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)
    empty_frame = FrameLandmarks()
    features = normalizer.normalize_frame(empty_frame)

    assert isinstance(features, np.ndarray)
    assert features.shape == (LandmarkNormalizer.TOTAL_STATIC_FEATURES,)
    assert np.all(features == 0.0)


def test_hand_normalization_translation_invariance():
    """Verify that shifting a hand in camera space results in identical normalized coordinates."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)

    # Hand at position A (wrist at 0.5, 0.5, 0.0)
    hand_a = np.zeros((21, 3), dtype=np.float32)
    hand_a[0] = [0.5, 0.5, 0.0]
    hand_a[9] = [0.5, 0.4, 0.0]  # Middle MCP 0.1 units away

    # Hand at position B (wrist at 0.2, 0.8, 0.1) with identical shape
    hand_b = hand_a.copy()
    shift = np.array([-0.3, 0.3, 0.1], dtype=np.float32)
    hand_b += shift

    frame_a = FrameLandmarks(right_hand=hand_a)
    frame_b = FrameLandmarks(right_hand=hand_b)

    feat_a = normalizer.normalize_frame(frame_a)
    feat_b = normalizer.normalize_frame(frame_b)

    # Right hand features are at indices 64:127
    rh_a = feat_a[64:127]
    rh_b = feat_b[64:127]

    assert np.allclose(rh_a, rh_b, atol=1e-5)


def test_hand_normalization_scale_invariance():
    """Verify that scaling hand size produces identical normalized coordinates."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)

    # Base hand shape
    wrist = np.array([0.5, 0.5, 0.1], dtype=np.float32)
    offsets = np.random.randn(21, 3).astype(np.float32) * 0.05
    offsets[0] = 0.0
    offsets[9] = [0.0, -0.06, 0.0]  # Middle finger MCP

    hand_small = wrist + offsets
    # Hand scaled by 2x relative to wrist
    hand_large = wrist + (offsets * 2.0)

    frame_s = FrameLandmarks(right_hand=hand_small)
    frame_l = FrameLandmarks(right_hand=hand_large)

    feat_s = normalizer.normalize_frame(frame_s)
    feat_l = normalizer.normalize_frame(frame_l)

    rh_s = feat_s[64:127]
    rh_l = feat_l[64:127]

    assert np.allclose(rh_s, rh_l, atol=1e-5)


def test_resample_sequence():
    """Verify that resample_sequence adjusts temporal length correctly."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)
    orig_seq = np.random.randn(45, LandmarkNormalizer.TOTAL_STATIC_FEATURES).astype(np.float32)

    resampled = normalizer.resample_sequence(orig_seq, target_length=30)
    assert resampled.shape == (30, LandmarkNormalizer.TOTAL_STATIC_FEATURES)

    # Resampling shorter sequence
    short_seq = np.random.randn(15, LandmarkNormalizer.TOTAL_STATIC_FEATURES).astype(np.float32)
    upsampled = normalizer.resample_sequence(short_seq, target_length=30)
    assert upsampled.shape == (30, LandmarkNormalizer.TOTAL_STATIC_FEATURES)


def test_mirror_hand_sequence():
    """Verify that mirroring flips x-coordinates and swaps left/right hands."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)

    # Create sequence where ONLY right hand is present
    frame = FrameLandmarks(
        right_hand=np.ones((21, 3), dtype=np.float32) * 0.5
    )
    matrix = np.tile(normalizer.normalize_frame(frame), (30, 1))

    # Before mirroring: Left hand absent (flag index 63 == 0), Right hand present (flag index 127 == 1)
    assert matrix[0, 63] == 0.0
    assert matrix[0, 127] == 1.0

    mirrored = normalizer.mirror_hand_sequence(matrix)

    # After mirroring: Left hand present (flag index 63 == 1), Right hand absent (flag index 127 == 0)
    assert mirrored[0, 63] == 1.0
    assert mirrored[0, 127] == 0.0


def test_motion_energy_calculation():
    """Verify motion energy correctly distinguishes static from moving sequences."""
    normalizer = LandmarkNormalizer(target_sequence_length=30)

    # Static sequence: all frames identical
    static_seq = np.ones((30, LandmarkNormalizer.TOTAL_STATIC_FEATURES), dtype=np.float32)
    energy_static = normalizer.compute_motion_energy(static_seq)
    assert energy_static == 0.0

    # Dynamic sequence: frame-to-frame change
    moving_seq = static_seq.copy()
    for i in range(30):
        moving_seq[i, :] += i * 0.1
    energy_moving = normalizer.compute_motion_energy(moving_seq)
    assert energy_moving > 0.0
