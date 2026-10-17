"""Unit tests for the MediaPipe landmark tracking subsystem."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from signtranslator.tracking.asset_manager import ModelAssetManager
from signtranslator.tracking.landmark_tracker import LandmarkTracker
from signtranslator.tracking.normalization import FrameLandmarks


def test_asset_manager_file_resolution():
    """Verify ModelAssetManager correctly identifies and resolves model paths."""
    manager = ModelAssetManager("models/mediapipe")
    hand_path = manager.get_model_path("hand_landmarker.task")
    assert isinstance(hand_path, Path)
    assert hand_path.exists()
    assert hand_path.stat().st_size > 0


def test_landmark_tracker_initialization():
    """Verify LandmarkTracker initializes all detectors without error."""
    tracker = LandmarkTracker(model_dir="models/mediapipe")
    assert tracker.hand_detector is not None
    assert tracker.pose_detector is not None
    assert tracker.face_detector is not None


def test_landmark_tracker_frame_processing():
    """Verify LandmarkTracker processes a frame and returns a valid FrameLandmarks dataclass."""
    tracker = LandmarkTracker(model_dir="models/mediapipe")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    landmarks = tracker.process_frame(img)
    assert isinstance(landmarks, FrameLandmarks)
    # Empty frame should have None for hands
    assert landmarks.left_hand is None
    assert landmarks.right_hand is None


def test_landmark_tracker_drawing():
    """Verify draw_landmarks overlays skeletons onto image without altering dimensions."""
    tracker = LandmarkTracker(model_dir="models/mediapipe")
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Synthetic landmarks
    hand_pts = np.linspace(0.4, 0.6, 21 * 3).reshape(21, 3).astype(np.float32)
    fake_landmarks = FrameLandmarks(
        right_hand=hand_pts,
        pose=np.ones((6, 3), dtype=np.float32) * 0.5,
    )

    annotated = tracker.draw_landmarks(img, fake_landmarks)
    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype
    # The annotated image should contain non-zero pixel colors drawn on it
    assert np.any(annotated > 0)
