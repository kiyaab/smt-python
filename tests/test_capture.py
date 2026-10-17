"""Unit tests for the video capture subsystem."""

import time
import numpy as np
import pytest

from signtranslator.capture.video_stream import FrameData, VideoStream


def test_video_stream_mock_mode_initialization():
    """Verify that VideoStream can initialize in mock mode without physical hardware."""
    stream = VideoStream(camera_index=0, width=320, height=240, fps_target=30, mock_mode=True)
    assert not stream.is_running
    assert stream.width == 320
    assert stream.height == 240


def test_video_stream_mock_capture_and_read():
    """Verify that VideoStream captures and emits valid FrameData in mock mode."""
    stream = VideoStream(camera_index=0, width=320, height=240, fps_target=30, mock_mode=True)
    started = stream.start()
    assert started
    assert stream.is_running

    # Read a few frames
    time.sleep(0.15)
    frame_data = stream.read(timeout=1.0)
    assert frame_data is not None
    assert isinstance(frame_data, FrameData)
    assert frame_data.frame.shape == (240, 320, 3)
    assert frame_data.frame.dtype == np.uint8
    assert frame_data.frame_index > 0
    assert frame_data.timestamp_ms > 0

    stream.stop()
    assert not stream.is_running


def test_video_stream_clean_stop():
    """Verify that stopping a stream cleans up queues and terminates the thread."""
    stream = VideoStream(mock_mode=True)
    stream.start()
    assert stream.is_running
    stream.stop()
    assert not stream.is_running
    # Second call to stop should be safe and idempotent
    stream.stop()
    assert not stream.is_running
