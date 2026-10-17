"""Thread-safe video capture with FPS control and frame buffer management."""

from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class FrameData:
    """Encapsulates a captured video frame and its associated metadata."""

    frame: np.ndarray
    timestamp_ms: int
    frame_index: int
    fps: float


class VideoStream:
    """Manages webcam capture in a decoupled background thread with drop-oldest buffering."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps_target: int = 30,
        flip_horizontal: bool = True,
        mock_mode: bool = False,
    ) -> None:
        """Initialize the video stream.

        Args:
            camera_index: Device index for cv2.VideoCapture.
            width: Target frame width.
            height: Target frame height.
            fps_target: Target frame rate limit.
            flip_horizontal: If True, horizontally mirror frames for natural signer view.
            mock_mode: If True, generate synthetic test frames without physical camera.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps_target = max(1, fps_target)
        self.frame_interval = 1.0 / self.fps_target
        self.flip_horizontal = flip_horizontal
        self.mock_mode = mock_mode

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()

        # Performance metrics
        self._frame_count = 0
        self._dropped_count = 0
        self._measured_fps = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_frame_counter = 0

    @property
    def is_running(self) -> bool:
        """Return True if the background capture thread is currently active."""
        return self._running.is_set()

    @property
    def measured_fps(self) -> float:
        """Return the current measured frame rate."""
        with self._lock:
            return self._measured_fps

    def start(self) -> bool:
        """Start the video capture thread. Returns True if started successfully."""
        if self.is_running:
            return True

        if not self.mock_mode:
            # Try DirectShow on Windows for fast webcam init
            self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default backend
                self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="VideoStreamThread")
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the capture thread and release hardware resources."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        # Empty the frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: float = 0.1) -> Optional[FrameData]:
        """Retrieve the latest frame from the buffer. Non-blocking with timeout."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        """Worker loop running in background thread."""
        next_frame_time = time.time()

        while self._running.is_set():
            now = time.time()
            if now < next_frame_time:
                time.sleep(max(0.001, next_frame_time - now))

            next_frame_time = time.time() + self.frame_interval

            if self.mock_mode:
                frame = self._generate_mock_frame()
                ret = True
            else:
                if self._cap is None or not self._cap.isOpened():
                    break
                ret, frame = self._cap.read()

            if not ret or frame is None:
                time.sleep(0.05)
                continue

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            # Ensure frame dimensions match configured target
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            self._frame_count += 1
            self._fps_frame_counter += 1

            # Update measured FPS once per second
            elapsed = time.time() - self._last_fps_calc_time
            if elapsed >= 1.0:
                with self._lock:
                    self._measured_fps = self._fps_frame_counter / elapsed
                self._fps_frame_counter = 0
                self._last_fps_calc_time = time.time()

            frame_data = FrameData(
                frame=frame,
                timestamp_ms=int(time.time() * 1000),
                frame_index=self._frame_count,
                fps=self._measured_fps,
            )

            # Drop-oldest queue policy
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self._dropped_count += 1
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame_data)
            except queue.Full:
                pass

    def _generate_mock_frame(self) -> np.ndarray:
        """Generate a synthetic frame for testing without physical webcam."""
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background gradient
        img[:, :] = [30, 30, 35]
        # Draw a synthetic moving target
        t = time.time()
        cx = int(self.width / 2 + 100 * np.sin(t * 2))
        cy = int(self.height / 2 + 60 * np.cos(t * 2))
        cv2.circle(img, (cx, cy), 40, (80, 180, 240), -1)
        cv2.putText(
            img,
            f"Mock VideoStream - Frame #{self._frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return img
