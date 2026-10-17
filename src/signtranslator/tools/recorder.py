"""PyQt6 Data Recorder tool for collecting landmark sequences with metadata."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

# Auto-resolve src directory so the script can be run directly
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import cv2
import numpy as np
from PyQt6.QtCore import QBasicTimer, QObject, QPoint, QRect, QSize, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import yaml

from signtranslator.capture.video_stream import VideoStream
from signtranslator.tracking.landmark_tracker import LandmarkTracker
from signtranslator.tracking.normalization import FrameLandmarks, LandmarkNormalizer


class CaptureWorker(QThread):
    """Background worker thread running video capture and MediaPipe landmark tracking."""

    frame_ready = pyqtSignal(np.ndarray, object, float)  # frame_bgr, landmarks, fps
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera_index: int = 0,
        model_dir: str = "models/mediapipe",
        mock_mode: bool = False,
    ) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.model_dir = model_dir
        self.mock_mode = mock_mode
        self._running = True

    def run(self) -> None:
        stream = VideoStream(camera_index=self.camera_index, mock_mode=self.mock_mode)
        if not stream.start():
            # If camera fails, switch automatically to mock mode for preview resilience
            stream.stop()
            stream = VideoStream(camera_index=self.camera_index, mock_mode=True)
            stream.start()

        try:
            tracker = LandmarkTracker(model_dir=self.model_dir)
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize tracker: {e}")
            stream.stop()
            return

        while self._running:
            frame_data = stream.read(timeout=0.05)
            if frame_data is None:
                continue

            try:
                landmarks = tracker.process_frame(frame_data.frame)
                annotated = tracker.draw_landmarks(frame_data.frame, landmarks)
                self.frame_ready.emit(annotated, landmarks, frame_data.fps)
            except Exception as e:
                # Still emit raw frame if tracker encounters a hiccup
                self.frame_ready.emit(frame_data.frame, FrameLandmarks(), frame_data.fps)

        stream.stop()

    def stop(self) -> None:
        self._running = False
        self.wait(1000)


class DataRecorderWindow(QMainWindow):
    """Main window for recording sign language landmark training sequences."""

    def __init__(
        self,
        config_path: str = "config/default.yaml",
        vocab_path: str = "config/vocabulary.yaml",
        output_dir: str = "data/raw",
        mock_mode: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Sign Language Data Recorder - Inclusive Communication")
        self.resize(1150, 750)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = LandmarkNormalizer(target_sequence_length=30)
        self.mock_mode = mock_mode

        # Load vocabulary
        self.vocab: List[Dict] = []
        if Path(vocab_path).exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.vocab = data.get("glosses", [])
        if not self.vocab:
            self.vocab = [{"gloss": "HELLO", "display_name": "Hello", "category": "greetings"}]

        # State variables
        self.is_recording = False
        self.countdown_val = 0
        self.recorded_buffer: List[FrameLandmarks] = []
        self.last_saved_file: Optional[Path] = None

        self._init_ui()
        self._start_capture_thread()

    def _init_ui(self) -> None:
        """Construct the UI layout and widgets."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left Column: Video Preview & Recording Status
        left_layout = QVBoxLayout()

        # Camera Header Bar
        cam_header = QHBoxLayout()
        self.status_badge = QLabel(" CAMERA ACTIVE ")
        self.status_badge.setStyleSheet(
            "background-color: #00aa44; color: white; font-weight: bold; border-radius: 4px; padding: 4px 8px;"
        )
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #aaaaaa; font-family: monospace;")
        cam_header.addWidget(self.status_badge)
        cam_header.addStretch()
        cam_header.addWidget(self.fps_label)
        left_layout.addLayout(cam_header)

        # Video Viewport
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #121214; border: 2px solid #2a2a2e; border-radius: 8px;")
        left_layout.addWidget(self.video_label)

        # Recording Status & Countdown Banner
        self.banner_label = QLabel("Ready to record")
        self.banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.banner_label.setStyleSheet(
            "background-color: #222226; color: #eeeeee; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 6px;"
        )
        left_layout.addWidget(self.banner_label)

        # Progress bar during take
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 30)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background-color: #222226; border-radius: 4px; }"
            "QProgressBar::chunk { background-color: #00aaff; border-radius: 4px; }"
        )
        left_layout.addWidget(self.progress_bar)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_record = QPushButton("Record Take (Space)")
        self.btn_record.setFixedHeight(45)
        self.btn_record.setStyleSheet(
            "background-color: #007acc; color: white; font-size: 14px; font-weight: bold; border-radius: 6px;"
        )
        self.btn_record.clicked.connect(self._trigger_countdown)

        self.btn_discard = QPushButton("Discard Last Take (Del)")
        self.btn_discard.setFixedHeight(45)
        self.btn_discard.setStyleSheet(
            "background-color: #aa3333; color: white; font-size: 14px; font-weight: bold; border-radius: 6px;"
        )
        self.btn_discard.clicked.connect(self._discard_last_take)
        self.btn_discard.setEnabled(False)

        btn_layout.addWidget(self.btn_record)
        btn_layout.addWidget(self.btn_discard)
        left_layout.addLayout(btn_layout)

        main_layout.addLayout(left_layout, stretch=3)

        # Right Column: Metadata & Controls
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #1a1a1e; border-radius: 8px; padding: 12px;")
        right_layout = QVBoxLayout(right_panel)

        # Group 1: Signer & Environment
        signer_box = QGroupBox("Signer & Environment")
        signer_box.setStyleSheet("color: white; font-weight: bold;")
        s_layout = QFormLayout(signer_box)

        self.signer_input = QLineEdit("signer_01")
        self.signer_input.setStyleSheet("background-color: #2a2a2e; color: white; padding: 6px; border-radius: 4px;")

        self.lighting_combo = QComboBox()
        self.lighting_combo.addItems(["Normal / Office", "Bright Sunlight", "Dim / Low Light"])
        self.lighting_combo.setStyleSheet(
            "background-color: #2a2a2e; color: white; padding: 6px; border-radius: 4px;"
        )

        self.hand_combo = QComboBox()
        self.hand_combo.addItems(["Right Dominant", "Left Dominant", "Both Hands"])
        self.hand_combo.setStyleSheet("background-color: #2a2a2e; color: white; padding: 6px; border-radius: 4px;")

        s_layout.addRow("Signer ID:", self.signer_input)
        s_layout.addRow("Lighting:", self.lighting_combo)
        s_layout.addRow("Dominance:", self.hand_combo)
        right_layout.addWidget(signer_box)

        # Group 2: Gloss Selection
        gloss_box = QGroupBox("Select Sign Gloss")
        gloss_box.setStyleSheet("color: white; font-weight: bold;")
        g_layout = QVBoxLayout(gloss_box)

        self.gloss_combo = QComboBox()
        for item in self.vocab:
            display = f"{item['gloss']} ({item.get('display_name', '')})"
            self.gloss_combo.addItem(display, item["gloss"])
        self.gloss_combo.setStyleSheet("background-color: #2a2a2e; color: white; padding: 8px; border-radius: 4px;")
        self.gloss_combo.currentIndexChanged.connect(self._update_stats)
        g_layout.addWidget(self.gloss_combo)

        self.gloss_count_label = QLabel("Takes recorded for this gloss: 0")
        self.gloss_count_label.setStyleSheet("color: #00ccff; font-size: 13px;")
        g_layout.addWidget(self.gloss_count_label)
        right_layout.addWidget(gloss_box)

        # Group 3: Session Stats
        stats_box = QGroupBox("Collection Summary")
        stats_box.setStyleSheet("color: white; font-weight: bold;")
        st_layout = QVBoxLayout(stats_box)

        self.total_takes_label = QLabel("Total takes in database: 0")
        self.total_takes_label.setStyleSheet("color: #dddddd; font-size: 13px;")
        st_layout.addWidget(self.total_takes_label)

        self.last_saved_label = QLabel("Last saved: None")
        self.last_saved_label.setStyleSheet("color: #888888; font-size: 11px;")
        st_layout.addWidget(self.last_saved_label)
        right_layout.addWidget(stats_box)

        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=2)

        # Apply dark theme styling
        self.setStyleSheet(
            "QMainWindow { background-color: #121214; }"
            "QLabel { color: #eeeeee; }"
            "QLineEdit { color: #ffffff; border: 1px solid #3a3a3e; }"
            "QComboBox { color: #ffffff; border: 1px solid #3a3a3e; }"
        )

        self._update_stats()

    def _start_capture_thread(self) -> None:
        """Start the background camera and landmark worker thread."""
        self.worker = CaptureWorker(mock_mode=self.mock_mode)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.error_occurred.connect(self._on_error)
        self.worker.start()

    @pyqtSlot(np.ndarray, object, float)
    def _on_frame_ready(self, frame_bgr: np.ndarray, landmarks: FrameLandmarks, fps: float) -> None:
        """Handle incoming frame from worker thread."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

        # If currently recording, append landmarks
        if self.is_recording:
            self.recorded_buffer.append(landmarks)
            self.progress_bar.setValue(len(self.recorded_buffer))

            # Stop when reached target length (e.g. 30 frames ~ 1.0s)
            if len(self.recorded_buffer) >= 30:
                self._finish_recording()

        # Render countdown on frame if active
        display_frame = frame_bgr.copy()
        if self.countdown_val > 0:
            h, w, _ = display_frame.shape
            cv2.putText(
                display_frame,
                str(self.countdown_val),
                (w // 2 - 40, h // 2 + 40),
                cv2.FONT_HERSHEY_DUPLEX,
                4.0,
                (0, 220, 255),
                8,
                cv2.LINE_AA,
            )

        # Convert to QPixmap
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @pyqtSlot(str)
    def _on_error(self, err_msg: str) -> None:
        self.status_badge.setText(" ERROR ")
        self.status_badge.setStyleSheet("background-color: #cc0000; color: white;")
        QMessageBox.warning(self, "Camera / Tracker Error", err_msg)

    def _trigger_countdown(self) -> None:
        """Start a 3-second visual countdown before recording starts."""
        if self.is_recording or self.countdown_val > 0:
            return

        self.btn_record.setEnabled(False)
        self.btn_discard.setEnabled(False)
        self.countdown_val = 3
        self.banner_label.setText("Get ready... 3")
        self.banner_label.setStyleSheet("background-color: #ffaa00; color: #111111; font-weight: bold; font-size: 16px;")

        self._countdown_step()

    def _countdown_step(self) -> None:
        """Advance countdown by 1 second."""
        if self.countdown_val > 1:
            self.countdown_val -= 1
            self.banner_label.setText(f"Get ready... {self.countdown_val}")
            QTimer.singleShot(1000, self._countdown_step)
        else:
            self.countdown_val = 0
            self._start_recording()

    def _start_recording(self) -> None:
        """Begin saving landmark frames."""
        self.is_recording = True
        self.recorded_buffer = []
        self.progress_bar.setValue(0)
        self.banner_label.setText("● RECORDING GESTURE...")
        self.banner_label.setStyleSheet("background-color: #cc2222; color: #ffffff; font-weight: bold; font-size: 16px;")

    def _finish_recording(self) -> None:
        """Finalize and save the recorded take."""
        self.is_recording = False
        self.banner_label.setText("Processing & saving take...")
        self.banner_label.setStyleSheet("background-color: #007acc; color: #ffffff; font-weight: bold; font-size: 16px;")

        gloss = self.gloss_combo.currentData()
        signer_id = self.signer_input.text().strip() or "signer_01"
        lighting = self.lighting_combo.currentText()
        hand_dom = self.hand_combo.currentText()

        # Normalize landmarks
        norm_matrix = self.normalizer.normalize_sequence(self.recorded_buffer)

        # Save to disk
        timestamp = int(time.time() * 1000)
        filename = f"{gloss}_{signer_id}_{timestamp}.npz"
        filepath = self.output_dir / filename

        meta = {
            "gloss": gloss,
            "signer_id": signer_id,
            "lighting": lighting,
            "dominant_hand": hand_dom,
            "timestamp": timestamp,
            "raw_frames": len(self.recorded_buffer),
            "is_synthetic": False,
        }

        np.savez_compressed(filepath, sequence=norm_matrix, metadata_json=json.dumps(meta))
        self.last_saved_file = filepath

        self.banner_label.setText(f"Take saved: {filename}")
        self.banner_label.setStyleSheet("background-color: #00aa44; color: #ffffff; font-weight: bold; font-size: 16px;")
        self.btn_record.setEnabled(True)
        self.btn_discard.setEnabled(True)

        self._update_stats()

    def _discard_last_take(self) -> None:
        """Delete the most recently recorded take."""
        if self.last_saved_file is not None and self.last_saved_file.exists():
            try:
                self.last_saved_file.unlink()
                self.banner_label.setText(f"Discarded take: {self.last_saved_file.name}")
                self.banner_label.setStyleSheet("background-color: #555555; color: #ffffff; font-size: 14px;")
                self.last_saved_file = None
                self.btn_discard.setEnabled(False)
                self._update_stats()
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Could not delete take: {e}")

    def _update_stats(self) -> None:
        """Update count labels based on filesystem contents."""
        gloss = self.gloss_combo.currentData()
        all_files = list(self.output_dir.glob("*.npz"))
        gloss_files = [f for f in all_files if f.name.startswith(f"{gloss}_")]

        self.gloss_count_label.setText(f"Takes recorded for '{gloss}': {len(gloss_files)}")
        self.total_takes_label.setText(f"Total takes in database: {len(all_files)}")
        if self.last_saved_file is not None:
            self.last_saved_label.setText(f"Last saved: {self.last_saved_file.name}")

    def keyPressEvent(self, event) -> None:
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            if self.btn_record.isEnabled():
                self._trigger_countdown()
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.btn_discard.isEnabled():
                self._discard_last_take()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        """Ensure worker thread is cleanly stopped on window close."""
        if hasattr(self, "worker"):
            self.worker.stop()
        event.accept()


# Import QTimer at module level for countdown
from PyQt6.QtCore import QTimer


def main() -> None:
    app = QApplication(sys.argv)
    window = DataRecorderWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
