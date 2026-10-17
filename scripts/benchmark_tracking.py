"""Benchmark script measuring CPU FPS and latency of the landmark tracking pipeline."""

from pathlib import Path
import sys
import time
import numpy as np

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signtranslator.capture.video_stream import VideoStream
from signtranslator.tracking.landmark_tracker import LandmarkTracker
from signtranslator.tracking.normalization import LandmarkNormalizer


def benchmark_pipeline(num_frames: int = 60) -> None:
    print(f"Benchmarking Landmark Tracking Pipeline on CPU ({num_frames} frames)...")

    # Initialize tracker and normalizer
    t0 = time.perf_counter()
    tracker = LandmarkTracker(model_dir="models/mediapipe", extract_pose=True, extract_face=True, pose_frequency=3)
    normalizer = LandmarkNormalizer(target_sequence_length=30)
    init_time = (time.perf_counter() - t0) * 1000.0
    print(f"Tracker Initialization Time: {init_time:.2f} ms")

    # Warmup
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        lm = tracker.process_frame(dummy)
        _ = tracker.draw_landmarks(dummy, lm)
        _ = normalizer.normalize_frame(lm)

    latencies = []
    norm_latencies = []

    for i in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = [30, 30, 35]

        # Tracking step
        t_start = time.perf_counter()
        landmarks = tracker.process_frame(frame)
        _ = tracker.draw_landmarks(frame, landmarks)
        t_track = (time.perf_counter() - t_start) * 1000.0
        latencies.append(t_track)

        # Normalization step
        t_norm_start = time.perf_counter()
        _ = normalizer.normalize_frame(landmarks)
        t_norm = (time.perf_counter() - t_norm_start) * 1000.0
        norm_latencies.append(t_norm)

    avg_track_ms = float(np.mean(latencies))
    p95_track_ms = float(np.percentile(latencies, 95))
    avg_norm_ms = float(np.mean(norm_latencies))
    total_ms = avg_track_ms + avg_norm_ms
    effective_fps = 1000.0 / total_ms

    print("\n--- Benchmark Results (CPU) ---")
    print(f"Tracking Latency (Mean): {avg_track_ms:.2f} ms")
    print(f"Tracking Latency (P95):  {p95_track_ms:.2f} ms")
    print(f"Normalization Latency:   {avg_norm_ms:.3f} ms")
    print(f"Total Pipeline Latency:  {total_ms:.2f} ms")
    print(f"Effective Processing:    {effective_fps:.1f} FPS")
    print("--------------------------------")


if __name__ == "__main__":
    benchmark_pipeline()
