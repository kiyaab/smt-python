"""Synthetic and bootstrap gesture sequence generator for automated testing and CI."""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml

from signtranslator.tracking.normalization import FrameLandmarks, LandmarkNormalizer


class SyntheticGestureGenerator:
    """Generates biomechanically plausible landmark motion sequences clearly tagged as synthetic.

    Uses canonical trajectory baselines augmented with temporal warping, spatial jitter,
    scale perturbation, and rotation.
    """

    def __init__(self, vocabulary_path: str = "config/vocabulary.yaml", sequence_length: int = 30) -> None:
        self.sequence_length = sequence_length
        self.normalizer = LandmarkNormalizer(target_sequence_length=sequence_length)

        # Load gloss list
        vocab_file = Path(vocabulary_path)
        if vocab_file.exists():
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.glosses = [item["gloss"] for item in data.get("glosses", [])]
        else:
            self.glosses = ["IDLE", "HELLO", "THANK_YOU", "HELP", "WATER", "ME", "YOU", "YES", "NO"]

    def generate_sample(
        self,
        gloss: str,
        signer_id: str = "signer_syn_01",
        sample_id: int = 0,
        jitter_std: float = 0.015,
        speed_factor: float = 1.0,
        scale_factor: float = 1.0,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate a single normalized sequence and associated metadata.

        Returns:
            Tuple of:
                - normalized_matrix: (sequence_length, 158) float32 array
                - metadata: dict with gloss, signer_id, synthetic flag, etc.
        """
        # Determine number of raw frames before temporal resampling
        num_frames = int(np.clip(self.sequence_length * speed_factor, 15, 60))

        # Base trajectories for left and right hands
        raw_frames: List[FrameLandmarks] = []
        for t in range(num_frames):
            progress = t / float(num_frames - 1)  # 0.0 to 1.0

            # Default pose (standard upper torso facing camera)
            pose = np.array([
                [0.40, 0.50, 0.0],  # 0: Left shoulder
                [0.60, 0.50, 0.0],  # 1: Right shoulder
                [0.35, 0.70, 0.05], # 2: Left elbow
                [0.65, 0.70, 0.05], # 3: Right elbow
                [0.38, 0.85, 0.1],  # 4: Left wrist
                [0.62, 0.85, 0.1],  # 5: Right wrist
            ], dtype=np.float32)

            # Default face (eyebrows & mouth)
            face = np.array([
                [0.46, 0.28, 0.0],  # Left eyebrow
                [0.54, 0.28, 0.0],  # Right eyebrow
                [0.50, 0.35, 0.0],  # Upper lip
                [0.50, 0.38, 0.0],  # Lower lip
            ], dtype=np.float32)

            # Generate hand coordinates based on gloss dynamics
            lh_coords, rh_coords = self._generate_gesture_hands(gloss, progress)

            # Apply scale factor and jitter
            if rh_coords is not None:
                rh_coords = (rh_coords - rh_coords[0]) * scale_factor + rh_coords[0]
                noise = np.random.normal(0, jitter_std, rh_coords.shape).astype(np.float32)
                rh_coords += noise

            if lh_coords is not None:
                lh_coords = (lh_coords - lh_coords[0]) * scale_factor + lh_coords[0]
                noise = np.random.normal(0, jitter_std, lh_coords.shape).astype(np.float32)
                lh_coords += noise

            raw_frames.append(FrameLandmarks(
                left_hand=lh_coords,
                right_hand=rh_coords,
                pose=pose,
                face=face,
            ))

        # Normalize and resample to fixed target length
        normalized_matrix = self.normalizer.normalize_sequence(raw_frames)

        metadata = {
            "gloss": gloss,
            "signer_id": signer_id,
            "sample_id": sample_id,
            "is_synthetic": True,
            "created_at": time.time(),
            "num_raw_frames": num_frames,
            "target_sequence_length": self.sequence_length,
            "features_per_frame": LandmarkNormalizer.TOTAL_STATIC_FEATURES,
        }

        return normalized_matrix, metadata

    def generate_dataset(
        self,
        output_dir: str = "data/raw",
        samples_per_class: int = 15,
        signers: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Generate a complete balanced bootstrap dataset partitioned across signers."""
        if signers is None:
            signers = ["signer_syn_train_A", "signer_syn_train_B", "signer_syn_val", "signer_syn_test"]

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        counts: Dict[str, int] = {gloss: 0 for gloss in self.glosses}

        for gloss in self.glosses:
            for i in range(samples_per_class):
                signer = signers[i % len(signers)]
                # Add natural variance
                speed = float(np.random.uniform(0.8, 1.25))
                scale = float(np.random.uniform(0.9, 1.1))
                jitter = float(np.random.uniform(0.005, 0.015))

                norm_matrix, meta = self.generate_sample(
                    gloss=gloss,
                    signer_id=signer,
                    sample_id=i,
                    jitter_std=jitter,
                    speed_factor=speed,
                    scale_factor=scale,
                )

                file_name = f"{gloss}_{signer}_{i:03d}.npz"
                np.savez_compressed(
                    out_path / file_name,
                    sequence=norm_matrix,
                    metadata_json=json.dumps(meta),
                )
                counts[gloss] += 1

        return counts

    def _generate_gesture_hands(
        self, gloss: str, progress: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate parameterized 3D hand landmarks for a given sign and progress (0.0 to 1.0)."""
        rh_template = self._create_open_hand_template(base_wrist=[0.65, 0.45, 0.1])
        lh_template = self._create_open_hand_template(base_wrist=[0.35, 0.45, 0.1])

        if gloss == "IDLE":
            # Both hands at rest down near waist/lap with minor drift
            rh = self._create_open_hand_template(base_wrist=[0.62, 0.85 + 0.01 * np.sin(progress * np.pi), 0.15])
            lh = self._create_open_hand_template(base_wrist=[0.38, 0.85 + 0.01 * np.sin(progress * np.pi), 0.15])
            return lh, rh

        elif gloss == "HELLO":
            # Right hand waving near temple
            dx = 0.08 * np.sin(progress * 4 * np.pi)
            dy = -0.02 * np.cos(progress * 2 * np.pi)
            rh = self._create_open_hand_template(base_wrist=[0.65 + dx, 0.30 + dy, 0.1])
            return None, rh

        elif gloss == "GOODBYE":
            # Right hand waving side to side
            dx = 0.10 * np.sin(progress * 3 * np.pi)
            rh = self._create_open_hand_template(base_wrist=[0.65 + dx, 0.35, 0.1])
            return None, rh

        elif gloss == "PLEASE":
            # Circular rubbing on chest with right hand
            angle = progress * 2 * np.pi
            cx, cy = 0.52 + 0.05 * np.cos(angle), 0.48 + 0.05 * np.sin(angle)
            rh = self._create_flat_hand_template(base_wrist=[cx, cy, 0.05])
            return None, rh

        elif gloss == "THANK_YOU":
            # Hand moves from chin (progress=0) forward and down (progress=1)
            y = 0.35 + 0.15 * progress
            z = -0.05 + 0.20 * progress
            rh = self._create_flat_hand_template(base_wrist=[0.52, y, z])
            return None, rh

        elif gloss == "HELP":
            # Left hand flat palm upward, right fist on it moving up together
            y = 0.65 - 0.15 * progress
            lh = self._create_flat_hand_template(base_wrist=[0.45, y, 0.15])
            rh = self._create_fist_template(base_wrist=[0.48, y - 0.05, 0.15])
            return lh, rh

        elif gloss == "WATER":
            # "W" hand tapping near mouth
            tap = 0.03 * np.sin(progress * 3 * np.pi)
            rh = self._create_w_hand_template(base_wrist=[0.55, 0.35 + tap, 0.05])
            return None, rh

        elif gloss == "FOOD":
            # Tapered hand tapping lips
            tap = 0.02 * np.sin(progress * 4 * np.pi)
            rh = self._create_pinched_hand_template(base_wrist=[0.50, 0.36 + tap, 0.02])
            return None, rh

        elif gloss == "ME":
            # Index finger pointing to chest
            rh = self._create_pointing_hand_template(base_wrist=[0.50, 0.50, 0.05])
            return None, rh

        elif gloss == "YOU":
            # Index finger pointing straight forward
            z = 0.10 + 0.15 * progress
            rh = self._create_pointing_hand_template(base_wrist=[0.58, 0.48, z])
            return None, rh

        elif gloss == "YES":
            # Fist nodding up and down
            nod = 0.06 * np.sin(progress * 3 * np.pi)
            rh = self._create_fist_template(base_wrist=[0.60, 0.45 + nod, 0.1])
            return None, rh

        elif gloss == "NO":
            # Fingers snapping shut
            rh = self._create_pinched_hand_template(base_wrist=[0.60, 0.42, 0.08])
            return None, rh

        else:
            # Generic distinct geometric trajectory for other glosses
            hash_val = hash(gloss) % 100
            freq = 1.0 + (hash_val % 4)
            y_shift = ((hash_val // 10) % 5) * 0.05
            dy = 0.06 * np.sin(progress * freq * np.pi)
            dx = 0.04 * np.cos(progress * freq * np.pi)
            rh = self._create_open_hand_template(base_wrist=[0.60 + dx, 0.42 + y_shift + dy, 0.1])
            return None, rh

    def _create_open_hand_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates for an open hand."""
        wrist = np.array(base_wrist, dtype=np.float32)
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = wrist

        # Offsets for 5 fingers (Thumb, Index, Middle, Ring, Pinky)
        finger_spreads = [-0.04, -0.02, 0.0, 0.02, 0.04]
        for f, spread in enumerate(finger_spreads):
            mcp_idx = 1 + f * 4
            for seg in range(4):
                idx = mcp_idx + seg
                dist = (seg + 1) * 0.025
                hand[idx] = wrist + np.array([spread, -dist, 0.01 * seg], dtype=np.float32)

        return hand

    def _create_flat_hand_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates for a flat palm."""
        wrist = np.array(base_wrist, dtype=np.float32)
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = wrist
        for f in range(5):
            mcp = 1 + f * 4
            for seg in range(4):
                hand[mcp + seg] = wrist + np.array([(f - 2) * 0.012, -(seg + 1) * 0.026, 0.0], dtype=np.float32)
        return hand

    def _create_fist_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates for a closed fist."""
        wrist = np.array(base_wrist, dtype=np.float32)
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = wrist
        for f in range(5):
            mcp = 1 + f * 4
            for seg in range(4):
                # Fingers curled inward
                hand[mcp + seg] = wrist + np.array([(f - 2) * 0.01, -0.02 + seg * 0.005, 0.02 * seg], dtype=np.float32)
        return hand

    def _create_pointing_hand_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates for an index-pointing hand."""
        hand = self._create_fist_template(base_wrist)
        # Extend index finger (joints 5, 6, 7, 8)
        wrist = hand[0]
        for seg in range(4):
            hand[5 + seg] = wrist + np.array([0.0, -(seg + 1) * 0.03, 0.0], dtype=np.float32)
        return hand

    def _create_w_hand_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates for a 'W' hand (index, middle, ring extended)."""
        hand = self._create_fist_template(base_wrist)
        wrist = hand[0]
        for f, spread in [(1, -0.02), (2, 0.0), (3, 0.02)]:
            mcp = 1 + f * 4
            for seg in range(4):
                hand[mcp + seg] = wrist + np.array([spread, -(seg + 1) * 0.03, 0.0], dtype=np.float32)
        return hand

    def _create_pinched_hand_template(self, base_wrist: List[float]) -> np.ndarray:
        """Create 21 landmark coordinates with fingertips pinched together."""
        wrist = np.array(base_wrist, dtype=np.float32)
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = wrist
        pinch_pt = wrist + np.array([0.0, -0.06, 0.0], dtype=np.float32)
        for f in range(5):
            mcp = 1 + f * 4
            for seg in range(4):
                factor = (seg + 1) / 4.0
                hand[mcp + seg] = wrist + factor * (pinch_pt - wrist) + np.array([(f - 2) * 0.005, 0, 0], dtype=np.float32)
        return hand
