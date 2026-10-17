"""Spatial landmark normalization, scale invariance, and temporal feature extraction."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FrameLandmarks:
    """Raw 3D landmarks for a single video frame.

    left_hand: (21, 3) float array or None if not detected
    right_hand: (21, 3) float array or None if not detected
    pose: (6, 3) float array [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist] or None
    face: (4, 3) float array [left_eyebrow, right_eyebrow, upper_lip, lower_lip] or None
    """

    left_hand: Optional[np.ndarray] = None
    right_hand: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    face: Optional[np.ndarray] = None


class LandmarkNormalizer:
    """Normalizes raw 3D landmark coordinates to achieve translation, scale, and signer invariance."""

    HAND_JOINTS = 21
    POSE_JOINTS = 6
    FACE_JOINTS = 4

    # Feature length per frame:
    # Left hand: 21 * 3 = 63
    # Right hand: 21 * 3 = 63
    # Pose: 6 * 3 = 18
    # Face: 4 * 3 = 12
    # Presence flags: left_present (1), right_present (1)
    # Total static features = 63 + 63 + 18 + 12 + 2 = 158
    TOTAL_STATIC_FEATURES = 158

    def __init__(self, target_sequence_length: int = 30) -> None:
        self.target_sequence_length = target_sequence_length

    def normalize_frame(self, frame_landmarks: FrameLandmarks) -> np.ndarray:
        """Convert a single frame's raw landmarks into a normalized 1D feature vector.

        Returns:
            1D np.ndarray of shape (TOTAL_STATIC_FEATURES,)
        """
        features: List[float] = []

        # 1. Left Hand
        if frame_landmarks.left_hand is not None and frame_landmarks.left_hand.shape == (self.HAND_JOINTS, 3):
            lh_norm = self._normalize_hand(frame_landmarks.left_hand)
            features.extend(lh_norm.flatten().tolist())
            features.append(1.0)  # Left hand present flag
        else:
            features.extend([0.0] * (self.HAND_JOINTS * 3))
            features.append(0.0)  # Left hand absent flag

        # 2. Right Hand
        if frame_landmarks.right_hand is not None and frame_landmarks.right_hand.shape == (self.HAND_JOINTS, 3):
            rh_norm = self._normalize_hand(frame_landmarks.right_hand)
            features.extend(rh_norm.flatten().tolist())
            features.append(1.0)  # Right hand present flag
        else:
            features.extend([0.0] * (self.HAND_JOINTS * 3))
            features.append(0.0)  # Right hand absent flag

        # 3. Pose Reference
        if frame_landmarks.pose is not None and frame_landmarks.pose.shape == (self.POSE_JOINTS, 3):
            pose_norm = self._normalize_pose(frame_landmarks.pose)
            features.extend(pose_norm.flatten().tolist())
        else:
            features.extend([0.0] * (self.POSE_JOINTS * 3))

        # 4. Face Reference
        if frame_landmarks.face is not None and frame_landmarks.face.shape == (self.FACE_JOINTS, 3):
            face_norm = self._normalize_face(frame_landmarks.face, frame_landmarks.pose)
            features.extend(face_norm.flatten().tolist())
        else:
            features.extend([0.0] * (self.FACE_JOINTS * 3))

        return np.array(features, dtype=np.float32)

    def normalize_sequence(self, sequence: List[FrameLandmarks]) -> np.ndarray:
        """Normalize a temporal sequence of frame landmarks into a fixed-length matrix.

        Returns:
            np.ndarray of shape (target_sequence_length, TOTAL_STATIC_FEATURES)
        """
        if not sequence:
            return np.zeros((self.target_sequence_length, self.TOTAL_STATIC_FEATURES), dtype=np.float32)

        # Normalize each frame
        frame_vectors = [self.normalize_frame(fl) for fl in sequence]
        matrix = np.stack(frame_vectors, axis=0)  # (N, TOTAL_STATIC_FEATURES)

        # Temporal resample to target_sequence_length
        return self.resample_sequence(matrix, self.target_sequence_length)

    @staticmethod
    def resample_sequence(matrix: np.ndarray, target_length: int) -> np.ndarray:
        """Uniformly resample temporal sequence to target_length frames."""
        current_length = matrix.shape[0]
        if current_length == target_length:
            return matrix.astype(np.float32)

        if current_length == 1:
            return np.repeat(matrix, target_length, axis=0).astype(np.float32)

        # Linear interpolation across temporal dimension
        orig_indices = np.linspace(0, current_length - 1, num=current_length)
        new_indices = np.linspace(0, current_length - 1, num=target_length)

        resampled = np.zeros((target_length, matrix.shape[1]), dtype=np.float32)
        for col in range(matrix.shape[1]):
            resampled[:, col] = np.interp(new_indices, orig_indices, matrix[:, col])

        return resampled

    @staticmethod
    def compute_motion_energy(sequence_matrix: np.ndarray) -> float:
        """Calculate motion energy (magnitude of frame-to-frame velocity) across the sequence."""
        if sequence_matrix.shape[0] < 2:
            return 0.0
        diff = np.diff(sequence_matrix, axis=0)
        # Average Euclidean norm of velocity vectors across frames
        return float(np.mean(np.linalg.norm(diff, axis=1)))

    def mirror_hand_sequence(self, sequence_matrix: np.ndarray) -> np.ndarray:
        """Generate a left-to-right mirrored version of a sequence for data augmentation.

        Flips x-coordinates and swaps left/right hand channels.
        """
        mirrored = sequence_matrix.copy()

        # Slice indices
        # Left hand: 0:63, presence: 63
        # Right hand: 64:127, presence: 127
        # Pose: 128:146
        # Face: 146:158

        lh_coords = mirrored[:, 0:63].copy()
        lh_pres = mirrored[:, 63:64].copy()
        rh_coords = mirrored[:, 64:127].copy()
        rh_pres = mirrored[:, 127:128].copy()

        # Flip X coordinates (every 3rd element starting from 0)
        lh_coords[:, 0::3] *= -1.0
        rh_coords[:, 0::3] *= -1.0

        # Swap hands
        mirrored[:, 0:63] = rh_coords
        mirrored[:, 63:64] = rh_pres
        mirrored[:, 64:127] = lh_coords
        mirrored[:, 127:128] = lh_pres

        # Flip pose X coordinates
        mirrored[:, 128:146:3] *= -1.0
        # Flip face X coordinates
        mirrored[:, 146:158:3] *= -1.0

        return mirrored

    def _normalize_hand(self, hand_coords: np.ndarray) -> np.ndarray:
        """Anchor hand to wrist (joint 0) and scale by palm span."""
        wrist = hand_coords[0]  # (3,)
        centered = hand_coords - wrist

        # Palm scale: distance between wrist (0) and middle MCP (9)
        palm_size = float(np.linalg.norm(hand_coords[9] - hand_coords[0]))
        if palm_size > 1e-4:
            scale = 1.0 / palm_size
        else:
            scale = 1.0

        return centered * scale

    def _normalize_pose(self, pose_coords: np.ndarray) -> np.ndarray:
        """Anchor pose to mid-shoulder and scale by shoulder width.

        Pose indices:
        0: Left shoulder, 1: Right shoulder, 2: Left elbow, 3: Right elbow, 4: Left wrist, 5: Right wrist
        """
        left_shoulder = pose_coords[0]
        right_shoulder = pose_coords[1]
        mid_shoulder = (left_shoulder + right_shoulder) / 2.0

        centered = pose_coords - mid_shoulder

        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        scale = 1.0 / shoulder_width if shoulder_width > 1e-4 else 1.0

        return centered * scale

    def _normalize_face(self, face_coords: np.ndarray, pose_coords: Optional[np.ndarray]) -> np.ndarray:
        """Anchor face to nose / mid-eyebrow and scale by inter-eyebrow distance."""
        left_eyebrow = face_coords[0]
        right_eyebrow = face_coords[1]
        mid_eyebrow = (left_eyebrow + right_eyebrow) / 2.0

        centered = face_coords - mid_eyebrow

        eyebrow_dist = float(np.linalg.norm(left_eyebrow - right_eyebrow))
        scale = 1.0 / eyebrow_dist if eyebrow_dist > 1e-4 else 1.0

        return centered * scale
