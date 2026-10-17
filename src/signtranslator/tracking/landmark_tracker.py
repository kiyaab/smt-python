"""Multi-task landmark tracking using MediaPipe Tasks Vision API."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from signtranslator.tracking.asset_manager import ModelAssetManager
from signtranslator.tracking.normalization import FrameLandmarks

logger = logging.getLogger(__name__)


class LandmarkTracker:
    """Extracts 3D landmarks for hands, upper-body pose, and facial non-manual markers."""

    # Selected Pose Landmark indices:
    # 11: Left Shoulder, 12: Right Shoulder, 13: Left Elbow, 14: Right Elbow, 15: Left Wrist, 16: Right Wrist
    POSE_INDICES = [11, 12, 13, 14, 15, 16]

    # Selected Face Landmark indices:
    # 70: Left Eyebrow outer, 300: Right Eyebrow outer, 13: Upper Lip center, 14: Lower Lip center
    FACE_INDICES = [70, 300, 13, 14]

    # MediaPipe Hand Skeleton connections (pairs of joint indices 0-20)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    def __init__(
        self,
        model_dir: str = "models/mediapipe",
        min_hand_confidence: float = 0.5,
        extract_pose: bool = True,
        extract_face: bool = True,
        pose_frequency: int = 1,
    ) -> None:
        self.model_dir = model_dir
        self.extract_pose = extract_pose
        self.extract_face = extract_face
        self.pose_frequency = max(1, pose_frequency)
        self._frame_counter = 0

        # Cached last detection for rate-limited components
        self._cached_pose: Optional[np.ndarray] = None
        self._cached_face: Optional[np.ndarray] = None

        # Ensure model files are ready
        asset_mgr = ModelAssetManager(model_dir)
        hand_model_path = asset_mgr.get_model_path("hand_landmarker.task")
        pose_model_path = asset_mgr.get_model_path("pose_landmarker.task")
        face_model_path = asset_mgr.get_model_path("face_landmarker.task")

        # 1. Initialize Hand Landmarker
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model_path)),
            num_hands=2,
            min_hand_detection_confidence=min_hand_confidence,
            min_hand_presence_confidence=min_hand_confidence,
            min_tracking_confidence=min_hand_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # 2. Initialize Pose Landmarker
        self.pose_detector = None
        if self.extract_pose and pose_model_path.exists():
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(pose_model_path)),
                running_mode=vision.RunningMode.IMAGE,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

        # 3. Initialize Face Landmarker
        self.face_detector = None
        if self.extract_face and face_model_path.exists():
            face_options = vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(face_model_path)),
                running_mode=vision.RunningMode.IMAGE,
            )
            self.face_detector = vision.FaceLandmarker.create_from_options(face_options)

    def process_frame(self, frame_bgr: np.ndarray) -> FrameLandmarks:
        """Process a video frame and return structured 3D landmarks for hands, pose, and face.

        Args:
            frame_bgr: BGR uint8 OpenCV image.

        Returns:
            FrameLandmarks dataclass.
        """
        self._frame_counter += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        left_hand_coords: Optional[np.ndarray] = None
        right_hand_coords: Optional[np.ndarray] = None

        # 1. Detect Hands
        hand_result = self.hand_detector.detect(mp_image)
        if hand_result.hand_landmarks:
            for i, hand_lms in enumerate(hand_result.hand_landmarks):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
                # MediaPipe handedness label
                handedness_label = "Right"
                if hand_result.handedness and i < len(hand_result.handedness):
                    handedness_label = hand_result.handedness[i][0].category_name

                if handedness_label == "Left":
                    left_hand_coords = coords
                else:
                    right_hand_coords = coords

        # 2. Detect Pose (upper body)
        pose_coords = self._cached_pose
        if self.pose_detector is not None and (self._frame_counter % self.pose_frequency == 0):
            pose_result = self.pose_detector.detect(mp_image)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                all_pose = pose_result.pose_landmarks[0]
                selected = []
                for idx in self.POSE_INDICES:
                    lm = all_pose[idx]
                    selected.append([lm.x, lm.y, lm.z])
                pose_coords = np.array(selected, dtype=np.float32)
                self._cached_pose = pose_coords

        # 3. Detect Face Keypoints
        face_coords = self._cached_face
        if self.face_detector is not None and (self._frame_counter % self.pose_frequency == 0):
            face_result = self.face_detector.detect(mp_image)
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                all_face = face_result.face_landmarks[0]
                selected_face = []
                for idx in self.FACE_INDICES:
                    lm = all_face[idx]
                    selected_face.append([lm.x, lm.y, lm.z])
                face_coords = np.array(selected_face, dtype=np.float32)
                self._cached_face = face_coords

        return FrameLandmarks(
            left_hand=left_hand_coords,
            right_hand=right_hand_coords,
            pose=pose_coords,
            face=face_coords,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray, landmarks: FrameLandmarks) -> np.ndarray:
        """Render high-contrast visual landmark skeletons directly onto an image frame."""
        h, w, _ = frame_bgr.shape
        annotated = frame_bgr.copy()

        # Colors (BGR)
        lh_joint_color = (255, 120, 0)   # Blue-Cyan for Left Hand
        lh_bone_color = (220, 80, 0)
        rh_joint_color = (0, 220, 100)   # Vibrant Green for Right Hand
        rh_bone_color = (0, 180, 70)
        pose_color = (255, 200, 50)      # Yellow-Gold for Pose
        face_color = (200, 100, 255)     # Magenta for Face

        # 1. Draw Left Hand
        if landmarks.left_hand is not None:
            self._render_hand(annotated, landmarks.left_hand, w, h, lh_joint_color, lh_bone_color)

        # 2. Draw Right Hand
        if landmarks.right_hand is not None:
            self._render_hand(annotated, landmarks.right_hand, w, h, rh_joint_color, rh_bone_color)

        # 3. Draw Pose
        if landmarks.pose is not None:
            points = []
            for pt in landmarks.pose:
                px = int(pt[0] * w)
                py = int(pt[1] * h)
                points.append((px, py))
                cv2.circle(annotated, (px, py), 4, pose_color, -1)

            # Connect shoulders and arms
            if len(points) == 6:
                # 0: L shoulder, 1: R shoulder, 2: L elbow, 3: R elbow, 4: L wrist, 5: R wrist
                cv2.line(annotated, points[0], points[1], pose_color, 2)  # Shoulders
                cv2.line(annotated, points[0], points[2], pose_color, 2)  # L upper arm
                cv2.line(annotated, points[2], points[4], pose_color, 2)  # L forearm
                cv2.line(annotated, points[1], points[3], pose_color, 2)  # R upper arm
                cv2.line(annotated, points[3], points[5], pose_color, 2)  # R forearm

        # 4. Draw Face non-manual markers
        if landmarks.face is not None:
            for pt in landmarks.face:
                px = int(pt[0] * w)
                py = int(pt[1] * h)
                cv2.circle(annotated, (px, py), 3, face_color, -1)

        return annotated

    def _render_hand(
        self,
        img: np.ndarray,
        coords: np.ndarray,
        w: int,
        h: int,
        joint_color: Tuple[int, int, int],
        bone_color: Tuple[int, int, int],
    ) -> None:
        """Render joints and skeletal connections for a 21-joint hand."""
        pixel_pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in coords]

        # Draw bones
        for start_idx, end_idx in self.HAND_CONNECTIONS:
            cv2.line(img, pixel_pts[start_idx], pixel_pts[end_idx], bone_color, 2)

        # Draw joint nodes
        for pt in pixel_pts:
            cv2.circle(img, pt, 4, joint_color, -1)
            cv2.circle(img, pt, 2, (255, 255, 255), -1)
