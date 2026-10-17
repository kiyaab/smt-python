"""Downloads and manages MediaPipe Tasks model bundles."""

import logging
import os
from pathlib import Path
from typing import Dict
import urllib.request

logger = logging.getLogger(__name__)

# Official Google MediaPipe Tasks model asset URLs (Lite / Float16 versions for fast CPU inference)
MODEL_URLS: Dict[str, str] = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}


class ModelAssetManager:
    """Ensures required MediaPipe .task bundles are present in the models directory."""

    def __init__(self, target_dir: str = "models/mediapipe") -> None:
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_filename: str) -> Path:
        """Return the path to a model file, downloading it if not present."""
        dest_path = self.target_dir / model_filename
        if not dest_path.exists():
            self.download_model(model_filename)
        return dest_path

    def is_model_available(self, model_filename: str) -> bool:
        """Check if a model file exists on disk and is non-empty."""
        path = self.target_dir / model_filename
        return path.exists() and path.stat().st_size > 0

    def download_model(self, model_filename: str, timeout: int = 30) -> bool:
        """Download a specified model file from the official CDN."""
        if model_filename not in MODEL_URLS:
            raise ValueError(f"Unknown model filename: {model_filename}. Supported: {list(MODEL_URLS.keys())}")

        url = MODEL_URLS[model_filename]
        dest_path = self.target_dir / model_filename

        logger.info(f"Downloading {model_filename} from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SignTranslator-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
            logger.info(f"Successfully saved {model_filename} ({dest_path.stat().st_size} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Could not download {model_filename}: {e}. Ensure internet connection or provide manually.")
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            return False

    def ensure_all_models(self) -> Dict[str, bool]:
        """Ensure all required MediaPipe task models are available on disk."""
        results = {}
        for filename in MODEL_URLS:
            if not self.is_model_available(filename):
                results[filename] = self.download_model(filename)
            else:
                results[filename] = True
        return results
