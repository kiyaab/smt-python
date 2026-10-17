"""Script to pre-download all required MediaPipe task models."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signtranslator.tracking.asset_manager import ModelAssetManager


def main() -> None:
    print("Checking and downloading MediaPipe task models...")
    manager = ModelAssetManager("models/mediapipe")
    results = manager.ensure_all_models()
    all_ok = True
    for model_name, ok in results.items():
        status = "READY" if ok else "FAILED"
        print(f"  [{status}] {model_name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll MediaPipe task models downloaded successfully.")
    else:
        print("\nSome models could not be downloaded. Offline fallback will be used.")


if __name__ == "__main__":
    main()
