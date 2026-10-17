"""Unit tests for synthetic dataset generator and sequence persistence."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from signtranslator.recognition.synthetic_data import SyntheticGestureGenerator
from signtranslator.tracking.normalization import LandmarkNormalizer


def test_synthetic_sample_generation():
    """Verify that SyntheticGestureGenerator produces a sequence matching target dimensions."""
    gen = SyntheticGestureGenerator(sequence_length=30)
    matrix, meta = gen.generate_sample("HELLO", signer_id="test_signer")

    assert matrix.shape == (30, LandmarkNormalizer.TOTAL_STATIC_FEATURES)
    assert matrix.dtype == np.float32
    assert meta["gloss"] == "HELLO"
    assert meta["signer_id"] == "test_signer"
    assert meta["is_synthetic"] is True


def test_synthetic_dataset_saving_and_loading():
    """Verify that generate_dataset writes readable .npz files with complete metadata."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = SyntheticGestureGenerator(sequence_length=30)
        # Generate 2 samples for 3 glosses
        gen.glosses = ["HELLO", "THANK_YOU", "WATER"]
        counts = gen.generate_dataset(output_dir=tmp_dir, samples_per_class=2)

        assert counts["HELLO"] == 2
        assert counts["THANK_YOU"] == 2
        assert counts["WATER"] == 2

        # Check saved files
        files = list(Path(tmp_dir).glob("*.npz"))
        assert len(files) == 6

        # Load one file and verify contents
        with np.load(files[0]) as data:
            assert "sequence" in data
            assert "metadata_json" in data
            assert data["sequence"].shape == (30, LandmarkNormalizer.TOTAL_STATIC_FEATURES)
            meta = json.loads(str(data["metadata_json"]))
            assert "gloss" in meta
            assert meta["is_synthetic"] is True
