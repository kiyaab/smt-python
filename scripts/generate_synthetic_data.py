"""CLI script to generate a synthetic bootstrap landmark sequence dataset."""

import argparse
from pathlib import Path
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signtranslator.recognition.synthetic_data import SyntheticGestureGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sign gesture landmark dataset.")
    parser.add_argument("--samples-per-class", type=int, default=15, help="Number of repetitions per gloss.")
    parser.add_argument("--vocab-path", type=str, default="config/vocabulary.yaml", help="Path to vocabulary.yaml.")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output folder for .npz files.")
    args = parser.parse_args()

    print(f"Generating synthetic gesture dataset ({args.samples_per_class} samples/class)...")
    generator = SyntheticGestureGenerator(vocabulary_path=args.vocab_path)
    counts = generator.generate_dataset(
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
    )

    total_samples = sum(counts.values())
    print(f"\nSuccessfully generated {total_samples} samples across {len(counts)} glosses in '{args.output_dir}'.")
    print("Sample distribution summary:")
    for gloss, count in list(counts.items())[:8]:
        print(f"  - {gloss}: {count} takes")
    if len(counts) > 8:
        print(f"  ... and {len(counts) - 8} more classes.")


if __name__ == "__main__":
    main()
