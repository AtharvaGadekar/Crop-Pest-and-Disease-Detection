#!/usr/bin/env python3
"""Classify image files with a trained maize pest model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from crop_recognition import PredictionSmoother, load_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify maize pest images without connecting the rover."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--confidence-threshold", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        classifier = load_artifact(
            args.model,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 2

    smoother = PredictionSmoother(
        classifier.class_names,
        classifier.confidence_threshold,
        window_size=1,
    )
    failed = False
    for path in args.images:
        try:
            probabilities, timestamp = classifier.predict_jpeg(path.read_bytes())
            prediction = smoother.add(probabilities, timestamp)
            if prediction.confident:
                print(
                    f"{path}: {prediction.label} "
                    f"({prediction.confidence:.1%})"
                )
            else:
                print(f"{path}: uncertain ({prediction.confidence:.1%})")
        except (OSError, RuntimeError, ValueError) as error:
            failed = True
            print(f"{path}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
