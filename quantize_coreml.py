"""Optional quantization script to reduce the SHARP CoreML model size.

This script applies post-training quantization to the CoreML model to reduce
its size for on-device deployment. Options include:

  - Palettization (6-bit or 4-bit): Reduces model to ~200-400MB
  - Linear quantization (INT8): Reduces model to ~200MB

Usage:
    # First convert the model:
    python convert_to_coreml.py -o SHARPPredictor.mlpackage

    # Then quantize:
    python quantize_coreml.py \
        --input SHARPPredictor.mlpackage \
        --output SHARPPredictor_quantized.mlpackage \
        --method palettize \
        --bits 6

Requirements:
    pip install coremltools

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def get_model_size_mb(path: Path) -> float:
    """Get total size of a .mlpackage directory in MB."""
    total = 0
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    else:
        total = path.stat().st_size
    return total / (1024 * 1024)


def palettize_model(input_path: Path, output_path: Path, n_bits: int = 6):
    """Apply weight palettization (k-means clustering) to reduce model size.

    Palettization groups weights into 2^n_bits clusters, significantly
    reducing model size with minimal quality loss for n_bits >= 4.

    Args:
        input_path: Path to input .mlpackage.
        output_path: Path for output .mlpackage.
        n_bits: Number of bits per weight (4, 6, or 8).
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    LOGGER.info("Loading model from %s", input_path)
    model = ct.models.MLModel(str(input_path))

    LOGGER.info("Applying %d-bit palettization...", n_bits)
    op_config = OpPalettizerConfig(nbits=n_bits)
    config = OptimizationConfig(global_config=op_config)
    compressed_model = palettize_weights(model, config)

    LOGGER.info("Saving quantized model to %s", output_path)
    compressed_model.save(str(output_path))

    original_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    LOGGER.info(
        "Size reduction: %.1f MB -> %.1f MB (%.1fx compression)",
        original_size, new_size, original_size / max(new_size, 0.1),
    )


def linear_quantize_model(input_path: Path, output_path: Path):
    """Apply linear INT8 quantization to reduce model size.

    Args:
        input_path: Path to input .mlpackage.
        output_path: Path for output .mlpackage.
    """
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    LOGGER.info("Loading model from %s", input_path)
    model = ct.models.MLModel(str(input_path))

    LOGGER.info("Applying INT8 linear quantization...")
    op_config = OpLinearQuantizerConfig(dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    compressed_model = linear_quantize_weights(model, config)

    LOGGER.info("Saving quantized model to %s", output_path)
    compressed_model.save(str(output_path))

    original_size = get_model_size_mb(input_path)
    new_size = get_model_size_mb(output_path)
    LOGGER.info(
        "Size reduction: %.1f MB -> %.1f MB (%.1fx compression)",
        original_size, new_size, original_size / max(new_size, 0.1),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quantize SHARP CoreML model for smaller on-device deployment"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("SHARPPredictor.mlpackage"),
        help="Input .mlpackage path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .mlpackage path (default: adds method suffix, or SHARPPredictor_quantized with --lightweight)",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use canonical name SHARPPredictor_quantized.mlpackage for Swift/visionOS (recommended)",
    )
    parser.add_argument(
        "--method",
        choices=["palettize", "linear_int8"],
        default="palettize",
        help="Quantization method (default: palettize)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 6, 8],
        default=6,
        help="Bits per weight for palettization (default: 6)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        LOGGER.error("Input model not found: %s", args.input)
        LOGGER.error("Run convert_to_coreml.py first to create the base model.")
        return

    if args.output is None:
        if args.lightweight:
            args.output = args.input.parent / "SHARPPredictor_quantized.mlpackage"
        else:
            suffix = f"_{args.method}_{args.bits}bit" if args.method == "palettize" else f"_{args.method}"
            args.output = args.input.with_name(args.input.stem + suffix + ".mlpackage")

    original_size = get_model_size_mb(args.input)
    LOGGER.info("Original model size: %.1f MB", original_size)

    if args.method == "palettize":
        palettize_model(args.input, args.output, args.bits)
    elif args.method == "linear_int8":
        linear_quantize_model(args.input, args.output)

    LOGGER.info("\nQuantized model saved to: %s", args.output)
    LOGGER.info(
        "Note: Test the quantized model quality before deploying.\n"
        "Run: python test_coreml_conversion.py --coreml %s --image data/teaser.jpg",
        args.output,
    )


if __name__ == "__main__":
    main()
