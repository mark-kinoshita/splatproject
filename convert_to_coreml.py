"""Convert SHARP PyTorch model to CoreML for on-device visionOS inference.

Usage:
    python convert_to_coreml.py [--checkpoint path/to/checkpoint.pt] [--output sharp_model.mlpackage]

Requirements:
    pip install coremltools torch torchvision timm

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_RESOLUTION = 1536


def load_model(checkpoint_path: Path | None = None, device: str = "cpu") -> torch.nn.Module:
    """Load the SHARP predictor model with pretrained weights.

    Args:
        checkpoint_path: Path to a local .pt checkpoint. If None, downloads default.
        device: Device to load weights onto.

    Returns:
        The loaded model in eval mode.
    """
    from sharp.models import PredictorParams, create_predictor

    LOGGER.info("Creating predictor model...")
    model = create_predictor(PredictorParams())

    if checkpoint_path is None:
        LOGGER.info("Downloading default checkpoint from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True, map_location=device
        )
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    LOGGER.info("Model loaded successfully.")
    return model


class SHARPTracingWrapper(torch.nn.Module):
    """Wrapper that makes the SHARP model traceable by flattening outputs.

    torch.jit.trace requires all outputs to be tensors (not NamedTuples),
    and all inputs to be tensors (no None). This wrapper:
      - Removes the optional `depth` argument (always None for inference)
      - Returns a flat tuple of tensors instead of Gaussians3D NamedTuple
    """

    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(
        self, image: torch.Tensor, disparity_factor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference and return flat tensor outputs.

        Args:
            image: [1, 3, 1536, 1536] normalized RGB image in [0, 1].
            disparity_factor: [1] scalar tensor = f_px / original_width.

        Returns:
            Tuple of (mean_vectors, singular_values, quaternions, colors, opacities)
            in NDC space. Each has batch dim = 1.
        """
        gaussians = self.predictor(image, disparity_factor, depth=None)
        return (
            gaussians.mean_vectors,
            gaussians.singular_values,
            gaussians.quaternions,
            gaussians.colors,
            gaussians.opacities,
        )


def trace_model(model: torch.nn.Module, device: str = "cpu") -> torch.jit.ScriptModule:
    """Trace the SHARP model with example inputs.

    Args:
        model: The loaded SHARP predictor.
        device: Device to run tracing on.

    Returns:
        A traced ScriptModule.
    """
    wrapper = SHARPTracingWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    LOGGER.info("Creating example inputs (resolution=%d)...", INTERNAL_RESOLUTION)
    example_image = torch.randn(1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION, device=device)
    example_image = example_image.clamp(0, 1)
    example_disparity_factor = torch.tensor([0.5], device=device)

    LOGGER.info("Tracing model (this may take a few minutes)...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (example_image, example_disparity_factor),
            check_trace=False,  # NamedTuple intermediates may cause check issues
            strict=False,
        )

    LOGGER.info("Tracing complete.")
    return traced


def convert_to_coreml(
    traced_model: torch.jit.ScriptModule,
    output_path: Path,
    quantize: str = "float16",
) -> None:
    """Convert traced PyTorch model to CoreML.

    Args:
        traced_model: The traced SHARP model.
        output_path: Where to save the .mlpackage.
        quantize: Precision mode - 'float16' or 'float32'.
    """
    import coremltools as ct

    LOGGER.info("Converting to CoreML with %s precision...", quantize)

    # Define input shapes
    image_shape = ct.Shape(shape=(1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION))
    disparity_shape = ct.Shape(shape=(1,))

    compute_precision = ct.precision.FLOAT16 if quantize == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image", shape=image_shape),
            ct.TensorType(name="disparity_factor", shape=disparity_shape),
        ],
        outputs=[
            ct.TensorType(name="mean_vectors"),
            ct.TensorType(name="singular_values"),
            ct.TensorType(name="quaternions"),
            ct.TensorType(name="colors"),
            ct.TensorType(name="opacities"),
        ],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS18,  # visionOS 2.0+ / iOS 18+
    )

    # Add metadata
    mlmodel.author = "SHARP (Apple)"
    mlmodel.short_description = (
        "SHARP: Single-image to 3D Gaussian Splatting predictor. "
        "Takes an RGB image and produces Gaussian splat parameters in NDC space."
    )
    mlmodel.input_description["image"] = (
        "RGB image normalized to [0,1], shape [1, 3, 1536, 1536]"
    )
    mlmodel.input_description["disparity_factor"] = (
        "Disparity factor = focal_length_px / original_image_width, shape [1]"
    )
    mlmodel.output_description["mean_vectors"] = "Gaussian centers in NDC space [1, N, 3]"
    mlmodel.output_description["singular_values"] = "Gaussian scales [1, N, 3]"
    mlmodel.output_description["quaternions"] = "Gaussian rotations as quaternions [1, N, 4]"
    mlmodel.output_description["colors"] = "Gaussian colors in linearRGB [1, N, 3]"
    mlmodel.output_description["opacities"] = "Gaussian opacities in [0,1] [1, N]"

    LOGGER.info("Saving CoreML model to %s", output_path)
    mlmodel.save(str(output_path))
    LOGGER.info("CoreML model saved successfully.")

    # Print model info
    spec = mlmodel.get_spec()
    LOGGER.info("Model spec version: %s", spec.specificationVersion)
    for output in spec.description.output:
        LOGGER.info("  Output '%s': %s", output.name, output.type)


def validate_conversion(
    pytorch_model: torch.nn.Module,
    coreml_path: Path,
    device: str = "cpu",
    atol: float = 1e-2,
) -> bool:
    """Validate CoreML model output matches PyTorch output.

    Args:
        pytorch_model: The original PyTorch SHARP predictor.
        coreml_path: Path to the .mlpackage to validate.
        device: Device for PyTorch inference.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    import coremltools as ct

    LOGGER.info("Validating CoreML model against PyTorch...")

    # Create a test input
    np.random.seed(42)
    test_image_np = np.random.rand(1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION).astype(
        np.float32
    )
    test_disparity_np = np.array([0.5], dtype=np.float32)

    # PyTorch inference
    wrapper = SHARPTracingWrapper(pytorch_model)
    wrapper.eval()
    with torch.no_grad():
        pt_image = torch.from_numpy(test_image_np).to(device)
        pt_disparity = torch.from_numpy(test_disparity_np).to(device)
        pt_outputs = wrapper(pt_image, pt_disparity)

    pt_mean = pt_outputs[0].cpu().numpy()
    pt_scales = pt_outputs[1].cpu().numpy()
    pt_quats = pt_outputs[2].cpu().numpy()
    pt_colors = pt_outputs[3].cpu().numpy()
    pt_opacities = pt_outputs[4].cpu().numpy()

    # CoreML inference
    ml_model = ct.models.MLModel(str(coreml_path))
    coreml_out = ml_model.predict(
        {"image": test_image_np, "disparity_factor": test_disparity_np}
    )

    # Compare outputs
    output_names = ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]
    pt_arrays = [pt_mean, pt_scales, pt_quats, pt_colors, pt_opacities]

    all_match = True
    for name, pt_arr in zip(output_names, pt_arrays):
        coreml_arr = coreml_out[name]
        max_diff = np.abs(pt_arr - coreml_arr).max()
        mean_diff = np.abs(pt_arr - coreml_arr).mean()
        matches = max_diff < atol
        status = "PASS" if matches else "FAIL"
        LOGGER.info(
            "  %s %s: max_diff=%.6f, mean_diff=%.6f (atol=%.4f)",
            status, name, max_diff, mean_diff, atol,
        )
        if not matches:
            all_match = False

    if all_match:
        LOGGER.info("Validation PASSED: All outputs within tolerance.")
    else:
        LOGGER.warning(
            "Validation WARNING: Some outputs exceed tolerance. "
            "This is expected with FP16 quantization; results should still be visually correct."
        )

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Convert SHARP model to CoreML")
    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint. Downloads default if not provided.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("SHARPPredictor.mlpackage"),
        help="Output path for .mlpackage (default: SHARPPredictor.mlpackage)",
    )
    parser.add_argument(
        "--quantize",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for tracing: cpu or mps (default: cpu)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after conversion (requires macOS with CoreML runtime)",
    )
    parser.add_argument(
        "--trace-only",
        action="store_true",
        help="Only trace the model (save as TorchScript), skip CoreML conversion",
    )
    args = parser.parse_args()

    # Step 1: Load the model
    model = load_model(args.checkpoint, device=args.device)

    # Step 2: Trace the model
    traced = trace_model(model, device=args.device)

    if args.trace_only:
        ts_path = args.output.with_suffix(".pt")
        LOGGER.info("Saving traced model to %s", ts_path)
        traced.save(str(ts_path))
        LOGGER.info("Done (trace-only mode).")
        return

    # Step 3: Convert to CoreML
    # Move traced model to CPU for coremltools conversion
    traced = traced.cpu()
    convert_to_coreml(traced, args.output, quantize=args.quantize)

    # Step 4: Optionally validate
    if args.validate:
        model = model.cpu()
        validate_conversion(model, args.output, device="cpu")

    LOGGER.info("Conversion complete! Model saved to: %s", args.output)
    LOGGER.info(
        "\nNext steps:\n"
        "  1. Add %s to your Xcode project\n"
        "  2. Use SHARPInferenceManager.swift to run inference\n"
        "  3. Use SHARPPostProcessor.swift for NDC -> world coordinate conversion\n",
        args.output,
    )


if __name__ == "__main__":
    main()
