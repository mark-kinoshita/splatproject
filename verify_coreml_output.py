"""Verify CoreML model output by comparing it with PyTorch output on the same image.

This script:
  1. Runs the original PyTorch SHARP model on an image → saves pytorch_output.ply
  2. Runs the CoreML model on the same image → saves coreml_output.ply
  3. Compares the raw output tensors numerically
  4. Uses the SAME Python post-processing for both, so any difference
     is purely from the model conversion.

Usage:
    python verify_coreml_output.py \
        --image data/teaser.jpg \
        --coreml SHARPPredictor.mlpackage \
        --output-dir verify_output/

Requirements:
    pip install coremltools  (only runs on macOS)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
INTERNAL_RESOLUTION = 1536


def preprocess_image(image_path: Path, device: str = "cpu"):
    """Load and preprocess an image exactly as SHARP does."""
    image, _, f_px = io.load_rgb(image_path)
    height, width = image.shape[:2]

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized = F.interpolate(
        image_pt[None],
        size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        mode="bilinear",
        align_corners=True,
    )

    return image_resized, disparity_factor, f_px, height, width


def run_pytorch(image_resized, disparity_factor, checkpoint_path, device="cpu"):
    """Run the original PyTorch model."""
    LOGGER.info("=== PyTorch Inference ===")

    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    model = create_predictor(PredictorParams())
    model.load_state_dict(state_dict)
    model.eval().to(device)

    with torch.no_grad():
        gaussians_ndc = model(image_resized.to(device), disparity_factor.to(device))

    LOGGER.info("  mean_vectors: min=%.4f, max=%.4f, mean=%.4f",
                gaussians_ndc.mean_vectors.min().item(),
                gaussians_ndc.mean_vectors.max().item(),
                gaussians_ndc.mean_vectors.mean().item())
    LOGGER.info("  singular_values: min=%.4f, max=%.4f",
                gaussians_ndc.singular_values.min().item(),
                gaussians_ndc.singular_values.max().item())
    LOGGER.info("  colors: min=%.4f, max=%.4f",
                gaussians_ndc.colors.min().item(),
                gaussians_ndc.colors.max().item())
    LOGGER.info("  opacities: min=%.4f, max=%.4f",
                gaussians_ndc.opacities.min().item(),
                gaussians_ndc.opacities.max().item())

    return gaussians_ndc


def run_coreml(image_resized, disparity_factor):
    """Run the CoreML model."""
    import coremltools as ct

    LOGGER.info("=== CoreML Inference ===")

    model = ct.models.MLModel(str(args.coreml))

    # Log model output spec
    spec = model.get_spec()
    LOGGER.info("  Model outputs:")
    for output in spec.description.output:
        LOGGER.info("    %s: %s", output.name, output.type)

    image_np = image_resized.numpy().astype(np.float32)
    disparity_np = disparity_factor.numpy().astype(np.float32)

    LOGGER.info("  Input image shape: %s, dtype: %s, min: %.4f, max: %.4f",
                image_np.shape, image_np.dtype, image_np.min(), image_np.max())
    LOGGER.info("  Disparity factor: %s", disparity_np)

    result = model.predict({
        "image": image_np,
        "disparity_factor": disparity_np,
    })

    # Check what keys came back
    LOGGER.info("  Output keys: %s", list(result.keys()))

    for key in result:
        arr = result[key]
        if isinstance(arr, np.ndarray):
            LOGGER.info("  %s: shape=%s, dtype=%s, min=%.4f, max=%.4f, mean=%.4f",
                        key, arr.shape, arr.dtype, arr.min(), arr.max(), arr.mean())
        else:
            LOGGER.info("  %s: type=%s", key, type(arr))

    gaussians_ndc = Gaussians3D(
        mean_vectors=torch.from_numpy(result["mean_vectors"]).float(),
        singular_values=torch.from_numpy(result["singular_values"]).float(),
        quaternions=torch.from_numpy(result["quaternions"]).float(),
        colors=torch.from_numpy(result["colors"]).float(),
        opacities=torch.from_numpy(result["opacities"]).float(),
    )

    return gaussians_ndc


def unproject_and_save(gaussians_ndc, f_px, height, width, output_path):
    """Unproject from NDC and save PLY using the standard SHARP code."""
    device = gaussians_ndc.mean_vectors.device

    intrinsics = torch.tensor([
        [f_px, 0, width / 2, 0],
        [0, f_px, height / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32).to(device)

    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_RESOLUTION / width
    intrinsics_resized[1] *= INTERNAL_RESOLUTION / height

    gaussians = unproject_gaussians(
        gaussians_ndc,
        torch.eye(4).to(device),
        intrinsics_resized,
        (INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
    )

    save_ply(gaussians, f_px, (height, width), output_path)
    LOGGER.info("  Saved PLY to: %s", output_path)
    return gaussians


def compare_outputs(pt_ndc, cm_ndc):
    """Compare PyTorch and CoreML outputs."""
    LOGGER.info("=== Comparison ===")
    fields = ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]
    for field in fields:
        pt = getattr(pt_ndc, field).cpu().numpy()
        cm = getattr(cm_ndc, field).cpu().numpy()
        max_diff = np.abs(pt - cm).max()
        mean_diff = np.abs(pt - cm).mean()
        rel_diff = (np.abs(pt - cm) / (np.abs(pt) + 1e-8)).mean()
        LOGGER.info("  %s: max_diff=%.6f, mean_diff=%.6f, rel_diff=%.4f%%",
                    field, max_diff, mean_diff, rel_diff * 100)

        if max_diff > 1.0:
            LOGGER.warning("    ^^^ LARGE DIFFERENCE - model conversion may be broken!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify CoreML model output")
    parser.add_argument("--image", type=Path, required=True, help="Test image path")
    parser.add_argument("--coreml", type=Path, default=Path("SHARPPredictor.mlpackage"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("verify_output"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess
    image_resized, disparity_factor, f_px, height, width = preprocess_image(
        args.image, device=args.device
    )

    # Run PyTorch
    pt_ndc = run_pytorch(
        image_resized, disparity_factor, args.checkpoint, device=args.device
    )
    pt_ndc_cpu = pt_ndc.to(torch.device("cpu"))
    unproject_and_save(
        pt_ndc_cpu, f_px, height, width,
        args.output_dir / "pytorch_output.ply"
    )

    # Run CoreML (if available)
    if args.coreml.exists():
        cm_ndc = run_coreml(image_resized.cpu(), disparity_factor.cpu())
        unproject_and_save(
            cm_ndc, f_px, height, width,
            args.output_dir / "coreml_output.ply"
        )
        compare_outputs(pt_ndc_cpu, cm_ndc)
    else:
        LOGGER.warning("CoreML model not found at %s -- skipping CoreML test.", args.coreml)
        LOGGER.info("Run convert_to_coreml.py first to create it.")

    LOGGER.info("\nDone! Compare the PLY files:")
    LOGGER.info("  - verify_output/pytorch_output.ply  (ground truth)")
    LOGGER.info("  - verify_output/coreml_output.ply   (CoreML conversion)")
    LOGGER.info("If coreml_output.ply looks wrong but pytorch_output.ply looks right,")
    LOGGER.info("the issue is in the CoreML conversion (torch.jit.trace or coremltools).")
