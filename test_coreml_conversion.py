"""Test script to validate the CoreML conversion against PyTorch.

Usage:
    # First, convert the model:
    python convert_to_coreml.py -o SHARPPredictor.mlpackage --validate

    # Or run this standalone test:
    python test_coreml_conversion.py --coreml SHARPPredictor.mlpackage

This script:
  1. Loads the SHARP PyTorch model and runs inference on a test image
  2. Loads the CoreML model and runs inference on the same test image
  3. Compares outputs numerically and reports differences
  4. Optionally runs inference on a real image and saves .ply output from both

Requirements:
    pip install coremltools torch torchvision timm numpy
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

INTERNAL_RESOLUTION = 1536


def benchmark_pytorch(model, device="cpu", num_runs=3):
    """Benchmark PyTorch inference speed."""
    from convert_to_coreml import SHARPTracingWrapper

    wrapper = SHARPTracingWrapper(model)
    wrapper.eval()
    wrapper.to(device)

    image = torch.randn(1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION, device=device).clamp(0, 1)
    disparity = torch.tensor([0.5], device=device)

    # Warmup
    with torch.no_grad():
        _ = wrapper(image, disparity)

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = wrapper(image, disparity)
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        LOGGER.info("  PyTorch run %d: %.2fs", i + 1, elapsed)

    avg = np.mean(times)
    LOGGER.info("  PyTorch avg: %.2fs (device=%s)", avg, device)
    return avg


def benchmark_coreml(coreml_path, num_runs=3):
    """Benchmark CoreML inference speed."""
    import coremltools as ct

    ml_model = ct.models.MLModel(str(coreml_path))

    image = np.random.rand(1, 3, INTERNAL_RESOLUTION, INTERNAL_RESOLUTION).astype(np.float32)
    disparity = np.array([0.5], dtype=np.float32)

    # Warmup
    _ = ml_model.predict({"image": image, "disparity_factor": disparity})

    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        _ = ml_model.predict({"image": image, "disparity_factor": disparity})
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        LOGGER.info("  CoreML run %d: %.2fs", i + 1, elapsed)

    avg = np.mean(times)
    LOGGER.info("  CoreML avg: %.2fs", avg)
    return avg


def test_with_real_image(
    model, coreml_path, image_path, output_dir, device="cpu"
):
    """Run both PyTorch and CoreML on a real image and save PLY outputs."""
    import torch.nn.functional as F

    from sharp.utils import io
    from sharp.utils.gaussians import save_ply, unproject_gaussians

    LOGGER.info("Testing with real image: %s", image_path)

    # Load image
    image, _, f_px = io.load_rgb(Path(image_path))
    height, width = image.shape[:2]

    # Preprocess
    image_pt = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
    image_resized = F.interpolate(
        image_pt[None],
        size=(INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        mode="bilinear",
        align_corners=True,
    )
    disparity_factor = torch.tensor([f_px / width]).float()

    # PyTorch inference
    from convert_to_coreml import SHARPTracingWrapper

    wrapper = SHARPTracingWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        pt_out = wrapper(image_resized.to(device), disparity_factor.to(device))

    from sharp.utils.gaussians import Gaussians3D

    pt_gaussians_ndc = Gaussians3D(
        mean_vectors=pt_out[0].cpu(),
        singular_values=pt_out[1].cpu(),
        quaternions=pt_out[2].cpu(),
        colors=pt_out[3].cpu(),
        opacities=pt_out[4].cpu(),
    )

    # Unproject and save PyTorch output
    intrinsics = torch.tensor(
        [[f_px, 0, width / 2, 0], [0, f_px, height / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_RESOLUTION / width
    intrinsics_resized[1] *= INTERNAL_RESOLUTION / height

    pt_gaussians = unproject_gaussians(
        pt_gaussians_ndc,
        torch.eye(4),
        intrinsics_resized,
        (INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_ply(pt_gaussians, f_px, (height, width), output_dir / "pytorch_output.ply")
    LOGGER.info("Saved PyTorch output to %s", output_dir / "pytorch_output.ply")

    # CoreML inference (if available)
    if coreml_path and Path(coreml_path).exists():
        import coremltools as ct

        ml_model = ct.models.MLModel(str(coreml_path))
        coreml_out = ml_model.predict({
            "image": image_resized.numpy(),
            "disparity_factor": disparity_factor.numpy(),
        })

        coreml_gaussians_ndc = Gaussians3D(
            mean_vectors=torch.from_numpy(coreml_out["mean_vectors"]),
            singular_values=torch.from_numpy(coreml_out["singular_values"]),
            quaternions=torch.from_numpy(coreml_out["quaternions"]),
            colors=torch.from_numpy(coreml_out["colors"]),
            opacities=torch.from_numpy(coreml_out["opacities"]),
        )

        coreml_gaussians = unproject_gaussians(
            coreml_gaussians_ndc,
            torch.eye(4),
            intrinsics_resized,
            (INTERNAL_RESOLUTION, INTERNAL_RESOLUTION),
        )

        save_ply(coreml_gaussians, f_px, (height, width), output_dir / "coreml_output.ply")
        LOGGER.info("Saved CoreML output to %s", output_dir / "coreml_output.ply")

        # Compare
        for name in ["mean_vectors", "singular_values", "quaternions", "colors", "opacities"]:
            pt_arr = getattr(pt_gaussians_ndc, name).numpy()
            cm_arr = coreml_out[name]
            max_diff = np.abs(pt_arr - cm_arr).max()
            LOGGER.info("  %s max_diff: %.6f", name, max_diff)


def main():
    parser = argparse.ArgumentParser(description="Test CoreML conversion")
    parser.add_argument(
        "--coreml",
        type=Path,
        default=Path("SHARPPredictor.mlpackage"),
        help="Path to CoreML model",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to PyTorch checkpoint (downloads default if not provided)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a test image for end-to-end validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_output"),
        help="Directory for test outputs",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for PyTorch (cpu/mps)"
    )
    args = parser.parse_args()

    # Load PyTorch model
    from convert_to_coreml import load_model

    model = load_model(args.checkpoint, device=args.device)

    if args.benchmark:
        LOGGER.info("=== Benchmarking PyTorch ===")
        benchmark_pytorch(model, device=args.device)

        if args.coreml.exists():
            LOGGER.info("=== Benchmarking CoreML ===")
            benchmark_coreml(args.coreml)

    if args.image:
        test_with_real_image(
            model, args.coreml, args.image, args.output_dir, device=args.device
        )

    LOGGER.info("Test complete.")


if __name__ == "__main__":
    main()
