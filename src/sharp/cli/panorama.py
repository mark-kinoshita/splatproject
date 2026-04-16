"""Contains `sharp panorama` CLI implementation.

Converts an equirectangular panorama image to a merged Gaussian splat .ply file
by extracting perspective views, running SHARP inference on each, and combining
the results with correct coordinate transforms.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image

from sharp.models import PredictorParams, create_predictor
from sharp.panorama.pipeline import PanoramaPipeline
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import save_ply

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to an equirectangular panorama image.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path for the output .ply file.",
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to .pt checkpoint. Downloads default if not provided.",
)
@click.option(
    "--face-size",
    type=int,
    default=1536,
    help="Resolution of each extracted perspective view (square).",
)
@click.option(
    "--strategy",
    type=click.Choice(["cubemap", "grid"]),
    default="cubemap",
    help="View generation strategy.",
)
@click.option(
    "--fov",
    type=float,
    default=90.0,
    help="Field of view in degrees for each view.",
)
@click.option(
    "--grid-rows",
    type=int,
    default=None,
    help="Number of latitude rows (grid strategy only).",
)
@click.option(
    "--grid-cols",
    type=int,
    default=None,
    help="Number of longitude columns (grid strategy only).",
)
@click.option(
    "--blend-margin",
    type=float,
    default=0.0,
    help="Edge opacity falloff fraction for overlapping views (0.0 = no blending).",
)
@click.option(
    "--save-faces",
    is_flag=True,
    help="Save extracted perspective views as debug images.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def panorama_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    face_size: int,
    strategy: str,
    fov: float,
    grid_rows: int | None,
    grid_cols: int | None,
    blend_margin: float,
    save_faces: bool,
    device: str,
    verbose: bool,
):
    """Convert an equirectangular panorama to a Gaussian splat .ply file."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    # Device selection
    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    torch_device = torch.device(device)
    LOGGER.info("Using device: %s", torch_device)

    # Load equirectangular image
    LOGGER.info("Loading panorama: %s", input_path)
    eq_pil = Image.open(input_path)
    eq_img = np.asarray(eq_pil)
    if eq_img.ndim == 2:
        eq_img = np.stack([eq_img] * 3, axis=-1)
    eq_img = eq_img[:, :, :3]  # Strip alpha if present
    LOGGER.info("Panorama shape: %s", eq_img.shape)

    h, w = eq_img.shape[:2]
    aspect = w / h
    if abs(aspect - 2.0) > 0.3:
        LOGGER.warning(
            "Image aspect ratio %.2f differs from expected 2:1 for equirectangular. "
            "Results may be distorted.",
            aspect,
        )

    # Load model
    LOGGER.info("Loading SHARP model...")
    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=torch_device)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval().to(torch_device)

    # Run pipeline
    pipeline = PanoramaPipeline(
        predictor=predictor,
        device=torch_device,
        face_size=face_size,
        strategy=strategy,
        fov_deg=fov,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        blend_margin=blend_margin,
    )

    save_faces_dir = output_path.parent / "faces" if save_faces else None
    merged = pipeline.run(eq_img, save_faces_dir=save_faces_dir)

    # Save output
    import math

    f_px = face_size / (2.0 * math.tan(math.radians(fov) / 2.0))
    image_shape = (face_size, face_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_ply(merged, f_px, image_shape, output_path)
    LOGGER.info("Saved merged PLY: %s", output_path)
    LOGGER.info("Done!")
