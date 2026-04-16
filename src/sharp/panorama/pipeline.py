"""Panorama-to-Gaussian-splat pipeline.

Orchestrates the conversion of an equirectangular panorama image into a merged
3D Gaussian splat by extracting perspective views, running SHARP inference on
each, and combining the results with correct coordinate transforms.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sharp.cli.predict import predict_image
from sharp.models import RGBGaussianPredictor
from sharp.utils.equirectangular import ViewConfig, equirect_to_perspective, generate_view_configs
from sharp.utils.gaussians import Gaussians3D, apply_transform

LOGGER = logging.getLogger(__name__)


class PanoramaPipeline:
    """Pipeline to convert an equirectangular panorama to a merged Gaussian splat."""

    def __init__(
        self,
        predictor: RGBGaussianPredictor,
        device: torch.device,
        face_size: int = 1536,
        strategy: str = "cubemap",
        fov_deg: float = 90.0,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        blend_margin: float = 0.0,
    ):
        """Initialize the pipeline.

        Args:
            predictor: Loaded SHARP model in eval mode.
            device: Torch device for inference.
            face_size: Resolution of each extracted perspective view (square).
            strategy: View generation strategy ("cubemap" or "grid").
            fov_deg: Field of view for each view in degrees.
            grid_rows: Number of latitude rows (grid strategy only).
            grid_cols: Number of longitude columns (grid strategy only).
            blend_margin: Fraction of view edge where opacity is faded (0.0 = off).
        """
        self.predictor = predictor
        self.device = device
        self.face_size = face_size
        self.blend_margin = blend_margin
        self.views = generate_view_configs(strategy, fov_deg, grid_rows, grid_cols)
        self.fov_deg = fov_deg

    def run(
        self,
        eq_img: np.ndarray,
        save_faces_dir: Path | None = None,
    ) -> Gaussians3D:
        """Execute the full pipeline.

        Args:
            eq_img: Equirectangular image (H, W, 3), uint8.
            save_faces_dir: If set, save extracted face images for debugging.

        Returns:
            Merged Gaussians3D with batch dim [1, N_total, ...].
        """
        LOGGER.info(
            "Pipeline: %d views, face_size=%d, fov=%.1f deg",
            len(self.views),
            self.face_size,
            self.fov_deg,
        )

        view_gaussians: list[Gaussians3D] = []

        for i, view in enumerate(self.views):
            LOGGER.info(
                "[%d/%d] Extracting view '%s' (yaw=%.1f, pitch=%.1f deg)...",
                i + 1,
                len(self.views),
                view.name,
                math.degrees(view.yaw),
                math.degrees(view.pitch),
            )

            # Extract perspective view
            face_img = equirect_to_perspective(
                eq_img, view.yaw, view.pitch, view.fov_h, self.face_size
            )

            if save_faces_dir is not None:
                save_faces_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(face_img).save(save_faces_dir / f"face_{view.name}.png")

            # Run SHARP inference
            LOGGER.info("[%d/%d] Running inference...", i + 1, len(self.views))
            gaussians = self._predict_view(face_img, view.fov_h)

            # Apply edge falloff for overlapping views
            if self.blend_margin > 0:
                gaussians = self._apply_edge_falloff(gaussians, view)

            # Transform to world coordinates
            gaussians = self._transform_to_world(gaussians, view)

            # Move to CPU to free GPU memory
            view_gaussians.append(gaussians.to(torch.device("cpu")))

            n = gaussians.mean_vectors.flatten(0, 1).shape[0]
            LOGGER.info("[%d/%d] Done: %d Gaussians", i + 1, len(self.views), n)

        return self._merge_views(view_gaussians)

    def _predict_view(self, image: np.ndarray, fov_h: float) -> Gaussians3D:
        """Run SHARP inference on a single perspective view."""
        f_px = self.face_size / (2.0 * math.tan(fov_h / 2.0))
        return predict_image(self.predictor, image, f_px, self.device)

    def _transform_to_world(
        self, gaussians: Gaussians3D, view: ViewConfig
    ) -> Gaussians3D:
        """Transform Gaussians from camera-local space to world space."""
        r = torch.from_numpy(view.rotation).float().to(gaussians.mean_vectors.device)
        transform = torch.zeros(3, 4, device=r.device, dtype=r.dtype)
        transform[:3, :3] = r
        return apply_transform(gaussians, transform)

    def _apply_edge_falloff(
        self, gaussians: Gaussians3D, view: ViewConfig
    ) -> Gaussians3D:
        """Reduce opacity of Gaussians near the edges of the view frustum.

        This helps blend overlapping views by fading edge Gaussians.
        """
        means = gaussians.mean_vectors  # [B, N, 3]
        x, y, z = means[..., 0], means[..., 1], means[..., 2]

        # Angular distance from view center
        angle = torch.atan2(torch.sqrt(x**2 + y**2), z.clamp(min=1e-6))
        max_angle = view.fov_h / 2.0

        # Linear ramp: 1.0 at center, 0.0 at edge
        edge_ratio = angle / max_angle
        fade_start = 1.0 - self.blend_margin
        scale = ((1.0 - edge_ratio) / self.blend_margin).clamp(0.0, 1.0)
        # Only apply fade in the margin zone
        scale = torch.where(edge_ratio < fade_start, torch.ones_like(scale), scale)

        return Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=gaussians.singular_values,
            quaternions=gaussians.quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities * scale.squeeze(-1) if scale.dim() > gaussians.opacities.dim() else gaussians.opacities * scale,
        )

    def _merge_views(self, view_gaussians: list[Gaussians3D]) -> Gaussians3D:
        """Concatenate all view Gaussians into a single cloud."""
        means = []
        svals = []
        quats = []
        colors = []
        opacities = []

        for g in view_gaussians:
            # Flatten batch dim if present: [1, N, D] -> [N, D]
            means.append(g.mean_vectors.flatten(0, -2) if g.mean_vectors.dim() > 2 else g.mean_vectors)
            svals.append(g.singular_values.flatten(0, -2) if g.singular_values.dim() > 2 else g.singular_values)
            quats.append(g.quaternions.flatten(0, -2) if g.quaternions.dim() > 2 else g.quaternions)
            colors.append(g.colors.flatten(0, -2) if g.colors.dim() > 2 else g.colors)
            opacities.append(g.opacities.flatten())

        total = sum(m.shape[0] for m in means)
        LOGGER.info("Merging %d views -> %d total Gaussians", len(view_gaussians), total)

        return Gaussians3D(
            mean_vectors=torch.cat(means, dim=0).unsqueeze(0),
            singular_values=torch.cat(svals, dim=0).unsqueeze(0),
            quaternions=torch.cat(quats, dim=0).unsqueeze(0),
            colors=torch.cat(colors, dim=0).unsqueeze(0),
            opacities=torch.cat(opacities, dim=0).unsqueeze(0),
        )
