"""Loss functions for SHARP Gaussian-head training.

Three supervisory signals are combined:

1. **Reconstruction loss** — render Gaussians back to the *source* viewpoint
   and compare with the input image (L1).  Ensures colours/opacities are
   grounded in the observed appearance.

2. **Novel-view-synthesis loss** — render Gaussians to a *target* viewpoint
   and compare with the actual target frame (L1 + SSIM).  This is the
   primary signal that forces the model to produce geometrically correct
   Gaussian placements.

3. **Depth loss** — compare the alpha-composited render depth with the
   LiDAR ground-truth depth from ARKitScenes (masked L1).  Penalises
   floaters and keeps the predicted Gaussian cloud close to the true
   surface.

Opacity regularisation discourages degenerate all-opaque solutions.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# SSIM helper (pure PyTorch, no extra deps)
# ---------------------------------------------------------------------------

def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def _ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable SSIM loss (1 - SSIM) averaged over channels.

    Args:
        pred:    [B, C, H, W] in [0, 1].
        target:  [B, C, H, W] in [0, 1].

    Returns:
        Scalar loss in [0, 1].
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)  # [W, W]
    # Expand to [1, 1, W, W] and repeat for all channels.
    C = pred.shape[1]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)  # [C, 1, W, W]

    pad = window_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sig_x = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_x_sq
    sig_y = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_y_sq
    sig_xy = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sig_x + sig_y + C2)
    )
    loss_map = (1.0 - ssim_map) / 2.0  # normalise to [0, 1]

    if reduction == "mean":
        return loss_map.mean()
    return loss_map


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

def photometric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor | None = None,
    use_ssim: bool = True,
    ssim_weight: float = 0.15,
) -> torch.Tensor:
    """L1 (+ optional SSIM) photometric loss.

    Args:
        pred:    [B, 3, H, W] rendered image in [0, 1].
        target:  [B, 3, H, W] ground-truth image in [0, 1].
        alpha:   [B, 1, H, W] accumulated alpha (optional mask).
        use_ssim: Whether to add the SSIM term.
        ssim_weight: Relative weight of the SSIM component.

    Returns:
        Scalar loss.
    """
    if alpha is not None:
        # Mask out background (alpha ≈ 0) so the model is not penalised for
        # background colour when scene coverage is incomplete.
        mask = (alpha > 0.1).float()
        pred = pred * mask
        target = target * mask

    l1 = F.l1_loss(pred, target)

    if use_ssim:
        ss = _ssim_loss(pred.clamp(0, 1), target.clamp(0, 1))
        return (1.0 - ssim_weight) * l1 + ssim_weight * ss

    return l1


def depth_loss(
    rendered_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    alpha: torch.Tensor | None = None,
    scale_invariant: bool = True,
) -> torch.Tensor:
    """Masked depth supervision loss.

    Args:
        rendered_depth: [B, 1, H, W] depth from Gaussian render (metres).
        gt_depth:       [B, 1, H, W] LiDAR depth (metres; 0 = invalid).
        alpha:          [B, 1, H, W] render alpha (skip near-zero coverage).
        scale_invariant: Whether to apply scale-invariant log loss in
                         addition to the direct L1 term.

    Returns:
        Scalar loss.
    """
    valid = gt_depth > 1e-3  # LiDAR invalid pixels are 0

    if alpha is not None:
        valid = valid & (alpha > 0.1)

    if valid.sum() < 10:
        return rendered_depth.sum() * 0.0  # no valid pixels → zero gradient

    pred_valid = rendered_depth[valid]
    gt_valid = gt_depth[valid]

    l1 = F.l1_loss(pred_valid, gt_valid)

    if scale_invariant:
        # Log-scale L1 reduces sensitivity to absolute scale errors.
        log_pred = torch.log(pred_valid.clamp(min=1e-3))
        log_gt = torch.log(gt_valid.clamp(min=1e-3))
        si_loss = F.l1_loss(log_pred, log_gt)
        return 0.5 * l1 + 0.5 * si_loss

    return l1


def opacity_regularisation(opacities: torch.Tensor) -> torch.Tensor:
    """Encourage sparsity by penalising intermediate opacity values.

    Uses a bimodal prior: good Gaussians should be either nearly opaque
    (surfaces) or nearly transparent (background).  The entropy
    ``H(o) = -o*log(o) - (1-o)*log(1-o)`` is maximised at o=0.5, so
    minimising it pushes opacities toward 0 or 1.

    Args:
        opacities: [B, N, 1] or [B, N] opacity values in (0, 1).

    Returns:
        Scalar regularisation loss.
    """
    o = opacities.clamp(1e-6, 1 - 1e-6)
    entropy = -(o * torch.log(o) + (1 - o) * torch.log(1 - o))
    return entropy.mean()


# ---------------------------------------------------------------------------
# Loss configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LossWeights:
    """Relative weights for each loss component.

    Recommended starting values (adjust based on training curves):
    - src_reconstruction: 1.0  — sanity constraint, keeps colours grounded.
    - nvs:                1.0  — primary training signal for 3D geometry.
    - depth:              0.1  — depth regulariser (don't overwhelm NVS).
    - opacity_reg:        0.01 — prevent opacity collapse.
    """
    src_reconstruction: float = 1.0
    nvs: float = 1.0
    depth: float = 0.1
    opacity_reg: float = 0.01


# ---------------------------------------------------------------------------
# Composite loss module
# ---------------------------------------------------------------------------

class GaussianTrainingLoss(nn.Module):
    """Combines all loss terms into a single scalar for back-propagation.

    Usage::

        loss_fn = GaussianTrainingLoss(weights=LossWeights())
        loss, breakdown = loss_fn(
            src_rendered_color=...,
            src_rendered_depth=...,
            src_rendered_alpha=...,
            src_target=...,
            src_gt_depth=...,
            tgt_rendered_color=...,
            tgt_rendered_alpha=...,
            tgt_target=...,
            gaussian_opacities=...,
        )
        loss.backward()
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        use_ssim: bool = True,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.use_ssim = use_ssim

    def forward(
        self,
        *,
        src_rendered_color: torch.Tensor,    # [B, 3, H, W]
        src_rendered_depth: torch.Tensor,    # [B, 1, H, W]
        src_rendered_alpha: torch.Tensor,    # [B, 1, H, W]
        src_target: torch.Tensor,            # [B, 3, H, W] input image
        src_gt_depth: torch.Tensor,          # [B, 1, H, W] ARKitScenes LiDAR
        tgt_rendered_color: torch.Tensor,    # [B, 3, H, W]
        tgt_rendered_alpha: torch.Tensor,    # [B, 1, H, W]
        tgt_target: torch.Tensor,            # [B, 3, H, W] target frame
        gaussian_opacities: torch.Tensor,    # [B, N, 1] or [B, N]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total weighted loss and per-term breakdown.

        Returns:
            (total_loss, breakdown_dict)
        """
        w = self.weights

        loss_src = photometric_loss(
            src_rendered_color, src_target,
            alpha=src_rendered_alpha,
            use_ssim=self.use_ssim,
        )

        loss_nvs = photometric_loss(
            tgt_rendered_color, tgt_target,
            alpha=tgt_rendered_alpha,
            use_ssim=self.use_ssim,
        )

        loss_depth = depth_loss(
            src_rendered_depth, src_gt_depth,
            alpha=src_rendered_alpha,
        )

        loss_opacity = opacity_regularisation(gaussian_opacities)

        total = (
            w.src_reconstruction * loss_src
            + w.nvs * loss_nvs
            + w.depth * loss_depth
            + w.opacity_reg * loss_opacity
        )

        breakdown = {
            "loss_src": loss_src.item(),
            "loss_nvs": loss_nvs.item(),
            "loss_depth": loss_depth.item(),
            "loss_opacity": loss_opacity.item(),
            "loss_total": total.item(),
        }

        return total, breakdown
