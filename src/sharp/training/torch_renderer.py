"""Pure-PyTorch differentiable Gaussian point-cloud renderer.

Used as a fallback when gsplat's CUDA extension is unavailable (e.g., Apple
Silicon MPS).  It is slower than gsplat but runs on any device.

Design
------
Each Gaussian is projected to 2D screen space.  Colors and inverse-depth
values are accumulated per pixel via ``scatter_add``.  The scatter operation
is differentiable with respect to the *values* (colors, depths, opacities),
so the head can learn better colors and depths from the photometric and depth
losses.  Gradient through the *positions* is not supported by scatter_add,
but depth consistency is enforced via a separate depth proxy loss.

This renderer is sufficient for training the regression head; for production-
quality scene reconstruction use gsplat on a CUDA device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sharp.utils.gaussians import Gaussians3D


def _perspective_project(
    means3d: torch.Tensor,   # [N, 3]  metric 3D positions in camera space
    intrinsics: torch.Tensor,  # [3, 3] or [4, 4]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D means to float pixel coords.  Returns (u_f, v_f, z)."""
    z = means3d[:, 2].clamp(min=1e-3)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u_f = fx * means3d[:, 0] / z + cx  # [N]
    v_f = fy * means3d[:, 1] / z + cy  # [N]
    return u_f, v_f, z


def torch_render_single(
    gaussians: Gaussians3D,
    intrinsics: torch.Tensor,  # [4, 4]
    width: int,
    height: int,
    sigma_px: float = 1.0,
    kernel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render a single (non-batched) Gaussians3D into an image.

    Splatting strategy:
    1.  Project Gaussian centres to 2D.
    2.  For each pixel in a (kernel × kernel) neighbourhood around the
        projected centre, accumulate colour and inverse-depth weighted by
        a 2D Gaussian kernel and opacity.
    3.  Normalise by total weight to get the final colour and depth maps.

    Returns:
        color:  [3, H, W] in [0, 1]
        depth:  [1, H, W] in metres (0 where no coverage)
        alpha:  [1, H, W] accumulated alpha ∈ [0, 1]
    """
    means3d = gaussians.mean_vectors   # [N, 3]
    colors = gaussians.colors          # [N, 3]
    opacities = gaussians.opacities    # [N] or [N, 1]
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)  # [N]

    device = means3d.device
    N = means3d.shape[0]

    u_f, v_f, z = _perspective_project(means3d, intrinsics)

    # Visibility mask: in front of camera and inside image (+margin)
    margin = kernel
    vis = (
        (u_f >= -margin) & (u_f < width + margin) &
        (v_f >= -margin) & (v_f < height + margin) &
        (z > 1e-3)
    )

    if vis.sum() == 0:
        return (
            torch.zeros(3, height, width, device=device),
            torch.zeros(1, height, width, device=device),
            torch.zeros(1, height, width, device=device),
        )

    u_f_v = u_f[vis]   # [M]
    v_f_v = v_f[vis]   # [M]
    z_v = z[vis]        # [M]
    c_v = colors[vis]   # [M, 3]
    o_v = opacities[vis]  # [M]

    color_accum = torch.zeros(height * width, 3, device=device)
    invdepth_accum = torch.zeros(height * width, device=device)
    weight_accum = torch.zeros(height * width, device=device)

    half_k = kernel // 2
    for di in range(-half_k, half_k + 1):
        for dj in range(-half_k, half_k + 1):
            # Gaussian kernel weight for this offset.
            kern_w = torch.exp(
                torch.tensor(-(di**2 + dj**2) / (2.0 * sigma_px**2), device=device)
            )

            ui = (u_f_v + dj).long().clamp(0, width - 1)   # [M]
            vi = (v_f_v + di).long().clamp(0, height - 1)  # [M]
            idx = vi * width + ui  # [M]

            w = (o_v * kern_w).float()  # [M]

            color_accum.scatter_add_(0, idx.unsqueeze(-1).expand(-1, 3), c_v.float() * w.unsqueeze(-1))
            invdepth_accum.scatter_add_(0, idx, (w / z_v.clamp(1e-3)).float())
            weight_accum.scatter_add_(0, idx, w)

    eps = 1e-6
    rendered_color = (color_accum / (weight_accum.unsqueeze(-1) + eps)
                      ).reshape(height, width, 3).permute(2, 0, 1)  # [3, H, W]
    rendered_depth = (weight_accum / (invdepth_accum + eps)
                      ).reshape(1, height, width)  # [1, H, W]
    rendered_alpha = (weight_accum / (weight_accum.max() + eps)
                      ).clamp(0, 1).reshape(1, height, width)  # [1, H, W]

    return rendered_color, rendered_depth, rendered_alpha


def torch_render_batch(
    gaussians: Gaussians3D,
    extrinsics: torch.Tensor,   # [B, 4, 4]  world-to-camera (identity = input view)
    intrinsics: torch.Tensor,   # [B, 4, 4]
    width: int,
    height: int,
    sigma_px: float = 1.0,
    kernel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched wrapper around torch_render_single.

    Args:
        gaussians:   Gaussians3D with batch dimension B.
        extrinsics:  [B, 4, 4] world-to-camera transforms (OpenCV).
        intrinsics:  [B, 4, 4] camera intrinsics.
        width, height: output resolution.

    Returns:
        color:  [B, 3, H, W]
        depth:  [B, 1, H, W]
        alpha:  [B, 1, H, W]
    """
    B = gaussians.mean_vectors.shape[0]
    colors_out, depths_out, alphas_out = [], [], []

    for b in range(B):
        m = gaussians.mean_vectors[b]    # [N, 3]
        c = gaussians.colors[b]          # [N, 3]
        o = gaussians.opacities[b]       # [N] or [N, 1]
        sv = gaussians.singular_values[b]  # [N, 3]
        q = gaussians.quaternions[b]       # [N, 4]

        # Apply world-to-camera transform to the means (differentiable for means).
        ext = extrinsics[b]   # [4, 4]
        R = ext[:3, :3]       # [3, 3]
        t = ext[:3, 3]        # [3]
        m_cam = m @ R.T + t   # [N, 3]

        g_single = Gaussians3D(
            mean_vectors=m_cam,
            singular_values=sv,
            quaternions=q,
            colors=c,
            opacities=o,
        )
        col, dep, alp = torch_render_single(
            g_single, intrinsics[b], width, height, sigma_px=sigma_px, kernel=kernel
        )
        colors_out.append(col)
        depths_out.append(dep)
        alphas_out.append(alp)

    return (
        torch.stack(colors_out),   # [B, 3, H, W]
        torch.stack(depths_out),   # [B, 1, H, W]
        torch.stack(alphas_out),   # [B, 1, H, W]
    )
