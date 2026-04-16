"""Equirectangular panorama projection utilities.

Provides functions for extracting perspective views from equirectangular images
and generating view configurations for panorama-to-splat pipelines.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class ViewConfig:
    """Configuration for a single perspective view extracted from a panorama."""

    name: str
    yaw: float  # radians, positive = turn right (+Z toward +X)
    pitch: float  # radians, positive = tilt down (+Z toward +Y)
    fov_h: float  # horizontal field of view in radians
    rotation: np.ndarray  # 3x3 camera-to-world rotation matrix


def compute_view_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Compute the camera-to-world rotation matrix for a given view direction.

    Convention: OpenCV (x right, y down, z forward).
    Camera starts at origin looking down +Z. Yaw rotates the view direction
    in the horizontal plane, pitch tilts up/down.

    Args:
        yaw: Horizontal angle in radians. 0 = forward (+Z).
             Positive = turn right (toward +X).
        pitch: Vertical angle in radians. 0 = horizon.
               Positive = tilt down (toward +Y).

    Returns:
        3x3 camera-to-world rotation matrix (to be used with apply_transform).
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    r_yaw = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ])

    r_pitch = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp],
    ])

    return r_yaw @ r_pitch


def equirect_to_perspective(
    eq_img: np.ndarray,
    yaw: float,
    pitch: float,
    fov_h: float,
    face_size: int,
) -> np.ndarray:
    """Sample a perspective view from an equirectangular image.

    Args:
        eq_img: Equirectangular image (H, W, C), uint8.
                Top row = +90 latitude (north pole), bottom = -90.
        yaw: Horizontal angle in radians. 0 = front center of panorama.
        pitch: Vertical angle in radians. 0 = horizon.
        fov_h: Horizontal field of view in radians.
        face_size: Output image resolution (square).

    Returns:
        Perspective image (face_size, face_size, C), same dtype as input.
    """
    h, w = eq_img.shape[:2]
    f_px = face_size / (2.0 * math.tan(fov_h / 2.0))

    # Pixel grid in the output face
    jj, ii = np.meshgrid(
        np.arange(face_size, dtype=np.float32) + 0.5,
        np.arange(face_size, dtype=np.float32) + 0.5,
        indexing="ij",
    )

    # Ray directions in camera-local space (OpenCV: x right, y down, z forward)
    x_cam = (ii - face_size / 2.0) / f_px
    y_cam = (jj - face_size / 2.0) / f_px
    z_cam = np.ones_like(x_cam)

    # Normalize
    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam, y_cam, z_cam = x_cam / norm, y_cam / norm, z_cam / norm

    # Rotate to world coordinates
    r = compute_view_rotation(yaw, pitch)
    dirs = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (face_size, face_size, 3)
    dirs_world = dirs @ r.T  # (face_size, face_size, 3)

    x_w = dirs_world[..., 0]
    y_w = dirs_world[..., 1]
    z_w = dirs_world[..., 2]

    # World directions to equirectangular coordinates
    # lon: -pi to pi, lat: -pi/2 to pi/2
    lon = np.arctan2(x_w, z_w)
    lat = np.arcsin(np.clip(-y_w, -1, 1))  # -y because equirect top = north = -Y in OpenCV

    # Map to pixel coordinates
    u_eq = (lon + math.pi) / (2.0 * math.pi)  # [0, 1]
    v_eq = (math.pi / 2.0 - lat) / math.pi  # [0, 1], top = north pole

    src_x = (u_eq * w) % w
    src_y = np.clip(v_eq * h, 0, h - 1.001)

    # Bilinear interpolation
    x0 = np.floor(src_x).astype(np.int32) % w
    y0 = np.floor(src_y).astype(np.int32)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (src_x - np.floor(src_x))[..., np.newaxis]
    fy = (src_y - np.floor(src_y))[..., np.newaxis]

    out = (
        (1 - fx) * (1 - fy) * eq_img[y0, x0]
        + fx * (1 - fy) * eq_img[y0, x1]
        + (1 - fx) * fy * eq_img[y1, x0]
        + fx * fy * eq_img[y1, x1]
    )
    return out.astype(eq_img.dtype)


def generate_view_configs(
    strategy: str = "cubemap",
    fov_deg: float = 90.0,
    rows: int | None = None,
    cols: int | None = None,
) -> list[ViewConfig]:
    """Generate view configurations for panorama coverage.

    Args:
        strategy: "cubemap" for standard 6-face coverage, or "grid" for a
                  configurable latitude/longitude grid.
        fov_deg: Field of view in degrees for each view.
        rows: Number of latitude rows (grid strategy only).
        cols: Number of longitude columns (grid strategy only).

    Returns:
        List of ViewConfig describing each perspective view.
    """
    fov_rad = math.radians(fov_deg)

    if strategy == "cubemap":
        configs = [
            ("front", 0.0, 0.0),
            ("back", math.pi, 0.0),
            ("right", math.pi / 2, 0.0),
            ("left", -math.pi / 2, 0.0),
            ("down", 0.0, -math.pi / 2),
            ("up", 0.0, math.pi / 2),
        ]
        return [
            ViewConfig(
                name=name,
                yaw=yaw,
                pitch=pitch,
                fov_h=fov_rad,
                rotation=compute_view_rotation(yaw, pitch),
            )
            for name, yaw, pitch in configs
        ]

    if strategy == "grid":
        grid_rows = rows if rows is not None else 3
        grid_cols = cols if cols is not None else 8
        views = []
        pitch_step = math.pi / grid_rows
        for r in range(grid_rows):
            pitch = -math.pi / 2.0 + pitch_step * (r + 0.5)
            for c in range(grid_cols):
                yaw = -math.pi + (2.0 * math.pi / grid_cols) * c
                name = f"row{r}_col{c}"
                views.append(
                    ViewConfig(
                        name=name,
                        yaw=yaw,
                        pitch=pitch,
                        fov_h=fov_rad,
                        rotation=compute_view_rotation(yaw, pitch),
                    )
                )
        return views

    raise ValueError(f"Unknown strategy: {strategy!r}. Use 'cubemap' or 'grid'.")
