"""ARKitScenes dataset loader for SHARP backbone-swap training.

Dataset layout expected (from Apple's official downloader):
    <root>/
        <split>/                  # e.g. "Training"
            <scene_id>/
                <scene_id>_frames/
                    lowres_wide/              # RGB PNGs  (1920×1440)
                    lowres_depth/             # Depth PNGs (uint16, millimetres)
                    lowres_wide_intrinsics/   # .pincam files ("w h fx fy cx cy")
                lowres_wide.traj              # per-frame poses
                                              # "timestamp tx ty tz qx qy qz qw"

Legal note
----------
ARKitScenes is released by Apple under a permissive non-commercial research
licence.  The *dataset* is used only at training time; trained weights
(your regression head) are your own IP and may be shipped commercially.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)

# ARKit camera convention (OpenGL-like): +X right, +Y up, +Z toward viewer.
# OpenCV convention used by SHARP/gsplat:  +X right, +Y down, +Z into scene.
# Flip Y and Z on the camera frame to convert.
_ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])  # 4×4


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_pincam(path: Path) -> tuple[int, int, float, float, float, float]:
    """Parse a .pincam intrinsics file → (w, h, fx, fy, cx, cy)."""
    parts = path.read_text().strip().split()
    w, h = int(parts[0]), int(parts[1])
    fx, fy = float(parts[2]), float(parts[3])
    cx, cy = float(parts[4]), float(parts[5])
    return w, h, fx, fy, cx, cy


def _build_intrinsics_4x4(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    K = np.eye(4, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def _load_traj(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Parse .traj file → (timestamps_array, list_of_c2w_arkit_4x4).

    ARKitScenes traj format (7 fields per line):
        timestamp  tx ty tz  rx ry rz
    where (rx, ry, rz) is a rotation *vector* (axis × angle in radians),
    NOT a quaternion.  ``scipy.spatial.transform.Rotation.from_rotvec``
    decodes it correctly.
    """
    timestamps: list[float] = []
    poses: list[np.ndarray] = []

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        ts = float(parts[0])
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        rx, ry, rz = float(parts[4]), float(parts[5]), float(parts[6])
        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]
        timestamps.append(ts)
        poses.append(c2w)

    return np.array(timestamps, dtype=np.float64), poses


def _nearest_pose(
    frame_ts: float,
    traj_timestamps: np.ndarray,
    traj_poses: list[np.ndarray],
    max_dt: float = 0.1,
) -> np.ndarray | None:
    """Return the c2w pose whose timestamp is nearest to frame_ts.

    Returns None if the closest pose is more than max_dt seconds away.
    """
    if len(traj_timestamps) == 0:
        return None
    idx = int(np.argmin(np.abs(traj_timestamps - frame_ts)))
    if abs(traj_timestamps[idx] - frame_ts) > max_dt:
        return None
    return traj_poses[idx]


def _arkit_c2w_to_opencv_c2w(c2w_arkit: np.ndarray) -> np.ndarray:
    """Convert ARKit camera-to-world to OpenCV camera-to-world.

    The ARKit camera frame has +Z pointing toward the viewer (+Y up).
    The OpenCV camera frame has +Z pointing into the scene (+Y down).
    We post-multiply by a flip matrix that re-maps the camera axes.
    """
    return (c2w_arkit @ _ARKIT_TO_OPENCV).astype(np.float32)


def _relative_extrinsics(src_c2w: np.ndarray, tgt_c2w: np.ndarray) -> np.ndarray:
    """World-to-target-cam expressed in source-cam coordinate frame.

    The SHARP model outputs Gaussians in the source-camera coordinate system
    (unprojected with identity extrinsics).  gsplat's ``viewmats`` expects a
    world-to-camera matrix, where "world" here IS the source camera frame.

    Returns the 4×4 matrix that maps source-cam points to target-cam points,
    i.e. ``T_{src → tgt} = inv(tgt_c2w) @ src_c2w``.
    """
    tgt_w2c = np.linalg.inv(tgt_c2w)
    return (tgt_w2c @ src_c2w).astype(np.float32)


def _load_rgb(path: Path) -> np.ndarray:
    """Load an RGB PNG as float32 [H, W, 3] in [0, 1]."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img


def _load_depth(path: Path) -> np.ndarray:
    """Load a uint16 depth PNG → float32 metres [H, W]."""
    depth_mm = np.array(Image.open(path), dtype=np.float32)
    return depth_mm / 1000.0  # millimetres → metres


def _resize_image(
    img: np.ndarray,
    size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize HWC numpy array to (size, size) → CHW tensor in [0, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1CHW
    t = F.interpolate(t, size=(size, size), mode=mode, align_corners=True)
    return t.squeeze(0)  # CHW


def _resize_depth(depth: np.ndarray, size: int) -> torch.Tensor:
    """Resize HW depth array to (size, size) using nearest-neighbour."""
    t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # 11HW
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0)  # 1HW


def _scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    orig_w: int, orig_h: int,
    new_size: int,
) -> tuple[float, float, float, float]:
    """Scale pinhole intrinsics after a non-square→square resize."""
    sx = new_size / orig_w
    sy = new_size / orig_h
    return fx * sx, fy * sy, cx * sx, cy * sy


# ---------------------------------------------------------------------------
# Per-scene index builder
# ---------------------------------------------------------------------------

class _FramePair:
    __slots__ = ("scene_id", "src_rgb", "src_depth", "src_intr", "src_ts",
                 "tgt_rgb", "tgt_depth", "tgt_intr", "tgt_ts",
                 "traj_path")

    def __init__(
        self,
        scene_id: str,
        src_rgb: Path, src_depth: Path, src_intr: Path, src_ts: str,
        tgt_rgb: Path, tgt_depth: Path, tgt_intr: Path, tgt_ts: str,
        traj_path: Path,
    ):
        self.scene_id = scene_id
        self.src_rgb, self.src_depth, self.src_intr = src_rgb, src_depth, src_intr
        self.src_ts = src_ts
        self.tgt_rgb, self.tgt_depth, self.tgt_intr = tgt_rgb, tgt_depth, tgt_intr
        self.tgt_ts = tgt_ts
        self.traj_path = traj_path


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class ARKitScenesDataset(Dataset):
    """ARKitScenes frame-pair dataset for SHARP regression head training.

    Each sample provides:
    - **source_image**      [3, S, S]  float32 in [0, 1]
    - **source_depth**      [1, S, S]  float32 metres (0 = invalid)
    - **source_intrinsics** [4, 4]     float32 pinhole K (square-resized)
    - **disparity_factor**  scalar     = fx_original / original_width
    - **target_image**      [3, S, S]  float32 in [0, 1]
    - **target_intrinsics** [4, 4]     float32 pinhole K (square-resized)
    - **rel_extrinsics**    [4, 4]     float32 world-to-target-cam
                                       (world = source-cam frame)

    Args:
        root:         Root of the ARKitScenes download.
        split:        "Training" or "Validation".
        input_size:   Square side length to resize images to (e.g. 768).
        frame_gap:    Number of frames between source and target in each pair.
        max_pairs_per_scene: Cap pairs per scene to balance dataset size.
        scene_ids:    Optional allowlist of scene IDs (for quick experiments).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "Training",
        input_size: int = 768,
        frame_gap: int = 10,
        max_pairs_per_scene: int = 200,
        scene_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.input_size = input_size
        self.frame_gap = frame_gap

        self._pairs: list[_FramePair] = []
        self._traj_cache: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}

        self._build_index(scene_ids, max_pairs_per_scene)
        LOGGER.info(
            "ARKitScenesDataset: split=%s  pairs=%d  input_size=%d",
            split, len(self._pairs), input_size,
        )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(
        self,
        scene_ids: list[str] | None,
        max_pairs: int,
    ) -> None:
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Download ARKitScenes with scripts/download_arkitscenes.py"
            )

        scene_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if scene_ids is not None:
            scene_dirs = [d for d in scene_dirs if d.name in scene_ids]

        for scene_dir in scene_dirs:
            pairs = self._index_scene(scene_dir, max_pairs)
            self._pairs.extend(pairs)

    def _index_scene(self, scene_dir: Path, max_pairs: int) -> list[_FramePair]:
        scene_id = scene_dir.name

        # ARKitScenes raw download: flat layout (no _frames/ subdir)
        rgb_dir = scene_dir / "lowres_wide"
        depth_dir = scene_dir / "lowres_depth"
        intr_dir = scene_dir / "lowres_wide_intrinsics"
        traj_path = scene_dir / "lowres_wide.traj"

        # Some scenes may be incomplete; skip silently.
        for d in (rgb_dir, depth_dir, intr_dir, traj_path):
            if not d.exists():
                LOGGER.debug("Skipping %s — missing %s", scene_id, d.name)
                return []

        # Collect sorted frame timestamps.
        rgb_files = sorted(rgb_dir.glob(f"{scene_id}_*.png"))
        if len(rgb_files) < self.frame_gap + 1:
            return []

        # Build a lookup: timestamp → (rgb_path, depth_path, intr_path)
        def ts_of(p: Path) -> str:
            # Filename pattern: <scene_id>_<timestamp>.png
            return p.stem.split("_", 1)[1]

        frame_ts = [ts_of(p) for p in rgb_files]

        def depth_path(ts: str) -> Path:
            return depth_dir / f"{scene_id}_{ts}.png"

        def intr_path(ts: str) -> Path:
            return intr_dir / f"{scene_id}_{ts}.pincam"

        # Filter to frames that have ALL required files.
        valid_ts = [
            ts for ts in frame_ts
            if depth_path(ts).exists() and intr_path(ts).exists()
        ]

        pairs: list[_FramePair] = []
        step = max(1, len(valid_ts) // max_pairs)  # thin out if too many frames

        for i in range(0, len(valid_ts) - self.frame_gap, step):
            j = i + self.frame_gap
            if j >= len(valid_ts):
                break

            src_ts, tgt_ts = valid_ts[i], valid_ts[j]
            pairs.append(_FramePair(
                scene_id=scene_id,
                src_rgb=rgb_dir / f"{scene_id}_{src_ts}.png",
                src_depth=depth_path(src_ts),
                src_intr=intr_path(src_ts),
                src_ts=src_ts,
                tgt_rgb=rgb_dir / f"{scene_id}_{tgt_ts}.png",
                tgt_depth=depth_path(tgt_ts),
                tgt_intr=intr_path(tgt_ts),
                tgt_ts=tgt_ts,
                traj_path=traj_path,
            ))
            if len(pairs) >= max_pairs:
                break

        return pairs

    # ------------------------------------------------------------------
    # Trajectory (lazy cached per scene)
    # ------------------------------------------------------------------

    def _get_traj(
        self, scene_id: str, traj_path: Path
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        if scene_id not in self._traj_cache:
            self._traj_cache[scene_id] = _load_traj(traj_path)
        return self._traj_cache[scene_id]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        pair = self._pairs[idx]

        # ---- Load source ----
        src_rgb_np = _load_rgb(pair.src_rgb)           # H×W×3 float32
        src_depth_np = _load_depth(pair.src_depth)     # H×W float32 metres
        src_w, src_h, src_fx, src_fy, src_cx, src_cy = _load_pincam(pair.src_intr)

        src_image = _resize_image(src_rgb_np, self.input_size)       # 3×S×S
        src_depth = _resize_depth(src_depth_np, self.input_size)     # 1×S×S

        # Scale intrinsics for the square resize.
        fx_s, fy_s, cx_s, cy_s = _scale_intrinsics(
            src_fx, src_fy, src_cx, src_cy, src_w, src_h, self.input_size
        )
        src_K = torch.from_numpy(
            _build_intrinsics_4x4(fx_s, fy_s, cx_s, cy_s)
        )  # [4, 4]

        # disparity_factor = fx_original / original_width  (SHARP convention)
        disparity_factor = torch.tensor(src_fx / src_w, dtype=torch.float32)

        # ---- Load target ----
        tgt_rgb_np = _load_rgb(pair.tgt_rgb)
        tgt_w, tgt_h, tgt_fx, tgt_fy, tgt_cx, tgt_cy = _load_pincam(pair.tgt_intr)

        tgt_image = _resize_image(tgt_rgb_np, self.input_size)       # 3×S×S

        fx_t, fy_t, cx_t, cy_t = _scale_intrinsics(
            tgt_fx, tgt_fy, tgt_cx, tgt_cy, tgt_w, tgt_h, self.input_size
        )
        tgt_K = torch.from_numpy(
            _build_intrinsics_4x4(fx_t, fy_t, cx_t, cy_t)
        )  # [4, 4]

        # ---- Relative pose ----
        traj_ts, traj_poses = self._get_traj(pair.scene_id, pair.traj_path)

        src_c2w_arkit = _nearest_pose(float(pair.src_ts), traj_ts, traj_poses)
        tgt_c2w_arkit = _nearest_pose(float(pair.tgt_ts), traj_ts, traj_poses)

        if src_c2w_arkit is None or tgt_c2w_arkit is None:
            # Fall back to identity relative pose if no nearby pose found.
            rel_ext = np.eye(4, dtype=np.float32)
        else:
            src_c2w = _arkit_c2w_to_opencv_c2w(src_c2w_arkit)
            tgt_c2w = _arkit_c2w_to_opencv_c2w(tgt_c2w_arkit)
            rel_ext = _relative_extrinsics(src_c2w, tgt_c2w)  # [4, 4]

        return {
            # Source inputs to the predictor.
            "source_image": src_image,                              # [3, S, S]
            "source_depth": src_depth,                              # [1, S, S]
            "source_intrinsics": src_K,                             # [4, 4]
            "disparity_factor": disparity_factor,                   # scalar

            # Target for novel-view-synthesis supervision.
            "target_image": tgt_image,                              # [3, S, S]
            "target_intrinsics": tgt_K,                             # [4, 4]

            # world-to-target-cam where "world" = source-cam frame.
            "rel_extrinsics": torch.from_numpy(rel_ext),            # [4, 4]

            # Metadata (useful for debugging / logging).
            "scene_id": pair.scene_id,
        }
