#!/usr/bin/env python3
"""Download a subset of ARKitScenes for SHARP backbone-swap training.

This script wraps Apple's official ARKitScenes download infrastructure.
It fetches only the data needed for training the regression head:
  • lowres_wide/      — RGB frames
  • lowres_depth/     — LiDAR depth maps
  • lowres_wide_intrinsics/ — camera intrinsics
  • lowres_wide.traj  — camera poses

Dataset licence: Apple Research (non-commercial).
The *dataset* is only used at training time.  The trained weights (your
regression head) are your own intellectual property and carry no dataset
licence restrictions.  See ARKitScenes README for full terms.

Usage
-----
    # Step 1: clone Apple's ARKitScenes repo once
    git clone https://github.com/apple/ARKitScenes /tmp/ARKitScenes

    # Step 2: download a small training subset (~50 scenes, ~20 GB)
    python scripts/download_arkitscenes.py \\
        --arkitscenes-repo /tmp/ARKitScenes \\
        --output /path/to/data \\
        --num-scenes 50 \\
        --split Training

    # Step 3: kick off training
    python train_arkit.py --data /path/to/data/ARKitScenes

Requirements
------------
    pip install boto3 tqdm  # for the official ARKitScenes downloader
    pip install pandas       # for parsing scene metadata

Reference
---------
    ARKitScenes: A Diverse Real-World Dataset for 3D Indoor Scene
    Understanding Using Mobile RGB-D Data
    https://github.com/apple/ARKitScenes
    https://arxiv.org/abs/2111.08897
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Asset names for --raw_dataset_assets (from official downloader --help).
REQUIRED_DATA_TYPES = [
    "lowres_wide",
    "lowres_depth",
    "lowres_wide_intrinsics",
    "lowres_wide.traj",    # camera pose trajectories
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ARKitScenes subset for SHARP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--arkitscenes-repo",
        type=Path,
        required=True,
        help="Local path to the cloned apple/ARKitScenes repository.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where dataset will be downloaded.",
    )
    p.add_argument(
        "--split",
        default="Training",
        choices=["Training", "Validation"],
        help="Dataset split to download [default: Training].",
    )
    p.add_argument(
        "--num-scenes",
        type=int,
        default=50,
        help="Maximum number of scenes to download (ordered by scene ID). "
             "50 scenes ≈ 20 GB and is enough for initial fine-tuning.",
    )
    p.add_argument(
        "--scene-ids",
        nargs="+",
        default=None,
        help="Download only these specific scene IDs (overrides --num-scenes).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def check_dependencies() -> None:
    """Verify that required Python packages for the ARKitScenes downloader are available."""
    missing = []
    for pkg in ("boto3", "tqdm", "pandas"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        LOGGER.error(
            "Missing packages required by ARKitScenes downloader: %s\n"
            "Install with:  pip install %s",
            " ".join(missing), " ".join(missing),
        )
        sys.exit(1)


def get_scene_ids_from_metadata(
    repo: Path,
    split: str,
    num_scenes: int,
) -> list[str]:
    """Parse the official scene-split CSV and return the first N scene IDs."""
    import pandas as pd  # noqa: PLC0415

    csv_candidates = [
        repo / "raw" / "raw_train_val_splits.csv",
        repo / "metadata" / "3dod_train_val_splits.csv",
        repo / "metadata" / "ARKitScenes_3dod_instance_split.csv",
    ]
    csv_path: Path | None = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break

    if csv_path is None:
        LOGGER.error(
            "Could not find scene-split CSV in %s/metadata/\n"
            "Expected one of: %s",
            repo, [c.name for c in csv_candidates],
        )
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Column names vary across ARKitScenes releases; try common variants.
    # Support both "split" and "fold" column names across dataset releases.
    split_col = next(
        (c for c in df.columns if "split" in c.lower() or c.lower() == "fold"), None
    )
    id_col = next(
        (c for c in df.columns if "video" in c.lower() or "scene" in c.lower() or c == "id"), None
    )

    if split_col is None or id_col is None:
        LOGGER.warning(
            "Unexpected CSV columns: %s\n"
            "Falling back to using all rows as scene IDs.",
            list(df.columns),
        )
        scene_ids = [str(v) for v in df.iloc[:, 0].tolist()]
    else:
        split_lower = split.lower()
        mask = df[split_col].str.lower() == split_lower
        scene_ids = [str(v) for v in df.loc[mask, id_col].tolist()]

    LOGGER.info("Found %d %s scenes in metadata CSV", len(scene_ids), split)
    return sorted(scene_ids)[:num_scenes]


def download_via_official_script(
    repo: Path,
    output: Path,
    split: str,
    scene_ids: list[str],
    data_types: list[str],
) -> None:
    """Invoke Apple's download_data.py with the appropriate arguments."""
    downloader = repo / "download_data.py"
    if not downloader.exists():
        LOGGER.error(
            "download_data.py not found at %s\n"
            "Make sure --arkitscenes-repo points to a full ARKitScenes clone.",
            downloader,
        )
        sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(downloader),
        "raw",
        "--split", split,
        "--download_dir", str(output),
        "--raw_dataset_assets", *data_types,
        "--video_id", *scene_ids,
    ]

    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        LOGGER.error(
            "ARKitScenes downloader exited with code %d.\n"
            "Check that your AWS credentials are configured or that the "
            "ARKitScenes servers are accessible.",
            result.returncode,
        )
        sys.exit(result.returncode)


def verify_download(output: Path, split: str, scene_ids: list[str]) -> None:
    """Quick sanity check: verify at least some frames were downloaded."""
    split_dir = output / split
    found = 0
    for sid in scene_ids:
        frames_dir = split_dir / sid / f"{sid}_frames" / "lowres_wide"
        if frames_dir.exists():
            n = len(list(frames_dir.glob("*.png")))
            if n > 0:
                found += 1

    LOGGER.info(
        "Verification: %d / %d requested scenes have lowres_wide frames.",
        found, len(scene_ids),
    )
    if found == 0:
        LOGGER.warning(
            "No frames found.  The download may have failed or the folder "
            "structure is different from expected."
        )
    else:
        LOGGER.info(
            "Dataset root: %s\n"
            "To start training, run:\n"
            "    python train_arkit.py --data %s",
            output, output,
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    check_dependencies()

    scene_ids = (
        args.scene_ids
        if args.scene_ids
        else get_scene_ids_from_metadata(
            args.arkitscenes_repo, args.split, args.num_scenes
        )
    )

    LOGGER.info(
        "Downloading %d scenes for split=%s → %s",
        len(scene_ids), args.split, args.output,
    )

    download_via_official_script(
        repo=args.arkitscenes_repo,
        output=args.output,
        split=args.split,
        scene_ids=scene_ids,
        data_types=REQUIRED_DATA_TYPES,
    )

    verify_download(args.output, args.split, scene_ids)


if __name__ == "__main__":
    main()
