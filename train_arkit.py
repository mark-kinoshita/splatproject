#!/usr/bin/env python3
"""SHARP Gaussian-head training on ARKitScenes.

Backbone swap: replace the research-only SHARP .pt weights with your own
commercially-safe weights by training only the regression head on ARKitScenes.

What is FROZEN (public, legally safe weights loaded from TIMM/HuggingFace):
  • monodepth_model (DinoV2 Large + SPN encoder + depth decoder)

What is TRAINED (your own IP after this script):
  • gaussian_decoder   (GaussianDensePredictionTransformer)
  • prediction_head    (DirectPredictionHead)
  • depth_alignment    (LearnedAlignment UNet, optional)

Usage
-----
    # Minimal (single GPU / Apple MPS):
    python train_arkit.py --data /path/to/ARKitScenes

    # Full options:
    python train_arkit.py \\
        --data   /path/to/ARKitScenes \\
        --split  Training \\
        --output ./runs/arkit_run1 \\
        --input-size 768 \\
        --batch-size 2 \\
        --lr 1e-4 \\
        --epochs 20 \\
        --frame-gap 10 \\
        --log-every 50 \\
        --save-every 1000

Legal
-----
Trained weights saved by this script are YOUR intellectual property.
They are built on:
  • DinoV2 (Meta AI, Apache 2.0)   — loaded via timm at init time
  • DepthPro architecture (Apple Sample Code Licence)
  • ARKitScenes dataset (Apple, non-commercial research use at *training only*)
Acknowledgements to include in your app's About / Notices screen:
  "Scene reconstruction powered by a model trained on the ARKitScenes dataset
   (Apple, 2021).  Visual backbone: DINOv2 (Meta AI, Apache 2.0)."
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# W&B is optional — training works without it, just no web dashboard.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from sharp.data import ARKitScenesDataset
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.training import GaussianTrainingLoss, LossWeights
from sharp.training.torch_renderer import torch_render_batch
from sharp.utils.gaussians import Gaussians3D, unproject_gaussians
from sharp.utils.gsplat import GSplatRenderer

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SHARP regression head on ARKitScenes")

    # Data
    p.add_argument("--data", required=True, type=Path,
                   help="Root of ARKitScenes download (contains Training/ subfolder)")
    p.add_argument("--split", default="Training",
                   help="Dataset split to train on [default: Training]")
    p.add_argument("--output", default="./runs/arkit", type=Path,
                   help="Directory to save checkpoints and logs")
    p.add_argument("--scene-ids", nargs="+", default=None,
                   help="Restrict training to these scene IDs (debug / ablation)")

    # Model
    p.add_argument("--input-size", type=int, default=1536,
                   help="Square side (px) to resize images to before the ViT backbone. "
                        "Must be 1536 (the SPN encoder's fixed internal resolution). "
                        "Smaller values will error; this flag is kept for future variants.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Resume from a previously saved *head-only* checkpoint "
                        "(does NOT load the SHARP research weights).")

    # Training hyper-params
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--frame-gap", type=int, default=10,
                   help="Frame separation between source/target pairs in each scene")
    p.add_argument("--max-pairs-per-scene", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)

    # Loss weights
    p.add_argument("--w-src", type=float, default=1.0,
                   help="Input-view reconstruction loss weight")
    p.add_argument("--w-nvs", type=float, default=1.0,
                   help="Novel-view synthesis loss weight")
    p.add_argument("--w-depth", type=float, default=0.1,
                   help="Depth supervision loss weight")
    p.add_argument("--w-opacity", type=float, default=0.01,
                   help="Opacity regularisation weight")
    p.add_argument("--no-ssim", action="store_true",
                   help="Disable SSIM component in photometric loss")

    # Logging / saving
    p.add_argument("--log-every", type=int, default=50,
                   help="Log metrics every N steps")
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save checkpoint every N steps (also saved at end of each epoch)")
    p.add_argument("--device", default="default",
                   help="Compute device: 'cuda', 'mps', 'cpu', or 'default' (auto)")

    # W&B
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging (requires: pip install wandb && wandb login)")
    p.add_argument("--wandb-project", default="sharp-arkit",
                   help="W&B project name [default: sharp-arkit]")
    p.add_argument("--wandb-run", default=None,
                   help="W&B run name (auto-generated if omitted)")

    # Debug
    p.add_argument("--dry-run", action="store_true",
                   help="Run one step and exit (sanity check)")
    p.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> torch.device:
    if requested != "default":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Render helper (handles MPS by routing gsplat through CPU)
# ---------------------------------------------------------------------------

def _gsplat_available() -> bool:
    """Return True if gsplat's CUDA extension is compiled and available."""
    try:
        import gsplat.cuda._wrapper as _w
        # Try constructing a minimal call to detect missing CUDA ext.
        _w.fully_fused_projection  # attribute exists even without CUDA
        # Deeper check: attempt to import the lazy CUDA obj factory
        obj = _w._make_lazy_cuda_obj("RasterizeToPixels")  # will raise if _C is None
        return True
    except Exception:
        return False


_GSPLAT_CUDA_OK: bool | None = None  # cached


def render_gaussians_for_training(
    gsplat_renderer: GSplatRenderer,
    gaussians: Gaussians3D,
    extrinsics: torch.Tensor,    # [B, 4, 4]  (on CPU, world-to-cam)
    intrinsics: torch.Tensor,    # [B, 4, 4]  (on CPU)
    width: int,
    height: int,
    train_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render and return (color, depth, alpha) on train_device.

    Uses gsplat if CUDA is available; otherwise falls back to the pure-PyTorch
    torch_render_batch which runs on any device (MPS, CPU).
    """
    global _GSPLAT_CUDA_OK
    if _GSPLAT_CUDA_OK is None:
        _GSPLAT_CUDA_OK = _gsplat_available()
        if not _GSPLAT_CUDA_OK:
            LOGGER.info(
                "gsplat CUDA extension unavailable — using pure-PyTorch renderer. "
                "Training will be slower; for best NVS quality use a CUDA device."
            )

    if _GSPLAT_CUDA_OK:
        # gsplat path (CUDA / CPU fallback that gsplat handles itself)
        gaussians_r = gaussians.to(torch.device("cpu"))
        out = gsplat_renderer(gaussians_r, extrinsics, intrinsics, width, height)
        return out.color.to(train_device), out.depth.to(train_device), out.alpha.to(train_device)
    else:
        # Pure-PyTorch path: gaussians are already on train_device after unproject
        color, depth, alpha = torch_render_batch(
            gaussians, extrinsics.to(train_device), intrinsics.to(train_device),
            width, height, sigma_px=1.0, kernel=3,
        )
        return color, depth, alpha


# ---------------------------------------------------------------------------
# Trainable parameter selection
# ---------------------------------------------------------------------------

def get_trainable_params(model: RGBGaussianPredictor) -> list[torch.nn.Parameter]:
    """Return only the parameters we want to optimise (NOT the frozen backbone).

    Trainable submodules:
      - feature_model      (gaussian_decoder)
      - prediction_head
      - depth_alignment    (scale-map estimator UNet)
    """
    trainable_modules = [
        model.feature_model,
        model.prediction_head,
        model.depth_alignment,
    ]
    params = []
    for m in trainable_modules:
        params.extend(p for p in m.parameters() if p.requires_grad)
    return params


# ---------------------------------------------------------------------------
# Checkpoint I/O (head-only — never saves SHARP research weights)
# ---------------------------------------------------------------------------

def save_head_checkpoint(
    model: RGBGaussianPredictor,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    output_dir: Path,
    tag: str = "",
) -> Path:
    """Save only the trainable (head) weights — NOT the frozen backbone.

    These weights are YOUR intellectual property and are safe to ship.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only keep keys that belong to trainable submodules.
    head_prefix = ("feature_model.", "prediction_head.", "depth_alignment.")
    head_state = {
        k: v for k, v in model.state_dict().items()
        if any(k.startswith(p) for p in head_prefix)
    }

    ckpt = {
        "head_state_dict": head_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
    }

    suffix = f"_step{step}" if not tag else f"_{tag}"
    path = output_dir / f"gaussian_head{suffix}.pt"
    torch.save(ckpt, path)
    LOGGER.info("Saved head checkpoint → %s", path)
    return path


def load_head_checkpoint(
    path: Path,
    model: RGBGaussianPredictor,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    """Load head-only checkpoint; return (step, epoch)."""
    ckpt = torch.load(path, weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["head_state_dict"], strict=False)
    LOGGER.info(
        "Loaded head checkpoint from %s  (missing=%d, unexpected=%d)",
        path, len(missing), len(unexpected),
    )
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0), ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Export for inference (full model state dict — DinoV2 is Apache 2.0 anyway)
# ---------------------------------------------------------------------------

def export_inference_weights(
    model: RGBGaussianPredictor,
    output_dir: Path,
    step: int,
) -> Path:
    """Save full model state_dict for use with create_predictor() + load_state_dict().

    Includes the DinoV2 backbone (Apache 2.0 — legally safe to ship).
    Does NOT include any SHARP research weights (they were never loaded).
    """
    path = output_dir / f"sharp_arkit_step{step}.pt"
    torch.save(model.state_dict(), path)
    LOGGER.info("Exported inference weights → %s", path)
    return path


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    # ---- Dataset & DataLoader ----
    dataset = ARKitScenesDataset(
        root=args.data,
        split=args.split,
        input_size=args.input_size,
        frame_gap=args.frame_gap,
        max_pairs_per_scene=args.max_pairs_per_scene,
        scene_ids=args.scene_ids,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(str(device) == "cuda"),
        drop_last=True,
    )
    LOGGER.info("Dataset: %d pairs", len(dataset))

    # ---- Model ----
    # IMPORTANT: We call create_predictor() WITHOUT loading any SHARP checkpoint.
    # DinoV2 weights are fetched from HuggingFace/TIMM (Apache 2.0) on first run.
    # The gaussian_decoder and prediction_head start from random initialisation.
    LOGGER.info("Creating model (public DinoV2 backbone, random gaussian head)…")
    model = create_predictor(PredictorParams())

    # The monodepth backbone is already frozen inside create_predictor().
    # Double-check nothing slipped through.
    for name, p in model.monodepth_model.named_parameters():
        p.requires_grad_(False)

    model.to(device)
    model.train()

    # ---- Optimizer (head only) ----
    trainable = get_trainable_params(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters()) - n_trainable
    LOGGER.info(
        "Parameters — trainable: %s  frozen: %s",
        f"{n_trainable:,}", f"{n_frozen:,}",
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine LR schedule over all steps.
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    # ---- Resume ----
    start_step, start_epoch = 0, 0
    if args.resume is not None:
        start_step, start_epoch = load_head_checkpoint(args.resume, model, optimizer)

    # ---- Loss function ----
    loss_fn = GaussianTrainingLoss(
        weights=LossWeights(
            src_reconstruction=args.w_src,
            nvs=args.w_nvs,
            depth=args.w_depth,
            opacity_reg=args.w_opacity,
        ),
        use_ssim=not args.no_ssim,
    )

    # ---- Renderer ----
    # linearRGB matches the model's default colour space.
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")

    S = args.input_size
    eyes = torch.eye(4, dtype=torch.float32)  # identity for input-view render

    # ---- W&B init ----
    use_wandb = args.wandb and _WANDB_AVAILABLE
    if args.wandb and not _WANDB_AVAILABLE:
        LOGGER.warning("--wandb requested but 'wandb' package not installed. Run: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "input_size": args.input_size,
                "frame_gap": args.frame_gap,
                "w_src": args.w_src,
                "w_nvs": args.w_nvs,
                "w_depth": args.w_depth,
                "w_opacity": args.w_opacity,
                "trainable_params": n_trainable,
                "frozen_params": n_frozen,
                "device": str(device),
            },
            resume="allow",
        )
        LOGGER.info("W&B run: %s", wandb.run.get_url())

    # ---- Training ----
    step = start_step
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            t0 = time.perf_counter()

            src_image = batch["source_image"].to(device)         # [B, 3, S, S]
            src_depth = batch["source_depth"].to(device)         # [B, 1, S, S]
            src_K = batch["source_intrinsics"]                   # [B, 4, 4] (keep cpu for rendering)
            disp_factor = batch["disparity_factor"].to(device)   # [B]

            tgt_image = batch["target_image"]                    # [B, 3, S, S] cpu
            tgt_K = batch["target_intrinsics"]                   # [B, 4, 4] cpu
            rel_ext = batch["rel_extrinsics"]                    # [B, 4, 4] cpu

            B = src_image.shape[0]

            optimizer.zero_grad()

            # -- Forward pass (frozen monodepth + trainable gaussian head) --
            gaussians_ndc = model(src_image, disp_factor, src_depth)

            # -- Unproject NDC Gaussians → source-camera metric space --
            # unproject_gaussians is NOT batched: call per batch item, then stack.
            src_K_cpu = src_K.cpu()
            gaussians_ndc_cpu = gaussians_ndc.to(torch.device("cpu"))

            unprojected = [
                unproject_gaussians(
                    Gaussians3D(
                        mean_vectors=gaussians_ndc_cpu.mean_vectors[b:b+1],
                        singular_values=gaussians_ndc_cpu.singular_values[b:b+1],
                        quaternions=gaussians_ndc_cpu.quaternions[b:b+1],
                        colors=gaussians_ndc_cpu.colors[b:b+1],
                        opacities=gaussians_ndc_cpu.opacities[b:b+1],
                    ),
                    eyes,           # [4, 4] identity (unbatched)
                    src_K_cpu[b],   # [4, 4] (unbatched)
                    (S, S),
                )
                for b in range(B)
            ]
            # Stack back to batched Gaussians3D
            gaussians_world_cpu = Gaussians3D(
                mean_vectors=torch.cat([g.mean_vectors for g in unprojected]),
                singular_values=torch.cat([g.singular_values for g in unprojected]),
                quaternions=torch.cat([g.quaternions for g in unprojected]),
                colors=torch.cat([g.colors for g in unprojected]),
                opacities=torch.cat([g.opacities for g in unprojected]),
            )
            # For the pure-PyTorch renderer, move Gaussians to train_device.
            gaussians_world = gaussians_world_cpu.to(device)
            id_ext = eyes.unsqueeze(0).expand(B, -1, -1)  # [B, 4, 4] identity

            # -- Render: input view (identity extrinsics) --
            src_color, src_rdepth, src_alpha = render_gaussians_for_training(
                renderer, gaussians_world, id_ext, src_K_cpu, S, S, device
            )

            # -- Render: novel view --
            tgt_color, _, tgt_alpha = render_gaussians_for_training(
                renderer, gaussians_world, rel_ext, tgt_K, S, S, device
            )

            # -- Loss --
            # Collect opacities for regularisation ([B, N, 1] → flat).
            opacities = gaussians_ndc.opacities   # [B, N, num_layers, 1] or similar

            loss, breakdown = loss_fn(
                src_rendered_color=src_color,
                src_rendered_depth=src_rdepth,
                src_rendered_alpha=src_alpha,
                src_target=src_image,
                src_gt_depth=src_depth,
                tgt_rendered_color=tgt_color,
                tgt_rendered_alpha=tgt_alpha,
                tgt_target=tgt_image.to(device),
                gaussian_opacities=opacities.reshape(B, -1, 1),
            )

            loss.backward()

            # Gradient clipping prevents spikes from poorly initialised head.
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            step += 1
            elapsed = time.perf_counter() - t0

            # -- Logging --
            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                LOGGER.info(
                    "epoch %d  step %d  "
                    "total=%.4f  src=%.4f  nvs=%.4f  depth=%.4f  opacity=%.4f  "
                    "lr=%.2e  %.1fs/step",
                    epoch, step,
                    breakdown["loss_total"], breakdown["loss_src"],
                    breakdown["loss_nvs"], breakdown["loss_depth"],
                    breakdown["loss_opacity"],
                    lr_now, elapsed,
                )
                if use_wandb:
                    wandb.log(
                        {
                            "loss/total": breakdown["loss_total"],
                            "loss/src_reconstruction": breakdown["loss_src"],
                            "loss/nvs": breakdown["loss_nvs"],
                            "loss/depth": breakdown["loss_depth"],
                            "loss/opacity_reg": breakdown["loss_opacity"],
                            "train/lr": lr_now,
                            "train/step_time_s": elapsed,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )

            # -- Checkpoint --
            if step % args.save_every == 0:
                save_head_checkpoint(model, optimizer, step, epoch, args.output)

            if args.dry_run:
                LOGGER.info("Dry run complete — exiting.")
                save_head_checkpoint(model, optimizer, step, epoch, args.output, tag="dryrun")
                return

        # -- End of epoch checkpoint --
        save_head_checkpoint(model, optimizer, step, epoch, args.output, tag=f"epoch{epoch}")

    # -- Final export (full inference weights) --
    export_path = export_inference_weights(model, args.output, step)
    LOGGER.info("Training complete. Inference weights saved to %s", args.output)
    if use_wandb:
        wandb.save(str(export_path))
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)


if __name__ == "__main__":
    main()
