#!/usr/bin/env bash
# gcp_train.sh — Provision a GCP GPU VM and launch SHARP regression-head training.
#
# Prerequisites (run once on your local machine):
#   brew install --cask google-cloud-sdk
#   gcloud auth login
#   gcloud auth application-default login
#
# Usage:
#   ./scripts/gcp_train.sh              # uses defaults below
#   GCP_PROJECT=my-project ./scripts/gcp_train.sh
#   GPU_TYPE=a100 ./scripts/gcp_train.sh   # upgrade to A100
#
# What this script does:
#   1. Creates a GCP Deep Learning VM (L4 GPU, CUDA 12, Ubuntu 22.04)
#   2. Uploads this repo to the VM via gcloud scp
#   3. SSHes in and installs Python dependencies
#   4. Downloads ARKitScenes data directly on the VM (fast GCP egress)
#   5. Launches training in a tmux session (detached — survives SSH disconnect)
#
# After the job starts you can disconnect freely.  Reconnect with:
#   gcloud compute ssh $INSTANCE_NAME --zone $ZONE -- -t "tmux attach -t train"

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GCP_PROJECT="${GCP_PROJECT:-}"                   # required — set here or via env
ZONE="${ZONE:-}"                                 # left empty = auto-select from candidates
INSTANCE_NAME="${INSTANCE_NAME:-sharp-train}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"    # 8 vCPU, 32 GB RAM, 1× L4 (24 GB)

# L4 zones to try in order (stockouts are common — script tries each automatically)
L4_ZONES=(
    us-central1-a us-central1-b us-central1-c us-central1-f
    us-east1-d us-east4-a us-east4-c
    us-west1-b us-west4-a
    europe-west4-a europe-west1-b
    asia-east1-a asia-southeast1-b
)
DISK_SIZE="${DISK_SIZE:-200GB}"                  # ARKitScenes (50 scenes) ≈ 20 GB
NUM_SCENES="${NUM_SCENES:-50}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
USE_SPOT="${USE_SPOT:-true}"                     # spot saves ~70%; set USE_SPOT=false to disable
WANDB_PROJECT="${WANDB_PROJECT:-sharp-arkit}"
WANDB_API_KEY="${WANDB_API_KEY:-}"               # set via env or left empty to skip W&B

# GPU type config
case "${GPU_TYPE:-l4}" in
    a100)
        MACHINE_TYPE="a2-highgpu-1g"
        ACCELERATOR="type=nvidia-tesla-a100,count=1"
        GPU_ZONES=(
            us-central1-a us-central1-b us-central1-c us-central1-f
            us-east1-b us-east1-c us-east1-d
            us-west1-a us-west1-b
            europe-west4-a europe-west4-b
            asia-east1-a asia-east1-c
        )
        ;;
    t4)
        MACHINE_TYPE="n1-standard-8"
        ACCELERATOR="type=nvidia-tesla-t4,count=1"
        BATCH_SIZE="${BATCH_SIZE:-2}"   # T4 has 16GB VRAM; lower default batch
        GPU_ZONES=(
            us-central1-a us-central1-b us-central1-c us-central1-f
            us-east1-b us-east1-c us-east1-d
            us-west1-a us-west1-b us-west2-b
            europe-west1-b europe-west1-c europe-west4-a
            asia-east1-a asia-east1-b asia-southeast1-b
        )
        ;;
    *)  # l4 default
        MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
        ACCELERATOR="type=nvidia-l4,count=1"
        GPU_ZONES=("${L4_ZONES[@]}")
        ;;
esac

# Deep Learning VM image — Ubuntu 22.04, CUDA 12.3, Python 3.10, PyTorch pre-installed
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO="/home/$(whoami)/ml-sharp-main"
# ──────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[gcp_train]${NC} $*"; }
warn()    { echo -e "${YELLOW}[gcp_train]${NC} $*"; }
die()     { echo -e "${RED}[gcp_train] ERROR:${NC} $*" >&2; exit 1; }

# ── Preflight checks ──────────────────────────────────────────────────────────
command -v gcloud >/dev/null 2>&1 || die "gcloud not found. Install: brew install --cask google-cloud-sdk"

if [[ -z "$GCP_PROJECT" ]]; then
    GCP_PROJECT=$(gcloud config get-value project 2>/dev/null || true)
    [[ -z "$GCP_PROJECT" ]] && die "GCP_PROJECT not set. Run: gcloud config set project YOUR_PROJECT_ID"
fi

SPOT_FLAGS=()
if [[ "$USE_SPOT" == "true" ]]; then
    SPOT_FLAGS=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

info "Project  : $GCP_PROJECT"
info "Zone     : $ZONE"
info "Instance : $INSTANCE_NAME ($MACHINE_TYPE)"
info "Disk     : $DISK_SIZE"
info "Spot     : $USE_SPOT  (saves ~70% vs on-demand)"
info "Scenes   : $NUM_SCENES  |  Epochs: $EPOCHS  |  Batch: $BATCH_SIZE"
[[ -n "$WANDB_API_KEY" ]] && info "W&B      : enabled (project=$WANDB_PROJECT)" || info "W&B      : disabled (set WANDB_API_KEY=<key> to enable)"
echo ""

# ── 1. Create VM ──────────────────────────────────────────────────────────────

# If ZONE was explicitly set, only try that zone; otherwise try all candidates.
if [[ -n "$ZONE" ]]; then
    ZONE_CANDIDATES=("$ZONE")
else
    ZONE_CANDIDATES=("${GPU_ZONES[@]}")
fi

# Check if instance already exists in any zone
EXISTING_ZONE=""
for z in "${ZONE_CANDIDATES[@]}"; do
    if gcloud compute instances describe "$INSTANCE_NAME" --zone="$z" --project="$GCP_PROJECT" &>/dev/null; then
        EXISTING_ZONE="$z"
        break
    fi
done

if [[ -n "$EXISTING_ZONE" ]]; then
    ZONE="$EXISTING_ZONE"
    warn "Instance '$INSTANCE_NAME' already exists in $ZONE — skipping creation."
else
    info "Creating instance (GPU=${GPU_TYPE:-l4}, trying ${#ZONE_CANDIDATES[@]} zones)..."
    CREATED=false
    for z in "${ZONE_CANDIDATES[@]}"; do
        info "  Trying zone $z..."
        if gcloud compute instances create "$INSTANCE_NAME" \
            --project="$GCP_PROJECT" \
            --zone="$z" \
            --machine-type="$MACHINE_TYPE" \
            --accelerator="$ACCELERATOR" \
            --maintenance-policy="TERMINATE" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size="$DISK_SIZE" \
            --boot-disk-type="pd-ssd" \
            --metadata="install-nvidia-driver=True" \
            --scopes="https://www.googleapis.com/auth/cloud-platform" \
            ${SPOT_FLAGS[@]+"${SPOT_FLAGS[@]}"} 2>&1; then
            ZONE="$z"
            CREATED=true
            info "Instance created in zone $ZONE"
            break
        else
            warn "  Zone $z unavailable (stockout), trying next..."
        fi
    done
    [[ "$CREATED" == true ]] || die "No ${GPU_TYPE:-l4} capacity found in any zone. Try again later or switch GPU types."

    info "Waiting for SSH to become ready..."
    sleep 30
    for i in {1..12}; do
        gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --project="$GCP_PROJECT" \
            --command="echo ready" -- -o StrictHostKeyChecking=no 2>/dev/null && break
        warn "  SSH not ready yet (attempt $i/12), retrying in 15s..."
        sleep 15
    done
fi

# ── 2. Upload repo ────────────────────────────────────────────────────────────
info "Uploading repo to VM..."
# Exclude large artefacts that don't belong on the training VM
gcloud compute scp --recurse \
    --zone="$ZONE" \
    --project="$GCP_PROJECT" \
    --compress \
    --scp-flag="-o StrictHostKeyChecking=no" \
    "$REPO_DIR" \
    "${INSTANCE_NAME}:${REMOTE_REPO}" \
    --exclude=".git,runs,__pycache__,*.pyc,.DS_Store,*.pt,*.onnx,*.mlpackage"

# ── 3. Remote setup + training launch (runs in tmux so it survives disconnect) ─
info "Running remote setup and launching training..."

gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$GCP_PROJECT" \
    -- -o StrictHostKeyChecking=no -T << REMOTE_EOF

set -euo pipefail

REMOTE_REPO="${REMOTE_REPO}"
ARKITSCENES_DIR="\$HOME/ARKitScenes"
ARKIT_REPO="\$HOME/ARKitScenes-repo"

echo "=== [1/5] Checking CUDA ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: nvidia-smi failed"

echo "=== [2/5] Installing Python dependencies ==="
cd "\$REMOTE_REPO"
pip install -q -e ".[training]" 2>&1 | tail -5
# gsplat — installs CUDA extension (fast on GCP because CUDA toolkit present)
pip install -q gsplat 2>&1 | tail -3
# ARKitScenes downloader deps
pip install -q boto3 tqdm pandas 2>&1 | tail -3
# W&B
pip install -q wandb 2>&1 | tail -3
# Log in to W&B if key provided
if [[ -n "${WANDB_API_KEY}" ]]; then
    wandb login "${WANDB_API_KEY}"
fi

echo "=== [3/5] Cloning ARKitScenes downloader repo ==="
if [ ! -d "\$ARKIT_REPO" ]; then
    git clone --depth 1 https://github.com/apple/ARKitScenes "\$ARKIT_REPO"
else
    echo "  Already cloned."
fi

echo "=== [4/5] Downloading ARKitScenes data (${NUM_SCENES} scenes) ==="
mkdir -p "\$ARKITSCENES_DIR"
if [ ! -d "\$ARKITSCENES_DIR/raw" ] || [ -z "\$(ls -A "\$ARKITSCENES_DIR/raw" 2>/dev/null)" ]; then
    python "\$REMOTE_REPO/scripts/download_arkitscenes.py" \
        --arkitscenes-repo "\$ARKIT_REPO" \
        --output "\$ARKITSCENES_DIR" \
        --num-scenes ${NUM_SCENES} \
        --split Training \
        --verbose
else
    echo "  Data already present — skipping download."
fi

echo "=== [5/5] Launching training in tmux ==="
tmux new-session -d -s train 2>/dev/null || true   # create if not exists
WANDB_FLAGS=""
if [[ -n "${WANDB_API_KEY}" ]]; then
    WANDB_FLAGS="--wandb --wandb-project ${WANDB_PROJECT}"
fi

tmux send-keys -t train "
cd \$REMOTE_REPO && \\
python train_arkit.py \\
    --data \$ARKITSCENES_DIR/raw \\
    --batch-size ${BATCH_SIZE} \\
    --workers 4 \\
    --epochs ${EPOCHS} \\
    --input-size 1536 \\
    --log-every 10 \\
    --save-every 500 \\
    --output ./runs/arkit_v1 \\
    \$WANDB_FLAGS \\
    2>&1 | tee ./runs/arkit_v1/train.log
" Enter

echo ""
echo "================================================================"
echo "  Training launched in tmux session 'train'."
echo "  Reconnect anytime with:"
echo "    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- -t 'tmux attach -t train'"
echo ""
echo "  Follow logs live:"
echo "    gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} -- 'tail -f \$HOME/ml-sharp-main/runs/arkit_v1/train.log'"
echo "================================================================"

REMOTE_EOF

# ── Local instructions ────────────────────────────────────────────────────────
echo ""
info "Done. Useful commands:"
echo ""
echo "  # Attach to training tmux (live output):"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- -t 'tmux attach -t train'"
echo ""
echo "  # Tail logs without attaching:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE -- 'tail -f ~/ml-sharp-main/runs/arkit_v1/train.log'"
echo ""
echo "  # Download the trained checkpoint when done:"
echo "  gcloud compute scp ${INSTANCE_NAME}:~/ml-sharp-main/runs/arkit_v1/sharp_arkit_final.pt ./runs/"
echo ""
echo "  # Stop the VM to avoid charges (data persists on disk):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
echo ""
echo "  # Delete the VM entirely when finished:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --project=$GCP_PROJECT"
