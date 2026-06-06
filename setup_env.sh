#!/usr/bin/env bash
#
# setup_env.sh — one-click environment setup for the TinyWorlds repo.
#
# Reproduces the full environment that was prepared on an NVIDIA GH200 (aarch64 + CUDA)
# box, so it can be re-created on a similar machine. Safe to re-run (idempotent).
#
# What it does:
#   1. Pre-flight checks (arch + NVIDIA driver/CUDA presence)
#   2. Installs Miniforge (conda) if conda is not already available
#   3. Runs `conda init bash` so `conda` works in new shells
#   4. Creates the conda env (default name: tiny) with Python 3.12
#   5. Installs PyTorch + torchvision from the CUDA *aarch64* wheels (cu128)
#   6. Installs the repo's requirements.txt
#   7. Adds the repo root to PYTHONPATH persistently (in ~/.bashrc)
#   8. Verifies CUDA works end-to-end
#   9. Prints a caution about the inference.yaml device gotcha
#
# Usage:
#   bash setup_env.sh
#
# Configurable via environment variables (with defaults):
#   ENV_NAME         conda env name                (default: tiny)
#   PYTHON_VERSION   python version for the env     (default: 3.12)
#   CONDA_DIR        miniforge install prefix        (default: $HOME/miniforge3)
#   TORCH_INDEX_URL  PyTorch wheel index            (default: cu128 index)
#
set -euo pipefail

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-tiny}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

# Repo root = directory containing this script.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS="$REPO_ROOT/requirements.txt"

log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err()  { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }

# ----------------------------------------------------------------------------
# 1. Pre-flight checks
# ----------------------------------------------------------------------------
log "Pre-flight checks"
ARCH="$(uname -m)"
echo "Architecture: $ARCH"
if [[ "$ARCH" != "aarch64" ]]; then
  warn "This script targets aarch64 (ARM, e.g. GH200). Detected '$ARCH'."
  warn "If this is an x86_64 box, change TORCH_INDEX_URL accordingly and re-run;"
  warn "the cu128 index also serves x86_64 wheels, so it will usually still work."
fi

# NVIDIA driver / CUDA check. We do NOT auto-install drivers: that needs root and
# is machine/distro specific. We only verify and instruct.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  warn "nvidia-smi not found — no NVIDIA driver detected on this machine."
  warn "PyTorch CUDA wheels bundle the CUDA runtime, but you still need a host"
  warn "NVIDIA *driver* installed (this requires root). Install it before training:"
  warn "  - Ubuntu:  sudo apt-get install -y nvidia-driver-<version>   (then reboot)"
  warn "  - Or use NVIDIA's official .run installer / your cloud image's GPU driver."
  warn "Continuing with env setup; GPU verification at the end will fail without a driver."
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
  err "requirements.txt not found at $REQUIREMENTS"
  err "Run this script from inside the repo (it should live at the repo root)."
  exit 1
fi

# ----------------------------------------------------------------------------
# 2. Install Miniforge (conda) if needed
# ----------------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
  CONDA_DIR="$(conda info --base)"
  log "conda already available at: $CONDA_DIR"
elif [[ -x "$CONDA_DIR/bin/conda" ]]; then
  log "Found existing Miniforge at: $CONDA_DIR"
else
  log "Installing Miniforge to $CONDA_DIR"
  case "$ARCH" in
    aarch64) MINIFORGE_ARCH="aarch64" ;;
    x86_64)  MINIFORGE_ARCH="x86_64"  ;;
    *)       MINIFORGE_ARCH="$ARCH"   ;;
  esac
  TMP_INSTALLER="$(mktemp /tmp/miniforge.XXXXXX.sh)"
  curl -fsSL -o "$TMP_INSTALLER" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINIFORGE_ARCH}.sh"
  bash "$TMP_INSTALLER" -b -p "$CONDA_DIR"
  rm -f "$TMP_INSTALLER"
fi

# Make conda usable in *this* (non-interactive) shell.
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

# ----------------------------------------------------------------------------
# 3. conda init bash (so `conda` works in future interactive shells)
# ----------------------------------------------------------------------------
log "Ensuring 'conda init bash' has run"
conda init bash >/dev/null 2>&1 || true

# ----------------------------------------------------------------------------
# 4. Create the conda env (idempotent)
# ----------------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists — reusing it"
else
  log "Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
echo "Active python: $(which python) ($(python --version 2>&1))"

# ----------------------------------------------------------------------------
# 5. Install PyTorch + torchvision from CUDA aarch64 wheels
# ----------------------------------------------------------------------------
log "Installing PyTorch + torchvision from $TORCH_INDEX_URL"
python -m pip install --upgrade pip
pip install torch torchvision --index-url "$TORCH_INDEX_URL"

# ----------------------------------------------------------------------------
# 6. Install repo requirements
# ----------------------------------------------------------------------------
log "Installing repo requirements ($REQUIREMENTS)"
pip install -r "$REQUIREMENTS"

# ----------------------------------------------------------------------------
# 7. Add repo root to PYTHONPATH persistently (idempotent)
# ----------------------------------------------------------------------------
BASHRC="$HOME/.bashrc"
PYTHONPATH_LINE="export PYTHONPATH=\"$REPO_ROOT:\$PYTHONPATH\""
if [[ -f "$BASHRC" ]] && grep -Fq "$PYTHONPATH_LINE" "$BASHRC"; then
  log "PYTHONPATH already set in $BASHRC"
else
  log "Adding repo root to PYTHONPATH in $BASHRC"
  {
    echo ""
    echo "# tinyworlds repo on PYTHONPATH (for utils/models/datasets imports)"
    echo "$PYTHONPATH_LINE"
  } >> "$BASHRC"
fi
# Also export for the current shell so verification below works.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# ----------------------------------------------------------------------------
# 8. Verify CUDA works end-to-end
# ----------------------------------------------------------------------------
log "Verifying PyTorch + CUDA"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    _ = (x @ x).sum().item()
    print("GPU matmul OK")
else:
    print("WARNING: CUDA not available — check the host NVIDIA driver.")
import torchvision, einops, h5py, cv2, omegaconf, wandb  # noqa: F401
print("repo deps import OK; torchvision:", torchvision.__version__)
PY

# ----------------------------------------------------------------------------
# 9. Done + caution
# ----------------------------------------------------------------------------
printf '\033[1;32m'
cat <<EOF

============================================================
 Environment ready.
============================================================

To use it:
  conda activate $ENV_NAME
  cd $REPO_ROOT
  # set W&B or disable it:
  export WANDB_API_KEY=<your_key>   # or: export WANDB_MODE=disabled
  python scripts/full_train.py --config configs/training.yaml

EOF
printf '\033[0m'

printf '\033[1;33m'
cat <<'EOF'
[CAUTION] Inference config gotcha:
  configs/inference.yaml ships with `device: mps` (the author's Mac).
  MPS is Apple-Silicon only and is NOT available on this CUDA/aarch64 box.
  Training already defaults to `cuda`, but for INFERENCE you must flip it:

      configs/inference.yaml ->  device: cuda

  Otherwise scripts/run_inference.py will fail with an unavailable-device error.
EOF
printf '\033[0m\n'
