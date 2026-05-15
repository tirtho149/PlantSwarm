#!/bin/bash
# ============================================================================
# scripts/setup_env.sh — the "environment setter".
#
# Takes care of the venv so you never hand-diagnose torch/CUDA again:
#   1. resolve + activate the venv (same logic as the sbatch scripts)
#   2. FAST PATH: if torch already imports AND torch.cuda is usable AND
#      transformers >= 4.49 -> print OK and exit 0 (idempotent, no pip)
#   3. otherwise detect the NVIDIA driver's max CUDA from nvidia-smi and
#      (re)install a torch/torchvision wheel that MATCHES that driver,
#      then the Phase 0R deps (transformers>=4.49, accelerate, pillow)
#   4. re-verify `torch.zeros(1).cuda()` actually works
#
# Why this exists: torch 2.x+cuXXX bundles its own CUDA userspace; if
# that CUDA is newer than the node's kernel driver you get
# "CUDA unknown error ... Setting the available devices to be zero"
# even though nvidia-smi works. This script picks the right wheel.
#
# Usage:
#   bash scripts/setup_env.sh                 # detect + heal + verify
#   PATHOME_FORCE_TORCH=cu121 bash scripts/setup_env.sh   # pin a wheel
#   PATHOME_VENV=/path/.venv bash scripts/setup_env.sh    # explicit venv
#
# Exit codes: 0 ok | 2 no venv | 3 CUDA still broken | 4 no nvidia-smi
# Network is only used when a (re)install is actually needed; run it on
# a login node (compute nodes often have no internet).
# ============================================================================
set -uo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- resolve + activate venv (mirrors submit_phase0r_regional.sh) ----------
VENV="${PATHOME_VENV:-$PATHOME_REPO/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$(dirname "$PATHOME_REPO")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PATHOME_REPO")/.venv"
  else
    echo "[env] ERROR: no venv found (tried PATHOME_VENV, $PATHOME_REPO/.venv, ../.venv)"
    exit 2
  fi
fi
echo "[env] venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

REQ_TF="4.49.0"   # first transformers with Qwen2.5-VL

py() { python -c "$1" 2>/dev/null; }

cuda_ok() {
  py "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
}
tf_ok() {
  py "import transformers,sys
from packaging.version import Version
sys.exit(0 if Version(transformers.__version__)>=Version('$REQ_TF') else 1)"
}

# ---- FAST PATH: already healthy -> nothing to do --------------------------
if py "import torch" && cuda_ok && tf_ok; then
  echo "[env] OK — torch=$(py 'import torch;print(torch.__version__)') " \
       "cuda=$(py 'import torch;print(torch.version.cuda)') " \
       "transformers=$(py 'import transformers;print(transformers.__version__)') " \
       "device=$(py 'import torch;print(torch.cuda.get_device_name(0))')"
  exit 0
fi
echo "[env] environment needs healing (torch.cuda or transformers not ready)"

# ---- detect the driver's max CUDA -----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[env] ERROR: nvidia-smi not found — run this on a GPU node."
  exit 4
fi
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
MAXCUDA="$(nvidia-smi 2>/dev/null | grep -m1 -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
echo "[env] NVIDIA driver: ${DRIVER:-?}   driver max CUDA: ${MAXCUDA:-?}"

# Map driver-max-CUDA -> the highest torch wheel index it can run.
choose_wheel() {
  local mc="${1:-}"
  [ -z "$mc" ] && { echo "cu121"; return; }   # safe default
  local major minor n
  major="${mc%%.*}"; minor="${mc##*.}"
  n=$((major * 10 + minor))                   # 12.4 -> 124
  if   [ "$n" -ge 128 ]; then echo "cu128"
  elif [ "$n" -ge 126 ]; then echo "cu126"
  elif [ "$n" -ge 124 ]; then echo "cu124"
  elif [ "$n" -ge 121 ]; then echo "cu121"
  elif [ "$n" -ge 118 ]; then echo "cu118"
  else                        echo "cu121"    # very old driver: best effort
  fi
}
WHEEL="${PATHOME_FORCE_TORCH:-$(choose_wheel "$MAXCUDA")}"
echo "[env] selected torch wheel index: $WHEEL  (override with PATHOME_FORCE_TORCH=cuXXX)"

# ---- (re)install matching torch + Phase 0R deps ---------------------------
IDX="https://download.pytorch.org/whl/${WHEEL}"
echo "[env] pip install torch/torchvision from $IDX (needs internet — login node)"
if ! pip install --quiet --force-reinstall torch torchvision --index-url "$IDX"; then
  echo "[env] ERROR: torch install failed (no internet on this node? run on a login node)."
  exit 3
fi
# Phase 0R model deps (Qwen2.5-VL needs transformers>=4.49 + accelerate + PIL)
pip install --quiet --upgrade "transformers>=${REQ_TF}" accelerate pillow "packaging" || true

# ---- verify a real CUDA context ------------------------------------------
if py "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()"; then
  echo "[env] HEALED — torch=$(py 'import torch;print(torch.__version__)') " \
       "cuda=$(py 'import torch;print(torch.version.cuda)') " \
       "device=$(py 'import torch;print(torch.cuda.get_device_name(0))')"
  exit 0
fi
echo "[env] ERROR: CUDA still not usable after reinstall."
echo "      driver=${DRIVER:-?} maxCUDA=${MAXCUDA:-?} wheel=$WHEEL"
echo "      Try a lower wheel: PATHOME_FORCE_TORCH=cu118 bash scripts/setup_env.sh"
exit 3
