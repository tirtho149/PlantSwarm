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

# Show the REAL CUDA error (never hide it behind 2>/dev/null).
cuda_probe_error() {
  python - <<'PY' 2>&1
import torch
try:
    torch.cuda.init()
    torch.zeros(1).cuda()
    print("CUDA_PROBE_OK", torch.cuda.get_device_name(0))
except BaseException as e:
    print(f"CUDA_PROBE_FAIL {type(e).__name__}: {e}")
PY
}

# ---- transformers (safe, CPU-only, fixes Qwen2.5-VL import) ----------------
if py "import transformers" && tf_ok; then :; else
  echo "[env] installing transformers>=$REQ_TF (+accelerate, pillow) ..."
  pip install --quiet --upgrade "transformers>=${REQ_TF}" accelerate pillow packaging \
    || echo "[env] WARN: transformers install failed (no internet here?)"
fi

# ---- FAST PATH: torch CUDA already works ----------------------------------
if py "import torch" && cuda_ok && tf_ok; then
  echo "[env] OK — torch=$(py 'import torch;print(torch.__version__)') " \
       "cuda=$(py 'import torch;print(torch.version.cuda)') " \
       "transformers=$(py 'import transformers;print(transformers.__version__)') " \
       "device=$(py 'import torch;print(torch.cuda.get_device_name(0))')"
  exit 0
fi

TORCH_PRESENT=0; py "import torch" && TORCH_PRESENT=1
TORCH_CUDA="$(py 'import torch;print(torch.version.cuda or "")')"   # "" if CPU-only
echo "[env] torch present=$TORCH_PRESENT  torch.version.cuda='${TORCH_CUDA:-none}'"

# Is this a PACKAGE problem (no torch / CPU-only torch / bundled CUDA
# NEWER than the driver) or a NODE problem (packages fine, CUDA still
# dead)? Decide before touching pip — reinstalling can't fix a node.
DRIVER=""; MAXCUDA=""; GPU_VISIBLE=1
if command -v nvidia-smi >/dev/null 2>&1; then
  SMI_RAW="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"
  if echo "$SMI_RAW" | grep -qiE 'No devices were found|Unknown Error|couldn'\''t communicate|has fallen off the bus'; then
    GPU_VISIBLE=0
    echo "[env] nvidia-smi CANNOT SEE A GPU on this node: '$SMI_RAW'"
  else
    DRIVER="$(echo "$SMI_RAW" | tr -d ' ')"
    MAXCUDA="$(nvidia-smi 2>/dev/null | grep -m1 -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')"
    echo "[env] NVIDIA driver: ${DRIVER:-?}   driver max CUDA: ${MAXCUDA:-?}"
  fi
fi

_cuda_int() { local v="${1:-}"; [ -z "$v" ] && { echo 0; return; }
  echo $(( ${v%%.*} * 10 + ${v##*.} )); }   # 12.8 -> 128

NEED_REINSTALL=0
if [ "$TORCH_PRESENT" = 0 ] || [ -z "$TORCH_CUDA" ]; then
  NEED_REINSTALL=1                           # missing or CPU-only torch
elif [ -n "$MAXCUDA" ] && \
     [ "$(_cuda_int "$TORCH_CUDA")" -gt "$(_cuda_int "$MAXCUDA")" ]; then
  NEED_REINSTALL=1                           # torch CUDA newer than driver
fi

if [ "$NEED_REINSTALL" = 1 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1 && [ -z "${PATHOME_FORCE_TORCH:-}" ]; then
    echo "[env] torch needs (re)install but no nvidia-smi here (login node)"
    echo "      and no PATHOME_FORCE_TORCH. The sbatch heals on the GPU"
    echo "      node automatically — just: sbatch scripts/submit_phase0r_regional.sh"
    exit 0
  fi
  choose_wheel() { local mc="${1:-}"; [ -z "$mc" ] && { echo cu121; return; }
    local n; n="$(_cuda_int "$mc")"
    if   [ "$n" -ge 128 ]; then echo cu128
    elif [ "$n" -ge 126 ]; then echo cu126
    elif [ "$n" -ge 124 ]; then echo cu124
    elif [ "$n" -ge 121 ]; then echo cu121
    else                        echo cu118; fi; }
  WHEEL="${PATHOME_FORCE_TORCH:-$(choose_wheel "$MAXCUDA")}"
  echo "[env] (re)installing torch/torchvision from cu wheel: $WHEEL"
  pip install --quiet --force-reinstall torch torchvision \
    --index-url "https://download.pytorch.org/whl/${WHEEL}" \
    || { echo "[env] ERROR: torch install failed (no internet on this node?)"; exit 3; }
else
  echo "[env] torch package looks correct for this driver — NOT reinstalling"
  echo "      (the real torch was already cu${TORCH_CUDA//./} and the driver"
  echo "       supports CUDA ${MAXCUDA:-?}; a reinstall cannot fix a node fault)."
fi

# ---- verify a real CUDA context, showing the ACTUAL error -----------------
PROBE="$(cuda_probe_error)"
echo "[env] CUDA probe: $PROBE"
if echo "$PROBE" | grep -q "CUDA_PROBE_OK"; then
  echo "[env] HEALED — torch=$(py 'import torch;print(torch.__version__)') " \
       "cuda=$(py 'import torch;print(torch.version.cuda)')"
  exit 0
fi

# Still dead AND we did not need a reinstall => NODE-LEVEL fault.
echo "[env] ====================================================================="
if [ "$GPU_VISIBLE" = 0 ]; then
echo "[env] NODE HARDWARE/DRIVER FAULT — nvidia-smi CANNOT SEE A GPU here."
echo "[env]   This is NOT a package or code problem. The GPU on this node"
echo "[env]   has fallen off / the driver<->kernel-module is out of sync"
echo "[env]   (typical right after a cluster driver rollout without reboot)."
else
echo "[env] NODE CUDA FAULT — packages are fine, the GPU node cannot init CUDA."
echo "[env]   driver=${DRIVER:-?} (supports CUDA ${MAXCUDA:-?}); torch cuda=${TORCH_CUDA:-none}"
echo "[env]   nvidia-smi works but the CUDA runtime does not -> typically a"
echo "[env]   missing/unhealthy /dev/nvidia-uvm or an Xid'd GPU on THIS node."
fi
echo "[env]   Quick checks on the node:"
echo "[env]     ls -l /dev/nvidia*            # is /dev/nvidia-uvm present?"
echo "[env]     nvidia-smi -q | grep -iE 'Xid|Pending|ECC mode|MIG'"
echo "[env]   Most reliable fix: resubmit so SLURM lands on a HEALTHY node,"
echo "[env]   and exclude this one, e.g.:"
echo "[env]     sbatch --exclude=\$SLURMD_NODENAME scripts/submit_phase0r_regional.sh"
echo "[env]   If many nodes fail the same way, it is a cluster issue -> contact"
echo "[env]   Nova support with the CUDA probe error above."
echo "[env] ====================================================================="
exit 3
