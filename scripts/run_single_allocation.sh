#!/usr/bin/env bash
# Run vLLM and the evaluation stack in ONE process tree (one Slurm/GPU allocation).
#
# Starts ``vllm serve`` on 127.0.0.1, waits for GET /v1/models, writes a temp YAML
# that sets model.vllm_base_url to match, then runs scripts/run_experiment_bundle.sh
# (or a custom command). Stops vLLM on EXIT/INT/TERM.
#
# Required env:
#   PYTHON_BIN, CONFIG_PATH, RESULTS_DIR
#   VLLM_MODEL   — Hugging Face model id (should match model.backbone in CONFIG_PATH)
#
# Optional:
#   VLLM_PORT (default 8000)
#   VLLM_HOST (default 127.0.0.1) — bind address for vLLM
#   VLLM_READY_TIMEOUT_SEC (default 7200) — max wait for server (many models need long load)
#   VLLM_EXTRA_ARGS — extra args passed to ``vllm serve`` (quoted string)
#   SINGLE_ALLOC_CMD — if set, run this instead of run_experiment_bundle.sh (e.g. smoke test)
#   SKIP_VLLM_START=1 — do not start vLLM; only merge URL and run (server already up)
#
# Example (interactive GPU node):
#   export PYTHON_BIN=python CONFIG_PATH=configs/qwen25_vl_3b_smoke.yaml
#   export RESULTS_DIR=results/single_alloc_test VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
#   bash scripts/run_single_allocation.sh
#
# Example (Slurm): see scripts/slurm/run_bundle_single_allocation.slurm

set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"
: "${RESULTS_DIR:?Set RESULTS_DIR}"

if [[ "${SKIP_VLLM_START:-0}" != "1" ]]; then
  : "${VLLM_MODEL:?Set VLLM_MODEL to the HF id served by vLLM (match model.backbone in YAML)}"
fi

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT_SEC:-7200}"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

mkdir -p "${RESULTS_DIR}/step_logs"

MERGED_CONFIG="${RESULTS_DIR}/.merged_config_${RANDOM}.yaml"
cleanup_config() {
  rm -f "${MERGED_CONFIG}" 2>/dev/null || true
}

"${PYTHON_BIN}" <<PY
import sys, yaml
inp, outp, url = "${CONFIG_PATH}", "${MERGED_CONFIG}", "${BASE_URL}"
with open(inp) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model", {})["vllm_base_url"] = url
with open(outp, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

export CONFIG_PATH="${MERGED_CONFIG}"
echo "Using merged config with model.vllm_base_url=${BASE_URL}"

VLLM_PID=""
cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "Stopping vLLM (pid ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
cleanup_all() {
  cleanup_vllm
  cleanup_config
}
trap cleanup_all EXIT INT TERM

if [[ "${SKIP_VLLM_START:-0}" == "1" ]]; then
  echo "SKIP_VLLM_START=1 — assuming vLLM already at ${BASE_URL}"
else
  VLLM_LOG="${RESULTS_DIR}/vllm_server.log"
  echo "Starting vLLM: model=${VLLM_MODEL} listen=${VLLM_HOST}:${VLLM_PORT}"
  # shellcheck disable=SC2086
  vllm serve "${VLLM_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.92 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    ${VLLM_EXTRA_ARGS:-} \
    >>"${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  echo "vLLM pid=${VLLM_PID} log=${VLLM_LOG}"

  _started=$(date +%s)
  while true; do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "vLLM ready (${BASE_URL})"
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[FATAL] vLLM exited before becoming ready. Tail ${VLLM_LOG}:"
      tail -n 80 "${VLLM_LOG}" || true
      exit 1
    fi
    _now=$(date +%s)
    if (( _now - _started > VLLM_READY_TIMEOUT_SEC )); then
      echo "[FATAL] Timeout waiting for vLLM after ${VLLM_READY_TIMEOUT_SEC}s"
      exit 1
    fi
    sleep 10
  done
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${SINGLE_ALLOC_CMD:-}" ]]; then
  echo "Running SINGLE_ALLOC_CMD: ${SINGLE_ALLOC_CMD}"
  bash -c "${SINGLE_ALLOC_CMD}"
else
  bash scripts/run_experiment_bundle.sh
fi

echo "Single-allocation run finished."
