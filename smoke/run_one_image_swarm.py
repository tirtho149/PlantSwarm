#!/usr/bin/env python
"""
smoke/run_one_image_swarm.py
============================
Dedicated, self-contained one-image smoke for the Phase 0R swarm.

Run it directly on a GPU node you already hold (salloc), with the
project venv active:

    python smoke/run_one_image_swarm.py

No shell wrapper needed — this file bootstraps its own environment
(loads the gitignored .env for HF_TOKEN, points the HF cache at
/work, enables hf_transfer, turns the Claude verifier OFF) BEFORE any
heavy import, then:

  1. picks ONE real Soybean (disease, state, cached-image) tuple using
     the *exact* production resolution (build_state_image_map +
     _resolve_cached_image),
  2. runs the FULL swarm via plantswarm.delta_pipeline.run_for_state:
       OrganDetectionAgent -> route to that organ's deep specialists
       -> round 1 -> blackboard -> round 2 (stigmergy)
       -> VisualDiagnosisAgent consolidator -> K-of-N -> merge,
  3. prints the chosen tuple, per-pass detected organ + active agent
     count, raw vs agreed deltas, final merged deltas,
  4. asserts the output is well-formed.

Env knobs (optional; faithful but quick defaults):
  CROP                 Soybean
  SMOKE_DISEASE        (auto)   force a disease, else first Soybean
                                tuple with a cached image
  VLLM_N_RUNS          3        stochastic passes (K-of-N)
  VLLM_AGREEMENT_MIN   2        K
  VLLM_SWARM_ROUNDS    2        2 = full swarm (round-2 blackboard)
  SWARM_GRANULARITY    routed   routed | grouped | specialists
  VLLM_MAX_NEW_TOKENS  512
  PATHOME_USABLE_CSV   BugWood_Diseases_usable.csv
  HF_HOME              <repo>/.hf_cache
  PATHOME_VENV / PATHOME_REPO   override paths if needed

Exit 0 = full swarm ran end-to-end, output well-formed (0 deltas is
         acceptable — the smoke validates the pipeline, not accuracy).
Exit 2 = setup problem (no registry / no cached image / venv).
Exit 3 = swarm ran but output malformed (a real failure).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment bootstrap — MUST run before importing torch/transformers/HF.
# ---------------------------------------------------------------------------

REPO = Path(os.environ.get("PATHOME_REPO")
            or Path(__file__).resolve().parent.parent)
os.chdir(REPO)
# Running a script file puts smoke/ (not the repo root) on sys.path[0],
# so `import pathome_kb` would fail. The repo is not pip-installed —
# put its root first on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE / export KEY=VALUE, no override of
    already-set vars. Never prints values."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(REPO / ".env")

# HF model cache on the big shared fs (download once, reuse on any node)
os.environ.setdefault("HF_HOME", str(REPO / ".hf_cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",
                       str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

# torch JIT/inductor caches off restricted /tmp -> persist on /work
# (removes the "kernel cache directory could not be created" warning
# and the slow recompile on every first generate).
_tc = REPO / ".torch_cache"
os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(_tc / "kernels"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_tc / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_tc / "triton"))
for _p in ("PYTORCH_KERNEL_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
           "TRITON_CACHE_DIR"):
    Path(os.environ[_p]).mkdir(parents=True, exist_ok=True)

# swarm-only smoke: no Claude verifier on a GPU node
os.environ.setdefault("PATHOME_USE_VERIFIER", "0")
os.environ.setdefault("PATHOME_IMAGE_CACHE_DIR", str(REPO / ".bugwood_cache"))

# faithful-but-quick defaults
os.environ.setdefault("CROP", "Soybean")
os.environ.setdefault("VLLM_N_RUNS", "3")
os.environ.setdefault("VLLM_AGREEMENT_MIN", "2")
os.environ.setdefault("VLLM_SWARM_ROUNDS", "2")     # 2 = full swarm

CROP = os.environ["CROP"]
CSV = os.environ.get("PATHOME_USABLE_CSV", "BugWood_Diseases_usable.csv")
N_RUNS = int(os.environ["VLLM_N_RUNS"])
K = int(os.environ["VLLM_AGREEMENT_MIN"])


def _fail(code: int, msg: str) -> None:
    print(f"\n[SMOKE FAIL] {msg}")
    sys.exit(code)


def main() -> None:
    reg_path = REPO / "artifacts" / "pathome_kb" / CROP / "final_registry.json"
    if not reg_path.is_file():
        _fail(2, f"no {reg_path} — run Phase 0/1 (canonical KB) first.")
    if not (REPO / CSV).is_file():
        _fail(2, f"no {CSV} — run Step 0 (filter) first.")

    try:
        from pathome_kb.regional_observation import (
            build_state_image_map, _resolve_cached_image,
        )
        from plantswarm.delta_pipeline import (
            run_for_state, existing_deltas_for_state,
        )
    except ModuleNotFoundError as e:
        _fail(2, f"import failed ({e}). Activate the project venv first: "
                 f"source {os.environ.get('PATHOME_VENV', '<repo>/.venv')}"
                 f"/bin/activate")

    import json
    import time

    reg = json.loads(reg_path.read_text())
    by_disease = {
        (d.get("disease_name") or "").strip(): d
        for d in reg.get("diseases", []) or []
        if (d.get("disease_name") or "").strip()
    }
    if not by_disease:
        _fail(2, f"{reg_path} has no diseases.")

    force = os.environ.get("SMOKE_DISEASE", "").strip()

    # Resolve a REAL (crop, disease, state, cached image) tuple exactly
    # like run_regional_observation does.
    smap = build_state_image_map(CSV)
    chosen = None
    for (c, disease, state), image_ids in smap.items():
        if c != CROP:
            continue
        if force and disease != force:
            continue
        if disease not in by_disease:
            continue
        for img_id in image_ids:
            p = _resolve_cached_image(img_id)
            if p:
                chosen = (disease, state, p, img_id)
                break
        if chosen:
            break

    if not chosen:
        _fail(2, f"no {CROP} tuple with a cached image found "
                 f"(cache empty? run scripts/ensure_state_image_cache.py).")

    disease, state, img_path, img_id = chosen
    drec = by_disease[disease]
    existing = existing_deltas_for_state(drec, state)

    print("=" * 64)
    print(f"ONE-IMAGE SWARM SMOKE — {CROP}")
    print("=" * 64)
    print(f"  disease : {disease}")
    print(f"  state   : {state}")
    print(f"  image   : {img_path}  (id={img_id})")
    print(f"  granularity={os.environ.get('SWARM_GRANULARITY','routed')} "
          f"N={N_RUNS} K={K} rounds={os.environ['VLLM_SWARM_ROUNDS']} "
          f"verifier={os.environ['PATHOME_USE_VERIFIER']}")
    print(f"  HF_HOME={os.environ['HF_HOME']}  "
          f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'unset'}")
    print(f"  existing deltas for this state: {len(existing)}")
    print("-" * 64)

    t0 = time.time()
    rec = run_for_state(
        crop=CROP, disease=disease, state=state,
        canonical_record=drec, image_path=Path(img_path),
        primary_image_id=img_id, existing_deltas=existing,
        n_runs=N_RUNS, agreement_min=K,
    )
    dt = time.time() - t0

    sm = rec.get("__swarm_meta__", {}) or {}
    deltas = rec.get("deltas", []) or []
    print(f"\n=== swarm finished in {dt:.0f}s ===")
    print(f"  granularity         : {sm.get('granularity')}")
    print(f"  detected organ/pass : {sm.get('detected_organ_per_pass')}")
    print(f"  active agents/pass  : {sm.get('n_active_agents_per_pass')}")
    print(f"  raw deltas/pass     : {sm.get('n_raw_per_pass')}")
    print(f"  after K-of-N        : {sm.get('n_after_agreement')}")
    print(f"  merge counts        : {sm.get('merge')}")
    print(f"  FINAL merged deltas : {len(deltas)}")
    for d in deltas[:12]:
        print(f"    - [{d.get('field')}] {str(d.get('image_shows',''))[:100]}")

    problems = []
    if "granularity" not in sm:
        problems.append("no __swarm_meta__.granularity (run_for_state contract broke)")
    n_raw = sm.get("n_raw_per_pass")
    if n_raw is None or len(n_raw) != N_RUNS:
        problems.append(f"expected {N_RUNS} passes in n_raw_per_pass, got {n_raw}")
    for d in deltas:
        if not isinstance(d, dict) or not d.get("field") or not d.get("image_shows"):
            problems.append(f"malformed delta: {d!r}")
            break

    if problems:
        for p in problems:
            print(f"  [bad] {p}")
        _fail(3, "swarm ran but output is malformed.")

    print("\n[SMOKE PASS] full swarm ran end-to-end on one image; "
          f"output well-formed ({len(deltas)} deltas; "
          "0 is acceptable if the image adds nothing).")
    sys.exit(0)


if __name__ == "__main__":
    main()
