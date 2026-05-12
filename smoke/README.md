# Pathome smoke test (2 crops)

A miniature end-to-end run of Phase 0 (canonical KB, Claude) and Phase
0R (regional deltas, Qwen swarm) on **Soybean + Tomato** only — ~25
(crop, disease) classes after threshold ≥ 15 and ~50 (disease, state)
tuples. Designed to validate every code path with as little compute as
possible.

---

## Directory layout

```
smoke/
├── BugWood_Diseases_smoke.csv          1,002 raw rows (Tomato + Soybean)
├── BugWood_Diseases_smoke_usable.csv   filtered + normalised (produced by Setup)
├── bugwood_pathome_smoke.yaml          swarm / model knobs
├── run_phase0_full.sh                  LOCAL: Phase 0 + Phase 0R (needs vLLM reachable)
├── run_phase0_local.sh                 LOCAL: canonical-only Phase 0 (no GPU needed)
└── README.md                           (this file)
```

## Two paths

### A. LOCAL + tunneled vLLM (one-shot)

If you can reach a vLLM endpoint serving Qwen2.5-VL-7B from your laptop
(e.g. via `ssh -L 8000:localhost:8000 nova-login`), run everything at
once:

```bash
VLLM_BASE_URL=http://localhost:8000/v1 bash smoke/run_phase0_full.sh
```

### B. LOCAL canonical, Nova Phase 0R (most common)

```bash
# 1. Canonical-only on LOCAL (no GPU needed)
bash smoke/run_phase0_local.sh

# 2. Push the canonical artefacts
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv \
           artifacts/pathome_kb/{Soybean,Tomato}/{discovery_results,final_registry}.json
git commit -m "smoke: Phase 0 canonical"
git push origin main

# 3. Phase 0R on Nova — the SLURM job boots vLLM in-place
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
PATHOME_ONLY_CROPS="Soybean,Tomato" PATHOME_SEED_QUICK=1 \
  PATHOME_USABLE_CSV=smoke/BugWood_Diseases_smoke_usable.csv \
  PATHOME_SEED_FILE=smoke/artifacts/pathome_seed/symptoms_seed.json \
  sbatch scripts/submit_phase0r_regional.sh
tail -f logs/pathome_phase0r-*.out
```

## Outputs

```
smoke/.bugwood_cache/<image_number>.jpg               cached photos (Setup)
artifacts/pathome_kb/<Crop>/
  ├── discovery_results.json                           URL cache (Phase 0)
  ├── raw_extractions.json                             per-source quotes (Phase 0)
  ├── final_registry.json                              canonical + regional deltas
  ├── final_registry.xlsx                              decision-tree view
  └── registry.md                                      human-readable canonical
smoke/artifacts/pathome_seed/symptoms_seed.json        ← terminal deliverable
```

## What's downscaled vs production

- Two crops only (`SMOKE_CROPS=Soybean,Tomato`) instead of 197.
- `--quick` caps the canonical discovery to a few sources per disease.
- `--quick` caps Phase 0R to 2 states per disease.
- The swarm reads the same prompts and runs the same agents as
  production — nothing about routing or delta extraction is mocked.

## Knobs

```bash
SMOKE_CROPS="Soybean,Tomato"     # default
SMOKE_THRESHOLD=15               # min rows per class
FULL_QUICK=1                     # cap canonical sources + regional states
FULL_KEEP_CACHE=1                # reuse cached final_registry.json (skip Phase 0)
FULL_SKIP_SETUP=1                # CSV already filtered
FULL_SKIP_CACHE=1                # image cache already topped up
FULL_SKIP_KB=1                   # skip the python -m pathome_kb call
FULL_SKIP_REGIONAL=1             # canonical-only (no Qwen needed)
VLLM_BASE_URL                    # default http://localhost:8000/v1
VLLM_MODEL                       # default Qwen/Qwen2.5-VL-7B-Instruct
```
