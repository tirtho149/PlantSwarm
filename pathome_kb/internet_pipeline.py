"""Internet Pipeline — Build disease registry from web sources.

Inputs: crop name, disease name list (optional, for targeted discovery)
Process: web discovery → extraction → reconciliation → output
Output: final_registry.json, registry.md, {Crop}_internet.xlsx
"""

import concurrent.futures
import json
import time
from pathlib import Path

from .config import (
    DISCOVERY_MAX_TURNS,
    DISCOVERY_MAX_TURNS_QUICK,
    DISCOVERY_TIMEOUT,
    DISCOVERY_TIMEOUT_QUICK,
    TARGETED_DISCOVERY_TIMEOUT,
    TARGETED_DISCOVERY_MAX_TURNS,
    TARGETED_DISCOVERY_MAX_TURNS_QUICK,
    MAX_PARALLEL_EXTRACTIONS,
    EXTRACTION_TIMEOUT,
    EXTRACTION_MAX_TURNS,
    EXTRACTION_QUICK_LIMIT,
    RECONCILIATION_BATCH_SIZE,
    MAX_PARALLEL_RECONCILIATIONS,
)
from .prompts import (
    DISCOVERY_PROMPT,
    DISCOVERY_SCHEMA,
    TARGETED_DISCOVERY_PROMPT,
    EXTRACTION_PROMPT,
    EXTRACTION_SCHEMA,
    RECONCILIATION_PROMPT,
    FINAL_REGISTRY_SCHEMA,
    NAME_NORMALIZATION_PROMPT,
    NAME_NORMALIZATION_SCHEMA,
)
from .shared import api_query, claude_query, match_names_to_folders, parse_json_result
from .utils import (
    chunk_list,
    fetch_page,
    get_crop_dir,
    load_json,
    registry_to_markdown,
    save_file,
    save_json,
    today_iso,
    write_enriched_xlsx,
)


# ─── Quick-mode discovery prompt ──────────────────────────────────────────

DISCOVERY_PROMPT_QUICK = """\
You are a research librarian specializing in plant pathology. For the crop "{crop}",
find authoritative web pages that document INDIVIDUAL diseases of this crop in detail.

Run ONLY these 3 searches:
1. "{crop} diseases complete list"
2. "{crop} fungal bacterial viral diseases extension"
3. "{crop} oomycete Phytophthora Pythium diseases"

CRITICAL: Prefer pages dedicated to a SINGLE disease with detailed symptom descriptions,
pathogen info, and management — NOT directory/index pages that just list disease names
with links. Index pages contain no extractable detail.

Pick 5-8 unique URLs. Prefer extension factsheets and university guides.

For each result, record:
- url, title, snippet, source_type, diseases_mentioned

Output the result as JSON matching the required schema.
"""


# ─── Stage 1: Discovery ─────────────────────────────────────────────────────


def _run_discovery(crop: str, quick: bool = False) -> dict:
    """Search the web for authoritative sources documenting diseases of {crop}."""
    mode = "QUICK" if quick else "FULL"
    print(f"\n{'='*60}")
    print(f"STAGE 1: DISCOVERY [{mode}] — Finding sources for {crop} diseases")
    print(f"{'='*60}")
    t0 = time.time()

    prompt = (DISCOVERY_PROMPT_QUICK if quick else DISCOVERY_PROMPT).format(crop=crop)

    raw = claude_query(
        prompt=prompt,
        allowed_tools=["WebSearch"],
        system_prompt=(
            "You are a research librarian specializing in plant pathology. "
            "Your job is to find authoritative web sources about crop diseases. "
            "Output ONLY valid JSON matching the required schema."
        ),
        json_schema=DISCOVERY_SCHEMA,
        max_turns=DISCOVERY_MAX_TURNS_QUICK if quick else DISCOVERY_MAX_TURNS,
        timeout_secs=DISCOVERY_TIMEOUT_QUICK if quick else DISCOVERY_TIMEOUT,
    )

    data = parse_json_result(raw, "discovery")
    n_sources = len(data.get("candidate_sources", []))
    n_queries = len(data.get("search_queries_run", []))
    print(f"  Found {n_sources} candidate sources from {n_queries} queries ({time.time()-t0:.0f}s)")
    return data


def _discover_single_disease(args: tuple) -> list[dict]:
    """Search for sources about one disease. Designed for parallel execution."""
    i, crop, disease_name, total, quick = args
    print(f"  [{i+1}/{total}] Searching: {disease_name}", flush=True)

    search_name = disease_name.replace("_", " ")
    prompt = TARGETED_DISCOVERY_PROMPT.format(crop=crop, disease_name=search_name)
    raw = claude_query(
        prompt=prompt,
        allowed_tools=["WebSearch"],
        system_prompt=(
            "You are a research librarian specializing in plant pathology. "
            "Find authoritative web sources for this disease. "
            "Output ONLY valid JSON matching the required schema."
        ),
        json_schema=DISCOVERY_SCHEMA,
        max_turns=TARGETED_DISCOVERY_MAX_TURNS_QUICK if quick else TARGETED_DISCOVERY_MAX_TURNS,
        timeout_secs=TARGETED_DISCOVERY_TIMEOUT,
    )

    data = parse_json_result(raw, f"discovery_{disease_name}")
    sources = data.get("candidate_sources", [])
    print(f"  [{i+1}/{total}] {disease_name}: {len(sources)} sources found")
    return sources


def _run_targeted_discovery(crop: str, disease_names: list[str], quick: bool = False) -> dict:
    """Search the web for sources about each disease individually, in parallel."""
    mode = "QUICK" if quick else "FULL"
    print(f"\n{'='*60}")
    print(f"STAGE 1: TARGETED DISCOVERY [{mode}] — {len(disease_names)} diseases, 1 search each")
    print(f"{'='*60}")
    t0 = time.time()

    args_list = [
        (i, crop, name, len(disease_names), quick)
        for i, name in enumerate(disease_names)
    ]

    all_sources = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as executor:
        futures = [executor.submit(_discover_single_disease, args) for args in args_list]
        for future in futures:
            all_sources.extend(future.result())

    # Deduplicate by URL
    seen_urls = set()
    unique_sources = []
    for s in all_sources:
        url = s.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    print(f"  Found {len(unique_sources)} unique sources ({time.time()-t0:.0f}s)")
    return {"crop": crop, "candidate_sources": unique_sources, "search_queries_run": []}


# ─── Stage 2: Extraction ────────────────────────────────────────────────────


def _extract_single_source(args: tuple) -> list[dict]:
    """Extract diseases from a single source. Designed for parallel execution."""
    i, source, total = args
    url = source["url"]
    title = source.get("title", "untitled")
    print(f"\n  --- Source {i+1}/{total}: {title[:60]} ---")

    # Step 1: Fetch page with httpx
    print(f"  [{i+1}] Fetching {url[:80]}...", flush=True)
    page_text = fetch_page(url)
    if not page_text:
        print(f"  [{i+1}] Skipping — could not fetch page")
        return []
    print(f"  [{i+1}] Fetched {len(page_text)} chars", flush=True)

    # Step 2: Extract via claude -p
    source_prompt = EXTRACTION_PROMPT.format(
        url=url, title=title, page_text=page_text
    )

    raw = claude_query(
        prompt=source_prompt,
        system_prompt=(
            "You are a data extraction agent for plant disease information. "
            "Extract disease data from the provided page text with verbatim quotes. "
            "NEVER fill in fields from your own knowledge. "
            "If the text doesn't state something, set it to null. "
            "Output ONLY valid JSON matching the required schema."
        ),
        json_schema=EXTRACTION_SCHEMA,
        max_turns=EXTRACTION_MAX_TURNS,
        timeout_secs=EXTRACTION_TIMEOUT,
    )

    source_data = parse_json_result(raw, f"extraction_source_{i+1}")
    source_extractions = source_data.get("extractions", [])
    n_diseases = sum(len(e.get("extracted_diseases", [])) for e in source_extractions)
    print(f"  [{i+1}] Done: {n_diseases} disease records")
    return source_extractions


def _run_extraction(sources: list[dict], quick: bool = False) -> dict:
    """Fetch each source URL with httpx, then extract disease data via claude -p (parallel)."""
    # Filter out PDFs — httpx gets binary, not useful text
    sources = [s for s in sources if not s.get("url", "").lower().endswith(".pdf")]

    if quick:
        sources = sources[:EXTRACTION_QUICK_LIMIT]

    print(f"\n{'='*60}")
    print(f"STAGE 2: EXTRACTION — Processing {len(sources)} sources (max {MAX_PARALLEL_EXTRACTIONS} parallel)")
    print(f"{'='*60}")
    t0 = time.time()

    all_extractions = []
    total = len(sources)
    task_args = [(i, source, total) for i, source in enumerate(sources)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as executor:
        futures = {executor.submit(_extract_single_source, args): args[0] for args in task_args}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                all_extractions.extend(result)
            except Exception as e:
                print(f"  [{idx+1}] ERROR: {e}")

    data = {"extractions": all_extractions}
    n_total = sum(len(e.get("extracted_diseases", [])) for e in all_extractions)
    print(f"\n  TOTAL: {n_total} disease records from {len(all_extractions)} sources ({time.time()-t0:.0f}s)")
    return data


# ─── Stage 3: Reconciliation ────────────────────────────────────────────────


def _normalize_disease_names(diseases: list[dict]) -> list[dict]:
    """Normalize disease names via one LLM call, then merge duplicates."""
    if not diseases:
        return diseases

    names = [d.get("disease_name", "") for d in diseases]
    print(f"  Normalizing {len(names)} disease names...", flush=True)

    prompt = NAME_NORMALIZATION_PROMPT.format(disease_names=json.dumps(names, indent=2))
    raw = api_query(
        prompt=prompt,
        system_prompt="You are a plant pathology nomenclature expert. Output JSON only.",
        json_schema=NAME_NORMALIZATION_SCHEMA,
    )
    result = parse_json_result(raw, "name_normalization")
    groups = result.get("groups", [])
    if not groups:
        print("  WARNING: Name normalization failed — skipping, returning diseases as-is")
        return diseases

    # Build mapping: original_name → canonical_name
    name_map = {}
    for group in groups:
        canonical = group.get("canonical_name", "")
        for orig in group.get("original_names", []):
            name_map[orig] = canonical

    # Apply mapping and dedup (keep entry with higher num_sources)
    merged = {}
    for disease in diseases:
        orig_name = disease.get("disease_name", "")
        canonical = name_map.get(orig_name, orig_name)
        disease["disease_name"] = canonical
        key = canonical.strip().lower()
        existing = merged.get(key)
        if existing is None:
            merged[key] = disease
        else:
            if disease.get("num_sources", 0) > existing.get("num_sources", 0):
                merged[key] = disease

    normalized = list(merged.values())
    if len(normalized) < len(diseases):
        print(f"  Name normalization: {len(diseases)} → {len(normalized)} (merged {len(diseases) - len(normalized)} duplicates)")
    else:
        print(f"  Name normalization: no duplicates found")
    return normalized


def _filter_to_input_diseases(registry: dict, disease_names: list[str]) -> dict:
    """Match registry diseases to folder names, rename, dedup, fill gaps."""
    diseases = registry.get("diseases", [])
    if not diseases:
        return registry

    registry_names = [d.get("disease_name", "") for d in diseases]
    print(f"  Matching {len(registry_names)} diseases to {len(disease_names)} folder names...", flush=True)

    name_map = match_names_to_folders(registry_names, disease_names)

    # Rename + dedup (keep entry with higher num_sources per folder name)
    merged: dict[str, dict] = {}
    for d in diseases:
        folder_name = name_map.get(d.get("disease_name", ""))
        if not folder_name:
            continue
        d["disease_name"] = folder_name
        key = folder_name.strip().lower()
        existing = merged.get(key)
        if existing is None or d.get("num_sources", 0) > existing.get("num_sources", 0):
            merged[key] = d

    # Fill empty entries for folder diseases with no web data
    _null_cited = {"value": None, "url": None, "quote": None}
    for name in disease_names:
        key = name.strip().lower()
        if key not in merged:
            merged[key] = {
                "disease_name": name,
                "pathogen_scientific_name": dict(_null_cited),
                "type_of_disease": dict(_null_cited),
                "affected_parts": dict(_null_cited),
                "visual_symptoms": {
                    "summary": dict(_null_cited),
                    "diagnostic_features": dict(_null_cited),
                    "look_alikes": dict(_null_cited),
                },
                "confidence": "low",
                "num_sources": 0,
                "conflicts": [],
            }

    registry["diseases"] = list(merged.values())
    matched = sum(1 for d in registry["diseases"] if d.get("num_sources", 0) > 0)
    print(f"  Matched: {matched}/{len(disease_names)} diseases have web data")
    return registry


def _reconcile_once(args: tuple) -> list[dict]:
    """Single reconciliation call via Anthropic API."""
    extractions_subset, taxonomy, label = args
    if label:
        print(f"  {label}", flush=True)

    prompt = RECONCILIATION_PROMPT.format(
        extractions=json.dumps(extractions_subset, indent=2),
        taxonomy=json.dumps(taxonomy, indent=2),
    )
    system = (
        "You are a data reconciliation agent for plant pathology. "
        "Merge disease records from multiple sources into canonical entries. "
        "Every field must have a citation with URL and verbatim quote."
    )

    raw = api_query(prompt=prompt, system_prompt=system, json_schema=FINAL_REGISTRY_SCHEMA)
    result = parse_json_result(raw, "reconciliation")
    return result.get("diseases", [])


def _run_reconciliation(extractions: dict, crop: str) -> dict:
    """Merge all extractions into a canonical registry. Batches in parallel if large."""
    print(f"\n{'='*60}")
    print(f"STAGE 3: RECONCILIATION — Building canonical registry")
    print(f"{'='*60}")
    t0 = time.time()

    taxonomy = {"validations": []}  # Taxonomy validation disabled
    all_ext = extractions.get("extractions", [])
    batches = chunk_list(all_ext, RECONCILIATION_BATCH_SIZE)
    print(f"  {len(all_ext)} source records → {len(batches)} batch(es) (parallel)")

    if len(batches) == 1:
        all_diseases = _reconcile_once(({"extractions": all_ext}, taxonomy, ""))
    else:
        task_args = [
            ({"extractions": batch}, taxonomy, f"Batch {i}/{len(batches)} ({len(batch)} records)")
            for i, batch in enumerate(batches, 1)
        ]
        batch_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_RECONCILIATIONS) as executor:
            futures = {executor.submit(_reconcile_once, args): i for i, args in enumerate(task_args)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results.append(future.result())
                except Exception as e:
                    print(f"  Batch error: {e}")

        all_diseases = [d for batch in batch_results for d in batch]
        print(f"  Collected {len(all_diseases)} diseases from {len(batch_results)} batches")

    data = {"diseases": all_diseases, "crop": crop, "generated_date": today_iso()}

    n_diseases = len(data.get("diseases", []))
    n_conflicts = sum(len(d.get("conflicts", [])) for d in data.get("diseases", []))
    print(f"  Registry: {n_diseases} diseases, {n_conflicts} conflicts ({time.time()-t0:.0f}s)")
    return data


# ─── Pipeline Orchestrator ──────────────────────────────────────────────────

STAGES = ["discovery", "extraction", "reconciliation"]


def run_internet_pipeline(
    crop: str,
    disease_names: list[str] | None = None,
    quick: bool = False,
    resume_from: str | None = None,
) -> dict:
    """Run the internet pipeline: web discovery → extraction → reconciliation.

    Args:
        crop: Crop name (e.g., "soybean")
        disease_names: Known disease names for targeted discovery (optional)
        quick: Quick mode (fewer sources, shorter timeouts)
        resume_from: Resume from a specific stage

    Returns:
        Final registry dict with diseases, crop, generated_date
    """
    print(f"\n{'='*60}")
    print(f"INTERNET PIPELINE — {crop.upper()} ({'QUICK' if quick else 'FULL'})")
    print(f"{'='*60}")

    output_dir = get_crop_dir(crop)

    start_idx = 0
    if resume_from:
        if resume_from not in STAGES:
            print(f"ERROR: Unknown stage '{resume_from}'. Valid: {STAGES}")
            return {}
        start_idx = STAGES.index(resume_from)
        print(f"  Resuming from stage: {resume_from}")

    # Stage 1: Discovery
    if start_idx <= 0:
        if disease_names:
            discovery = _run_targeted_discovery(crop, disease_names, quick=quick)
        else:
            discovery = _run_discovery(crop, quick=quick)
        save_json("discovery_results.json", discovery, output_dir=output_dir)
    else:
        print("\n  Loading cached discovery_results.json...")
        discovery = load_json("discovery_results.json", output_dir=output_dir)

    sources = discovery.get("candidate_sources", [])
    if not sources:
        print("ERROR: No sources found. Cannot proceed.")
        return {}

    # Stage 2: Extraction
    if start_idx <= 1:
        extractions = _run_extraction(sources, quick=quick)
        save_json("raw_extractions.json", extractions, output_dir=output_dir)
    else:
        print("\n  Loading cached raw_extractions.json...")
        extractions = load_json("raw_extractions.json", output_dir=output_dir)

    # Stage 3: Reconciliation
    if start_idx <= 2:
        registry = _run_reconciliation(extractions, crop)

        # When input list provided: filter + rename to input names + dedup
        # When free-form: just normalize names and dedup
        if disease_names:
            registry = _filter_to_input_diseases(registry, disease_names)
        else:
            registry["diseases"] = _normalize_disease_names(registry.get("diseases", []))

        save_json("final_registry.json", registry, output_dir=output_dir)
        save_file("registry.md", registry_to_markdown(registry), output_dir=output_dir)
    else:
        print("\n  Loading cached final_registry.json...")
        registry = load_json("final_registry.json", output_dir=output_dir)

    # Write xlsx
    xlsx_path = str(output_dir / "internet.xlsx")
    write_enriched_xlsx(registry, None, xlsx_path)

    return registry
