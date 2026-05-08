"""
pathome_kb/
===========
Phase 0 of the Pathome pipeline — provenance-tracked knowledge-base
construction. Ports the SAGE/disease_registry internet track to PathomeDB:

    discovery (claude -p WebSearch)  →  per-source extraction (LLM with
    verbatim quotes)  →  reconciliation across sources (per-field merge,
    citations preserved)  →  PathomeDB SymptomProfile JSON.

Source of disease names is the (NormCrop, NormDisease) pairs in
``BugWood_Diseases_usable.csv`` (484 classes). The pipeline groups by
crop, runs the internet track per crop (so the discovery prompt can
focus the search on one crop's disease catalogue), and then merges the
per-crop registries into a single ``symptoms_seed.json`` consumable by
``pathome.SymptomLibrary.load``.

CLI entry point: ``python -m pathome_kb --csv BugWood_Diseases_usable.csv``.
"""
