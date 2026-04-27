---
name: dataset-lineage-map
description: Map end-to-end dataset lineage from raw media through preprocessing, manifests, and model inputs to detect stale or undocumented transforms. Use when debugging data provenance, reproducibility gaps, or split inconsistencies.
---

# Dataset Lineage Map

## Purpose

Provide a traceable data-flow map across the AVATAR pipeline.

## Mapping Scope

1. Source inputs (raw media and metadata origins).
2. Preprocessing scripts and transformation steps.
3. Intermediate artifacts and generated files.
4. Manifest generation logic and split assignment rules.
5. Training-time loaders and feature extraction behavior.
6. Eval-time differences from training data flow.

## Quality Checks

- Identify undocumented transformations.
- Identify stale intermediates still referenced by configs.
- Identify non-deterministic or seed-sensitive steps.
- Identify places where train and eval pipelines diverge unintentionally.

## Output Format

- Ordered lineage stages with file/script references.
- Blockers and ambiguity list.
- Suggested fixes for provenance and reproducibility.
- Optional short "single source of truth" recommendation.
