---
name: manifest-audit
description: Audit AVATAR manifest CSV files for integrity, split hygiene, and training readiness. Use when the user mentions train_manifest.csv, val_manifest.csv, data quality checks, schema validation, duplicates, leakage, or manifest debugging.
---

# Manifest Audit

## Purpose

Validate dataset manifests before training or release.

## Checklist

1. Confirm required columns exist and are consistently typed.
2. Detect missing/invalid paths and unreadable media references.
3. Detect duplicate sample IDs or duplicated file paths.
4. Flag outlier durations (too short, too long, or zero-length).
5. Check train/val split leakage by shared IDs, clips, or speakers.
6. Report class or speaker imbalance if labels are present.
7. Flag malformed rows, delimiter issues, and encoding problems.

## Severity Model

- `critical`: likely to break training/eval or invalidate metrics.
- `warning`: likely to degrade quality or create unstable training.
- `info`: cleanup opportunities with low immediate risk.

## Reporting Format

Return:

1. Executive result (`pass` / `pass-with-warnings` / `fail`).
2. Counts by severity.
3. Top findings with concrete row examples.
4. A short remediation plan with safe next commands.

## Guardrails

- Do not rewrite manifests unless the user asks.
- Prefer deterministic checks over heuristics.
- When uncertain, state assumptions and exact fields inspected.
