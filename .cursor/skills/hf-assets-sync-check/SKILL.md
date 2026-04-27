---
name: hf-assets-sync-check
description: Verify synchronization between local AVATAR artifacts and Hugging Face Hub assets, including missing files, revision mismatches, and upload/download actions. Use when working with check_hf_assets.py, model artifacts, or Hub publishing workflows.
---

# HF Assets Sync Check

## Purpose

Ensure local artifacts and Hugging Face Hub repositories are aligned.

## Workflow

1. Inspect expected assets from project config/readme/script defaults.
2. Run `scripts/check_hf_assets.py` (or equivalent checks) and collect results.
3. Classify differences as:
   - local-only
   - hub-only
   - version/revision mismatch
   - corrupted or incomplete artifacts
4. Propose exact sync actions (upload, download, or skip with reason).

## Reliability Rules

- Never assume latest revision is correct without verification.
- Distinguish intentional drift from accidental drift.
- For large files, include checksum/size comparison when available.
- Flag gated/private access blockers explicitly.

## Output Format

- Status summary by repository.
- Table-like bullet list of mismatched artifacts.
- Minimal command list to converge local and remote state safely.
- Risks section (overwrite risk, stale checkpoint risk, bandwidth cost).
