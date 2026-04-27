---
name: checkpoint-health-check
description: Inspect model checkpoint files for structural integrity and runtime compatibility before training or inference. Use when handling .pth files, loading failures, key mismatch errors, or migration across model versions.
---

# Checkpoint Health Check

## Purpose

Detect checkpoint issues early and prevent expensive failed runs.

## Core Checks

1. File readability and deserialization success.
2. Expected top-level keys exist (`state_dict`, optimizer, scheduler, metadata).
3. Tensor shapes are compatible with current model definition.
4. Missing/unexpected keys are summarized by module.
5. Precision and device assumptions are safe for target runtime.
6. Linked assets (tokenizer/LM/head) are version-compatible.

## Diagnostics

- Distinguish harmless partial-load differences from fatal mismatches.
- Highlight likely cause (architecture change, renamed modules, adapter mismatch).
- Provide safe options: strict load, non-strict load, key remap, or retrain.

## Output

- Health verdict (`healthy`, `loadable-with-warnings`, `broken`).
- Concise mismatch report with counts and representative keys.
- Recommended next action with lowest risk path.
