---
name: paper-to-experiment
description: Translate research papers into concrete AVATAR experiments with implementation mapping, ablation matrix, and measurable success criteria. Use when the user shares a paper, asks for replication plans, or requests paper-inspired improvements.
---

# Paper To Experiment

## Purpose

Turn paper claims into executable experiments in this codebase.

## Procedure

1. Extract paper claims, assumptions, and reported gains.
2. Separate core method from optional engineering tricks.
3. Map each claim to existing project modules/config knobs.
4. Identify gaps requiring new code or data preparation.
5. Build a minimal experiment matrix:
   - baseline,
   - claimed method,
   - key ablations,
   - sensitivity checks.
6. Define expected metric deltas and failure criteria.

## Comparison Rules

- Compare against the strongest relevant baseline already in repo.
- Keep one change at a time per ablation unless interaction is the point.
- State confounders (data mismatch, compute budget mismatch, preprocessing differences).

## Deliverable

- Implementation mapping checklist.
- Prioritized experiment table (small, medium, full).
- Risk notes and estimated effort.
