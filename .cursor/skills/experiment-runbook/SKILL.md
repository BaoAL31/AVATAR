---
name: experiment-runbook
description: Standardize AVATAR training and evaluation runs into a reproducible runbook with launch commands, expected checkpoints, and failure triage. Use when the user starts experiments, compares runs, or asks for reproducibility.
---

# Experiment Runbook

## Purpose

Make each experiment reproducible, comparable, and easy to resume.

## Runbook Steps

1. Define experiment ID, goal, hypothesis, and success metric.
2. Record dataset/manifests and exact config paths.
3. Record environment assumptions (GPU, memory, package versions).
4. Generate canonical launch command and output directory.
5. Define checkpoint cadence and eval cadence.
6. Define stop criteria and early-failure signals.
7. Define resume procedure and artifact retention rules.

## Logging Standard

Always capture:

- git commit SHA (if repo is available),
- config snapshot,
- key metrics per eval point,
- best checkpoint path,
- final conclusion against hypothesis.

## Output Template

- `goal`
- `setup`
- `command`
- `monitoring`
- `failure-signatures`
- `resume-plan`
- `done-criteria`
