---
name: train-config-lint
description: Lint and sanity-check AVATAR training configuration files for compatibility, resource fit, and unsafe defaults. Use when editing YAML configs in models/usr/conf, preparing training runs, or troubleshooting config-related failures.
---

# Train Config Lint

## Purpose

Catch configuration mistakes before a training run starts.

## Checks

1. Validate required keys and schema consistency.
2. Validate path fields (checkpoints, tokenizer, manifests, outputs).
3. Validate hyperparameter coherence (batch size, LR, warmup, scheduler).
4. Validate memory-fit assumptions for target hardware profiles (for example 8GB).
5. Validate LoRA/module settings against selected backbone.
6. Validate mixed precision and gradient settings for compatibility.
7. Validate resume/fine-tune flags against checkpoint type.

## Failure Prevention

- Flag contradictory options (for example frozen layers with incompatible adapters).
- Flag silently ignored keys due to misspelling or nesting mistakes.
- Flag likely OOM settings and propose conservative alternatives.

## Output

- `ready` / `needs-fixes` verdict.
- Prioritized issue list with exact field paths.
- Suggested patch values for each blocking issue.
