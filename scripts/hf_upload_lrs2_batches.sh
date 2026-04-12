#!/usr/bin/env bash
# Upload LRS2-style data in batches: many clip folders per hf upload (same FS hardlinks).
#
# Layout: SRC/<stem>/<stem>.mp4 + json, txt, wav
#
# Usage (from bash, conda env must have `hf` on PATH):
#   export HF_TOKEN=...   # or: hf auth login
#   ./hf_upload_lrs2_batches.sh /path/to/lrs2_webdataset HBaoAL/LRS2 1000 0
#
# Args: SRC_DIR REPO_ID BATCH_SIZE START_BATCH_INDEX
#   START_BATCH_INDEX: resume (0 = from first batch)
#
# Each batch uploads to: data/batch_<NNNNNN>/ on the Hub (path-in-repo).
#
set -euo pipefail

SRC=$(realpath "$1")
REPO=$2
BATCH_SIZE=${3:-1000}
START_BATCH=${4:-0}

if [[ ! -d "$SRC" ]]; then
  echo "Not a directory: $1" >&2
  exit 1
fi

STAGE_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/hf_lrs2_stage.XXXXXX")
cleanup() { rm -rf "$STAGE_ROOT"; }
trap cleanup EXIT

mapfile -t DIRS < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d | sort)
TOTAL=${#DIRS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No subdirectories under $SRC" >&2
  exit 1
fi

echo "Found $TOTAL clip folders under $SRC"
echo "Batch size=$BATCH_SIZE, starting at batch index $START_BATCH"

batch_idx=0
uploaded=0
for ((i = 0; i < TOTAL; i += BATCH_SIZE)); do
  if ((batch_idx < START_BATCH)); then
    ((batch_idx++)) || true
    continue
  fi

  chunk=("${DIRS[@]:i:BATCH_SIZE}")
  n=${#chunk[@]}
  label=$(printf '%06d' "$batch_idx")

  stage="$STAGE_ROOT/batch_$label"
  rm -rf "$stage"
  mkdir -p "$stage"

  echo ""
  echo "=== Batch $label  (folders $((i + 1))..$((i + n)) of $TOTAL) ==="

  for d in "${chunk[@]}"; do
    base=$(basename "$d")
    cp -al "$d" "$stage/$base"
  done

  # path-in-repo: data/batch_000042/<stem>/...
  hf upload "$REPO" "$stage" "data/batch_$label" --repo-type dataset

  ((uploaded++)) || true
  ((batch_idx++)) || true
done

echo ""
echo "Done. Finished at batch index $((batch_idx - 1)); uploaded $uploaded batch(es) in this run (resume from $START_BATCH)."
