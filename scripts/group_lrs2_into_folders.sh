#!/usr/bin/env bash
# Group flat LRS2-style files into one folder per clip stem:
#   stem.mp4, stem.json, stem.txt, stem.wav  ->  stem/
#
# Usage:
#   ./group_lrs2_into_folders.sh /path/to/lrs2_webdataset
#   DRY_RUN=1 ./group_lrs2_into_folders.sh .   # print moves only
#
# Only touches files directly in DIR (not recursive). Skips if stem/ already exists
# and is not empty (unless FORCE=1).

set -euo pipefail

DIR="${1:-.}"
if [[ ! -d "$DIR" ]]; then
  echo "Not a directory: $DIR" >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

cd "$DIR"

# Unique stems from any of the four extensions (flat dir only).
stems=$(
  shopt -s nullglob
  for ext in mp4 json txt wav; do
    for f in *."$ext"; do
      [[ -f "$f" ]] || continue
      printf '%s\n' "${f%.$ext}"
    done
  done | sort -u
)

while IFS= read -r stem; do
  [[ -n "$stem" ]] || continue

  target="$stem"
  if [[ -e "$target" && ! -d "$target" ]]; then
    echo "Skip $stem: $target exists and is not a directory" >&2
    continue
  fi

  if [[ -d "$target" ]] && [[ -n "$(find "$target" -mindepth 1 -maxdepth 1 2>/dev/null | head -1)" ]]; then
    if [[ "$FORCE" != "1" ]]; then
      echo "Skip $stem: directory $target already has contents (set FORCE=1 to override)" >&2
      continue
    fi
  fi

  mkdir -p "$target"

  moved=0
  for ext in mp4 json txt wav; do
    f="${stem}.${ext}"
    if [[ -f "$f" ]]; then
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "would: mv -n $(printf '%q' "$f") $(printf '%q' "$target/")"
      else
        mv -n -- "$f" "$target/"
      fi
      moved=1
    fi
  done

  if [[ "$moved" -eq 0 ]]; then
    rmdir "$target" 2>/dev/null || true
  fi
done <<< "$stems"

echo "Done. Processed directory: $(pwd)"
