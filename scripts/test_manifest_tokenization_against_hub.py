#!/usr/bin/env python3
"""Validate manifest label ids against Hub transcript text.

For each sampled row:
- Resolve shard repo from tag (lrs2_XX -> {repo_prefix}_XX)
- Download/read <stem>/<stem>.txt from Hub
- Recompute ids using unigram1000 rules (same as build_lrs2_usr_manifest.py)
- Compare with manifest ids and report mismatches
"""

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Check manifest ids against Hub transcript text.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv"),
    )
    ap.add_argument(
        "--units",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/models/usr/utils/labels/unigram1000_units.txt"),
    )
    ap.add_argument("--repo-prefix", type=str, default="HBaoAL/LRS2")
    ap.add_argument(
        "--sample-count",
        type=int,
        default=200,
        help="Rows to check (0 = all).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--revision", type=str, default=None)
    ap.add_argument(
        "--show-limit",
        type=int,
        default=20,
        help="How many mismatch examples to print.",
    )
    return ap.parse_args()


def load_unigram_table(units_path: Path) -> Tuple[Dict[str, int], int]:
    table = {}
    with units_path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            tok, tid = s.rsplit(None, 1)
            table[tok] = int(tid)
    return table, table.get("<unk>", 1)


def words_to_token_strings(text: str) -> List[str]:
    words = re.findall(r"[A-Z0-9']+", text.upper())
    return ["▁" + w for w in words]


def text_to_ids(text: str, table: Dict[str, int], unk_id: int) -> List[int]:
    return [table.get(tok, unk_id) for tok in words_to_token_strings(text)]


def parse_transcript_txt(path: Path) -> Optional[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.lower().startswith("text:"):
                return s.split(":", 1)[1].strip()
    return None


def read_manifest(path: Path) -> List[Tuple[int, str, str, List[int]]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            parts = s.split(",", 3)
            if len(parts) != 4:
                continue
            tag = parts[0].strip()
            rel = parts[1].strip().replace("\\", "/")
            ids = [int(x) for x in parts[3].strip().split()] if parts[3].strip() else []
            rows.append((ln, tag, rel, ids))
    return rows


def repo_from_tag(tag: str, repo_prefix: str) -> str:
    m = re.match(r"^lrs2_(\d{2})$", tag)
    if m:
        return f"{repo_prefix}_{m.group(1)}"
    if tag == "lrs2":
        return repo_prefix
    raise ValueError(f"Unsupported tag format: {tag}")


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    units = args.units.resolve()

    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}")
        return 1
    if not units.is_file():
        print(f"ERROR: units not found: {units}")
        return 1

    table, unk_id = load_unigram_table(units)
    rows = read_manifest(manifest)
    if not rows:
        print(f"ERROR: no rows parsed from {manifest}")
        return 1

    rng = random.Random(args.seed)
    if args.sample_count > 0 and len(rows) > args.sample_count:
        rows = rng.sample(rows, args.sample_count)

    checked = 0
    failed_download = 0
    failed_parse = 0
    mismatches = []

    if hf_hub_download is None:
        print("ERROR: huggingface_hub is not installed in this Python env.")
        print("Use your usr_env (or install with: pip install huggingface_hub).")
        return 1

    for ln, tag, rel, ids_manifest in rows:
        stem = Path(rel).stem
        folder = Path(rel).parent.as_posix()
        txt_rel = f"{folder}/{stem}.txt" if folder != "." else f"{stem}.txt"
        repo_id = repo_from_tag(tag, args.repo_prefix)
        checked += 1

        try:
            txt_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=txt_rel,
                    revision=args.revision,
                )
            )
        except Exception as e:
            failed_download += 1
            mismatches.append(f"L{ln} {repo_id} missing {txt_rel}: {e}")
            continue

        text = parse_transcript_txt(txt_path)
        if not text:
            failed_parse += 1
            mismatches.append(f"L{ln} {repo_id} bad transcript format in {txt_rel}")
            continue

        ids_expected = text_to_ids(text, table, unk_id)
        if ids_expected != ids_manifest:
            preview_t = text[:80] + ("..." if len(text) > 80 else "")
            mismatches.append(
                f"L{ln} {repo_id} {rel}\n"
                f"  text: {preview_t}\n"
                f"  expected[:16]={ids_expected[:16]}\n"
                f"  manifest[:16]={ids_manifest[:16]}"
            )

    print(f"Manifest: {manifest}")
    print(f"Checked rows: {checked}")
    print(f"Download failures: {failed_download}")
    print(f"Transcript parse failures: {failed_parse}")
    print(f"Tokenization mismatches: {len(mismatches) - failed_download - failed_parse}")
    if mismatches:
        print("\nExamples:")
        for x in mismatches[: args.show_limit]:
            print(x)
        if len(mismatches) > args.show_limit:
            print(f"... and {len(mismatches) - args.show_limit} more")
        return 1
    print("PASS: all checked rows match transcript tokenization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

