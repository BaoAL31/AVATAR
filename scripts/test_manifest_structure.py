#!/usr/bin/env python3
"""Structural sanity checks for USR manifest CSVs.

Checks:
- row format: tag, rel_video, frame_count, token_ids
- rel_video extension policy (default: .avi only)
- positive frame counts (optionally allow zero)
- token id parseability and basic distribution stats
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Set


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate USR manifest structure.")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("/home/hoangbng/AVATAR/AVATAR/data/val_manifest.csv"),
    )
    ap.add_argument(
        "--allow-ext",
        type=str,
        default=".avi",
        help="Comma-separated allowed rel_video extensions (default: .avi).",
    )
    ap.add_argument(
        "--allow-zero-frames",
        action="store_true",
        help="Allow frame_count=0 rows.",
    )
    ap.add_argument(
        "--show-bad-limit",
        type=int,
        default=20,
        help="Print up to N failing rows per category.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    allowed_exts = {
        x.strip().lower() if x.strip().startswith(".") else f".{x.strip().lower()}"
        for x in args.allow_ext.split(",")
        if x.strip()
    }

    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}")
        return 1

    n_rows = 0
    tag_counts: Counter[str] = Counter()
    bad_cols: List[str] = []
    bad_ext: List[str] = []
    bad_fc: List[str] = []
    bad_ids: List[str] = []
    empty_ids: List[str] = []
    id_1_only = 0
    id_total = 0
    id_unk_1 = 0

    with manifest.open("r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s:
                continue
            n_rows += 1
            parts = s.split(",", 3)
            if len(parts) != 4:
                bad_cols.append(f"L{ln}: {s}")
                continue

            tag, rel_video, frame_count_s, ids_s = (
                parts[0].strip(),
                parts[1].strip().replace("\\", "/"),
                parts[2].strip(),
                parts[3].strip(),
            )
            tag_counts[tag] += 1

            ext = Path(rel_video).suffix.lower()
            if allowed_exts and ext not in allowed_exts:
                bad_ext.append(f"L{ln}: {rel_video}")

            try:
                fc = int(frame_count_s)
            except ValueError:
                bad_fc.append(f"L{ln}: non-int frame_count={frame_count_s}")
                fc = -1
            if (not args.allow_zero_frames and fc <= 0) or (args.allow_zero_frames and fc < 0):
                bad_fc.append(f"L{ln}: frame_count={fc}")

            if not ids_s:
                empty_ids.append(f"L{ln}: empty label ids")
                continue

            ids_ok = True
            ids = []
            for tok in ids_s.split():
                try:
                    ids.append(int(tok))
                except ValueError:
                    ids_ok = False
                    break
            if not ids_ok:
                bad_ids.append(f"L{ln}: non-int ids: {ids_s[:120]}")
                continue
            if not ids:
                empty_ids.append(f"L{ln}: empty ids after parse")
                continue

            id_total += len(ids)
            id_unk_1 += sum(1 for x in ids if x == 1)
            if all(x == 1 for x in ids):
                id_1_only += 1

    print(f"Manifest: {manifest}")
    print(f"Rows: {n_rows}")
    print(f"Tags: {dict(sorted(tag_counts.items()))}")
    print(f"Allowed extensions: {sorted(allowed_exts)}")
    if id_total > 0:
        print(f"Token id stats: total={id_total} id==1={id_unk_1} ({100.0*id_unk_1/id_total:.2f}%)")
    print(f"Rows with all ids==1: {id_1_only}")

    def show(name: str, rows: List[str]) -> None:
        print(f"{name}: {len(rows)}")
        for x in rows[: args.show_bad_limit]:
            print(f"  {x}")
        if len(rows) > args.show_bad_limit:
            print(f"  ... and {len(rows) - args.show_bad_limit} more")

    show("Bad column count", bad_cols)
    show("Bad extension", bad_ext)
    show("Bad frame_count", bad_fc)
    show("Bad token ids", bad_ids)
    show("Empty ids", empty_ids)

    fail = any([bad_cols, bad_ext, bad_fc, bad_ids, empty_ids])
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

