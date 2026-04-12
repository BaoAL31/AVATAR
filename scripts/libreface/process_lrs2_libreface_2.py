import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration (clips live under HF staging shards LRS2_00 .. LRS2_14)
INPUT_DIR = "/home/hoangbng/lrs2_hf_staging"
OUTPUT_DIR = "/home/hoangbng/lrs2/libreface_out"
BATCH_SIZE = 256
NUM_WORKERS = 4
WEIGHTS_DOWNLOAD_DIR = "/home/hoangbng/lrs2/libreface_weights"
TEMP_DIR = "/home/hoangbng/lrs2/libreface_out/temp"
DEVICE = "cuda:0"

# Subdirectories under INPUT_DIR to ignore when using nested layout (not clip folders).
SKIP_SUBDIRS = frozenset({"libreface_out", "temp", "libreface_weights"})

# Parallelization Configuration
# These variables allow you to run multiple copies of this script simultaneously
# to process the dataset faster by splitting the workload.
#
# HOW TO USE:
# 1. Decide how many parallel processes you want to run (e.g., 4).
# 2. Set TOTAL_INSTANCES to that number (e.g., 4) in ALL terminal sessions.
# 3. Open 4 separate terminals.
# 4. In Terminal 1, set INSTANCE_ID = 0 and run the script.
# 5. In Terminal 2, set INSTANCE_ID = 1 and run the script.
# 6. In Terminal 3, set INSTANCE_ID = 2 and run the script.
# 7. In Terminal 4, set INSTANCE_ID = 3 and run the script.
#
# Each instance will automatically process a unique subset of videos
# (e.g., Instance 0 gets videos 0, 4, 8... Instance 1 gets videos 1, 5, 9...).
# They safely share the same output directory and resume state.
INSTANCE_ID = 2
TOTAL_INSTANCES = 4


def _results_to_npz_dict(results) -> dict:
    """LibreFace may return a DataFrame; np.savez needs ndarray kwargs."""
    import pandas as pd

    if isinstance(results, pd.DataFrame):
        return {str(c): results[c].to_numpy() for c in results.columns}
    if isinstance(results, dict):
        return {str(k): np.asarray(v) for k, v in results.items()}
    raise TypeError(
        f"Unexpected get_facial_attributes return type {type(results).__name__!r}; "
        "expected DataFrame or dict"
    )


def discover_mp4_videos(root: Path) -> tuple[list[Path], str]:
    """
    Find all clip MP4 paths.

    - Flat: root/*.mp4
    - Sharded (HF staging): root/LRS2_XX/<stem>/<stem>.mp4
    - Nested (grouped clips): root/<stem>/<stem>.mp4
    """
    flat = sorted(root.glob("*.mp4"))
    if flat:
        return flat, "flat"

    if not root.is_dir():
        return [], "nested"

    shard_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"LRS2_\d{2}", p.name)
    )
    if shard_dirs:
        nested: list[Path] = []
        for shard in shard_dirs:
            for d in sorted(p for p in shard.iterdir() if p.is_dir()):
                if d.name.startswith("."):
                    continue
                if d.name in SKIP_SUBDIRS:
                    continue
                candidate = d / f"{d.name}.mp4"
                if candidate.is_file():
                    nested.append(candidate)
        return sorted(nested), "sharded"

    nested = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("."):
            continue
        if d.name in SKIP_SUBDIRS:
            continue
        candidate = d / f"{d.name}.mp4"
        if candidate.is_file():
            nested.append(candidate)

    return sorted(nested), "nested"


def process_video(
    video_path: Path,
    output_path: Path,
    *,
    weights_download_dir: str,
    temp_dir: str,
    device: str,
) -> bool:
    """Process a single video file with LibreFace and save results to NPZ."""
    import libreface

    try:
        results = libreface.get_facial_attributes(
            str(video_path),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            weights_download_dir=weights_download_dir,
            temp_dir=temp_dir,
            device=device,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **_results_to_npz_dict(results))
        return True
    except Exception as e:
        print(f"\nError processing {video_path.name}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LibreFace NPZ export for LRS2 (flat, sharded LRS2_XX/*, or stem/stem.mp4)."
    )
    ap.add_argument("--input-dir", type=Path, default=Path(INPUT_DIR))
    ap.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    ap.add_argument("--weights-dir", type=Path, default=Path(WEIGHTS_DOWNLOAD_DIR))
    ap.add_argument("--temp-dir", type=Path, default=Path(TEMP_DIR))
    ap.add_argument("--device", type=str, default=DEVICE)
    args = ap.parse_args()

    try:
        import libreface  # noqa: F401
    except ImportError:
        raise SystemExit("libreface not found. Please install it: pip install libreface")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    weights_dir = str(args.weights_dir.resolve())
    temp_dir = str(args.temp_dir.resolve())
    device = args.device

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = {f.stem + ".mp4" for f in output_dir.glob("*.npz")}
    all_videos, layout = discover_mp4_videos(input_dir)
    videos_to_process = [v for v in all_videos if v.name not in processed]
    videos_to_process = videos_to_process[INSTANCE_ID::TOTAL_INSTANCES]

    print(f"Instance {INSTANCE_ID}/{TOTAL_INSTANCES}")
    print(f"Input layout: {layout} (under {input_dir})")
    print(f"Total videos found: {len(all_videos)}")
    print(f"Already processed: {len(processed)}")
    print(f"Assigned to this instance: {len(videos_to_process)}")

    for video_path in tqdm(videos_to_process, desc=f"Processing (Instance {INSTANCE_ID})"):
        output_path = output_dir / f"{video_path.stem}.npz"
        try:
            process_video(
                video_path,
                output_path,
                weights_download_dir=weights_dir,
                temp_dir=temp_dir,
                device=device,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Run again to resume.")
            break
        except Exception as e:
            print(f"\nFailed to process {video_path.name}: {e}")

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
