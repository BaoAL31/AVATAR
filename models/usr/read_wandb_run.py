import argparse
import csv
import pathlib

import wandb

parser = argparse.ArgumentParser(description="Download WandB run history to CSV.")
parser.add_argument(
    "--run-path",
    default="/baoal31/semi_supervised_usr/runs/5srm9gnh",
    help="WandB run path in the form entity/project/runs/run_id",
)
parser.add_argument(
    "--output-dir",
    default=pathlib.Path("/home/hoangbng/AVATAR/AVATAR/models/usr/checkpoints/lora"),
    type=pathlib.Path,
    help="Base directory where history CSV will be saved (experiment name will be appended).",
)
parser.add_argument(
    "--experiment-name",
    type=str,
    default=None,
    help="Experiment name for organizing output (defaults to run name or run ID).",
)
parser.add_argument(
    "--samples",
    type=int,
    default=1000000,
    help="Maximum number of history samples to download.",
)
parser.add_argument(
    "--every",
    type=int,
    default=1,
    help="Keep every Nth history row in the output CSV.",
)
parser.add_argument(
    "--last",
    type=int,
    default=0,
    help="After downsampling, keep only the last N rows.",
)
parser.add_argument(
    "--max-rows",
    type=int,
    default=0,
    help="Compress history to at most this many rows by downsampling.",
)
parser.add_argument(
    "--pandas",
    action="store_true",
    help="Use pandas if available for CSV export.",
)
args = parser.parse_args()

api = wandb.Api()
run = api.run(args.run_path)

print(run)
print(run.summary)
print(run.config)

# Determine experiment name for output organization
if args.experiment_name:
    experiment_name = args.experiment_name
elif run.name and run.name != run.id:
    experiment_name = run.name
else:
    experiment_name = run.id

# Create output directory with experiment name
output_dir = args.output_dir / experiment_name
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{run.id}.csv"

print(f"Downloading history to {output_path}")
print(f"Experiment: {experiment_name}")

try:
    history = run.history(samples=args.samples)
except TypeError:
    history = run.history(samples=args.samples, pandas=False)

rows = list(history) if not hasattr(history, "to_csv") else None

if hasattr(history, "to_csv"):
    if args.every == 1 and args.last == 0 and args.max_rows == 0:
        history.to_csv(output_path, index=False)
        print(f"Saved full history: {output_path}")
    else:
        rows = list(history)

if rows is not None:
    if not rows:
        print(f"No history rows found for run {run.id}")
    else:
        if args.every > 1:
            rows = [row for idx, row in enumerate(rows) if idx % args.every == 0]
        if args.max_rows > 0 and len(rows) > args.max_rows:
            step = len(rows) // args.max_rows
            rows = rows[::step][:args.max_rows]  # Ensure at most max_rows
        if args.last > 0:
            rows = rows[-args.last:]

        all_keys = []
        key_set = set()
        for row in rows:
            for key in row.keys():
                if key not in key_set:
                    key_set.add(key)
                    all_keys.append(key)

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in all_keys})
        print(f"Saved downsampled history: {output_path}")
